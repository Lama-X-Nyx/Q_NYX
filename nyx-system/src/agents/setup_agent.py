"""
Setup Agent

Detects and validates SMC patterns on setup timeframe.
"""

import numpy as np
import pandas as pd
from typing import Dict
from src.agents.contracts import AgentResult
from src.core.smc import SMCDetector
from src.core.hsmm import SemiMarkovHMM


class SetupAgent:
    """
    Setup Agent - SMC Pattern Detection

    Timeframe: 15M
    Role: Detect SMC patterns (OB, FVG) and validate alignment via HSMM

    Answers: "Is there a valid SMC setup?"

    Alignment score = P(Trend+) if bullish context, P(Trend-) if bearish context,
    computed by running HSMM Forward-Backward on the 15M DataFrame.
    """

    def __init__(self, config: Dict):
        """
        Initialize Setup Agent

        Args:
            config: Configuration dict
        """
        self.config = config
        self.name = 'setup'

        # Extract timeframe
        mtf_config = config.get('mtf', {})
        timeframes = mtf_config.get('timeframes', {})
        self.timeframe = timeframes.get('setup', '15m')

        # Initialize SMC Detector with explicit scalar parameters
        smc_config = config.get('smc_detector', {})
        self.smc = SMCDetector(
            ob_range_threshold=smc_config.get('ob_range_threshold', 0.015),
            fvg_min_gap=smc_config.get('fvg_min_gap', 0.005),
            liquidity_lookback=smc_config.get('liquidity_lookback', 20)
        )

        # HSMM for real alignment probability — P4b: 6-state (matches RegimeAgent)
        # Squeeze on 15M = pre-breakout compression → partially bullish
        # Distribution on 15M = bearish exhaustion → blocks bullish setups
        self.hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation'])

        # Threshold — calibrated for 5-state model:
        #   3-state base P ≈ 0.33 → was 0.60 (1.8× base)
        #   5-state base P ≈ 0.20 → new default 0.28 (~1.4× base)
        # Combined formula (P(Trend+) + 0.5×P(Squeeze) for bullish) raises
        # effective probability, so threshold can stay modest.
        mtf_conditions = config.get('strategy', {}).get('mtf_conditions', {})
        self.alignment_min = mtf_conditions.get('alignment_15m_min', 0.28)

    def _prepare_data(self, df: pd.DataFrame, df_htf: "pd.DataFrame | None" = None) -> pd.DataFrame:
        """
        Prepare 15M data for HSMM (add returns, ATR, sma_20, sma_50).

        Mirrors RegimeAgent._prepare_data() — HSMM heuristic labeling requires
        sma_20 and sma_50.

        Args:
            df:     OHLCV DataFrame at the setup timeframe (15M).
            df_htf: Optional higher-timeframe (1H) OHLCV DataFrame. When
                    provided, an 'htf_pos' column is added: the normalised
                    distance of the current 15M close from the 1H SMA20.
                    Uses merge_asof to avoid any look-ahead.
        """
        df_prepared = df.copy()

        if 'returns' not in df_prepared.columns:
            df_prepared['returns'] = df_prepared['close'].pct_change()

        if 'atr_14' not in df_prepared.columns:
            high = np.asarray(df_prepared['high'].values)
            low = np.asarray(df_prepared['low'].values)
            close = np.asarray(df_prepared['close'].values)
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )
            tr[0] = high[0] - low[0]
            atr_series: pd.Series = pd.Series(tr).rolling(14).mean()
            df_prepared['atr_14'] = atr_series.values

        if 'sma_20' not in df_prepared.columns:
            df_prepared['sma_20'] = df_prepared['close'].rolling(window=20).mean()

        if 'sma_50' not in df_prepared.columns:
            df_prepared['sma_50'] = df_prepared['close'].rolling(window=50).mean()

        # Required for 5-state Squeeze detection (ATR compression)
        if 'atr_50' not in df_prepared.columns and 'atr_14' in df_prepared.columns:
            df_prepared['atr_50'] = df_prepared['atr_14'].rolling(window=50).mean()

        # Required for 5-state Distribution detection (high volume + range-bound)
        if 'volume_ma20' not in df_prepared.columns and 'volume' in df_prepared.columns:
            df_prepared['volume_ma20'] = df_prepared['volume'].rolling(window=20).mean()

        # Add HTF context feature: (close - htf_sma20) / htf_sma20
        # Uses pd.merge_asof for O(n log n) alignment without look-ahead.
        if df_htf is not None and not df_htf.empty and 'htf_pos' not in df_prepared.columns:
            _htf_close: pd.Series = df_htf['close']
            htf_sma20: pd.Series = _htf_close.rolling(20).mean().rename('_htf_sma20')
            htf_ref = htf_sma20.reset_index()
            htf_ref.columns = ['_ts', '_htf_sma20']
            htf_ref = htf_ref.dropna(subset=['_htf_sma20']).sort_values('_ts')

            cur_ref = df_prepared[['close']].copy().reset_index()
            cur_ref.columns = ['_ts', '_close']
            cur_ref = cur_ref.sort_values('_ts')

            merged = pd.merge_asof(
                cur_ref, htf_ref,
                on='_ts',
                direction='backward'   # strictly use the last completed HTF bar
            )
            merged.index = df_prepared.index
            with np.errstate(invalid='ignore', divide='ignore'):
                htf_pos = (merged['_close'] - merged['_htf_sma20']) / merged['_htf_sma20'].replace(0, np.nan)
            df_prepared['htf_pos'] = htf_pos.values

        return df_prepared

    def pretrain(self, df: pd.DataFrame, n_iter: int = 30, tol: float = 1e-4,
                 df_htf: "pd.DataFrame | None" = None) -> list:
        """
        Train the alignment HSMM via Baum-Welch EM on historical 15M data.

        Call once on a training period before running backtests/live.
        Afterwards, _compute_hsmm_alignment() uses the learned parameters
        and skips re-initialization (fingerprint cache hit).

        Args:
            df:     Historical OHLCV DataFrame (15M timeframe)
            n_iter: Maximum EM iterations
            tol:    Convergence tolerance on log-likelihood
            df_htf: Optional higher-timeframe (1H) DataFrame for htf_pos feature

        Returns:
            List of log-likelihoods per EM iteration
        """
        df_prepared = self._prepare_data(df, df_htf=df_htf)
        return self.hsmm.initialize_parameters_with_em(df_prepared, n_iter=n_iter, tol=tol)

    def _compute_hsmm_alignment(self, df: pd.DataFrame, context_state: str,
                                 df_htf: "pd.DataFrame | None" = None) -> Dict:
        """
        Run HSMM Forward-Backward on 15M data and return alignment probability.

        5-state alignment formula:
          Bullish: P(Trend+) + 0.5×P(Squeeze)
            Squeeze = ATR compression before breakout → partially bullish
          Bearish:  P(Trend-) + P(Distribution)
            Distribution = high-vol bearish exhaustion → reinforces bearish
          Neutral:  max(bullish_score, bearish_score)

        State index lookup uses self.hsmm.state_to_idx — safe for any n-state model.

        Returns:
            dict with keys: alignment (float), p_trend_plus (float),
                            p_trend_minus (float), p_range (float),
                            p_squeeze (float), p_distribution (float), hsmm_ok (bool)
        """
        fallback = {
            'alignment': 0.0,
            'p_trend_plus': 0.0,
            'p_trend_minus': 0.0,
            'p_range': 1.0,
            'p_squeeze': 0.0,
            'p_distribution': 0.0,
            'hsmm_ok': False
        }
        try:
            df_prepared = self._prepare_data(df)
            self.hsmm.initialize_parameters(df_prepared)

            window_size = min(50, len(df_prepared))
            window = df_prepared.tail(window_size)

            observations = []
            for i in range(len(window)):
                row = window.iloc[i]
                obs = {
                    'price': float(row['returns']) if not pd.isna(row['returns']) else 0.0,
                    'atr': (
                        float(row['atr_14'])
                        if 'atr_14' in window.columns and not pd.isna(row['atr_14'])
                        else float(row['close']) * 0.02
                    )
                }
                observations.append(obs)

            state_probs = self.hsmm.forward_backward(observations)

            if len(state_probs) == 0:
                return fallback

            latest = state_probs[-1]

            # Index lookup by state name — robust for 3, 5, or 6-state models
            s2i = self.hsmm.state_to_idx
            p_trend_plus   = float(latest[s2i['Trend+']])
            p_range        = float(latest[s2i['Range']])
            p_trend_minus  = float(latest[s2i['Trend-']])
            p_squeeze      = float(latest[s2i['Squeeze']])      if 'Squeeze'      in s2i else 0.0
            p_distribution = float(latest[s2i['Distribution']]) if 'Distribution' in s2i else 0.0
            p_liquidation  = float(latest[s2i['Liquidation']])  if 'Liquidation'  in s2i else 0.0

            # Liquidation = black swan event on 15M TF → block all entries
            if p_liquidation > 0.5:
                return {
                    'alignment': 0.0, 'p_trend_plus': p_trend_plus,
                    'p_trend_minus': p_trend_minus, 'p_range': p_range,
                    'p_squeeze': p_squeeze, 'p_distribution': p_distribution,
                    'p_liquidation': p_liquidation, 'hsmm_ok': True,
                    'liquidation_block': True
                }

            # Nuanced alignment (6-state P4b):
            # Bullish: Trend+ full credit, Squeeze 50% (pre-breakout), Range 25%
            #          (consolidation in a bull market is acceptable backdrop for LONGs)
            # Bearish: Trend- full credit, Distribution full (exhaustion continuation)
            # Liquidation drains from both sides proportionally.
            bullish_score = p_trend_plus + 0.5 * p_squeeze + 0.25 * p_range
            bearish_score = p_trend_minus + p_distribution

            if context_state == 'bullish':
                alignment = bullish_score
            elif context_state == 'bearish':
                alignment = bearish_score
            else:
                alignment = max(bullish_score, bearish_score)

            return {
                'alignment':       alignment,
                'p_trend_plus':    p_trend_plus,
                'p_trend_minus':   p_trend_minus,
                'p_range':         p_range,
                'p_squeeze':       p_squeeze,
                'p_distribution':  p_distribution,
                'p_liquidation':   p_liquidation,
                'hsmm_ok':         True,
                'liquidation_block': False
            }

        except Exception:
            return fallback

    def analyze(self, df: pd.DataFrame, context_state: str = "",
                df_htf: "pd.DataFrame | None" = None) -> AgentResult:
        """
        Analyze SMC patterns and alignment

        Args:
            df: DataFrame for setup timeframe (15M)
            context_state: Context bias from Context Agent
            df_htf: Higher-timeframe DataFrame (1H) for HTF context feature

        Returns:
            AgentResult with setup decision
        """
        context_state = context_state or ""

        # Get minimum bars from config
        readiness_config = self.config.get('fractal_readiness', {})
        min_bars = readiness_config.get('setup_min_bars', 50)

        # Readiness check - CRITICAL: Check this FIRST
        if len(df) < min_bars:
            return AgentResult(
                agent=self.name,
                state='not_ready',
                score=0.0,
                passed=False,
                ready=False,
                blocked_by_readiness=True,
                reason=f'Insufficient data ({len(df)} bars, need {min_bars})',
                metadata={'timeframe': self.timeframe, 'bars': len(df), 'min_bars': min_bars}
            )

        # Agent is READY - now do SMC logic

        # Detect SMC patterns
        try:
            patterns = self.smc.detect_all(df)
        except Exception as e:
            return AgentResult(
                agent=self.name,
                state='detector_error',
                score=0.0,
                passed=False,
                ready=True,
                blocked_by_readiness=False,
                reason=f'SMCDetector error: {type(e).__name__}: {str(e)}',
                metadata={
                    'timeframe': self.timeframe,
                    'detector_error': True,
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'bars': len(df),
                    'min_bars': min_bars
                }
            )

        # Check pattern existence — v2 includes ChoCH and price-at-zone
        has_bullish = (patterns.get('bullish_ob', False)
                       or patterns.get('bullish_fvg', False)
                       or patterns.get('bullish_choch', False))
        has_bearish = (patterns.get('bearish_ob', False)
                       or patterns.get('bearish_fvg', False)
                       or patterns.get('bearish_choch', False))

        # SMC quality score from aggregate (0-1); falls back to 0 for v1 dicts
        smc_bull_score = float(patterns.get('smc_score_bullish', 0.0))
        smc_bear_score = float(patterns.get('smc_score_bearish', 0.0))

        # No context provided - neutral
        hsmm_result = None
        if not context_state:
            if has_bullish or has_bearish:
                state = 'pattern_found'
                score = 0.5
                passed = False
                reason = 'Pattern found but no Context for alignment'
            else:
                state = 'no_pattern'
                score = 0.0
                passed = False
                reason = 'No SMC patterns detected'
            alignment = score

        # Context provided - compute real HSMM alignment
        elif context_state in ('bullish', 'bearish'):
            hsmm_result = self._compute_hsmm_alignment(df, context_state, df_htf=df_htf)

            # Liquidation detected on 15M → hard block (crash/flash-crash event)
            if hsmm_result.get('liquidation_block', False):
                return AgentResult(
                    agent=self.name,
                    state='liquidation',
                    score=0.0,
                    passed=False,
                    ready=True,
                    blocked_by_readiness=False,
                    reason=f'LIQUIDATION detected on 15M (P={hsmm_result.get("p_liquidation", 0):.2f}) — all entries blocked',
                    metadata={
                        'timeframe': self.timeframe,
                        'liquidation_block': True,
                        'p_liquidation': hsmm_result.get('p_liquidation', 0),
                        'bars': len(df),
                        'min_bars': min_bars,
                    }
                )

            alignment = hsmm_result['alignment']
            score = alignment

            # State label derived from SMC pattern direction (informational)
            if context_state == 'bullish':
                if has_bullish and not has_bearish:
                    state = 'valid_setup'
                elif has_bullish and has_bearish:
                    state = 'mixed_signals'
                else:
                    state = 'misaligned'
                pattern_aligned = has_bullish   # SMC pattern in trade direction required
            else:  # bearish
                if has_bearish and not has_bullish:
                    state = 'valid_setup'
                elif has_bearish and has_bullish:
                    state = 'mixed_signals'
                else:
                    state = 'misaligned'
                pattern_aligned = has_bearish   # SMC pattern in trade direction required

            # Score = HSMM alignment + SMC quality bonus (additive, capped at 1.0).
            # SMC can only LIFT the score (ChoCH, at-zone, OB confluence),
            # never reduce it — prevents filtering valid HSMM setups.
            # Bonus weight 0.15 means: excellent SMC (+0.15) brings a 0.70 HSMM
            # trade to 0.85, just past the 0.84 entry threshold.
            smc_quality = smc_bull_score if context_state == 'bullish' else smc_bear_score
            score = min(alignment + 0.15 * smc_quality, 1.0)

            # Soft penalty only when NO pattern exists in trade direction
            if not pattern_aligned:
                score = score * 0.85

            passed = alignment >= self.alignment_min

            direction_label = 'P(Trend+)' if context_state == 'bullish' else 'P(Trend-)'
            hsmm_tag    = '' if hsmm_result['hsmm_ok'] else ' [hsmm_fallback]'
            pattern_tag = '' if pattern_aligned else ' [no SMC -15%]'
            choch_tag   = ' [ChoCH]' if patterns.get(f'{context_state}_choch') else ''
            zone_tag    = ' [at-zone]' if patterns.get(f'{context_state}_at_zone') else ''
            reason = (
                f'{state} | HSMM={alignment:.3f} SMC={smc_quality:.2f}'
                f' score={score:.3f}{hsmm_tag}{pattern_tag}{choch_tag}{zone_tag}'
            )

        else:  # neutral
            state = 'no_bias'
            score = 0.40
            passed = False
            reason = 'Neutral Context - no directional setup required'
            alignment = 0.40
            hsmm_result = None

        # Determine pattern type
        pattern_type = None
        if patterns.get('bullish_ob'):
            pattern_type = 'bullish_ob'
        elif patterns.get('bullish_fvg'):
            pattern_type = 'bullish_fvg'
        elif patterns.get('bearish_ob'):
            pattern_type = 'bearish_ob'
        elif patterns.get('bearish_fvg'):
            pattern_type = 'bearish_fvg'

        # Build metadata
        meta = {
            'timeframe': self.timeframe,
            'patterns': patterns,
            'pattern_type': pattern_type,
            'alignment': alignment,
            'context_state': context_state,
            'bars': len(df),
            'min_bars': min_bars
        }
        if context_state in ('bullish', 'bearish') and hsmm_result is not None:
            meta.update({
                'hsmm_p_trend_plus':   hsmm_result['p_trend_plus'],
                'hsmm_p_trend_minus':  hsmm_result['p_trend_minus'],
                'hsmm_p_range':        hsmm_result['p_range'],
                'hsmm_p_squeeze':      hsmm_result.get('p_squeeze', 0.0),
                'hsmm_p_distribution': hsmm_result.get('p_distribution', 0.0),
                'hsmm_ok':             hsmm_result['hsmm_ok']
            })

        return AgentResult(
            agent=self.name,
            state=state,
            score=score,
            passed=passed,
            ready=True,
            blocked_by_readiness=False,
            reason=reason,
            metadata=meta
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("Setup Agent Test")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='15T')
    prices = np.linspace(40000, 41000, 200)

    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.rand(200) * 1000
    }, index=dates)

    config = {
        'mtf': {'timeframes': {'setup': '15m'}},
        'strategy': {'mtf_conditions': {'alignment_15m_min': 0.28}}
    }

    agent = SetupAgent(config)
    result = agent.analyze(df, context_state='bullish')

    print(f"\n✅ Result:")
    print(f"  State: {result.state}")
    print(f"  Score: {result.score:.4f}")
    print(f"  Passed: {result.passed}")
    print(f"  Reason: {result.reason}")
    m = result.metadata
    print(f"  HSMM P(Trend+):      {m.get('hsmm_p_trend_plus', 'n/a'):.4f}")
    print(f"  HSMM P(Trend-):      {m.get('hsmm_p_trend_minus', 'n/a'):.4f}")
    print(f"  HSMM P(Range):       {m.get('hsmm_p_range', 'n/a'):.4f}")
    print(f"  HSMM P(Squeeze):     {m.get('hsmm_p_squeeze', 'n/a'):.4f}")
    print(f"  HSMM P(Distribution):{m.get('hsmm_p_distribution', 'n/a'):.4f}")
