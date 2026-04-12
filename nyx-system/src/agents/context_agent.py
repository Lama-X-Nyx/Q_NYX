"""
Context Agent

Determines macro bias and strategic direction (Intent_1D) from higher timeframe.

P1: Intent_1D = argmax(π_4H · A^k) — HSMM projection instead of SMA heuristic.
Falls back to SMA when no 4H regime result is provided (backward compatible).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from src.agents.contracts import AgentResult


class ContextAgent:
    """
    Context Agent - Higher Timeframe Bias

    Timeframe: 1D
    Role: Compute Intent_1D = argmax(π_4H · A^k), the projected 4H state
          distribution k steps forward.

    Answers: "What's the strategic Intent for today?"
    """

    def __init__(self, config: Dict):
        """
        Initialize Context Agent

        Args:
            config: Configuration dict
        """
        self.config = config
        self.name = 'context'

        # Extract timeframe from config
        mtf_config = config.get('mtf', {})
        timeframes = mtf_config.get('timeframes', {})
        self.timeframe = timeframes.get('context', '1d')

        # SMA threshold (fallback only)
        self.trend_threshold = 0.02

        # Projection horizon k (configurable, default 2 steps forward on 4H)
        strategy = config.get('strategy', {})
        self.projection_k = strategy.get('intent_daily_projection_steps', 2)

    # ------------------------------------------------------------------
    # HSMM projection helpers
    # ------------------------------------------------------------------

    def _compute_intent_from_projection(
        self,
        hsmm_states: Dict[str, float],
        transition_matrix_list: list,
        k: int
    ) -> Tuple[str, np.ndarray]:
        """
        Intent_1D = argmax(π_4H · A^k)

        Handles both the legacy 3-state HSMM and the current 5-state model
        (Trend+, Range, Trend-, Squeeze, Distribution).  For N-state models
        the projected probabilities are collapsed to the 3-class scheme:
          bullish_p  = P(Trend+) + 0.5×P(Squeeze)
          neutral_p  = P(Range)
          bearish_p  = P(Trend-) + P(Distribution)

        Args:
            hsmm_states: dict of state → probability for all N states
            transition_matrix_list: A as nested Python list (N×N)
            k: projection horizon (steps forward)

        Returns:
            (context_state, projected_3)
            context_state ∈ {'bullish', 'neutral', 'bearish'}
            projected_3: np.ndarray shape (3,) [bullish, neutral, bearish]
        """
        A = np.array(transition_matrix_list, dtype=float)
        n = A.shape[0]

        # Build the full N-dimensional initial distribution
        all_states = list(hsmm_states.keys())
        pi_full = np.array([hsmm_states.get(s, 1.0 / n) for s in all_states], dtype=float)
        pi_full = pi_full / pi_full.sum()

        if pi_full.shape[0] != n:
            raise ValueError(
                f'Dimension mismatch: hsmm_states has {len(all_states)} entries '
                f'but transition_matrix is {n}×{n}'
            )

        A_k = np.linalg.matrix_power(A, k)
        projected_full = pi_full @ A_k
        projected_full = projected_full / projected_full.sum()

        # Map N-state projection → 3 semantic classes
        state_idx = {s: i for i, s in enumerate(all_states)}
        bullish_p = projected_full[state_idx['Trend+']] \
                    + 0.5 * projected_full[state_idx.get('Squeeze', -1)] \
                    if 'Squeeze' in state_idx else projected_full[state_idx['Trend+']]
        bearish_p = projected_full[state_idx['Trend-']] \
                    + projected_full[state_idx.get('Distribution', -1)] \
                    if 'Distribution' in state_idx else projected_full[state_idx['Trend-']]
        neutral_p = projected_full[state_idx.get('Range', 0)]

        # Handle out-of-range index from .get() returning -1
        if 'Squeeze' not in state_idx:
            bullish_p = projected_full[state_idx['Trend+']]
        if 'Distribution' not in state_idx:
            bearish_p = projected_full[state_idx['Trend-']]

        projected_3 = np.array([bullish_p, neutral_p, bearish_p])
        projected_3 = projected_3 / projected_3.sum()

        dominant_idx = int(np.argmax(projected_3))
        context_map = {0: 'bullish', 1: 'neutral', 2: 'bearish'}
        return context_map[dominant_idx], projected_3

    # ------------------------------------------------------------------
    # SMA fallback
    # ------------------------------------------------------------------

    def _compute_intent_from_sma(self, df: pd.DataFrame) -> Tuple[str, float, str]:
        """
        Macro bias from price position relative to SMA200 (1D).

        SMA200 is the industry-standard macro trend filter.  Price above SMA200
        = bull market; price below = bear market.  Much more stable than SMA10/30
        which whipsaws on short-term corrections inside a bull year.

        In BTC 2023 (+156%), price stayed above SMA200 for most of the year
        after the February golden cross — this avoids the cascade of false
        SHORT signals that SMA10/30 produced during intra-year pullbacks.

        Returns (state, score, reason)
        """
        close = df['close'].values
        sma_200 = pd.Series(close).rolling(200).mean().values[-1]

        if np.isnan(sma_200):
            return 'neutral', 0.0, 'SMA200 NaN — insufficient data (need 200 bars)'

        current_close = float(close[-1])
        diff_pct = (current_close - sma_200) / sma_200

        if diff_pct > self.trend_threshold:
            state = 'bullish'
            score = min(0.5 + diff_pct * 5, 1.0)
            reason = f'Price {diff_pct:.1%} above SMA200 — macro BULLISH'
        elif diff_pct < -self.trend_threshold:
            state = 'bearish'
            score = min(0.5 + abs(diff_pct) * 5, 1.0)
            reason = f'Price {abs(diff_pct):.1%} below SMA200 — macro BEARISH'
        else:
            state = 'neutral'
            score = 0.4
            reason = f'Price within {self.trend_threshold:.1%} of SMA200 — neutral'

        return state, score, reason

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        regime_4h_result: Optional[AgentResult] = None
    ) -> AgentResult:
        """
        Analyze higher timeframe context.

        If regime_4h_result is provided and contains HSMM data, computes
        Intent_1D = argmax(π_4H · A^k).  Otherwise falls back to SMA heuristic.

        Args:
            df: DataFrame for context timeframe (1D)
            regime_4h_result: AgentResult from RegimeAgent run on 4H data
                              (must contain 'hsmm_states' and 'transition_matrix'
                              in metadata).  Optional — backwards compatible.

        Returns:
            AgentResult with context decision
        """
        # Get minimum bars from config
        readiness_config = self.config.get('fractal_readiness', {})
        min_bars = readiness_config.get('context_min_bars', 50)

        # Readiness check
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

        # ---- Try HSMM projection first --------------------------------
        intent_method = 'sma_heuristic'
        hsmm_meta: Dict = {}
        score: float = 0.0
        passed: bool = False
        reason: str = ''

        if (
            regime_4h_result is not None
            and regime_4h_result.ready
            and 'hsmm_states' in regime_4h_result.metadata
            and 'transition_matrix' in regime_4h_result.metadata
        ):
            hsmm_states = regime_4h_result.metadata['hsmm_states']
            tm_list = regime_4h_result.metadata['transition_matrix']

            try:
                state, projected = self._compute_intent_from_projection(
                    hsmm_states, tm_list, self.projection_k
                )
                score = float(np.max(projected))
                passed = state != 'neutral'
                reason = (
                    f'HSMM projection (k={self.projection_k}): '
                    f'Intent_1D={state} | P={score:.3f}'
                )
                intent_method = 'hsmm_projection'
                hsmm_meta = {
                    'projected_distribution': {
                        'bullish': float(projected[0]),
                        'neutral': float(projected[1]),
                        'bearish': float(projected[2]),
                    },
                    'projection_k': self.projection_k,
                    'source_4h_hsmm_states': hsmm_states,
                }
            except Exception as exc:
                # HSMM projection failed — fall through to SMA
                state = None
                reason = f'HSMM projection failed ({exc!r}), falling back to SMA'
                score = 0.0
                passed = False

        else:
            state = None  # Will be set by SMA below

        # ---- SMA fallback -------------------------------------------
        if state is None:
            state, score, reason = self._compute_intent_from_sma(df)
            passed = state != 'neutral'
            intent_method = 'sma_heuristic'

            # SMA NaN edge case → not_ready
            if score == 0.0 and 'NaN' in reason:
                return AgentResult(
                    agent=self.name,
                    state='not_ready',
                    score=0.0,
                    passed=False,
                    ready=False,
                    blocked_by_readiness=True,
                    reason=reason,
                    metadata={'timeframe': self.timeframe}
                )

        # ---- Trend strength (informational, from 1D data) -----------
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_close = df['close'].values[-1]
        if recent_high > recent_low:
            trend_strength = (current_close - recent_low) / (recent_high - recent_low)
        else:
            trend_strength = 0.5

        meta = {
            'timeframe': self.timeframe,
            'intent_method': intent_method,
            'trend_strength': trend_strength,
            'structure': 'uptrend' if state == 'bullish' else 'downtrend' if state == 'bearish' else 'sideways',
            'bars': len(df),
            'min_bars': min_bars
        }
        meta.update(hsmm_meta)

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
    print("Context Agent Test")

    dates = pd.date_range('2023-01-01', periods=100, freq='1D')
    prices = np.linspace(40000, 50000, 100)

    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.rand(100) * 1000
    }, index=dates)

    agent = ContextAgent({'mtf': {'timeframes': {'context': '1d'}}})
    result = agent.analyze(df)

    print(f"\n✅ SMA fallback:")
    print(f"  State: {result.state}")
    print(f"  Score: {result.score:.4f}")
    print(f"  Method: {result.metadata['intent_method']}")
    print(f"  Reason: {result.reason}")

    # Simulate HSMM regime result
    from src.agents.contracts import AgentResult as AR
    import numpy as np

    fake_regime = AR(
        agent='regime',
        state='trend_plus',
        score=0.75,
        passed=True,
        reason='test',
        metadata={
            'hsmm_states': {'Trend+': 0.70, 'Range': 0.20, 'Trend-': 0.10},
            'transition_matrix': [
                [0.75, 0.125, 0.125],
                [0.125, 0.75, 0.125],
                [0.125, 0.125, 0.75]
            ]
        }
    )

    result2 = agent.analyze(df, regime_4h_result=fake_regime)
    print(f"\n✅ HSMM projection:")
    print(f"  State: {result2.state}")
    print(f"  Score: {result2.score:.4f}")
    print(f"  Method: {result2.metadata['intent_method']}")
    print(f"  Reason: {result2.reason}")
    print(f"  Projected: {result2.metadata['projected_distribution']}")
