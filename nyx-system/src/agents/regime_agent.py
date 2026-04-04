"""
Regime Agent

Detects market regime using HSMM on operational timeframe.
"""

import pandas as pd
import numpy as np
from typing import Dict
from src.agents.contracts import AgentResult
from src.core.hsmm import SemiMarkovHMM


class RegimeAgent:
    """
    Regime Agent - Market Regime Detection
    
    Timeframe: 4H or 1H
    Role: Detect regime (Trend+, Range, Trend-) with HSMM
    
    Answers: "What's the current market regime?"
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Regime Agent
        
        Args:
            config: Configuration dict
        """
        self.config = config
        self.name = 'regime'
        
        # Extract timeframe
        mtf_config = config.get('mtf', {})
        timeframes = mtf_config.get('timeframes', {})
        self.timeframe = timeframes.get('regime', '4h')
        
        # Initialize HSMM — P4a: 5-state (Trend+, Range, Trend-, Squeeze, Distribution)
        self.hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        
        # Thresholds from config
        mtf_conditions = config.get('strategy', {}).get('mtf_conditions', {})
        self.sdc_min = mtf_conditions.get('sdc_min', 5.0)
        self.stability_min = mtf_conditions.get('stability_4h_min', 0.60)
    
    def pretrain(self, df: pd.DataFrame, n_iter: int = 30, tol: float = 1e-4) -> list:
        """
        Train HSMM via Baum-Welch EM on historical data.

        Call once on a training period before running backtests/live.
        Afterwards, forward_backward() uses the learned parameters.

        Args:
            df: Historical OHLCV DataFrame (at regime timeframe)
            n_iter: Maximum EM iterations
            tol: Convergence tolerance on log-likelihood

        Returns:
            List of log-likelihoods per EM iteration
        """
        df_prepared = self._prepare_data(df)
        return self.hsmm.initialize_parameters_with_em(df_prepared, n_iter=n_iter, tol=tol)

    def analyze(self, df: pd.DataFrame, context_state: str = None) -> AgentResult:
        """
        Analyze market regime using HSMM
        
        Args:
            df: DataFrame for regime timeframe (4H or 1H)
            context_state: Optional context bias from Context Agent
        
        Returns:
            AgentResult with regime decision
        """
        
        # Get minimum bars from config
        readiness_config = self.config.get('fractal_readiness', {})
        min_bars = readiness_config.get('regime_min_bars', 100)
        
        # Readiness check - CRITICAL: Check this FIRST
        if len(df) < min_bars:
            return AgentResult(
                agent=self.name,
                state='not_ready',
                score=0.0,
                passed=False,
                ready=False,
                blocked_by_readiness=True,
                reason=f'Insufficient data for HSMM ({len(df)} bars, need {min_bars})',
                metadata={'timeframe': self.timeframe, 'bars': len(df), 'min_bars': min_bars}
            )
        
        # Agent is READY - now do HSMM logic
        
        # Store for metadata
        bars_count = len(df)
        
        # Prepare data for HSMM
        df_prepared = self._prepare_data(df)
        
        # Initialize HSMM with data
        self.hsmm.initialize_parameters(df_prepared)
        
        # Build observations (last 50 bars)
        window_size = min(50, len(df_prepared))
        window = df_prepared.tail(window_size)
        
        observations = []
        for idx in range(len(window)):
            obs = {
                'price': window.iloc[idx]['returns'] if 'returns' in window.columns else 0.0,
                'atr': window.iloc[idx]['atr_14'] if 'atr_14' in window.columns else window.iloc[idx]['close'] * 0.02
            }
            observations.append(obs)
        
        try:
            # Run HSMM forward-backward
            state_probs = self.hsmm.forward_backward(observations)
            
            if len(state_probs) == 0:
                raise ValueError("HSMM returned empty state probabilities")
            
            # Current state probabilities
            current_probs = state_probs[-1]
            
            # Map to states (all active states in model)
            hsmm_states = {
                s: float(current_probs[i])
                for i, s in enumerate(self.hsmm.states)
            }

            # Get most likely state
            state_name = max(hsmm_states.items(), key=lambda x: x[1])[0]
            state_prob = hsmm_states[state_name]
            
            # Compute SdC (Score de Confiance)
            sdc = float(10 * state_prob)

            # Compute stability (persistence probability)
            # Cast to Python float — numpy.float64 propagates to numpy.bool_ in comparisons,
            # which breaks AgentResult's isinstance(passed, bool) contract.
            current_state_idx = int(np.argmax(current_probs))
            stability = float(self.hsmm.transition_matrix[current_state_idx, current_state_idx])
            
            # Map to standardized state names (P4a: 5-state)
            state_mapping = {
                'Trend+':       'trend_plus',
                'Range':        'range',
                'Trend-':       'trend_minus',
                'Squeeze':      'squeeze',
                'Distribution': 'distribution',
                'Liquidation':  'liquidation',
            }
            state = state_mapping.get(state_name, 'range')
            
            # Check conditions
            sdc_passed = sdc > self.sdc_min

            # State-specific stability thresholds (P4a: 5-state model)
            # Squeeze and Distribution are TRANSIENT by design — their prior
            # self-transitions (0.42 / 0.18) are lower than the 3-state threshold.
            # Squeeze is a pre-breakout state → do not require high persistence.
            # Distribution is bearish → block for LONG (no stability relaxation needed).
            if state == 'squeeze':
                stability_min_effective = 0.35
            else:
                stability_min_effective = self.stability_min
            stability_passed = stability >= stability_min_effective
            
            # Context alignment (if provided)
            # P4a: Squeeze/Distribution are ambiguous — don't hard-block on context
            context_aligned = True
            if context_state:
                if context_state == 'bullish' and state in ('trend_minus', 'distribution', 'liquidation'):
                    context_aligned = False
                elif context_state == 'bearish' and state in ('trend_plus',):
                    context_aligned = False
                elif context_state == 'neutral' and state not in ('range', 'squeeze'):
                    context_aligned = False
            
            # Overall pass
            passed = sdc_passed and stability_passed and context_aligned
            
            # Score (normalized SdC)
            score = min(sdc / 10, 1.0)
            
            # Reason
            reasons = []
            if not sdc_passed:
                reasons.append(f'SdC {sdc:.1f} < {self.sdc_min}')
            if not stability_passed:
                reasons.append(f'Stability {stability:.2f} < {self.stability_min}')
            if not context_aligned:
                reasons.append(f'Regime {state} conflicts with Context {context_state}')
            
            if passed:
                reason = f'HSMM {state_name} with SdC={sdc:.1f}, Stability={stability:.2f}'
            else:
                reason = f'Blocked: {", ".join(reasons)}'
            
            return AgentResult(
                agent=self.name,
                state=state,
                score=score,
                passed=passed,
                ready=True,  # Agent is READY (has enough data)
                blocked_by_readiness=False,  # Not blocked by readiness (logic block if passed=False)
                reason=reason,
                metadata={
                    'timeframe': self.timeframe,
                    'sdc': sdc,
                    'stability': stability,
                    'hsmm_states': hsmm_states,
                    # transition_matrix exposed so ContextAgent can compute Intent_1D = argmax(π_4H · A^k)
                    'transition_matrix': self.hsmm.transition_matrix.tolist(),
                    'context_aligned': context_aligned,
                    'bars': len(df),
                    'min_bars': min_bars
                }
            )
            
        except Exception as e:
            # Fallback on error
            return AgentResult(
                agent=self.name,
                state='not_ready',
                score=0.0,
                passed=False,
                ready=False,
                blocked_by_readiness=True,
                reason=f'HSMM error: {str(e)}',
                metadata={'timeframe': self.timeframe, 'error': str(e)}
            )
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for HSMM (add returns, ATR, and moving averages)
        
        CRITICAL: HSMM._label_states_heuristic() requires sma_20 and sma_50
        for proper initialization. Without them, HSMM defaults to Range state,
        causing structural bias in regime detection.
        """
        df_prepared = df.copy()
        
        # Add returns
        if 'returns' not in df_prepared.columns:
            df_prepared['returns'] = df_prepared['close'].pct_change()
        
        # Add ATR
        if 'atr_14' not in df_prepared.columns:
            high = df_prepared['high'].values
            low = df_prepared['low'].values
            close = df_prepared['close'].values
            
            tr = np.maximum(high - low, 
                           np.maximum(np.abs(high - np.roll(close, 1)),
                                     np.abs(low - np.roll(close, 1))))
            tr[0] = high[0] - low[0]
            
            atr = pd.Series(tr).rolling(14).mean().values
            df_prepared['atr_14'] = atr
        
        # Add SMA_20 (required by HSMM heuristic labeling)
        if 'sma_20' not in df_prepared.columns:
            df_prepared['sma_20'] = df_prepared['close'].rolling(window=20).mean()
        
        # Add SMA_50 (required by HSMM heuristic labeling)
        if 'sma_50' not in df_prepared.columns:
            df_prepared['sma_50'] = df_prepared['close'].rolling(window=50).mean()

        # Add ATR_50 — needed for Squeeze detection (P4a)
        if 'atr_50' not in df_prepared.columns and 'atr_14' in df_prepared.columns:
            df_prepared['atr_50'] = df_prepared['atr_14'].rolling(window=50).mean()

        # Add volume_ma20 — needed for Distribution detection (P4a)
        if 'volume_ma20' not in df_prepared.columns and 'volume' in df_prepared.columns:
            df_prepared['volume_ma20'] = df_prepared['volume'].rolling(window=20).mean()

        return df_prepared


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.agents.regime_agent import RegimeAgent
    
    print("Regime Agent Test")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='4H')
    prices = np.linspace(40000, 50000, 200)  # Uptrend
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.rand(200) * 1000
    }, index=dates)
    
    # Test agent
    config = {
        'mtf': {'timeframes': {'regime': '4h'}},
        'strategy': {'mtf_conditions': {'sdc_min': 5.0, 'stability_4h_min': 0.60}}
    }
    
    agent = RegimeAgent(config)
    result = agent.analyze(df, context_state='bullish')
    
    print(f"\n✅ Result:")
    print(f"  State: {result.state}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Passed: {result.passed}")
    print(f"  Reason: {result.reason}")
    print(f"  SdC: {result.metadata.get('sdc', 0):.2f}")
    print(f"  Stability: {result.metadata.get('stability', 0):.2f}")
