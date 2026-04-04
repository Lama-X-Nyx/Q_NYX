"""
Context Agent

Determines macro bias and structural direction from higher timeframe.
"""

import pandas as pd
import numpy as np
from typing import Dict
from src.agents.contracts import AgentResult


class ContextAgent:
    """
    Context Agent - Higher Timeframe Bias
    
    Timeframe: 1D or 4H
    Role: Determine strategic bias (bullish/bearish/neutral)
    
    Answers: "What's the higher timeframe bias?"
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
        
        # Thresholds
        self.trend_threshold = 0.02  # 2% separation for trend detection
    
    def analyze(self, df: pd.DataFrame) -> AgentResult:
        """
        Analyze higher timeframe context
        
        Args:
            df: DataFrame for context timeframe (1D or 4H)
        
        Returns:
            AgentResult with context decision
        """
        
        # Get minimum bars from config
        readiness_config = self.config.get('fractal_readiness', {})
        min_bars = readiness_config.get('context_min_bars', 50)
        
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
        
        # Agent is READY - now do logic checks
        # Simple trend detection using SMAs
        close = df['close'].values
        sma_20 = pd.Series(close).rolling(20).mean().values[-1]
        sma_50 = pd.Series(close).rolling(50).mean().values[-1]
        
        if np.isnan(sma_20) or np.isnan(sma_50):
            return AgentResult(
                agent=self.name,
                state='not_ready',
                score=0.0,
                passed=False,
                ready=False,
                blocked_by_readiness=True,
                reason='Insufficient data for SMA calculation',
                metadata={'timeframe': self.timeframe}
            )
        
        # Determine bias (LOGIC - agent is ready)
        diff_pct = (sma_20 - sma_50) / sma_50
        
        if diff_pct > self.trend_threshold:
            state = 'bullish'
            score = min(0.5 + diff_pct * 10, 1.0)  # Scale to [0.5, 1.0]
            passed = True
            reason = f'Bullish bias: SMA20 {diff_pct:.1%} above SMA50'
        elif diff_pct < -self.trend_threshold:
            state = 'bearish'
            score = min(0.5 + abs(diff_pct) * 10, 1.0)
            passed = True
            reason = f'Bearish bias: SMA20 {abs(diff_pct):.1%} below SMA50'
        else:
            state = 'neutral'
            score = 0.4
            passed = False  # Neutral blocks (no clear bias)
            reason = f'Neutral: SMA20 within {self.trend_threshold:.1%} of SMA50'
        
        # Calculate trend strength
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_close = close[-1]
        
        if recent_high > recent_low:
            trend_strength = (current_close - recent_low) / (recent_high - recent_low)
        else:
            trend_strength = 0.5
        
        return AgentResult(
            agent=self.name,
            state=state,
            score=score,
            passed=passed,
            ready=True,  # Agent is READY (has enough data)
            blocked_by_readiness=False,  # Not blocked by readiness (blocked by logic if passed=False)
            reason=reason,
            metadata={
                'timeframe': self.timeframe,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'diff_pct': diff_pct,
                'trend_strength': trend_strength,
                'structure': 'uptrend' if diff_pct > 0 else 'downtrend' if diff_pct < 0 else 'sideways',
                'bars': len(df),
                'min_bars': min_bars
            }
        )


if __name__ == "__main__":
    print("Context Agent Test")
    
    # Create sample data (uptrend)
    dates = pd.date_range('2023-01-01', periods=100, freq='1D')
    prices = np.linspace(40000, 50000, 100)  # Uptrend
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.rand(100) * 1000
    }, index=dates)
    
    # Test agent
    agent = ContextAgent({'mtf': {'timeframes': {'context': '1d'}}})
    result = agent.analyze(df)
    
    print(f"\n✅ Result:")
    print(f"  State: {result.state}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Passed: {result.passed}")
    print(f"  Reason: {result.reason}")
    print(f"  Metadata: {result.metadata}")
    
    # Test downtrend
    df_down = df.copy()
    df_down['close'] = prices[::-1]  # Reverse for downtrend
    result_down = agent.analyze(df_down)
    
    print(f"\n✅ Downtrend Result:")
    print(f"  State: {result_down.state}")
    print(f"  Passed: {result_down.passed}")
