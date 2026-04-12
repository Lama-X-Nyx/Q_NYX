"""
Entry Agent

Validates entry timing on finest timeframe.
"""

import pandas as pd
import numpy as np
from typing import Dict
from src.agents.contracts import AgentResult


class EntryAgent:
    """
    Entry Agent - Entry Timing Validation
    
    Timeframe: 5M
    Role: Validate entry timing and micro-structure
    
    Answers: "Is timing ready for entry?"
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Entry Agent
        
        Args:
            config: Configuration dict
        """
        self.config = config
        self.name = 'entry'
        
        # Extract timeframe
        mtf_config = config.get('mtf', {})
        timeframes = mtf_config.get('timeframes', {})
        self.timeframe = timeframes.get('entry', '5m')
    
    def analyze(self, df: pd.DataFrame, setup_state: "str | None" = None) -> AgentResult:
        """
        Analyze entry timing

        Args:
            df: DataFrame for entry timeframe (5M)
            setup_state: Setup state from Setup Agent

        Returns:
            AgentResult with entry decision
        """
        setup_state = setup_state or ""
        
        # Require minimum data
        if len(df) < 20:
            return AgentResult(
                agent=self.name,
                state='not_ready',
                score=0.0,
                passed=False,
                reason=f'Insufficient data ({len(df)} bars)',
                metadata={'timeframe': self.timeframe}
            )
        
        # If no valid setup, entry is not ready
        if setup_state and setup_state not in ['valid_setup', 'pattern_found']:
            return AgentResult(
                agent=self.name,
                state='no_setup',
                score=0.0,
                passed=False,
                reason=f'Setup not valid ({setup_state})',
                metadata={'timeframe': self.timeframe, 'setup_state': setup_state}
            )
        
        # Simple timing check - momentum alignment
        close = df['close'].values
        
        # Check recent momentum (last 5 bars)
        recent_changes = np.diff(np.asarray(close[-6:]))
        momentum_up = np.sum(recent_changes > 0)
        momentum_down = np.sum(recent_changes < 0)
        
        # Micro-structure check
        if len(close) >= 10:
            ema_5 = pd.Series(close).ewm(span=5).mean().values[-1]
            current_price = close[-1]
            
            # Check if price near EMA
            price_near_ema = abs(current_price - ema_5) / ema_5 < 0.005  # Within 0.5%
        else:
            price_near_ema = False
            ema_5 = close[-1]
        
        # Determine timing quality
        if momentum_up > momentum_down:
            momentum_direction = 'bullish'
            timing_score = 0.5 + (momentum_up / 5) * 0.3
        elif momentum_down > momentum_up:
            momentum_direction = 'bearish'
            timing_score = 0.5 + (momentum_down / 5) * 0.3
        else:
            momentum_direction = 'neutral'
            timing_score = 0.4
        
        # Entry timing states
        if timing_score > 0.55 and price_near_ema:
            state = 'ready'
            score = timing_score
            passed = True
            reason = f'Entry timing ready ({momentum_direction} momentum)'
        elif timing_score > 0.55:
            state = 'ready'
            score = timing_score * 0.9
            passed = timing_score >= 0.6
            reason = f'Timing OK but price not at structure ({momentum_direction})'
        else:
            state = 'early'
            score = timing_score
            passed = False
            reason = f'Entry timing not ready (weak momentum: {momentum_direction})'
        
        return AgentResult(
            agent=self.name,
            state=state,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                'timeframe': self.timeframe,
                'momentum_direction': momentum_direction,
                'momentum_up': int(momentum_up),
                'momentum_down': int(momentum_down),
                'price_near_ema': price_near_ema,
                'timing_score': timing_score
            }
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    print("Entry Agent Test")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=50, freq='5T')
    prices = np.linspace(40000, 40100, 50)  # Slight uptrend
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.random.rand(50) * 1000
    }, index=dates)
    
    # Test
    config = {
        'mtf': {'timeframes': {'entry': '5m'}}
    }
    
    agent = EntryAgent(config)
    result = agent.analyze(df, setup_state='valid_setup')
    
    print(f"\n✅ Result:")
    print(f"  State: {result.state}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Passed: {result.passed}")
    print(f"  Reason: {result.reason}")
    print(f"  Metadata: {result.metadata}")
