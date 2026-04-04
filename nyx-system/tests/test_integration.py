"""
Integration Test - NYXEngine End-to-End

Tests that NYXEngine works end-to-end without errors:
- Loads real data
- Initializes HSMM with DataFrame
- Generates signal
- No crashes

Run:
    pytest tests/test_integration.py -v
"""

import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.nyx_engine import NYXEngine


class TestNYXEngineIntegration:
    """End-to-end integration tests"""
    
    def test_engine_initializes(self):
        """Engine initializes without error"""
        config = {
            'strategy': {
                'hsmm': {'sdc_threshold': 3.5, 'prob_threshold': 0.55}
            },
            'risk': {
                'base_size': 0.05,
                'max_position': 0.25,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'bull_multiplier': 2.5,
                'bear_multiplier': 0.7
            }
        }
        
        engine = NYXEngine(config)
        
        assert engine is not None
        assert hasattr(engine, 'hsmm')
        assert hasattr(engine, 'smc')
        assert hasattr(engine, 'macro')
    
    def test_engine_generates_signal_on_real_data(self):
        """
        CRITICAL: Engine generates signal without crashing
        
        This is the bug GPT found - HSMM gets ndarray instead of DataFrame
        """
        config = {
            'strategy': {
                'hsmm': {'sdc_threshold': 3.5, 'prob_threshold': 0.55}
            },
            'risk': {
                'base_size': 0.05,
                'max_position': 0.25,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'bull_multiplier': 2.5,
                'bear_multiplier': 0.7
            }
        }
        
        engine = NYXEngine(config)
        
        # Create realistic sample data (200 bars minimum for HSMM)
        np.random.seed(42)
        n = 250
        
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        # Generate realistic OHLCV
        close_base = 68000
        returns = np.random.normal(0.0001, 0.01, n).cumsum()
        closes = close_base * (1 + returns)
        
        data = pd.DataFrame({
            'open': closes * (1 + np.random.normal(0, 0.001, n)),
            'high': closes * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': closes * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': closes,
            'volume': np.random.lognormal(10, 0.5, n)
        }, index=dates)
        
        # CRITICAL TEST: This should NOT crash with "ndarray has no attribute iterrows"
        try:
            signal = engine.generate_signal(
                pair='BTCUSDT',
                data=data,
                current_date='2024-01-10'
            )
            
            # If we get here, no crash = PASS
            assert signal is not None
            assert 'action' in signal
            assert signal['action'] in ['BUY', 'SELL', 'HOLD']
            
        except AttributeError as e:
            if 'iterrows' in str(e):
                pytest.fail("HSMM bug still present: ndarray passed instead of DataFrame")
            else:
                raise
    
    def test_signal_structure_complete(self):
        """Signal has all required fields"""
        config = {
            'strategy': {'hsmm': {'sdc_threshold': 3.5}},
            'risk': {
                'base_size': 0.05,
                'max_position': 0.25,
                'stop_loss': 0.05,
                'take_profit': 0.15
            }
        }
        
        engine = NYXEngine(config)
        
        # Sample data
        n = 200
        data = pd.DataFrame({
            'open': np.random.uniform(67000, 69000, n),
            'high': np.random.uniform(68000, 70000, n),
            'low': np.random.uniform(66000, 68000, n),
            'close': np.random.uniform(67000, 69000, n),
            'volume': np.random.lognormal(10, 0.5, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='1h'))
        
        signal = engine.generate_signal('BTCUSDT', data, '2024-01-10')
        
        # Required fields
        required = [
            'action', 'confidence', 'position_size',
            'regime', 'macro_signal', 'macro_strength', 'reasons'
        ]
        
        for field in required:
            assert field in signal, f"Missing field: {field}"
    
    def test_multiple_signals_no_crash(self):
        """Can generate multiple signals sequentially"""
        config = {
            'strategy': {'hsmm': {'sdc_threshold': 3.5}},
            'risk': {'base_size': 0.05, 'max_position': 0.25}
        }
        
        engine = NYXEngine(config)
        
        # Generate data
        n = 200
        data = pd.DataFrame({
            'open': np.random.uniform(67000, 69000, n),
            'high': np.random.uniform(68000, 70000, n),
            'low': np.random.uniform(66000, 68000, n),
            'close': np.random.uniform(67000, 69000, n),
            'volume': np.random.lognormal(10, 0.5, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='1h'))
        
        # Generate 5 signals
        for i in range(5):
            signal = engine.generate_signal('BTCUSDT', data, '2024-01-10')
            assert signal is not None
            assert 'action' in signal


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
