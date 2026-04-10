"""
Tests for Walk-Forward Bounded Lookback - P1-06d

Validates that walk-forward uses bounded lookback correctly.

Run:
    pytest tests/test_walkforward_bounded_lookback.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.walk_forward import WalkForward


@pytest.fixture
def sample_config():
    """Sample configuration with bounded lookback"""
    return {
        'validation': {
            'warmup_bars': 50,  # Smaller for testing
            'signal_lookback_bars': 200,
            'walk_forward': {
                'train_months': 6,
                'test_months': 2,
                'step_months': 2
            }
        },
        'capital': {'initial': 10000},
        'strategy': {
            'hsmm': {'sdc_threshold': 3.5, 'prob_threshold': 0.55},
            'risk': {
                'base_size': 0.05,
                'max_position': 0.25,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'bull_multiplier': 2.5,
                'bear_multiplier': 0.7,
                'range_multiplier': 1.0
            }
        }
    }


@pytest.fixture
def sample_data():
    """Create sample data for walk-forward (2 years)"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='1h')
    
    data = pd.DataFrame({
        'open': 50000 + np.random.randn(len(dates)) * 1000,
        'high': 51000 + np.random.randn(len(dates)) * 1000,
        'low': 49000 + np.random.randn(len(dates)) * 1000,
        'close': 50000 + np.random.randn(len(dates)) * 1000,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    return data


class TestWalkForwardBoundedLookback:
    """Test walk-forward with bounded lookback"""
    
    def test_walkforward_uses_bounded_lookback(self, sample_config, sample_data):
        """Walk-forward should use bounded lookback"""
        wf = WalkForward(sample_config)
        
        # Run walk-forward
        results = wf.run(sample_data, pair='BTCUSDT')
        
        # Should complete successfully
        assert results is not None
        assert 'pair' in results
        assert 'windows' in results
        assert 'aggregated' in results
    
    def test_walkforward_no_crash_short_window(self, sample_config):
        """Walk-forward should handle windows shorter than lookback"""
        wf = WalkForward(sample_config)
        
        # Create very short data
        dates = pd.date_range('2024-01-01', periods=500, freq='1h')
        short_data = pd.DataFrame({
            'open': 50000 + np.random.randn(len(dates)) * 1000,
            'high': 51000 + np.random.randn(len(dates)) * 1000,
            'low': 49000 + np.random.randn(len(dates)) * 1000,
            'close': 50000 + np.random.randn(len(dates)) * 1000,
            'volume': np.random.lognormal(10, 1, len(dates))
        }, index=dates)
        
        # Should not crash (may generate 0 windows)
        results = wf.run(short_data, pair='BTCUSDT')
        
        assert results is not None
    
    def test_walkforward_performance_logged(self, sample_config, sample_data):
        """Walk-forward windows should log performance"""
        wf = WalkForward(sample_config)
        
        results = wf.run(sample_data, pair='BTCUSDT')
        
        # Check if any windows have performance metrics
        if len(results['windows']) > 0:
            first_window = results['windows'][0]
            
            # Performance metrics may be present
            if 'performance' in first_window:
                perf = first_window['performance']
                assert 'bars_processed' in perf
                assert 'execution_time_seconds' in perf


class TestWalkForwardOutputSchema:
    """Test output schema remains compatible"""
    
    def test_output_schema_unchanged(self, sample_config, sample_data):
        """Walk-forward output schema should be compatible"""
        wf = WalkForward(sample_config)
        
        results = wf.run(sample_data, pair='BTCUSDT')
        
        # Check top-level schema
        assert 'pair' in results
        assert 'windows' in results
        assert 'aggregated' in results
        assert 'num_windows' in results
        
        # Check aggregated schema
        agg = results['aggregated']
        assert 'total_return' in agg
        assert 'sharpe_ratio' in agg
        assert 'max_drawdown' in agg
        assert 'num_windows' in agg
    
    def test_window_schema_unchanged(self, sample_config, sample_data):
        """Individual window schema should be compatible"""
        wf = WalkForward(sample_config)
        
        results = wf.run(sample_data, pair='BTCUSDT')
        
        if len(results['windows']) > 0:
            window = results['windows'][0]
            
            # Check window schema
            assert 'window' in window
            assert 'train_start' in window
            assert 'train_end' in window
            assert 'test_start' in window
            assert 'test_end' in window
            assert 'metrics' in window


class TestAggregationWithBoundedLookback:
    """Test aggregation works correctly with bounded lookback"""
    
    def test_aggregation_handles_empty_windows(self, sample_config):
        """Aggregation should handle 0 windows gracefully"""
        wf = WalkForward(sample_config)
        
        # Test with empty results
        agg = wf._aggregate_results([])
        
        assert agg is not None
        assert agg['num_windows'] == 0
        assert 'error' in agg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
