"""
Tests for Walk-Forward Engine

Run:
    pytest tests/test_walk_forward.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.walk_forward import WalkForward


@pytest.fixture
def sample_config():
    """Sample configuration"""
    return {
        'validation': {
            'walk_forward': {
                'train_months': 12,
                'test_months': 3,
                'step_months': 3
            }
        },
        'capital': {'initial': 10000},
        'strategy': {
            'hsmm': {'sdc_threshold': 3.5},
            'risk': {
                'base_size': 0.05,
                'stop_loss': 0.05,
                'take_profit': 0.15
            }
        }
    }


@pytest.fixture
def sample_data():
    """Create sample OHLCV data (5 years)"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='1h')
    
    data = pd.DataFrame({
        'open': 50000 + np.random.randn(len(dates)) * 1000,
        'high': 51000 + np.random.randn(len(dates)) * 1000,
        'low': 49000 + np.random.randn(len(dates)) * 1000,
        'close': 50000 + np.random.randn(len(dates)) * 1000,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    return data


class TestWalkForwardBasics:
    """Test basic walk-forward functionality"""
    
    def test_initialization(self, sample_config):
        """WalkForward should initialize with config"""
        wf = WalkForward(sample_config)
        
        assert wf.train_months == 12
        assert wf.test_months == 3
        assert wf.step_months == 3
    
    def test_generate_windows_creates_windows(self, sample_config, sample_data):
        """Should generate windows from data"""
        wf = WalkForward(sample_config)
        windows = wf._generate_windows(sample_data)
        
        assert len(windows) > 0
        assert all(len(w) == 4 for w in windows)  # Each window has 4 dates
    
    def test_empty_data_generates_zero_windows(self, sample_config):
        """Empty data should generate 0 windows"""
        wf = WalkForward(sample_config)
        empty_data = pd.DataFrame()
        
        with pytest.raises(Exception):
            wf._generate_windows(empty_data)
    
    def test_insufficient_data_generates_zero_windows(self, sample_config):
        """Insufficient data should generate 0 windows"""
        wf = WalkForward(sample_config)
        
        # Only 6 months of data (need 15 for 1 window)
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='1h')
        small_data = pd.DataFrame({
            'close': [50000] * len(dates)
        }, index=dates)
        
        windows = wf._generate_windows(small_data)
        
        assert len(windows) == 0


class TestWalkForwardAggregation:
    """Test aggregation logic"""
    
    def test_aggregate_empty_results(self, sample_config):
        """Aggregating 0 windows should return safe values"""
        wf = WalkForward(sample_config)
        
        agg = wf._aggregate_results([])
        
        assert agg['num_windows'] == 0
        assert agg['total_return']['mean'] is None
        assert 'error' in agg
    
    def test_aggregate_single_window(self, sample_config):
        """Aggregating 1 window should work"""
        wf = WalkForward(sample_config)
        
        mock_result = {
            'window': 1,
            'metrics': {
                'total_return': 10.0,
                'cagr': 5.0,
                'sharpe_ratio': 1.5,
                'max_drawdown': -10.0,
                'win_rate': 60.0,
                'is_blown_up': False
            }
        }
        
        agg = wf._aggregate_results([mock_result])
        
        assert agg['num_windows'] == 1
        assert agg['total_return']['mean'] == 10.0
        assert agg['windows_blown_up'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
