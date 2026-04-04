"""
Tests for Out-of-Sample Report

Run:
    pytest tests/test_oos_report.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.oos_report import OOSReport


@pytest.fixture
def sample_config():
    """Sample configuration"""
    return {
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
    """Create sample OHLCV data"""
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


class TestOOSReportBasics:
    """Test basic OOS functionality"""
    
    def test_initialization(self, sample_config):
        """OOSReport should initialize with config"""
        oos = OOSReport(sample_config)
        
        assert oos.oos_split == 0.80
    
    def test_compare_performance(self, sample_config):
        """Should compare IS vs OOS metrics"""
        oos = OOSReport(sample_config)
        
        is_metrics = {
            'total_return': 20.0,
            'sharpe_ratio': 1.5,
            'max_drawdown': -10.0,
            'win_rate': 60.0
        }
        
        oos_metrics = {
            'total_return': 15.0,
            'sharpe_ratio': 1.2,
            'max_drawdown': -12.0,
            'win_rate': 55.0
        }
        
        comparison = oos._compare_performance(is_metrics, oos_metrics)
        
        assert 'total_return' in comparison
        assert comparison['total_return']['is'] == 20.0
        assert comparison['total_return']['oos'] == 15.0
        assert comparison['total_return']['change_pct'] == -25.0


class TestOOSVerdict:
    """Test verdict generation"""
    
    def test_verdict_stable(self, sample_config):
        """Small degradation should be STABLE"""
        oos = OOSReport(sample_config)
        
        comparison = {
            'total_return': {'change_pct': -10.0},
            'sharpe_ratio': {'change_pct': -15.0}
        }
        
        verdict = oos._generate_verdict(comparison)
        
        assert verdict['status'] == 'STABLE'
    
    def test_verdict_warning(self, sample_config):
        """Medium degradation should be WARNING"""
        oos = OOSReport(sample_config)
        
        comparison = {
            'total_return': {'change_pct': -35.0},
            'sharpe_ratio': {'change_pct': -20.0}
        }
        
        verdict = oos._generate_verdict(comparison)
        
        assert verdict['status'] == 'WARNING'
    
    def test_verdict_failed(self, sample_config):
        """Large degradation should be FAILED"""
        oos = OOSReport(sample_config)
        
        comparison = {
            'total_return': {'change_pct': -60.0},
            'sharpe_ratio': {'change_pct': -30.0}
        }
        
        verdict = oos._generate_verdict(comparison)
        
        assert verdict['status'] == 'FAILED'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
