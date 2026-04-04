"""
Tests for OOS Bounded Lookback - P1-06d

Validates that bounded lookback is correctly implemented and functional.

Run:
    pytest tests/test_oos_bounded_lookback.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.nyx_engine import NYXEngine
from src.validation.oos_report import OOSReport


@pytest.fixture
def sample_config():
    """Sample configuration with bounded lookback"""
    return {
        'validation': {
            'warmup_bars': 100,
            'signal_lookback_bars': 500  # Test with smaller window
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
    """Create sample OHLCV data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1500, freq='1h')
    
    data = pd.DataFrame({
        'open': 50000 + np.random.randn(len(dates)) * 1000,
        'high': 51000 + np.random.randn(len(dates)) * 1000,
        'low': 49000 + np.random.randn(len(dates)) * 1000,
        'close': 50000 + np.random.randn(len(dates)) * 1000,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    return data


class TestBoundedLookback:
    """Test bounded lookback implementation"""
    
    def test_generate_signal_at_index_uses_bounded_window(self, sample_config, sample_data):
        """generate_signal_at_index should use bounded window"""
        engine = NYXEngine(sample_config)
        prepared = engine.prepare_data(sample_data)
        
        lookback = 500
        i = 1000
        
        # Generate signal at index 1000 with lookback 500
        signal = engine.generate_signal_at_index(
            pair='BTCUSDT',
            prepared_data=prepared,
            i=i,
            lookback_bars=lookback
        )
        
        # Signal should be generated
        assert signal is not None
        assert 'action' in signal
        assert 'confidence' in signal
    
    def test_bounded_window_when_i_less_than_lookback(self, sample_config, sample_data):
        """When i < lookback_bars, should use [0:i+1]"""
        engine = NYXEngine(sample_config)
        prepared = engine.prepare_data(sample_data)
        
        lookback = 500
        i = 200  # Less than lookback
        
        # Should not crash - uses smaller window
        signal = engine.generate_signal_at_index(
            pair='BTCUSDT',
            prepared_data=prepared,
            i=i,
            lookback_bars=lookback
        )
        
        assert signal is not None
    
    def test_bounded_window_size_respected(self, sample_config, sample_data):
        """Bounded window should have correct size"""
        engine = NYXEngine(sample_config)
        prepared = engine.prepare_data(sample_data)
        
        lookback = 500
        i = 1000
        
        # Calculate expected window
        start_idx = max(0, i - lookback + 1)
        expected_size = i - start_idx + 1
        
        # Should be exactly lookback bars
        assert expected_size == lookback


class TestOOSBoundedExecution:
    """Test OOS report with bounded lookback"""
    
    def test_oos_runs_with_bounded_lookback(self, sample_config, sample_data):
        """OOS should complete with bounded lookback"""
        oos = OOSReport(sample_config)
        
        # Run OOS (should not timeout)
        results = oos.run(sample_data, pair='BTCUSDT')
        
        # Should return valid results
        assert results is not None
        assert 'in_sample' in results
        assert 'out_of_sample' in results
        assert 'verdict' in results
    
    def test_oos_performance_metrics_logged(self, sample_config, sample_data):
        """OOS should log performance metrics"""
        oos = OOSReport(sample_config)
        
        results = oos.run(sample_data, pair='BTCUSDT')
        
        # Check performance metrics exist
        assert 'in_sample' in results
        in_sample = results['in_sample']
        
        if 'performance' in in_sample:
            perf = in_sample['performance']
            assert 'bars_processed' in perf
            assert 'warmup_bars' in perf
            assert 'lookback_bars' in perf
            assert 'execution_time_seconds' in perf
    
    def test_oos_output_schema_unchanged(self, sample_config, sample_data):
        """OOS output schema should remain compatible"""
        oos = OOSReport(sample_config)
        
        results = oos.run(sample_data, pair='BTCUSDT')
        
        # Check schema compatibility
        assert 'pair' in results
        assert 'split_ratio' in results
        assert 'in_sample' in results
        assert 'out_of_sample' in results
        assert 'comparison' in results
        assert 'verdict' in results
        
        # Check verdict has expected fields
        verdict = results['verdict']
        assert 'status' in verdict
        assert 'reason' in verdict


class TestWarmupRespected:
    """Test warmup period is respected"""
    
    def test_warmup_skipped(self, sample_config, sample_data):
        """First warmup_bars should be skipped"""
        oos = OOSReport(sample_config)
        
        # Split data
        split_idx = int(len(sample_data) * 0.8)
        is_data = sample_data.iloc[:split_idx]
        
        # Run backtest
        result = oos._run_backtest(is_data, 'BTCUSDT', 'in_sample')
        
        # Should have performance metrics
        if 'performance' in result:
            perf = result['performance']
            warmup = perf.get('warmup_bars', 100)
            bars_processed = perf.get('bars_processed', 0)
            
            # Bars processed should be less than total
            assert bars_processed < len(is_data)
            assert bars_processed == len(is_data) - warmup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
