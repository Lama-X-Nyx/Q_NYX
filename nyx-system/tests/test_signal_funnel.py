"""
Tests for Signal Funnel Diagnostics

Run:
    pytest tests/test_signal_funnel.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.signal_funnel import SignalFunnel, save_funnel_report


@pytest.fixture
def sample_config():
    """Sample configuration"""
    return {
        'validation': {
            'warmup_bars': 50,
            'signal_lookback_bars': 50
        },
        'strategy': {
            'hsmm': {
                'sdc_threshold': 3.5,
                'prob_threshold': 0.55
            },
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
    """Create sample data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    
    data = pd.DataFrame({
        'open': 50000 + np.random.randn(len(dates)) * 1000,
        'high': 51000 + np.random.randn(len(dates)) * 1000,
        'low': 49000 + np.random.randn(len(dates)) * 1000,
        'close': 50000 + np.random.randn(len(dates)) * 1000,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    return data


class TestSignalFunnel:
    """Test signal funnel diagnostics"""
    
    def test_funnel_runs(self, sample_config, sample_data):
        """Funnel should run without error"""
        funnel = SignalFunnel(sample_config)
        report = funnel.run(sample_data, pair='BTCUSDT')
        
        assert report is not None
        assert 'pair' in report
        assert report['pair'] == 'BTCUSDT'
    
    def test_funnel_counters(self, sample_config, sample_data):
        """Funnel should have all required counters"""
        funnel = SignalFunnel(sample_config)
        report = funnel.run(sample_data, pair='BTCUSDT')
        
        counters = report['counters']
        
        assert 'bars_analyzed' in counters
        assert 'warmup_skipped' in counters
        assert 'raw_opportunities' in counters
        assert 'hsmm_passed' in counters
        assert 'smc_passed' in counters
        assert 'confidence_passed' in counters
        assert 'macro_passed' in counters
        assert 'risk_passed' in counters
        assert 'trades_executed' in counters
    
    def test_funnel_pass_rates(self, sample_config, sample_data):
        """Funnel should calculate pass rates"""
        funnel = SignalFunnel(sample_config)
        report = funnel.run(sample_data, pair='BTCUSDT')
        
        pass_rates = report['pass_rates']
        
        assert 'hsmm_rate' in pass_rates
        assert 'smc_rate' in pass_rates
        assert 'confidence_rate' in pass_rates
        assert 'macro_rate' in pass_rates
        assert 'risk_rate' in pass_rates
        assert 'execution_rate' in pass_rates
    
    def test_funnel_rejections(self, sample_config, sample_data):
        """Funnel should track rejection reasons"""
        funnel = SignalFunnel(sample_config)
        report = funnel.run(sample_data, pair='BTCUSDT')
        
        assert 'top_rejections' in report
        assert isinstance(report['top_rejections'], list)
    
    def test_funnel_sides(self, sample_config, sample_data):
        """Funnel should track long/short"""
        funnel = SignalFunnel(sample_config)
        report = funnel.run(sample_data, pair='BTCUSDT')
        
        sides = report['sides']
        
        assert 'long_opportunities' in sides
        assert 'short_opportunities' in sides
        assert 'long_executed' in sides
        assert 'short_executed' in sides
    
    def test_funnel_summary(self, sample_config, sample_data):
        """Funnel should generate text summary"""
        funnel = SignalFunnel(sample_config)
        report = funnel.run(sample_data, pair='BTCUSDT')
        
        assert 'summary' in report
        assert isinstance(report['summary'], str)
        assert 'SIGNAL FUNNEL' in report['summary']
    
    def test_funnel_warmup_respected(self, sample_config, sample_data):
        """Warmup bars should be skipped"""
        funnel = SignalFunnel(sample_config)
        report = funnel.run(sample_data, pair='BTCUSDT')
        
        warmup = sample_config['validation']['warmup_bars']
        bars_analyzed = report['counters']['bars_analyzed']
        raw_opportunities = report['counters']['raw_opportunities']
        
        assert raw_opportunities == bars_analyzed - warmup


class TestSaveFunnelReport:
    """Test funnel report saving"""
    
    def test_save_creates_json(self, sample_config, sample_data, tmp_path):
        """save_funnel_report should create JSON file"""
        funnel = SignalFunnel(sample_config)
        report = funnel.run(sample_data, pair='BTCUSDT')
        
        output_dir = tmp_path / 'funnel'
        save_funnel_report(report, str(output_dir))
        
        json_file = output_dir / 'BTCUSDT_signal_funnel.json'
        assert json_file.exists()
    
    def test_saved_json_valid(self, sample_config, sample_data, tmp_path):
        """Saved JSON should be valid and loadable"""
        import json
        
        funnel = SignalFunnel(sample_config)
        report = funnel.run(sample_data, pair='BTCUSDT')
        
        output_dir = tmp_path / 'funnel'
        save_funnel_report(report, str(output_dir))
        
        json_file = output_dir / 'BTCUSDT_signal_funnel.json'
        
        with open(json_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['pair'] == 'BTCUSDT'
        assert 'counters' in loaded
        assert 'pass_rates' in loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
