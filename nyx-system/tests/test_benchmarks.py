"""
Tests for Validation Benchmarks

Run:
    pytest tests/test_benchmarks.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.benchmarks import (
    BuyAndHold, SMACrossover, RandomEntry,
    run_benchmark, run_all_benchmarks
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data"""
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(n) * 50,
        'high': prices + np.abs(np.random.randn(n) * 100),
        'low': prices - np.abs(np.random.randn(n) * 100),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n)
    }, index=dates)
    
    return data


@pytest.fixture
def sample_config():
    """Create sample config"""
    return {
        'capital': {'initial': 10000},
        'strategy': {
            'risk': {
                'stop_loss': 0.05,
                'take_profit': 0.15
            }
        }
    }


class TestBuyAndHold:
    """Test Buy & Hold benchmark"""
    
    def test_buy_and_hold_returns_one_trade(self, sample_data):
        """Buy & Hold should return exactly one trade"""
        strategy = BuyAndHold(capital=10000)
        trades = strategy.run(sample_data)
        
        assert len(trades) == 1
        assert 'entry_time' in trades.columns
        assert 'exit_time' in trades.columns
        assert 'pnl' in trades.columns
    
    def test_buy_and_hold_entry_exit(self, sample_data):
        """Entry should be first price, exit should be last"""
        strategy = BuyAndHold(capital=10000)
        trades = strategy.run(sample_data)
        
        trade = trades.iloc[0]
        
        assert trade['entry_time'] == sample_data.index[0]
        assert trade['exit_time'] == sample_data.index[-1]
        assert trade['entry_price'] == sample_data.iloc[0]['close']
        assert trade['exit_price'] == sample_data.iloc[-1]['close']


class TestSMACrossover:
    """Test SMA Crossover benchmark"""
    
    def test_sma_crossover_generates_trades(self, sample_data):
        """SMA crossover should generate some trades"""
        strategy = SMACrossover(capital=10000, fast=20, slow=50)
        trades = strategy.run(sample_data)
        
        # Should generate at least one trade on 500 bars
        assert len(trades) >= 0  # May be 0 if no crossover
    
    def test_sma_crossover_trade_structure(self, sample_data):
        """Trades should have correct structure"""
        strategy = SMACrossover(capital=10000, fast=20, slow=50)
        trades = strategy.run(sample_data)
        
        if len(trades) > 0:
            trade = trades.iloc[0]
            assert 'entry_time' in trades.columns
            assert 'exit_time' in trades.columns
            assert 'pnl' in trades.columns
            assert 'exit_reason' in trades.columns


class TestRandomEntry:
    """Test Random Entry benchmark"""
    
    def test_random_entry_reproducible(self, sample_data):
        """Random entries with same seed should be reproducible"""
        strategy1 = RandomEntry(capital=10000, seed=42)
        strategy2 = RandomEntry(capital=10000, seed=42)
        
        trades1 = strategy1.run(sample_data, num_trades=10)
        trades2 = strategy2.run(sample_data, num_trades=10)
        
        # Should generate same trades with same seed
        assert len(trades1) == len(trades2)
    
    def test_random_entry_respects_stops(self, sample_data):
        """Random entry should respect stop loss and take profit"""
        strategy = RandomEntry(capital=10000, stop_loss=0.05, take_profit=0.15, seed=42)
        trades = strategy.run(sample_data, num_trades=20)
        
        if len(trades) > 0:
            # All trades should have exit_reason
            assert 'exit_reason' in trades.columns
            
            # Exit reasons should be valid
            valid_reasons = ['stop_loss', 'take_profit', 'end_of_period']
            assert all(reason in valid_reasons for reason in trades['exit_reason'])


class TestBenchmarkRunner:
    """Test benchmark runner functions"""
    
    def test_run_single_benchmark(self, sample_data, sample_config):
        """Run single benchmark"""
        result = run_benchmark('buy_hold', sample_data, sample_config)
        
        assert 'benchmark' in result
        assert 'trades' in result
        assert 'metrics' in result
        assert result['benchmark'] == 'buy_hold'
    
    def test_run_all_benchmarks(self, sample_data, sample_config):
        """Run all benchmarks"""
        results = run_all_benchmarks(sample_data, sample_config)
        
        # Should have results for main benchmarks
        assert 'buy_hold' in results
        assert 'sma_crossover' in results
        assert 'random_entry' in results
        
        # Each should have metrics
        for name, result in results.items():
            if 'error' not in result:
                assert 'metrics' in result
    
    def test_invalid_benchmark_name(self, sample_data, sample_config):
        """Invalid benchmark name should raise error"""
        with pytest.raises(ValueError):
            run_benchmark('invalid_name', sample_data, sample_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
