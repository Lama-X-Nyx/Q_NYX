"""
Tests for Validation Metrics Library

Run:
    pytest tests/test_metrics.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.metrics import calculate_metrics, compare_metrics, metrics_to_dict


class TestMetricsCalculation:
    """Test metrics calculation"""
    
    def test_empty_trades(self):
        """Empty trades should return zero metrics"""
        empty_df = pd.DataFrame()
        
        metrics = calculate_metrics(empty_df)
        
        assert metrics['total_return'] == 0
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0
    
    def test_single_winning_trade(self):
        """Single winning trade"""
        trades = pd.DataFrame({
            'entry_time': ['2024-01-01'],
            'exit_time': ['2024-01-02'],
            'pnl': [100],
            'pnl_pct': [1.0]
        })
        
        metrics = calculate_metrics(trades, initial_capital=10000)
        
        assert metrics['total_trades'] == 1
        assert metrics['winners'] == 1
        assert metrics['losers'] == 0
        assert metrics['win_rate'] == 100
        assert metrics['total_pnl'] == 100
        assert metrics['total_return'] == 1.0
    
    def test_mixed_trades(self):
        """Mix of winning and losing trades"""
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2024-01-01', periods=5, freq='D'),
            'exit_time': pd.date_range('2024-01-02', periods=5, freq='D'),
            'pnl': [100, -50, 150, -30, 80],
            'pnl_pct': [1.0, -0.5, 1.5, -0.3, 0.8]
        })
        
        metrics = calculate_metrics(trades, initial_capital=10000)
        
        assert metrics['total_trades'] == 5
        assert metrics['winners'] == 3
        assert metrics['losers'] == 2
        assert metrics['win_rate'] == 60
        assert metrics['total_pnl'] == 250
        assert metrics['profit_factor'] > 0
    
    def test_profit_factor(self):
        """Profit factor calculation"""
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2024-01-01', periods=4, freq='D'),
            'exit_time': pd.date_range('2024-01-02', periods=4, freq='D'),
            'pnl': [100, 100, -50, -50],  # 200 profit, 100 loss = PF 2.0
            'pnl_pct': [1.0, 1.0, -0.5, -0.5]
        })
        
        metrics = calculate_metrics(trades)
        
        assert metrics['profit_factor'] == pytest.approx(2.0, abs=0.01)
    
    def test_max_drawdown(self):
        """Max drawdown calculation"""
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2024-01-01', periods=5, freq='D'),
            'exit_time': pd.date_range('2024-01-02', periods=5, freq='D'),
            'pnl': [100, 100, -300, 100, 100],  # DD after -300
            'pnl_pct': [1.0, 1.0, -3.0, 1.0, 1.0]
        })
        
        metrics = calculate_metrics(trades, initial_capital=10000)
        
        assert metrics['max_drawdown'] < 0  # Should be negative
    
    def test_cagr_calculation(self):
        """CAGR should be calculated correctly"""
        # 1 year of trades, 20% return
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2024-01-01', periods=10, freq='30D'),
            'exit_time': pd.date_range('2024-01-02', periods=10, freq='30D'),
            'pnl': [200] * 10,  # 2000 total = 20%
            'pnl_pct': [2.0] * 10
        })
        
        metrics = calculate_metrics(trades, initial_capital=10000)
        
        assert metrics['cagr'] > 0


class TestMetricsComparison:
    """Test metrics comparison"""
    
    def test_compare_two_metrics(self):
        """Compare two metric sets"""
        metrics_a = {'total_return': 10, 'sharpe_ratio': 1.5}
        metrics_b = {'total_return': 15, 'sharpe_ratio': 2.0}
        
        comparison = compare_metrics(metrics_a, metrics_b, labels=('Strategy A', 'Strategy B'))
        
        assert 'Strategy A' in comparison.columns
        assert 'Strategy B' in comparison.columns
        assert 'Difference' in comparison.columns


class TestMetricsFormatting:
    """Test metrics formatting"""
    
    def test_metrics_to_dict_rounds(self):
        """Metrics should be rounded"""
        metrics = {
            'total_return': 12.3456789,
            'sharpe_ratio': 1.987654321,
            'total_trades': 100
        }
        
        rounded = metrics_to_dict(metrics, round_decimals=2)
        
        assert rounded['total_return'] == 12.35
        assert rounded['sharpe_ratio'] == 1.99
        assert rounded['total_trades'] == 100


class TestMetricsEdgeCases:
    """Test edge cases and blown up strategies"""
    
    def test_blown_up_strategy(self):
        """Strategy that destroys all capital"""
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2024-01-01', periods=3, freq='D'),
            'exit_time': pd.date_range('2024-01-02', periods=3, freq='D'),
            'pnl': [-3000, -4000, -5000],  # Total loss > initial capital
            'pnl_pct': [-30.0, -40.0, -50.0]
        })
        
        metrics = calculate_metrics(trades, initial_capital=10000)
        
        assert metrics['is_blown_up'] == True
        assert metrics['final_capital'] <= 0
        assert metrics['cagr'] is None  # Not meaningful when blown up
        assert metrics['max_drawdown'] == -100.0  # Convention
        assert metrics['sharpe_ratio'] is None
        assert metrics['calmar_ratio'] is None
    
    def test_cagr_edge_case_negative_capital(self):
        """CAGR should be None when capital becomes negative"""
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2024-01-01', periods=2, freq='D'),
            'exit_time': pd.date_range('2024-01-02', periods=2, freq='D'),
            'pnl': [-8000, -3000],  # Destroys capital
            'pnl_pct': [-80.0, -30.0]
        })
        
        metrics = calculate_metrics(trades, initial_capital=10000)
        
        assert metrics['cagr'] is None
        assert metrics['is_blown_up'] == True
    
    def test_max_drawdown_capped(self):
        """Max drawdown should be capped at -100%"""
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2024-01-01', periods=1, freq='D'),
            'exit_time': pd.date_range('2024-01-02', periods=1, freq='D'),
            'pnl': [-15000],  # Loss > 100%
            'pnl_pct': [-150.0]
        })
        
        metrics = calculate_metrics(trades, initial_capital=10000)
        
        assert metrics['max_drawdown'] >= -100.0  # Not worse than -100%
        assert metrics['is_blown_up'] == True
    
    def test_absurd_cagr_flagged(self):
        """Absurdly high CAGR should be flagged as None"""
        # This would happen with extreme leverage or errors
        trades = pd.DataFrame({
            'entry_time': ['2024-01-01'],
            'exit_time': ['2024-01-02'],
            'pnl': [100000000],  # Absurd profit
            'pnl_pct': [1000000.0]
        })
        
        metrics = calculate_metrics(trades, initial_capital=10000)
        
        # CAGR > 1000% should be flagged
        if metrics['cagr'] is not None:
            assert abs(metrics['cagr']) <= 1000
    
    def test_normal_strategy_not_blown_up(self):
        """Normal profitable strategy should not be marked blown up"""
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2024-01-01', periods=5, freq='D'),
            'exit_time': pd.date_range('2024-01-02', periods=5, freq='D'),
            'pnl': [100, -50, 150, -30, 80],
            'pnl_pct': [1.0, -0.5, 1.5, -0.3, 0.8]
        })
        
        metrics = calculate_metrics(trades, initial_capital=10000)
        
        assert metrics['is_blown_up'] == False
        assert metrics['final_capital'] > 0
        assert metrics['cagr'] is not None or metrics['cagr'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
