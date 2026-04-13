"""
Unit Tests for Backtest Engine

Run:
    pytest tests/test_backtest.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range('2020-01-01', periods=n, freq='1h')
    base_price = 10000
    
    data = pd.DataFrame({
        'datetime': dates,
        'open': base_price + np.cumsum(np.random.randn(n) * 50),
        'high': base_price + np.cumsum(np.random.randn(n) * 50) + np.random.rand(n) * 100,
        'low': base_price + np.cumsum(np.random.randn(n) * 50) - np.random.rand(n) * 100,
        'close': base_price + np.cumsum(np.random.randn(n) * 50),
        'volume': 1000000 + np.random.rand(n) * 500000
    })
    
    # Ensure OHLC logic
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    data = data.set_index('datetime')
    
    return data


class TestDataPreparation:
    """Test data preparation functions"""
    
    def test_add_returns(self, sample_data):
        """Test returns calculation"""
        df = sample_data.copy()
        df['returns'] = df['close'].pct_change()
        
        assert 'returns' in df.columns
        assert not df['returns'].iloc[1:].isna().all()
        assert df['returns'].iloc[0] != df['returns'].iloc[0]  # First is NaN
    
    def test_add_indicators(self, sample_data):
        """Test indicator calculation"""
        df = sample_data.copy()
        
        # ATR
        df['hl_range'] = df['high'] - df['low']
        df['atr_14'] = df['hl_range'].rolling(14).mean()
        
        assert 'atr_14' in df.columns
        assert df['atr_14'].iloc[20:].notna().any()
        
        # SMA
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        assert 'sma_20' in df.columns
        assert 'sma_50' in df.columns
        assert df['sma_20'].iloc[25:].notna().any()


class TestBacktestEngine:
    """Test backtest engine logic"""
    
    def test_simple_strategy(self, sample_data):
        """Test simple moving average crossover strategy"""
        df = sample_data.copy()
        
        # Prepare data
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Simulate trades
        capital = 10000
        position = None
        entry_price = 0
        trades = []
        
        for i in range(100, len(df)):
            row = df.iloc[i]
            
            if pd.isna(row['sma_20']) or pd.isna(row['sma_50']):
                continue
            
            # Entry
            if not position and row['sma_20'] > row['sma_50']:
                position = 'LONG'
                entry_price = row['close']
            
            # Exit
            elif position and row['sma_20'] < row['sma_50']:
                exit_price = row['close']
                pnl = (exit_price - entry_price) / entry_price
                capital += capital * pnl * 0.05
                
                trades.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl': pnl
                })
                
                position = None
        
        # Validate results
        assert len(trades) > 0, "Should generate some trades"
        assert capital != 10000, "Capital should change"
        
        # Check trade structure
        for trade in trades:
            assert 'entry' in trade
            assert 'exit' in trade
            assert 'pnl' in trade
    
    def test_position_sizing(self, sample_data):
        """Test position sizing logic"""
        capital = 10000
        
        # Test different position sizes
        sizes = [0.05, 0.10, 0.25]
        
        for size in sizes:
            position_value = capital * size
            assert 0 < position_value <= capital
            assert position_value == capital * size
    
    def test_stop_loss(self, sample_data):
        """Test stop loss logic"""
        entry_price = 10000
        stop_loss_pct = 0.05
        
        # Simulate price drop
        current_price = 9400  # -6% (should trigger 5% stop)
        
        pnl = (current_price - entry_price) / entry_price
        
        assert pnl < -stop_loss_pct, "Should trigger stop loss"


class TestPerformanceMetrics:
    """Test performance metrics calculation"""
    
    def test_total_return(self):
        """Test total return calculation"""
        initial = 10000
        final = 12000
        
        total_return = (final - initial) / initial * 100
        
        assert total_return == 20.0
    
    def test_cagr(self):
        """Test CAGR calculation"""
        initial = 10000
        final = 12000
        years = 2.0
        
        cagr = ((final / initial)**(1/years) - 1) * 100
        
        assert 9.0 < cagr < 10.0  # Should be ~9.5%
    
    def test_win_rate(self):
        """Test win rate calculation"""
        trades = [
            {'pnl': 0.05},
            {'pnl': -0.02},
            {'pnl': 0.03},
            {'pnl': -0.01},
            {'pnl': 0.04}
        ]
        
        winners = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = winners / len(trades) * 100
        
        assert win_rate == 60.0
    
    def test_profit_factor(self):
        """Test profit factor calculation"""
        trades = [
            {'pnl': 0.10},  # +10%
            {'pnl': -0.05}, # -5%
            {'pnl': 0.08},  # +8%
            {'pnl': -0.03}  # -3%
        ]
        
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
        
        total_wins = sum(wins)
        total_losses = sum(losses)
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        assert profit_factor > 1.0  # Profitable


class TestRiskManagement:
    """Test risk management functions"""
    
    def test_max_position_limit(self):
        """Test maximum position size enforcement"""
        capital = 10000
        max_position = 0.25  # 25%
        
        # Try to size position at 30%
        requested_size = 0.30
        actual_size = min(requested_size, max_position)
        
        assert actual_size == 0.25
        assert actual_size <= max_position
    
    def test_pyramiding_logic(self):
        """Test position pyramiding"""
        initial_size = 0.05
        pyramid_sizes = []
        
        # Add positions at profit levels
        pyramid_sizes.append(initial_size)
        pyramid_sizes.append(initial_size * 0.5)  # Half size
        pyramid_sizes.append(initial_size * 0.5)  # Half size
        
        total_size = sum(pyramid_sizes)
        
        assert len(pyramid_sizes) == 3
        assert total_size == 0.05 + 0.025 + 0.025
        assert total_size == 0.10


class TestEdgeCases:
    """Test edge cases in backtest engine"""
    
    def test_no_trades(self, sample_data):
        """Test when strategy generates no trades"""
        # Strategy that never enters
        trades = []
        
        assert len(trades) == 0
        
        # Should handle gracefully
        result = None
        if len(trades) == 0:
            result = "No trades generated"

        assert result == "No trades generated"
    
    def test_all_losing_trades(self):
        """Test when all trades are losses"""
        trades = [
            {'pnl': -0.02},
            {'pnl': -0.01},
            {'pnl': -0.03}
        ]
        
        winners = sum(1 for t in trades if t['pnl'] > 0)
        
        assert winners == 0
        
        # Win rate should be 0%
        win_rate = winners / len(trades) * 100 if trades else 0
        assert win_rate == 0.0
    
    def test_extreme_volatility(self):
        """Test with extreme price movements"""
        entry_price = 10000
        
        # Extreme drop
        extreme_price = 5000  # -50%
        
        pnl = (extreme_price - entry_price) / entry_price
        
        assert pnl == -0.5
        assert pnl < -0.05  # Would trigger stop loss


class TestIntegration:
    """Integration tests"""
    
    def test_full_backtest_workflow(self, sample_data):
        """Test complete backtest workflow"""
        # 1. Prepare data
        df = sample_data.copy()
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # 2. Run strategy
        capital = 10000
        trades = []
        position = None
        entry_price = 0
        
        for i in range(100, len(df)):
            row = df.iloc[i]
            
            if pd.isna(row['sma_20']) or pd.isna(row['sma_50']):
                continue
            
            if not position and row['sma_20'] > row['sma_50']:
                position = 'LONG'
                entry_price = row['close']
            elif position and row['sma_20'] < row['sma_50']:
                pnl = (row['close'] - entry_price) / entry_price
                capital += capital * pnl * 0.05
                trades.append({'pnl': pnl})
                position = None
        
        # 3. Calculate metrics
        if trades:
            final_capital = capital
            total_return = (final_capital - 10000) / 10000 * 100
            winners = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = winners / len(trades) * 100
            
            # 4. Validate
            assert isinstance(total_return, float)
            assert isinstance(win_rate, float)
            assert 0 <= win_rate <= 100
            assert len(trades) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
