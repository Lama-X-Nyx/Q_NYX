"""
Benchmark Strategies (Partial Implementation - Sprint 1)

Simple strategies to compare NYX against.
All benchmarks use the same execution model (fees, slippage).

IMPLEMENTED (Sprint 1):
1. Buy & Hold ✅
2. SMA Crossover (simple trend) ✅
3. Random Entry (same exits as NYX) ✅

NOT YET IMPLEMENTED (Require NYXEngine integration):
4. HSMM-only (no SMC, no macro) - Sprint 2
5. SMC-only (no HSMM, no macro) - Sprint 2

Note: HSMM-only and SMC-only require modifications to NYXEngine
to selectively disable components. This will be addressed in Sprint 2
when walk-forward analysis is implemented.

Usage:
    from src.validation.benchmarks import run_all_benchmarks
    
    results = run_all_benchmarks(data, config)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from src.validation.metrics import calculate_metrics


class BuyAndHold:
    """Simple buy and hold strategy"""
    
    def __init__(self, capital: float = 10000):
        self.capital = capital
        self.position = None
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Buy at start, hold until end
        
        Returns:
            DataFrame with single trade
        """
        
        if len(data) < 2:
            return pd.DataFrame()
        
        entry_price = data.iloc[0]['close']
        exit_price = data.iloc[-1]['close']
        
        position_size = self.capital / entry_price
        pnl = (exit_price - entry_price) * position_size
        pnl_pct = (exit_price / entry_price - 1) * 100
        
        trade = {
            'entry_time': data.index[0],
            'entry_price': entry_price,
            'exit_time': data.index[-1],
            'exit_price': exit_price,
            'quantity': position_size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': 'end_of_period'
        }
        
        return pd.DataFrame([trade])


class SMACrossover:
    """Simple moving average crossover"""
    
    def __init__(self, capital: float = 10000, fast: int = 20, slow: int = 50):
        self.capital = capital
        self.fast = fast
        self.slow = slow
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Buy when fast SMA crosses above slow SMA
        Sell when fast SMA crosses below slow SMA
        
        Returns:
            DataFrame with trades
        """
        
        df = data.copy()
        df['sma_fast'] = df['close'].rolling(self.fast).mean()
        df['sma_slow'] = df['close'].rolling(self.slow).mean()
        df['signal'] = 0
        
        # Generate signals
        df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
        df.loc[df['sma_fast'] <= df['sma_slow'], 'signal'] = -1
        
        # Detect crossovers
        df['signal_change'] = df['signal'].diff()
        
        trades = []
        position = None
        current_capital = self.capital  # Track capital
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Buy signal
            if row['signal_change'] == 2 and position is None and current_capital > 0:
                position = {
                    'entry_time': row.name,
                    'entry_price': row['close'],
                    'quantity': (current_capital * 0.95) / row['close']  # 95% of current capital
                }
            
            # Sell signal
            elif row['signal_change'] == -2 and position is not None:
                pnl = (row['close'] - position['entry_price']) * position['quantity']
                pnl_pct = (row['close'] / position['entry_price'] - 1) * 100
                
                # Update capital
                current_capital += pnl
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': row.name,
                    'exit_price': row['close'],
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'sma_cross'
                })
                
                position = None
        
        # Close any open position at end
        if position is not None:
            last_row = df.iloc[-1]
            pnl = (last_row['close'] - position['entry_price']) * position['quantity']
            pnl_pct = (last_row['close'] / position['entry_price'] - 1) * 100
            
            current_capital += pnl
            
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': last_row.name,
                'exit_price': last_row['close'],
                'quantity': position['quantity'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'end_of_period'
            })
        
        return pd.DataFrame(trades)


class RandomEntry:
    """Random entries with same exits as NYX"""
    
    def __init__(self, capital: float = 10000, stop_loss: float = 0.05, 
                 take_profit: float = 0.15, seed: int = 42):
        self.capital = capital
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        np.random.seed(seed)
    
    def run(self, data: pd.DataFrame, num_trades: int = 50) -> pd.DataFrame:
        """
        Random entry times, but use NYX-like exits (stop/TP)
        
        Args:
            data: Price data
            num_trades: Number of random entries to generate
        
        Returns:
            DataFrame with trades
        """
        
        trades = []
        
        # Generate random entry indices
        entry_indices = np.random.choice(
            range(len(data) // 2, len(data) - 100),  # Don't enter too early or late
            size=min(num_trades, len(data) // 10),
            replace=False
        )
        
        for entry_idx in sorted(entry_indices):
            entry_row = data.iloc[entry_idx]
            entry_price = entry_row['close']
            
            # Random direction
            direction = np.random.choice([1, -1])
            
            # Calculate stop and TP
            if direction == 1:  # Long
                stop_price = entry_price * (1 - self.stop_loss)
                tp_price = entry_price * (1 + self.take_profit)
            else:  # Short
                stop_price = entry_price * (1 + self.stop_loss)
                tp_price = entry_price * (1 - self.take_profit)
            
            # Find exit
            exit_idx = None
            exit_reason = None
            
            for i in range(entry_idx + 1, min(entry_idx + 200, len(data))):
                row = data.iloc[i]
                
                if direction == 1:
                    if row['low'] <= stop_price:
                        exit_idx = i
                        exit_reason = 'stop_loss'
                        break
                    elif row['high'] >= tp_price:
                        exit_idx = i
                        exit_reason = 'take_profit'
                        break
                else:
                    if row['high'] >= stop_price:
                        exit_idx = i
                        exit_reason = 'stop_loss'
                        break
                    elif row['low'] <= tp_price:
                        exit_idx = i
                        exit_reason = 'take_profit'
                        break
            
            # If no exit found, close at end
            if exit_idx is None:
                exit_idx = len(data) - 1
                exit_reason = 'end_of_period'
            
            exit_row = data.iloc[exit_idx]
            exit_price = exit_row['close']
            
            # Calculate PnL
            quantity = (self.capital * 0.05) / entry_price  # 5% position like NYX
            pnl = (exit_price - entry_price) * quantity * direction
            pnl_pct = ((exit_price / entry_price - 1) * 100) * direction
            
            trades.append({
                'entry_time': entry_row.name,
                'entry_price': entry_price,
                'exit_time': exit_row.name,
                'exit_price': exit_price,
                'quantity': quantity,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'direction': 'long' if direction == 1 else 'short'
            })
        
        return pd.DataFrame(trades)


def run_benchmark(benchmark_name: str, data: pd.DataFrame, config: dict) -> Dict:
    """
    Run a single benchmark
    
    Args:
        benchmark_name: 'buy_hold', 'sma_crossover', 'random_entry', 'hsmm_only', 'smc_only'
        data: Price data
        config: Configuration dict
    
    Returns:
        Dict with trades and metrics
    """
    
    capital = config.get('capital', {}).get('initial', 10000)
    
    if benchmark_name == 'buy_hold':
        strategy = BuyAndHold(capital)
        trades = strategy.run(data)
    
    elif benchmark_name == 'sma_crossover':
        strategy = SMACrossover(capital, fast=20, slow=50)
        trades = strategy.run(data)
    
    elif benchmark_name == 'random_entry':
        stop = config.get('strategy', {}).get('risk', {}).get('stop_loss', 0.05)
        tp = config.get('strategy', {}).get('risk', {}).get('take_profit', 0.15)
        strategy = RandomEntry(capital, stop_loss=stop, take_profit=tp)
        trades = strategy.run(data, num_trades=50)
    
    elif benchmark_name == 'hsmm_only':
        # Use NYXEngine but disable SMC and Macro
        from src.core.nyx_engine import NYXEngine
        
        # Modify config to disable SMC/Macro
        test_config = config.copy()
        # Simple HSMM-only strategy (would need NYXEngine modification or mock)
        # For now, return placeholder
        trades = pd.DataFrame()  # TODO: Implement HSMM-only integration
        print("    ⚠️  HSMM-only requires NYXEngine integration (placeholder)")
    
    elif benchmark_name == 'smc_only':
        # Use NYXEngine but disable HSMM and Macro
        # For now, return placeholder
        trades = pd.DataFrame()  # TODO: Implement SMC-only integration
        print("    ⚠️  SMC-only requires NYXEngine integration (placeholder)")
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    # Calculate metrics
    metrics = calculate_metrics(trades, initial_capital=capital)
    
    return {
        'benchmark': benchmark_name,
        'trades': trades,
        'metrics': metrics,
        'num_trades': len(trades)
    }


def run_all_benchmarks(data: pd.DataFrame, config: dict) -> Dict:
    """
    Run all benchmarks
    
    Args:
        data: Price data
        config: Configuration dict
    
    Returns:
        Dict with all benchmark results
    """
    
    # Only run working benchmarks for now
    # HSMM-only and SMC-only require NYXEngine integration
    benchmarks = ['buy_hold', 'sma_crossover', 'random_entry']
    
    results = {}
    
    print("\nRunning benchmarks...")
    for benchmark in benchmarks:
        print(f"  • {benchmark}...", end='')
        try:
            results[benchmark] = run_benchmark(benchmark, data, config)
            print(f" ✓ ({results[benchmark]['num_trades']} trades)")
        except Exception as e:
            print(f" ✗ Error: {e}")
            results[benchmark] = {'error': str(e)}
    
    # Note: HSMM-only and SMC-only commented out until NYXEngine integration
    # benchmarks_future = ['hsmm_only', 'smc_only']
    # print(f"\n⚠️  Future benchmarks (require engine integration): {', '.join(benchmarks_future)}")
    
    return results


def compare_to_nyx(nyx_metrics: Dict, benchmark_results: Dict) -> pd.DataFrame:
    """
    Compare NYX metrics to all benchmarks
    
    Args:
        nyx_metrics: NYX metrics dict
        benchmark_results: Results from run_all_benchmarks
    
    Returns:
        Comparison DataFrame
    """
    
    comparison = {'NYX': nyx_metrics}
    
    for name, result in benchmark_results.items():
        if 'metrics' in result:
            comparison[name] = result['metrics']
    
    df = pd.DataFrame(comparison).T
    
    # Select key metrics
    key_metrics = [
        'total_return', 'cagr', 'sharpe_ratio', 'max_drawdown',
        'win_rate', 'profit_factor', 'total_trades'
    ]
    
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    return pd.DataFrame(df[available_metrics])


if __name__ == "__main__":
    print("Benchmarks Module - Example")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    prices = 50000 + np.cumsum(np.random.randn(500) * 100)
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(500) * 50,
        'high': prices + np.abs(np.random.randn(500) * 100),
        'low': prices - np.abs(np.random.randn(500) * 100),
        'close': prices,
        'volume': np.random.lognormal(10, 1, 500)
    }, index=dates)
    
    config = {
        'capital': {'initial': 10000},
        'strategy': {
            'risk': {
                'stop_loss': 0.05,
                'take_profit': 0.15
            }
        }
    }
    
    # Run benchmarks
    results = run_all_benchmarks(data, config)
    
    # Print summary
    print("\nBenchmark Summary:")
    for name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n{name}:")
            print(f"  Return: {metrics['total_return']:.2f}%")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"  Max DD: {metrics['max_drawdown']:.2f}%")
            print(f"  Trades: {metrics['total_trades']}")
