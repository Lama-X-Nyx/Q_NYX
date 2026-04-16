"""
Walk-Forward Analysis Engine

Tests strategy on rolling windows to validate robustness.

Default configuration:
- Train window: 12 months
- Test window: 3 months
- Step: 3 months (rolling)

Usage:
    from src.validation.walk_forward import WalkForward
    
    wf = WalkForward(config)
    results = wf.run(data, pair='BTCUSDT')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, cast
from datetime import datetime, timedelta
import yaml


class WalkForward:
    """Walk-forward validation engine"""
    
    def __init__(self, config: dict):
        """
        Initialize walk-forward engine
        
        Args:
            config: Configuration dict with walk_forward settings
        """
        self.config = config
        
        # Get walk-forward parameters
        wf_config = config.get('validation', {}).get('walk_forward', {})
        self.train_months = wf_config.get('train_months', 12)
        self.test_months = wf_config.get('test_months', 3)
        self.step_months = wf_config.get('step_months', 3)
    
    def run(self, data: pd.DataFrame, pair: str) -> Dict:
        """
        Run walk-forward analysis
        
        Args:
            data: OHLCV data with datetime index
            pair: Trading pair
        
        Returns:
            Dict with walk-forward results
        """
        
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD ANALYSIS - {pair}")
        print(f"{'='*80}")
        print(f"Train window: {self.train_months} months")
        print(f"Test window:  {self.test_months} months")
        print(f"Step:         {self.step_months} months")
        
        # Generate windows
        windows = self._generate_windows(data)
        
        print(f"\nGenerated {len(windows)} windows")
        
        # Run backtest on each window
        results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\n[Window {i+1}/{len(windows)}]")
            print(f"  Train: {train_start.date()} to {train_end.date()}")
            print(f"  Test:  {test_start.date()} to {test_end.date()}")
            
            # Extract train and test data
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]
            
            print(f"  Train rows: {len(train_data)}, Test rows: {len(test_data)}")
            
            # Run on test window (no training needed - NYX params are frozen)
            window_result = self._run_window(
                train_data, test_data, pair, i+1
            )
            
            results.append(window_result)
        
        # Aggregate results
        aggregated = self._aggregate_results(results)
        
        return {
            'pair': pair,
            'windows': results,
            'aggregated': aggregated,
            'num_windows': len(windows)
        }
    
    def _generate_windows(self, data: pd.DataFrame) -> List[Tuple]:
        """
        Generate rolling windows
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        
        windows = []
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_start = start_date
        
        while True:
            # Calculate train window
            train_start = current_start
            train_end = train_start + pd.DateOffset(months=self.train_months)
            
            # Calculate test window
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            # Check if we have enough data
            if test_end > end_date:
                break
            
            windows.append((train_start, train_end, test_start, test_end))
            
            # Step forward
            current_start = current_start + pd.DateOffset(months=self.step_months)
        
        return windows
    
    def _run_window(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                    pair: str, window_num: int) -> Dict:
        """
        Run backtest on a single window (OPTIMIZED with BOUNDED LOOKBACK)
        
        Note: NYX parameters are FROZEN, so we don't actually train.
        We just run the frozen strategy on the test period.
        
        Args:
            train_data: Training data (not used - params frozen)
            test_data: Test data
            pair: Trading pair
            window_num: Window number
        
        Returns:
            Window results dict
        """
        
        # Import NYXEngine
        from src.core.nyx_engine_v08 import NYXEngine  # legacy v0.8 (Ticket 04)
        from src.validation.metrics import calculate_metrics
        import time
        
        # Initialize engine with frozen config
        engine = NYXEngine(self.config)
        
        # Prepare data once
        prepared_data = engine.prepare_data(test_data)
        
        # Get config params
        val_config = self.config.get('validation', {})
        warmup = val_config.get('warmup_bars', 100)
        lookback_bars = val_config.get('signal_lookback_bars', 1000)
        
        # Simulate trading on test period
        trades = []
        position = None
        capital = self.config.get('capital', {}).get('initial', 10000)
        
        start_time = time.time()
        
        # Start after warmup
        for i in range(warmup, len(prepared_data)):
            current_date = cast(pd.Timestamp, prepared_data.index[i])
            
            # Get signal (bounded lookback)
            signal = engine.generate_signal_at_index(
                pair=pair,
                prepared_data=prepared_data,
                i=i,
                current_date=current_date.isoformat(),
                lookback_bars=lookback_bars
            )
            
            # Simple execution logic
            if signal['action'] == 'BUY' and position is None:
                position = {
                    'entry_time': current_date,
                    'entry_price': prepared_data.iloc[i]['close'],
                    'quantity': (capital * signal['position_size']) / prepared_data.iloc[i]['close']
                }
            
            elif signal['action'] == 'SELL' and position is not None:
                exit_price = prepared_data.iloc[i]['close']
                pnl = (exit_price - position['entry_price']) * position['quantity']
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': current_date,
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price / position['entry_price'] - 1) * 100,
                    'exit_reason': 'signal'
                })
                
                capital += pnl
                position = None
        
        # Close any open position
        if position is not None:
            last_row = prepared_data.iloc[-1]
            exit_price = last_row['close']
            pnl = (exit_price - position['entry_price']) * position['quantity']
            
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': prepared_data.index[-1],
                'exit_price': last_row['close'],
                'quantity': position['quantity'],
                'pnl': pnl,
                'pnl_pct': (exit_price / position['entry_price'] - 1) * 100,
                'exit_reason': 'end_of_test'
            })
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        metrics = calculate_metrics(trades_df, initial_capital=capital)
        
        return {
            'window': window_num,
            'train_start': cast(pd.Timestamp, train_data.index[0]).isoformat(),
            'train_end': cast(pd.Timestamp, train_data.index[-1]).isoformat(),
            'test_start': cast(pd.Timestamp, prepared_data.index[0]).isoformat(),
            'test_end': cast(pd.Timestamp, prepared_data.index[-1]).isoformat(),
            'trades': trades_df,
            'metrics': metrics,
            'performance': {
                'bars_processed': len(prepared_data) - warmup,
                'execution_time_seconds': elapsed
            }
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate metrics across all windows
        
        Args:
            results: List of window results
        
        Returns:
            Aggregated metrics
        """
        
        # Handle empty results
        if not results:
            return {
                'total_return': {'mean': None, 'std': None, 'min': None, 'max': None},
                'cagr': {'mean': None, 'std': None},
                'sharpe_ratio': {'mean': None, 'std': None},
                'max_drawdown': {'mean': None, 'worst': None},
                'win_rate': {'mean': None, 'std': None},
                'num_windows': 0,
                'windows_blown_up': 0,
                'error': 'No windows generated - insufficient data'
            }
        
        # Collect all metrics
        all_metrics = [r['metrics'] for r in results]
        
        # Calculate statistics
        def safe_mean(values):
            """Mean excluding None"""
            valid = [v for v in values if v is not None]
            return np.mean(valid) if valid else None
        
        def safe_std(values):
            """Std excluding None"""
            valid = [v for v in values if v is not None]
            return np.std(valid) if valid else None
        
        def safe_min(values):
            """Min excluding None"""
            valid = [v for v in values if v is not None]
            return min(valid) if valid else None
        
        def safe_max(values):
            """Max excluding None"""
            valid = [v for v in values if v is not None]
            return max(valid) if valid else None
        
        aggregated = {
            'total_return': {
                'mean': safe_mean([m['total_return'] for m in all_metrics]),
                'std': safe_std([m['total_return'] for m in all_metrics]),
                'min': safe_min([m['total_return'] for m in all_metrics]),
                'max': safe_max([m['total_return'] for m in all_metrics])
            },
            'cagr': {
                'mean': safe_mean([m['cagr'] for m in all_metrics]),
                'std': safe_std([m['cagr'] for m in all_metrics])
            },
            'sharpe_ratio': {
                'mean': safe_mean([m['sharpe_ratio'] for m in all_metrics]),
                'std': safe_std([m['sharpe_ratio'] for m in all_metrics])
            },
            'max_drawdown': {
                'mean': safe_mean([m['max_drawdown'] for m in all_metrics]),
                'worst': safe_min([m['max_drawdown'] for m in all_metrics])
            },
            'win_rate': {
                'mean': safe_mean([m['win_rate'] for m in all_metrics]),
                'std': safe_std([m['win_rate'] for m in all_metrics])
            },
            'num_windows': len(results),
            'windows_blown_up': sum(1 for m in all_metrics if m.get('is_blown_up', False))
        }
        
        return aggregated


if __name__ == "__main__":
    print("Walk-Forward Engine - Example")
    
    # Load config
    with open('config/validation_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='1h')
    data = pd.DataFrame({
        'open': 50000 + np.random.randn(len(dates)) * 1000,
        'high': 51000 + np.random.randn(len(dates)) * 1000,
        'low': 49000 + np.random.randn(len(dates)) * 1000,
        'close': 50000 + np.random.randn(len(dates)) * 1000,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    # Run walk-forward
    wf = WalkForward(config)
    results = wf.run(data, pair='BTCUSDT')
    
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS")
    print(f"{'='*80}")
    print(f"Windows: {results['num_windows']}")
    print(f"Mean Return: {results['aggregated']['total_return']['mean']:.2f}%")
    print(f"Mean Sharpe: {results['aggregated']['sharpe_ratio']['mean']:.2f}")
