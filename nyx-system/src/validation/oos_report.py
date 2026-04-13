"""
Out-of-Sample (OOS) Report Generator

Splits data into in-sample and out-of-sample periods and compares performance.

Convention:
- In-sample: First 80% of data
- Out-of-sample: Last 20% of data

Usage:
    from src.validation.oos_report import OOSReport
    
    oos = OOSReport(config)
    results = oos.run(data, pair='BTCUSDT')
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, cast
import json
from pathlib import Path
from src.validation.metrics import calculate_metrics, compare_metrics


class OOSReport:
    """Out-of-sample validation report"""
    
    def __init__(self, config: dict):
        """
        Initialize OOS report generator
        
        Args:
            config: Configuration dict
        """
        self.config = config
        
        # OOS split ratio (80/20 by default)
        self.oos_split = 0.80
    
    def run(self, data: pd.DataFrame, pair: str) -> Dict:
        """
        Run OOS analysis
        
        Args:
            data: OHLCV data with datetime index
            pair: Trading pair
        
        Returns:
            Dict with OOS results
        """
        
        print(f"\n{'='*80}")
        print(f"OUT-OF-SAMPLE ANALYSIS - {pair}")
        print(f"{'='*80}")
        
        # Split data
        split_idx = int(len(data) * self.oos_split)
        
        is_data = data.iloc[:split_idx]
        oos_data = data.iloc[split_idx:]
        
        is_start = is_data.index[0]
        is_end = is_data.index[-1]
        oos_start = oos_data.index[0]
        oos_end = oos_data.index[-1]
        
        print(f"\nIn-Sample:     {is_start.date()} to {is_end.date()} ({len(is_data)} rows)")
        print(f"Out-of-Sample: {oos_start.date()} to {oos_end.date()} ({len(oos_data)} rows)")
        
        # Run backtest on both periods
        print("\nRunning In-Sample...")
        is_results = self._run_backtest(is_data, pair, 'in_sample')
        
        print("\nRunning Out-of-Sample...")
        oos_results = self._run_backtest(oos_data, pair, 'out_of_sample')
        
        # Compare metrics
        comparison = self._compare_performance(is_results['metrics'], oos_results['metrics'])
        
        # Generate verdict
        verdict = self._generate_verdict(comparison)
        
        return {
            'pair': pair,
            'split_ratio': self.oos_split,
            'in_sample': is_results,
            'out_of_sample': oos_results,
            'comparison': comparison,
            'verdict': verdict
        }
    
    def _run_backtest(self, data: pd.DataFrame, pair: str, period_name: str) -> Dict:
        """
        Run backtest on a period (OPTIMIZED with BOUNDED LOOKBACK)
        
        Uses precomputed features and bounded lookback window
        for practical performance on large datasets.
        
        Args:
            data: OHLCV data
            pair: Trading pair
            period_name: 'in_sample' or 'out_of_sample'
        
        Returns:
            Results dict
        """
        
        from src.core.nyx_engine import NYXEngine
        import time
        
        # Initialize engine
        engine = NYXEngine(self.config)
        
        # Prepare data once (precompute all features)
        prepared_data = engine.prepare_data(data)
        
        # Get config params
        val_config = self.config.get('validation', {})
        warmup = val_config.get('warmup_bars', 100)
        lookback_bars = val_config.get('signal_lookback_bars', 1000)
        
        print(f"  Warmup: {warmup} bars")
        print(f"  Lookback: {lookback_bars} bars")
        print(f"  Processing: {len(prepared_data) - warmup} bars")
        
        # Simulate trading
        trades = []
        position = None
        capital = self.config.get('capital', {}).get('initial', 10000)
        
        signals_generated = 0
        start_time = time.time()
        
        # Start after warmup
        for i in range(warmup, len(prepared_data)):
            current_date = cast(pd.Timestamp, prepared_data.index[i])
            
            # Get signal (bounded lookback - constant cost per bar)
            signal = engine.generate_signal_at_index(
                pair=pair,
                prepared_data=prepared_data,
                i=i,
                current_date=current_date.isoformat(),
                lookback_bars=lookback_bars
            )
            signals_generated += 1
            
            # Simple execution
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
        
        # Close open position
        if position is not None:
            last_row = prepared_data.iloc[-1]
            exit_price = last_row['close']
            pnl = (exit_price - position['entry_price']) * position['quantity']
            
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': prepared_data.index[-1],
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'pnl_pct': (exit_price / position['entry_price'] - 1) * 100,
                'exit_reason': 'end_of_period'
            })
        
        # Performance logging
        elapsed = time.time() - start_time
        bars_processed = len(prepared_data) - warmup
        avg_time_per_bar = elapsed / bars_processed if bars_processed > 0 else 0
        
        print(f"  Execution time: {elapsed:.2f}s")
        print(f"  Avg time/bar: {avg_time_per_bar*1000:.2f}ms")
        print(f"  Signals generated: {signals_generated}")
        print(f"  Trades executed: {len(trades)}")
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        metrics = calculate_metrics(trades_df, initial_capital=capital)
        
        return {
            'period': period_name,
            'start_date': cast(pd.Timestamp, prepared_data.index[0]).isoformat(),
            'end_date': cast(pd.Timestamp, prepared_data.index[-1]).isoformat(),
            'trades': trades_df,
            'metrics': metrics,
            'performance': {
                'bars_processed': bars_processed,
                'warmup_bars': warmup,
                'lookback_bars': lookback_bars,
                'execution_time_seconds': elapsed,
                'avg_time_per_bar_ms': avg_time_per_bar * 1000,
                'signals_generated': signals_generated
            }
        }
    
    def _compare_performance(self, is_metrics: Dict, oos_metrics: Dict) -> Dict:
        """
        Compare IS vs OOS performance
        
        Args:
            is_metrics: In-sample metrics
            oos_metrics: Out-of-sample metrics
        
        Returns:
            Comparison dict
        """
        
        def pct_change(is_val, oos_val):
            """Calculate percentage change"""
            if is_val is None or oos_val is None:
                return None
            if is_val == 0:
                return None
            return ((oos_val - is_val) / abs(is_val)) * 100
        
        comparison = {
            'total_return': {
                'is': is_metrics['total_return'],
                'oos': oos_metrics['total_return'],
                'change_pct': pct_change(is_metrics['total_return'], oos_metrics['total_return'])
            },
            'sharpe_ratio': {
                'is': is_metrics['sharpe_ratio'],
                'oos': oos_metrics['sharpe_ratio'],
                'change_pct': pct_change(is_metrics['sharpe_ratio'], oos_metrics['sharpe_ratio'])
            },
            'max_drawdown': {
                'is': is_metrics['max_drawdown'],
                'oos': oos_metrics['max_drawdown'],
                'change_pct': pct_change(is_metrics['max_drawdown'], oos_metrics['max_drawdown'])
            },
            'win_rate': {
                'is': is_metrics['win_rate'],
                'oos': oos_metrics['win_rate'],
                'change_pct': pct_change(is_metrics['win_rate'], oos_metrics['win_rate'])
            }
        }
        
        return comparison
    
    def _generate_verdict(self, comparison: Dict) -> Dict:
        """
        Generate OOS verdict
        
        Args:
            comparison: Comparison dict
        
        Returns:
            Verdict dict
        """
        
        # Simple rules for degradation
        total_return_change = comparison['total_return']['change_pct']
        sharpe_change = comparison['sharpe_ratio']['change_pct']
        
        if total_return_change is None or sharpe_change is None:
            status = "INCONCLUSIVE"
            reason = "Metrics not comparable (None values)"
        elif total_return_change < -50:
            status = "FAILED"
            reason = f"Total return degraded {abs(total_return_change):.1f}% (>50% degradation)"
        elif sharpe_change < -50:
            status = "DEGRADED"
            reason = f"Sharpe ratio degraded {abs(sharpe_change):.1f}% (>50% degradation)"
        elif total_return_change < -30:
            status = "WARNING"
            reason = f"Total return degraded {abs(total_return_change):.1f}% (30-50% degradation)"
        else:
            status = "STABLE"
            reason = f"Performance degradation within acceptable range (<30%)"
        
        return {
            'status': status,
            'reason': reason,
            'total_return_change_pct': total_return_change,
            'sharpe_change_pct': sharpe_change
        }


def save_oos_report(results: Dict, output_dir: str):
    """
    Save OOS report to JSON and markdown
    
    Args:
        results: OOS results dict
        output_dir: Output directory
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pair = results['pair']
    
    # Save JSON
    json_file = output_path / f"{pair}_oos.json"
    
    # Convert DataFrames to dict for JSON
    results_copy = results.copy()
    if 'in_sample' in results_copy and 'trades' in results_copy['in_sample']:
        results_copy['in_sample']['trades'] = results_copy['in_sample']['trades'].to_dict('records')
    if 'out_of_sample' in results_copy and 'trades' in results_copy['out_of_sample']:
        results_copy['out_of_sample']['trades'] = results_copy['out_of_sample']['trades'].to_dict('records')
    
    with open(json_file, 'w') as f:
        json.dump(results_copy, f, indent=2, default=str)
    
    print(f"\n✓ OOS report saved: {json_file}")


if __name__ == "__main__":
    print("OOS Report - Example")
    
    import yaml
    
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
    
    # Run OOS
    oos = OOSReport(config)
    results = oos.run(data, pair='BTCUSDT')
    
    print(f"\n{'='*80}")
    print("OOS VERDICT")
    print(f"{'='*80}")
    print(f"Status: {results['verdict']['status']}")
    print(f"Reason: {results['verdict']['reason']}")
