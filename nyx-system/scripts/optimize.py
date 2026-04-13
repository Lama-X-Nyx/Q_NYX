#!/usr/bin/env python3
"""
NYX Parameter Optimizer - Grid search for optimal parameters

Usage:
    python scripts/optimize.py --pair BTCUSDT --param sdc_threshold
    python scripts/optimize.py --pair ETHUSDT --param all
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from itertools import product
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ParameterOptimizer:
    """
    Parameter optimization using grid search
    
    Optimizes:
    - sdc_threshold: 2.5 - 5.0
    - prob_threshold: 0.40 - 0.70
    - pyramid_threshold: 0.05 - 0.20
    - stop_loss: 0.02 - 0.10
    """
    
    PARAM_RANGES = {
        'sdc_threshold': np.arange(2.5, 5.1, 0.5),
        'prob_threshold': np.arange(0.40, 0.71, 0.05),
        'pyramid_threshold': np.arange(0.05, 0.21, 0.03),
        'stop_loss': np.arange(0.02, 0.11, 0.02)
    }
    
    def __init__(self, data: pd.DataFrame, base_config: dict):
        self.data = data
        self.base_config = base_config
        self.results = []
    
    def optimize_single_param(self, param_name: str, metric: str = 'cagr'):
        """
        Optimize single parameter
        
        Args:
            param_name: Parameter to optimize
            metric: Metric to optimize ('cagr', 'sharpe', 'profit_factor')
        
        Returns:
            Best value and results
        """
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZING: {param_name}")
        print(f"Target metric: {metric.upper()}")
        print(f"{'='*80}")
        
        if param_name not in self.PARAM_RANGES:
            print(f"✗ Unknown parameter: {param_name}")
            return None
        
        values = self.PARAM_RANGES[param_name]
        
        print(f"\nTesting {len(values)} values: {values[0]:.2f} - {values[-1]:.2f}")
        
        results = []
        
        for i, value in enumerate(values):
            # Update config
            config = self.base_config.copy()
            
            if param_name == 'sdc_threshold' or param_name == 'prob_threshold':
                config.setdefault('strategy', {}).setdefault('hsmm', {})[param_name] = value
            else:
                config.setdefault('risk', {})[param_name] = value
            
            # Run backtest (simplified)
            result = self._run_backtest_simplified(config)
            
            results.append({
                'value': value,
                'metric': result.get(metric, 0),
                'return': result.get('total_return', 0),
                'trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0)
            })
            
            print(f"  [{i+1}/{len(values)}] {param_name}={value:.2f} → {metric}={result.get(metric, 0):.2f}")
        
        # Find best
        best = max(results, key=lambda x: x['metric'])
        
        print(f"\n{'='*80}")
        print(f"BEST RESULT")
        print(f"{'='*80}")
        print(f"  {param_name}: {best['value']:.2f}")
        print(f"  {metric.upper()}: {best['metric']:.2f}")
        print(f"  Return: {best['return']:.2f}%")
        print(f"  Trades: {best['trades']}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        
        return best, results
    
    def optimize_all(self, metric: str = 'cagr', max_combinations: int = 100):
        """
        Optimize all parameters (grid search)
        
        Warning: Can be slow with many combinations
        
        Args:
            metric: Target metric
            max_combinations: Max combinations to test (safety limit)
        """
        
        print(f"\n{'█'*80}")
        print("FULL GRID SEARCH OPTIMIZATION")
        print(f"{'█'*80}")
        
        # Create grid
        param_names = list(self.PARAM_RANGES.keys())
        param_values = [self.PARAM_RANGES[p] for p in param_names]
        
        total_combinations = np.prod([len(v) for v in param_values])
        
        print(f"\nParameters: {', '.join(param_names)}")
        print(f"Total combinations: {total_combinations:,}")
        
        if total_combinations > max_combinations:
            print(f"\n⚠ Too many combinations (> {max_combinations})")
            print(f"  Using reduced grid...")
            
            # Reduce grid
            param_values = [v[::2] for v in param_values]  # Every 2nd value
            total_combinations = np.prod([len(v) for v in param_values])
            print(f"  Reduced to: {total_combinations} combinations")
        
        results = []
        
        for i, combo in enumerate(product(*param_values)):
            if i >= max_combinations:
                break
            
            # Create config with this combination
            config = self.base_config.copy()
            
            params_dict = dict(zip(param_names, combo))
            
            config.setdefault('strategy', {}).setdefault('hsmm', {})['sdc_threshold'] = params_dict['sdc_threshold']
            config['strategy']['hsmm']['prob_threshold'] = params_dict['prob_threshold']
            
            config.setdefault('risk', {})['pyramid_threshold'] = params_dict['pyramid_threshold']
            config['risk']['stop_loss'] = params_dict['stop_loss']
            
            # Run backtest
            result: dict = self._run_backtest_simplified(config)

            result['params'] = params_dict
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{total_combinations}")
        
        # Find best
        best = max(results, key=lambda x: x.get(metric, 0))
        
        print(f"\n{'='*80}")
        print("BEST PARAMETER COMBINATION")
        print(f"{'='*80}")
        print(f"\nParameters:")
        for param, value in best['params'].items():
            print(f"  {param:20s}: {value:.2f}")
        
        print(f"\nResults:")
        print(f"  {metric.upper():20s}: {best.get(metric, 0):.2f}")
        print(f"  Return:              {best.get('total_return', 0):.2f}%")
        print(f"  Trades:              {best.get('total_trades', 0)}")
        print(f"  Win Rate:            {best.get('win_rate', 0):.1f}%")
        
        return best, results
    
    def _run_backtest_simplified(self, config):
        """Simplified backtest for optimization"""
        
        # Prepare data
        data = self.data.copy()
        
        # Add indicators
        data['returns'] = data['close'].pct_change()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['sma_200'] = data['close'].rolling(200).mean()
        
        # Simple backtest
        capital = 10000
        trades = []
        position = None
        entry_price = 0
        
        sdc_threshold = config.get('strategy', {}).get('hsmm', {}).get('sdc_threshold', 3.5)
        stop_loss = config.get('risk', {}).get('stop_loss', 0.05)
        
        for i in range(100, len(data)):
            row = data.iloc[i]
            
            # Simple signal (trend-following)
            if pd.notna(row['sma_20']) and pd.notna(row['sma_50']):
                # Manage position
                if position:
                    pnl_pct = (row['close'] - entry_price) / entry_price
                    
                    # Exit on stop or reverse
                    should_exit = False
                    
                    if position == 'LONG':
                        if pnl_pct < -stop_loss or row['sma_20'] < row['sma_50']:
                            should_exit = True
                    else:
                        if -pnl_pct < -stop_loss or row['sma_20'] > row['sma_50']:
                            should_exit = True
                    
                    if should_exit:
                        pnl = capital * pnl_pct * 0.05
                        capital += pnl
                        trades.append({'pnl': pnl, 'pnl_pct': pnl_pct * 100})
                        position = None
                
                # Entry
                if not position:
                    if row['sma_20'] > row['sma_50'] and row['close'] > row['sma_20']:
                        position = 'LONG'
                        entry_price = row['close']
                    elif row['sma_20'] < row['sma_50'] and row['close'] < row['sma_20']:
                        position = 'SHORT'
                        entry_price = row['close']
        
        # Results
        if not trades:
            return {'total_return': 0, 'cagr': 0, 'total_trades': 0, 'win_rate': 0, 'profit_factor': 0}
        
        total_return = (capital - 10000) / 10000 * 100
        winners = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = winners / len(trades) * 100
        
        years = (data.index[-1] - data.index[0]).days / 365.25
        cagr = ((capital / 10000)**(1/years) - 1) * 100
        
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl'] > 0]) if winners > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl'] < 0]) if winners < len(trades) else 0
        
        profit_factor = (avg_win * winners / (avg_loss * (len(trades) - winners))) if avg_loss > 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }


def load_config():
    """Load base configuration"""
    
    config_path = 'config/config.yaml'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    return {
        'strategy': {'hsmm': {'sdc_threshold': 3.5, 'prob_threshold': 0.55}},
        'risk': {'stop_loss': 0.05, 'pyramid_threshold': 0.10}
    }


def main():
    parser = argparse.ArgumentParser(description='Optimize NYX parameters')
    
    parser.add_argument('--pair', type=str, required=True, help='Trading pair')
    parser.add_argument('--param', type=str, default='sdc_threshold', 
                       help='Parameter to optimize (sdc_threshold, prob_threshold, pyramid_threshold, stop_loss, all)')
    parser.add_argument('--metric', type=str, default='cagr', help='Optimization metric (cagr, profit_factor, sharpe)')
    parser.add_argument('--output', type=str, default='data/results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    data_file = f"data/raw/{args.pair}_1h.csv"
    
    if not os.path.exists(data_file):
        print(f"✗ Data not found: {data_file}")
        return
    
    print(f"\n{'█'*80}")
    print("NYX PARAMETER OPTIMIZER")
    print(f"{'█'*80}")
    print(f"\nPair: {args.pair}")
    print(f"Parameter: {args.param}")
    print(f"Metric: {args.metric}")
    
    data = pd.read_csv(data_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    print(f"\nData loaded: {len(data):,} bars")
    print(f"Period: {data.index[0]} → {data.index[-1]}")
    
    # Load config
    config = load_config()
    
    # Optimize
    optimizer = ParameterOptimizer(data, config)
    
    if args.param == 'all':
        best, results = optimizer.optimize_all(metric=args.metric, max_combinations=100)
    else:
        opt_result = optimizer.optimize_single_param(args.param, metric=args.metric)
        if opt_result is None:
            return
        best, results = opt_result
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(
        args.output,
        f"optimize_{args.pair}_{args.param}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump({
            'pair': args.pair,
            'param': args.param,
            'metric': args.metric,
            'best': best,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
