"""
Order Block Coverage Audit

Analyzes why Order Block detection returns zero on observed samples.

Goal: Determine if OB is broken, too strict, or market-incompatible.

Usage:
    from src.validation.order_block_audit import OrderBlockAudit
    
    audit = OrderBlockAudit(config)
    report = audit.run(data, pair='BTCUSDT')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from copy import deepcopy


class OrderBlockAudit:
    """
    Order Block Detection Audit
    
    Analyzes OB detection failures and coverage
    """
    
    def __init__(self, config: Dict):
        """
        Initialize OB audit
        
        Args:
            config: Configuration dict
        """
        self.config = config
        
        # Counters
        self.counters = {
            'bars_analyzed': 0,
            'evaluation_count': 0,
            'bullish_ob_detected': 0,
            'bearish_ob_detected': 0,
            'failed_range_threshold': 0,
            'failed_followthrough_move': 0,
            'failed_window_constraints': 0
        }
    
    def run(self, data: pd.DataFrame, pair: str, 
            sensitivity: bool = False, quiet: bool = False) -> Dict:
        """
        Run OB audit on dataset
        
        Args:
            data: OHLCV DataFrame
            pair: Trading pair
            sensitivity: If True, run sensitivity analysis
            quiet: If True, suppress print output
        
        Returns:
            Audit report dict
        """
        
        if not quiet:
            print(f"\n{'='*80}")
            print(f"ORDER BLOCK AUDIT - {pair}")
            print(f"{'='*80}")
        
        # Get config params
        smc_config = self.config.get('strategy', {}).get('smc', {})
        ob_lookback = smc_config.get('ob_lookback', 5)
        ob_range_threshold = smc_config.get('ob_range_threshold', 0.015)
        
        if not quiet:
            print(f"\nConfig:")
            print(f"  OB lookback:        {ob_lookback}")
            print(f"  OB range threshold: {ob_range_threshold*100:.2f}%")
        
        # Run audit with current config
        self._audit_order_blocks(data, ob_lookback, ob_range_threshold)
        
        # Calculate rates
        total_evals = self.counters['evaluation_count']
        bullish_rate = (self.counters['bullish_ob_detected'] / total_evals * 100) if total_evals > 0 else 0
        bearish_rate = (self.counters['bearish_ob_detected'] / total_evals * 100) if total_evals > 0 else 0
        overall_rate = ((self.counters['bullish_ob_detected'] + self.counters['bearish_ob_detected']) / total_evals * 100) if total_evals > 0 else 0
        
        # Build report
        report = {
            'pair': pair,
            'bars_analyzed': self.counters['bars_analyzed'],
            'config_used': {
                'ob_lookback': ob_lookback,
                'ob_range_threshold': ob_range_threshold
            },
            'ob_audit': self.counters.copy(),
            'coverage': {
                'bullish_ob_rate': bullish_rate,
                'bearish_ob_rate': bearish_rate,
                'overall_ob_rate': overall_rate
            },
            'summary': self._generate_summary()
        }
        
        # Run sensitivity if requested
        if sensitivity:
            if not quiet:
                print(f"\n{'='*80}")
                print("SENSITIVITY ANALYSIS")
                print(f"{'='*80}")
            
            sensitivity_results = self._run_sensitivity(data, quiet)
            report['sensitivity'] = sensitivity_results
        
        return report
    
    def _audit_order_blocks(self, data: pd.DataFrame, 
                           ob_lookback: int, ob_range_threshold: float):
        """
        Audit OB detection with given parameters
        
        Args:
            data: OHLCV DataFrame
            ob_lookback: Lookback window
            ob_range_threshold: Range threshold
        """
        
        self.counters['bars_analyzed'] = len(data)
        
        # Need at least ob_lookback + 1 bars
        if len(data) < ob_lookback + 1:
            return
        
        # Analyze each potential OB location
        for i in range(ob_lookback, len(data) - 1):
            self.counters['evaluation_count'] += 1
            
            # Get lookback window
            window = data.iloc[i-ob_lookback:i+1]
            
            # Check for bullish OB (low consolidation followed by up move)
            bullish_detected, bullish_failure = self._check_bullish_ob(
                window, data.iloc[i+1], ob_range_threshold
            )
            
            if bullish_detected:
                self.counters['bullish_ob_detected'] += 1
            elif bullish_failure:
                self.counters[bullish_failure] += 1
            
            # Check for bearish OB (high consolidation followed by down move)
            bearish_detected, bearish_failure = self._check_bearish_ob(
                window, data.iloc[i+1], ob_range_threshold
            )
            
            if bearish_detected:
                self.counters['bearish_ob_detected'] += 1
            elif bearish_failure:
                self.counters[bearish_failure] += 1
    
    def _check_bullish_ob(self, window: pd.DataFrame, next_bar: pd.Series,
                         threshold: float) -> Tuple[bool, str]:
        """
        Check for bullish OB pattern
        
        Returns:
            (detected: bool, failure_reason: str)
        """
        
        # Get window range
        window_high = window['high'].max()
        window_low = window['low'].min()
        window_range = window_high - window_low
        
        # Check range threshold
        avg_price = (window_high + window_low) / 2
        range_pct = window_range / avg_price
        
        if range_pct > threshold:
            return False, 'failed_range_threshold'
        
        # Check follow-through (next bar moves up)
        last_close = window.iloc[-1]['close']
        next_close = next_bar['close']
        
        move_pct = (next_close - last_close) / last_close
        
        if move_pct <= 0.005:  # Require at least 0.5% up move
            return False, 'failed_followthrough_move'
        
        # Bullish OB detected
        return True, ""

    def _check_bearish_ob(self, window: pd.DataFrame, next_bar: pd.Series,
                         threshold: float) -> Tuple[bool, str]:
        """
        Check for bearish OB pattern
        
        Returns:
            (detected: bool, failure_reason: str)
        """
        
        # Get window range
        window_high = window['high'].max()
        window_low = window['low'].min()
        window_range = window_high - window_low
        
        # Check range threshold
        avg_price = (window_high + window_low) / 2
        range_pct = window_range / avg_price
        
        if range_pct > threshold:
            return False, 'failed_range_threshold'
        
        # Check follow-through (next bar moves down)
        last_close = window.iloc[-1]['close']
        next_close = next_bar['close']
        
        move_pct = (next_close - last_close) / last_close
        
        if move_pct >= -0.005:  # Require at least 0.5% down move
            return False, 'failed_followthrough_move'
        
        # Bearish OB detected
        return True, ""
    
    def _run_sensitivity(self, data: pd.DataFrame, quiet: bool = False) -> List[Dict]:
        """
        Run sensitivity analysis with different configs
        
        Args:
            data: OHLCV DataFrame
            quiet: Suppress output
        
        Returns:
            List of sensitivity results
        """
        
        # Test combinations
        lookbacks = [5, 10, 20]
        thresholds = [0.015, 0.01, 0.005]
        
        results = []
        
        for lookback in lookbacks:
            for threshold in thresholds:
                if not quiet:
                    print(f"\nTesting: lookback={lookback}, threshold={threshold*100:.2f}%")
                
                # Reset counters
                test_counters = {
                    'evaluation_count': 0,
                    'bullish_ob_detected': 0,
                    'bearish_ob_detected': 0,
                    'failed_range_threshold': 0,
                    'failed_followthrough_move': 0,
                    'failed_window_constraints': 0
                }
                
                # Run audit
                temp_audit = OrderBlockAudit(self.config)
                temp_audit.counters = test_counters
                temp_audit._audit_order_blocks(data, lookback, threshold)
                
                # Store result
                result = {
                    'ob_lookback': lookback,
                    'ob_range_threshold': threshold,
                    'bullish_ob_detected': test_counters['bullish_ob_detected'],
                    'bearish_ob_detected': test_counters['bearish_ob_detected'],
                    'total_detected': test_counters['bullish_ob_detected'] + test_counters['bearish_ob_detected']
                }
                
                results.append(result)
                
                if not quiet:
                    print(f"  Bullish: {result['bullish_ob_detected']}, Bearish: {result['bearish_ob_detected']}")
        
        return results
    
    def _generate_summary(self) -> str:
        """Generate text summary"""
        
        total_detected = self.counters['bullish_ob_detected'] + self.counters['bearish_ob_detected']
        
        summary = f"""
Bars analyzed:         {self.counters['bars_analyzed']}
Evaluation count:      {self.counters['evaluation_count']}

Bullish OB detected:   {self.counters['bullish_ob_detected']}
Bearish OB detected:   {self.counters['bearish_ob_detected']}
Total OB detected:     {total_detected}

Failure breakdown:
  Range threshold fail:      {self.counters['failed_range_threshold']}
  Follow-through fail:       {self.counters['failed_followthrough_move']}
  Window constraint fail:    {self.counters['failed_window_constraints']}
"""
        
        # Add conclusion
        if total_detected == 0:
            if self.counters['failed_range_threshold'] > self.counters['evaluation_count'] * 0.5:
                summary += "\nConclusion:\nOrder Block detection appears too restrictive on this sample.\nMain bottleneck: range threshold.\n"
            else:
                summary += "\nConclusion:\nOrder Block detection appears inactive.\nMain bottleneck: follow-through move requirements.\n"
        else:
            summary += f"\nConclusion:\nOrder Block detection is active ({total_detected} detected).\n"
        
        return summary


def save_ob_audit(report: Dict, output_dir: str):
    """
    Save OB audit to JSON
    
    Args:
        report: Audit report dict
        output_dir: Output directory
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pair = report.get('pair', 'UNKNOWN')
    json_file = output_path / f"{pair}_order_block_audit.json"
    
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ OB audit saved: {json_file}")


if __name__ == "__main__":
    print("Order Block Coverage Audit")
    
    import yaml
    
    # Load config
    with open('config/validation_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure SMC config exists
    if 'smc' not in config.get('strategy', {}):
        config['strategy']['smc'] = {}
    config['strategy']['smc']['ob_lookback'] = 5
    config['strategy']['smc']['ob_range_threshold'] = 0.015
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
    data = pd.DataFrame({
        'open': 50000 + np.random.randn(len(dates)) * 1000,
        'high': 51000 + np.random.randn(len(dates)) * 1000,
        'low': 49000 + np.random.randn(len(dates)) * 1000,
        'close': 50000 + np.random.randn(len(dates)) * 1000,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    # Run audit
    audit = OrderBlockAudit(config)
    report = audit.run(data, pair='BTCUSDT', sensitivity=True)
    
    print(report['summary'])
