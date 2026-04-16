"""
Signal Funnel Diagnostics

Tracks where NYX signals die before execution.

Goal: Understand if NYX is silent by discipline or by strangulation.

Usage:
    from src.validation.signal_funnel import SignalFunnel
    
    funnel = SignalFunnel(config)
    report = funnel.run(data, pair='BTCUSDT')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, cast
from collections import Counter
import json
from pathlib import Path


class SignalFunnel:
    """
    Signal funnel diagnostics
    
    Tracks opportunity flow through NYX pipeline:
    1. Raw opportunities (every bar)
    2. HSMM filter
    3. SMC filter
    4. Confidence threshold
    5. Macro filter
    6. Risk filter
    7. Executed trades
    """
    
    def __init__(self, config: Dict):
        """
        Initialize funnel tracker
        
        Args:
            config: Configuration dict
        """
        self.config = config
        
        # Counters
        self.counters = {
            'bars_analyzed': 0,
            'warmup_skipped': 0,
            'raw_opportunities': 0,
            'insufficient_data': 0,
            'hsmm_initialization_failed': 0,
            'hsmm_passed': 0,
            'smc_passed': 0,
            'confidence_passed': 0,
            'macro_passed': 0,
            'risk_passed': 0,
            'trades_executed': 0
        }
        
        # Side tracking
        self.sides = {
            'long_opportunities': 0,
            'short_opportunities': 0,
            'long_rejected': 0,
            'short_rejected': 0,
            'long_executed': 0,
            'short_executed': 0
        }
        
        # Rejection reasons
        self.rejections = Counter()
    
    @staticmethod
    def compare_min_bars(data: pd.DataFrame, pair: str, base_config: Dict, 
                         min_bars_list: List[int] = [50, 100]) -> Dict:
        """
        Compare funnel results with different min_required_bars settings
        
        Args:
            data: OHLCV DataFrame
            pair: Trading pair
            base_config: Base configuration
            min_bars_list: List of min_required_bars values to test
        
        Returns:
            Comparison dict
        """
        
        print(f"\n{'='*80}")
        print(f"MIN_REQUIRED_BARS COMPARISON - {pair}")
        print(f"{'='*80}")
        
        results = {}
        
        for min_bars in min_bars_list:
            print(f"\nTesting min_required_bars = {min_bars}...")
            
            # Create config copy with modified min_required_bars
            config = base_config.copy()
            if 'strategy' not in config:
                config['strategy'] = {}
            config['strategy']['min_required_bars'] = min_bars
            
            # Run funnel
            funnel = SignalFunnel(config)
            report = funnel.run(data, pair, quiet=True)
            
            # Extract key metrics
            results[str(min_bars)] = {
                'insufficient_data': report['counters']['insufficient_data'],
                'hsmm_passed': report['counters']['hsmm_passed'],
                'confidence_passed': report['counters']['confidence_passed'],
                'smc_passed': report['counters']['smc_passed'],
                'macro_passed': report['counters']['macro_passed'],
                'risk_passed': report['counters']['risk_passed'],
                'trades_executed': report['counters']['trades_executed'],
                'top_rejection': report['top_rejections'][0] if report['top_rejections'] else ('none', 0)
            }
        
        # Generate comparison summary
        comparison = {
            'pair': pair,
            'min_bars_tested': min_bars_list,
            'results': results,
            'summary': _generate_comparison_summary(results, min_bars_list)
        }
        
        return comparison
    
    def run(self, data: pd.DataFrame, pair: str, quiet: bool = False) -> Dict:
        """
        Run funnel diagnostics on dataset
        
        Args:
            data: OHLCV DataFrame
            pair: Trading pair
            quiet: If True, suppress print output
        
        Returns:
            Funnel report dict
        """
        
        if not quiet:
            print(f"\n{'='*80}")
            print(f"SIGNAL FUNNEL DIAGNOSTICS - {pair}")
            print(f"{'='*80}")
        
        from src.core.nyx_engine_v08 import NYXEngine  # legacy v0.8 (Ticket 04)
        
        # Initialize engine
        engine = NYXEngine(self.config)
        
        # Prepare data
        prepared_data = engine.prepare_data(data)
        
        # Get config
        val_config = self.config.get('validation', {})
        warmup = val_config.get('warmup_bars', 100)
        lookback = val_config.get('signal_lookback_bars', 50)
        
        self.counters['bars_analyzed'] = len(prepared_data)
        self.counters['warmup_skipped'] = warmup
        
        if not quiet:
            print(f"\nAnalyzing {len(prepared_data)} bars...")
            print(f"Warmup: {warmup} bars")
            print(f"Lookback: {lookback} bars")
        
        # Track through pipeline
        for i in range(warmup, len(prepared_data)):
            current_date = cast(pd.Timestamp, prepared_data.index[i])
            
            # Every bar is a raw opportunity
            self.counters['raw_opportunities'] += 1
            
            # Get full signal with diagnostics
            signal = engine.generate_signal_at_index(
                pair=pair,
                prepared_data=prepared_data,
                i=i,
                current_date=current_date.isoformat(),
                lookback_bars=lookback
            )
            
            # Track through funnel
            self._track_signal(signal)
        
        # Calculate pass rates
        pass_rates = self._calculate_pass_rates()
        
        # Generate report
        report = {
            'pair': pair,
            'period': {
                'start': cast(pd.Timestamp, prepared_data.index[0]).isoformat(),
                'end': cast(pd.Timestamp, prepared_data.index[-1]).isoformat()
            },
            'counters': self.counters,
            'pass_rates': pass_rates,
            'sides': self.sides,
            'top_rejections': self.rejections.most_common(10),
            'summary': self._generate_summary()
        }
        
        return report
    
    def _track_signal(self, signal: Dict):
        """
        Track signal through funnel stages
        
        Args:
            signal: Signal dict from NYXEngine
        """
        
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        reasons = signal.get('reasons', [])
        
        # Convert reasons to string for checking
        reasons_str = ' '.join(str(r) for r in reasons)
        
        # Track side intent
        if action == 'BUY':
            self.sides['long_opportunities'] += 1
        elif action == 'SELL':
            self.sides['short_opportunities'] += 1
        
        # Stage 0: Check for insufficient data (before HSMM)
        if 'Insufficient data' in reasons_str:
            self.counters['insufficient_data'] += 1
            self.rejections['insufficient_data'] += 1
            if action == 'BUY':
                self.sides['long_rejected'] += 1
            elif action == 'SELL':
                self.sides['short_rejected'] += 1
            return
        
        # Stage 0b: HSMM initialization failure
        if 'HSMM initialization failed' in reasons_str:
            self.counters['hsmm_initialization_failed'] += 1
            self.rejections['hsmm_initialization_failed'] += 1
            if action == 'BUY':
                self.sides['long_rejected'] += 1
            elif action == 'SELL':
                self.sides['short_rejected'] += 1
            return
        
        # Stage 1: HSMM
        if 'HSMM' in reasons_str or confidence > 0:
            self.counters['hsmm_passed'] += 1
        else:
            self.rejections['hsmm_no_signal'] += 1
            if action == 'BUY':
                self.sides['long_rejected'] += 1
            elif action == 'SELL':
                self.sides['short_rejected'] += 1
            return
        
        # Stage 2: Confidence threshold
        sdc_threshold = self.config.get('strategy', {}).get('hsmm', {}).get('sdc_threshold', 3.5)
        if confidence >= sdc_threshold:
            self.counters['confidence_passed'] += 1
        else:
            self.rejections['hsmm_confidence_too_low'] += 1
            if action == 'BUY':
                self.sides['long_rejected'] += 1
            elif action == 'SELL':
                self.sides['short_rejected'] += 1
            return
        
        # Stage 3: SMC patterns
        smc_patterns = signal.get('smc_patterns', {})
        has_smc = any([
            smc_patterns.get('bullish_ob'),
            smc_patterns.get('bearish_ob'),
            smc_patterns.get('bullish_fvg'),
            smc_patterns.get('bearish_fvg')
        ])
        
        if has_smc:
            self.counters['smc_passed'] += 1
        else:
            self.rejections['smc_no_pattern'] += 1
            if action == 'BUY':
                self.sides['long_rejected'] += 1
            elif action == 'SELL':
                self.sides['short_rejected'] += 1
            return
        
        # Stage 4: Macro filter
        macro_signal = signal.get('macro_signal', 'NEUTRAL')
        if action == 'BUY' and macro_signal in ['BULLISH', 'NEUTRAL']:
            self.counters['macro_passed'] += 1
        elif action == 'SELL' and macro_signal in ['BEARISH', 'NEUTRAL']:
            self.counters['macro_passed'] += 1
        else:
            self.rejections['macro_conflict'] += 1
            if action == 'BUY':
                self.sides['long_rejected'] += 1
            elif action == 'SELL':
                self.sides['short_rejected'] += 1
            return
        
        # Stage 5: Risk filter (position size > 0)
        position_size = signal.get('position_size', 0)
        if position_size > 0:
            self.counters['risk_passed'] += 1
        else:
            self.rejections['risk_rejected'] += 1
            if action == 'BUY':
                self.sides['long_rejected'] += 1
            elif action == 'SELL':
                self.sides['short_rejected'] += 1
            return
        
        # Stage 6: Execution (if action != HOLD)
        if action in ['BUY', 'SELL']:
            self.counters['trades_executed'] += 1
            if action == 'BUY':
                self.sides['long_executed'] += 1
            else:
                self.sides['short_executed'] += 1
        else:
            self.rejections['position_blocked'] += 1
            if action == 'BUY':
                self.sides['long_rejected'] += 1
            elif action == 'SELL':
                self.sides['short_rejected'] += 1
    
    def _calculate_pass_rates(self) -> Dict:
        """Calculate pass-through rates"""
        
        raw = self.counters['raw_opportunities']
        if raw == 0:
            return {}
        
        return {
            'hsmm_rate': (self.counters['hsmm_passed'] / raw) * 100 if raw > 0 else 0,
            'smc_rate': (self.counters['smc_passed'] / raw) * 100 if raw > 0 else 0,
            'confidence_rate': (self.counters['confidence_passed'] / raw) * 100 if raw > 0 else 0,
            'macro_rate': (self.counters['macro_passed'] / raw) * 100 if raw > 0 else 0,
            'risk_rate': (self.counters['risk_passed'] / raw) * 100 if raw > 0 else 0,
            'execution_rate': (self.counters['trades_executed'] / raw) * 100 if raw > 0 else 0
        }
    
    def _generate_summary(self) -> str:
        """Generate text summary"""
        
        raw = self.counters['raw_opportunities']
        
        summary = f"""
Bars analyzed: {self.counters['bars_analyzed']}
Warmup skipped: {self.counters['warmup_skipped']}

SIGNAL FUNNEL:
Raw opportunities:    {raw:6d}
Insufficient data:    {self.counters['insufficient_data']:6d}  ({self.counters['insufficient_data']/raw*100:.1f}%)
After HSMM:           {self.counters['hsmm_passed']:6d}  ({self.counters['hsmm_passed']/raw*100:.1f}%)
After confidence:     {self.counters['confidence_passed']:6d}  ({self.counters['confidence_passed']/raw*100:.1f}%)
After SMC:            {self.counters['smc_passed']:6d}  ({self.counters['smc_passed']/raw*100:.1f}%)
After macro:          {self.counters['macro_passed']:6d}  ({self.counters['macro_passed']/raw*100:.1f}%)
After risk:           {self.counters['risk_passed']:6d}  ({self.counters['risk_passed']/raw*100:.1f}%)
Trades executed:      {self.counters['trades_executed']:6d}  ({self.counters['trades_executed']/raw*100:.1f}%)

SIDES:
Long opportunities:   {self.sides['long_opportunities']:6d}
Long executed:        {self.sides['long_executed']:6d}
Short opportunities:  {self.sides['short_opportunities']:6d}
Short executed:       {self.sides['short_executed']:6d}

TOP REJECTION REASONS:
"""
        
        for reason, count in self.rejections.most_common(5):
            summary += f"  {reason:30s} {count:6d}\n"
        
        return summary


def save_funnel_report(report: Dict, output_dir: str):
    """
    Save funnel report to JSON
    
    Args:
        report: Funnel report dict
        output_dir: Output directory
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pair = report.get('pair', 'UNKNOWN')
    json_file = output_path / f"{pair}_signal_funnel.json"
    
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Funnel report saved: {json_file}")


def _generate_comparison_summary(results: Dict, min_bars_list: List[int]) -> str:
    """Generate comparison summary text"""
    
    summary = "\nCOMPARISON RESULTS:\n"
    summary += "="*80 + "\n\n"
    
    # Header
    summary += f"{'Metric':<30s}"
    for min_bars in min_bars_list:
        summary += f"  {str(min_bars):>10s}"
    summary += "\n" + "-"*80 + "\n"
    
    # Metrics
    metrics = [
        'insufficient_data',
        'hsmm_passed',
        'confidence_passed',
        'smc_passed',
        'macro_passed',
        'risk_passed',
        'trades_executed'
    ]
    
    for metric in metrics:
        summary += f"{metric:<30s}"
        for min_bars in min_bars_list:
            value = results[str(min_bars)].get(metric, 0)
            summary += f"  {value:>10d}"
        summary += "\n"
    
    summary += "\n" + "="*80 + "\n"
    
    # Top rejections
    summary += "\nTOP REJECTIONS:\n"
    for min_bars in min_bars_list:
        reason, count = results[str(min_bars)]['top_rejection']
        summary += f"  {min_bars} bars: {reason} ({count})\n"
    
    return summary
    """
    Save funnel report to JSON
    
    Args:
        report: Funnel report dict
        output_dir: Output directory
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pair = report['pair']
    json_file = output_path / f"{pair}_signal_funnel.json"
    
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Funnel report saved: {json_file}")


if __name__ == "__main__":
    print("Signal Funnel Diagnostics")
    
    import yaml
    
    # Load config
    with open('config/validation_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    data = pd.DataFrame({
        'open': 50000 + np.random.randn(len(dates)) * 1000,
        'high': 51000 + np.random.randn(len(dates)) * 1000,
        'low': 49000 + np.random.randn(len(dates)) * 1000,
        'close': 50000 + np.random.randn(len(dates)) * 1000,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    # Run funnel
    funnel = SignalFunnel(config)
    report = funnel.run(data, pair='BTCUSDT')
    
    print(report['summary'])
