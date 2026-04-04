"""
SMC Diagnostics - Pattern Coverage Analysis

Analyzes why SMC rejects HSMM-passed candidates.

Goal: Understand if SMC is a healthy filter or excessive bottleneck.

Usage:
    from src.validation.smc_diagnostics import SMCDiagnostics
    
    diag = SMCDiagnostics(config)
    report = diag.run(data, pair='BTCUSDT')
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import json
from pathlib import Path


class SMCDiagnostics:
    """
    SMC Pattern Coverage Diagnostics
    
    Analyzes pattern availability for HSMM-passed candidates
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SMC diagnostics
        
        Args:
            config: Configuration dict
        """
        self.config = config
        
        # Counters
        self.counters = {
            'bars_analyzed': 0,
            'hsmm_passed': 0,
            'with_any_pattern': 0,
            'without_pattern': 0,
            'bullish_ob': 0,
            'bearish_ob': 0,
            'bullish_fvg': 0,
            'bearish_fvg': 0,
            'multiple_patterns': 0
        }
        
        # Side tracking
        self.sides = {
            'long_candidates': 0,
            'short_candidates': 0,
            'long_with_pattern': 0,
            'short_with_pattern': 0,
            'long_without_pattern': 0,
            'short_without_pattern': 0
        }
    
    def run(self, data: pd.DataFrame, pair: str, quiet: bool = False) -> Dict:
        """
        Run SMC diagnostics on dataset
        
        Args:
            data: OHLCV DataFrame
            pair: Trading pair
            quiet: If True, suppress print output
        
        Returns:
            Diagnostics report dict
        """
        
        if not quiet:
            print(f"\n{'='*80}")
            print(f"SMC DIAGNOSTICS - {pair}")
            print(f"{'='*80}")
        
        from src.core.nyx_engine import NYXEngine
        
        # Initialize engine
        engine = NYXEngine(self.config)
        
        # Prepare data
        prepared_data = engine.prepare_data(data)
        
        # Get config
        val_config = self.config.get('validation', {})
        warmup = val_config.get('warmup_bars', 100)
        lookback = val_config.get('signal_lookback_bars', 50)
        
        self.counters['bars_analyzed'] = len(prepared_data)
        
        if not quiet:
            print(f"\nAnalyzing {len(prepared_data)} bars...")
            print(f"Warmup: {warmup} bars")
            print(f"Lookback: {lookback} bars")
        
        # Analyze patterns
        for i in range(warmup, len(prepared_data)):
            current_date = prepared_data.index[i]
            
            # Get signal
            signal = engine.generate_signal_at_index(
                pair=pair,
                prepared_data=prepared_data,
                i=i,
                current_date=current_date.isoformat(),
                lookback_bars=lookback
            )
            
            # Track patterns
            self._track_patterns(signal)
        
        # Calculate ratios
        ratios = self._calculate_ratios()
        
        # Generate report
        report = {
            'pair': pair,
            'period': {
                'start': prepared_data.index[0].isoformat(),
                'end': prepared_data.index[-1].isoformat()
            },
            'bars_analyzed': self.counters['bars_analyzed'],
            'hsmm_passed': self.counters['hsmm_passed'],
            'smc_summary': {
                'with_any_pattern': self.counters['with_any_pattern'],
                'without_pattern': self.counters['without_pattern'],
                'bullish_ob': self.counters['bullish_ob'],
                'bearish_ob': self.counters['bearish_ob'],
                'bullish_fvg': self.counters['bullish_fvg'],
                'bearish_fvg': self.counters['bearish_fvg'],
                'multiple_patterns': self.counters['multiple_patterns']
            },
            'side_breakdown': self.sides,
            'coverage_ratios': ratios,
            'summary': self._generate_summary(ratios)
        }
        
        return report
    
    def _track_patterns(self, signal: Dict):
        """
        Track SMC patterns for a signal
        
        Args:
            signal: Signal dict from NYXEngine
        """
        
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        reasons = signal.get('reasons', [])
        reasons_str = ' '.join(str(r) for r in reasons)
        
        # Check if passed HSMM
        # (confidence > 0 or not insufficient_data)
        if 'Insufficient data' in reasons_str:
            return
        
        if 'HSMM initialization failed' in reasons_str:
            return
        
        # Get HSMM threshold
        sdc_threshold = self.config.get('strategy', {}).get('hsmm', {}).get('sdc_threshold', 3.5)
        
        # Check if passed HSMM confidence
        if confidence < sdc_threshold:
            return
        
        # This is an HSMM-passed candidate
        self.counters['hsmm_passed'] += 1
        
        # Track side
        if action == 'BUY':
            self.sides['long_candidates'] += 1
        elif action == 'SELL':
            self.sides['short_candidates'] += 1
        
        # Get SMC patterns
        smc_patterns = signal.get('smc_patterns', {})
        
        bullish_ob = smc_patterns.get('bullish_ob', False)
        bearish_ob = smc_patterns.get('bearish_ob', False)
        bullish_fvg = smc_patterns.get('bullish_fvg', False)
        bearish_fvg = smc_patterns.get('bearish_fvg', False)
        
        # Count individual patterns
        if bullish_ob:
            self.counters['bullish_ob'] += 1
        if bearish_ob:
            self.counters['bearish_ob'] += 1
        if bullish_fvg:
            self.counters['bullish_fvg'] += 1
        if bearish_fvg:
            self.counters['bearish_fvg'] += 1
        
        # Check if has any pattern
        has_any = bullish_ob or bearish_ob or bullish_fvg or bearish_fvg
        
        if has_any:
            self.counters['with_any_pattern'] += 1
            
            # Track side with pattern
            if action == 'BUY':
                self.sides['long_with_pattern'] += 1
            elif action == 'SELL':
                self.sides['short_with_pattern'] += 1
            
            # Count multiple patterns
            pattern_count = sum([bullish_ob, bearish_ob, bullish_fvg, bearish_fvg])
            if pattern_count > 1:
                self.counters['multiple_patterns'] += 1
        else:
            self.counters['without_pattern'] += 1
            
            # Track side without pattern
            if action == 'BUY':
                self.sides['long_without_pattern'] += 1
            elif action == 'SELL':
                self.sides['short_without_pattern'] += 1
    
    def _calculate_ratios(self) -> Dict:
        """Calculate coverage ratios"""
        
        hsmm = self.counters['hsmm_passed']
        
        if hsmm == 0:
            return {
                'pattern_coverage_ratio': 0,
                'long_pattern_coverage_ratio': 0,
                'short_pattern_coverage_ratio': 0,
                'coverage_level': 'N/A (no HSMM candidates)'
            }
        
        pattern_coverage = (self.counters['with_any_pattern'] / hsmm) * 100
        
        long_total = self.sides['long_candidates']
        short_total = self.sides['short_candidates']
        
        long_coverage = (self.sides['long_with_pattern'] / long_total * 100) if long_total > 0 else 0
        short_coverage = (self.sides['short_with_pattern'] / short_total * 100) if short_total > 0 else 0
        
        # Interpret coverage level
        if pattern_coverage < 20:
            level = 'very low coverage'
        elif pattern_coverage < 40:
            level = 'low coverage'
        elif pattern_coverage < 60:
            level = 'moderate coverage'
        else:
            level = 'high coverage'
        
        return {
            'pattern_coverage_ratio': pattern_coverage,
            'long_pattern_coverage_ratio': long_coverage,
            'short_pattern_coverage_ratio': short_coverage,
            'coverage_level': level
        }
    
    def _generate_summary(self, ratios: Dict) -> str:
        """Generate text summary"""
        
        hsmm = self.counters['hsmm_passed']
        
        summary = f"""
Bars analyzed: {self.counters['bars_analyzed']}

HSMM-passed candidates:  {hsmm:6d}
With any SMC pattern:    {self.counters['with_any_pattern']:6d}  ({ratios['pattern_coverage_ratio']:.1f}%)
Without SMC pattern:     {self.counters['without_pattern']:6d}  ({(self.counters['without_pattern']/hsmm*100 if hsmm > 0 else 0):.1f}%)

Pattern breakdown:
  Bullish OB:            {self.counters['bullish_ob']:6d}
  Bearish OB:            {self.counters['bearish_ob']:6d}
  Bullish FVG:           {self.counters['bullish_fvg']:6d}
  Bearish FVG:           {self.counters['bearish_fvg']:6d}
  Multiple patterns:     {self.counters['multiple_patterns']:6d}

Side breakdown:
  Long candidates:       {self.sides['long_candidates']:6d}
  Long with pattern:     {self.sides['long_with_pattern']:6d}  ({ratios['long_pattern_coverage_ratio']:.1f}%)
  Short candidates:      {self.sides['short_candidates']:6d}
  Short with pattern:    {self.sides['short_with_pattern']:6d}  ({ratios['short_pattern_coverage_ratio']:.1f}%)

Coverage level: {ratios['coverage_level']}
"""
        
        return summary


def save_smc_diagnostics(report: Dict, output_dir: str):
    """
    Save SMC diagnostics to JSON
    
    Args:
        report: Diagnostics report dict
        output_dir: Output directory
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pair = report.get('pair', 'UNKNOWN')
    json_file = output_path / f"{pair}_smc_diagnostics.json"
    
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ SMC diagnostics saved: {json_file}")


if __name__ == "__main__":
    print("SMC Diagnostics")
    
    import yaml
    
    # Load config
    with open('config/validation_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set min_required_bars to 50
    config['strategy']['min_required_bars'] = 50
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
    data = pd.DataFrame({
        'open': 50000 + np.random.randn(len(dates)) * 1000,
        'high': 51000 + np.random.randn(len(dates)) * 1000,
        'low': 49000 + np.random.randn(len(dates)) * 1000,
        'close': 50000 + np.random.randn(len(dates)) * 1000,
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    # Run diagnostics
    diag = SMCDiagnostics(config)
    report = diag.run(data, pair='BTCUSDT')
    
    print(report['summary'])
