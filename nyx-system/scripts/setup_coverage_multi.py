"""
Multi-Period Setup Coverage Audit

Compare Setup behavior across Bull/Bear/Range market periods.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

from scripts.setup_bottleneck import run_setup_bottleneck


# Default reference dates
DEFAULT_DATES = [
    ('2022-06-15', 'bear'),   # Bear/stress market
    ('2023-03-15', 'bull'),   # Bull market
    ('2023-12-15', 'range'),  # Range/quiet market
]


def run_setup_coverage_multi(pair: str, config: dict, output_dir: Path, dates: list = None):
    """
    Compare Setup coverage across multiple periods
    
    Determines if setup silence is regime-dependent or global.
    """
    
    print(f"\n{'='*80}")
    print(f"MULTI-PERIOD SETUP COVERAGE - {pair}")
    print(f"{'='*80}")
    
    # Use default dates if not provided
    if dates is None:
        test_dates = DEFAULT_DATES
    else:
        # Parse custom dates (assume all are unknown regime)
        test_dates = [(d, 'unknown') for d in dates]
    
    print(f"\nTesting {len(test_dates)} periods:")
    for date, regime in test_dates:
        print(f"  - {date} ({regime})")
    
    # Run setup_bottleneck for each period
    periods = {}
    
    for date_str, regime_label in test_dates:
        print(f"\n{'='*60}")
        print(f"Testing period: {date_str} ({regime_label})")
        print(f"{'='*60}")
        
        try:
            # Run setup bottleneck for this period
            result = run_setup_bottleneck(
                pair=pair,
                config=config,
                output_dir=output_dir,
                sample_date=date_str
            )
            
            # Load the generated JSON
            bottleneck_path = output_dir / 'fractal' / f'{pair}_setup_bottleneck.json'
            with open(bottleneck_path) as f:
                bottleneck_report = json.load(f)
            
            # Extract key metrics
            periods[date_str] = {
                'regime_label': regime_label,
                'bars_ready_for_decision': bottleneck_report['bars_ready_for_decision'],
                'setup_pass_rate': bottleneck_report['setup_summary']['pass_rate'],
                'block_reasons': bottleneck_report['block_reasons'],
                'pattern_observed': bottleneck_report['pattern_observed'],
                'primary_setup_issue': bottleneck_report['verdict']['primary_setup_issue'],
                'sample_blocked_setups': bottleneck_report['sample_blocked_setups'][:5]  # Keep 5 samples
            }
            
        except Exception as e:
            print(f"❌ Failed to test {date_str}: {e}")
            # Add minimal entry
            periods[date_str] = {
                'regime_label': regime_label,
                'error': str(e)
            }
    
    # Compute cross-period summary
    print(f"\n{'='*80}")
    print(f"CROSS-PERIOD ANALYSIS")
    print(f"{'='*80}")
    
    cross_summary = _compute_cross_period_summary(periods, test_dates)
    
    # Print summary
    _print_summary(periods, cross_summary, test_dates)
    
    # Save comparative JSON
    output_dir_fractal = Path(output_dir) / 'fractal'
    output_dir_fractal.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir_fractal / f"{pair}_setup_coverage_multi.json"
    
    report = {
        'pair': pair,
        'periods': periods,
        'cross_period_summary': cross_summary
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Multi-period report saved: {report_path}")
    
    return {
        'status': 'completed',
        'pair': pair,
        'periods_tested': len(periods),
        'verdict': cross_summary['global_interpretation'],
        'output': str(report_path)
    }


def _compute_cross_period_summary(periods: dict, test_dates: list) -> dict:
    """Compute cross-period statistics and verdict"""
    
    # Extract metrics per regime
    bull_metrics = []
    bear_metrics = []
    range_metrics = []
    
    for date_str, regime_label in test_dates:
        if date_str not in periods or 'error' in periods[date_str]:
            continue
        
        period = periods[date_str]
        
        if regime_label == 'bull':
            bull_metrics.append(period)
        elif regime_label == 'bear':
            bear_metrics.append(period)
        elif regime_label == 'range':
            range_metrics.append(period)
    
    # Compute averages
    all_pass_rates = [p['setup_pass_rate'] for p in periods.values() if 'setup_pass_rate' in p]
    avg_pass_rate = sum(all_pass_rates) / len(all_pass_rates) if all_pass_rates else 0.0
    
    # Analyze behavior per regime
    bull_behavior = _summarize_regime_behavior(bull_metrics, 'bull')
    bear_behavior = _summarize_regime_behavior(bear_metrics, 'bear')
    range_behavior = _summarize_regime_behavior(range_metrics, 'range')
    
    # Determine global verdict
    verdict = _determine_cross_period_verdict(periods, bull_behavior, bear_behavior, range_behavior)
    
    return {
        'average_setup_pass_rate': round(avg_pass_rate, 1),
        'bull_behavior': bull_behavior,
        'bear_behavior': bear_behavior,
        'range_behavior': range_behavior,
        'global_interpretation': verdict
    }


def _summarize_regime_behavior(metrics: list, regime: str) -> str:
    """Summarize Setup behavior for a regime"""
    
    if not metrics:
        return f"no {regime} periods tested"
    
    # Average pass rate
    avg_pass = sum(m['setup_pass_rate'] for m in metrics) / len(metrics)
    
    # Total patterns observed
    total_patterns = 0
    for m in metrics:
        patterns = m['pattern_observed']
        total_patterns += patterns.get('bullish_fvg', 0)
        total_patterns += patterns.get('bearish_fvg', 0)
        total_patterns += patterns.get('bullish_ob', 0)
        total_patterns += patterns.get('bearish_ob', 0)
    
    # Dominant block reason
    all_block_reasons = {}
    for m in metrics:
        for reason, count in m['block_reasons'].items():
            all_block_reasons[reason] = all_block_reasons.get(reason, 0) + count
    
    if all_block_reasons:
        dominant_reason = max(all_block_reasons.items(), key=lambda x: x[1])[0]
    else:
        dominant_reason = 'unknown'
    
    # Categorize
    if avg_pass < 5.0:
        if total_patterns == 0:
            return "very low pattern coverage"
        else:
            return "patterns present but mostly rejected"
    elif avg_pass < 20.0:
        return "moderate pattern presence with alignment issues"
    else:
        return "good pattern coverage"


def _determine_cross_period_verdict(periods: dict, bull: str, bear: str, range_: str) -> str:
    """Determine cross-period verdict (Cas A, B, C, or D)"""
    
    # Count how many periods have very low coverage
    very_low_count = 0
    moderate_count = 0
    good_count = 0
    
    for behavior in [bull, bear, range_]:
        if "very low" in behavior:
            very_low_count += 1
        elif "moderate" in behavior:
            moderate_count += 1
        elif "good" in behavior:
            good_count += 1
    
    # Check if patterns exist but are rejected
    patterns_exist_but_rejected = False
    alignment_is_dominant = False
    
    for period in periods.values():
        if 'pattern_observed' not in period:
            continue
        
        patterns = period['pattern_observed']
        total_patterns = sum(patterns.values()) - patterns.get('no_pattern', 0)
        
        if total_patterns > 0:
            patterns_exist_but_rejected = True
            
            # Check if misaligned_pattern or alignment_score_too_low dominates
            blocks = period['block_reasons']
            alignment_blocks = blocks.get('misaligned_pattern', 0) + blocks.get('alignment_score_too_low', 0)
            total_blocks = sum(blocks.values())
            
            if alignment_blocks > total_blocks * 0.5:
                alignment_is_dominant = True
    
    # Determine verdict
    if very_low_count == 3:
        # All periods have very low coverage
        return "Detector is globally too silent across all tested regimes"  # Cas B
    
    elif very_low_count >= 1 and "very low" in range_:
        # Range has very low coverage but others better
        return "Detector silence is regime-dependent and strongest in range markets"  # Cas A
    
    elif alignment_is_dominant:
        # Patterns exist but alignment rejects them
        return "Patterns exist across periods, but alignment is the dominant blocker"  # Cas C
    
    else:
        # Mixed behavior
        return "Setup behavior is mixed and needs deeper decomposition"  # Cas D


def _print_summary(periods: dict, cross_summary: dict, test_dates: list):
    """Print human-readable summary"""
    
    print(f"\n{'='*80}")
    print(f"MULTI-PERIOD SETUP COVERAGE SUMMARY")
    print(f"{'='*80}\n")
    
    for date_str, regime_label in test_dates:
        if date_str not in periods:
            continue
        
        period = periods[date_str]
        
        if 'error' in period:
            print(f"{date_str} ({regime_label})")
            print(f"  ERROR: {period['error']}\n")
            continue
        
        print(f"{date_str} ({regime_label.upper()})")
        print(f"  Ready bars:         {period['bars_ready_for_decision']:>6}")
        print(f"  Setup pass rate:    {period['setup_pass_rate']:>5.1f}%")
        print(f"  Primary issue:      {period['primary_setup_issue']}")
        
        patterns = period['pattern_observed']
        print(f"  Patterns: BFVG {patterns.get('bullish_fvg', 0)} / " +
              f"BeFVG {patterns.get('bearish_fvg', 0)} / " +
              f"BOB {patterns.get('bullish_ob', 0)} / " +
              f"BeOB {patterns.get('bearish_ob', 0)}")
        print()
    
    print(f"{'='*80}")
    print(f"CROSS-PERIOD CONCLUSION")
    print(f"{'='*80}")
    print(f"Average setup pass rate: {cross_summary['average_setup_pass_rate']:.1f}%")
    print(f"\nBull behavior:  {cross_summary['bull_behavior']}")
    print(f"Bear behavior:  {cross_summary['bear_behavior']}")
    print(f"Range behavior: {cross_summary['range_behavior']}")
    print(f"\nGlobal interpretation:")
    print(f"{cross_summary['global_interpretation']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import yaml
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_setup_coverage_multi('BTCUSDT', config, Path('reports/validation'))
