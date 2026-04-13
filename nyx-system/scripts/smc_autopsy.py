"""
SMC Autopsy CLI Script

Run technical autopsy of SMC detector to understand why it's silent.
"""

import sys
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.insert(0, '.')

from src.data.mtf_loader import load_fractal_context
from src.validation.smc_autopsy import SMCAutopsy


def run_smc_autopsy(pair: str, config: dict, output_dir: Path,
                    sample_date: str = "", sensitivity: bool = False):
    """
    Run SMC detector autopsy
    
    Args:
        pair: Trading pair
        config: System configuration
        output_dir: Output directory
        sample_date: Reference date
        sensitivity: Run sensitivity analysis
    """
    
    print(f"\n{'='*80}")
    print(f"SMC DETECTOR AUTOPSY - {pair}")
    print(f"{'='*80}")
    
    # Parse sample date
    if sample_date:
        current_date = datetime.strptime(sample_date, '%Y-%m-%d')
    else:
        current_date = datetime(2024, 1, 1)
    
    print(f"\nReference date: {current_date.strftime('%Y-%m-%d')}")
    
    # Load fractal context
    print(f"\n📊 Loading fractal context...")
    
    try:
        mtf_data = load_fractal_context(pair, current_date, config)
    except Exception as e:
        print(f"❌ Failed to load fractal context: {e}")
        return {'status': 'error', 'reason': str(e)}
    
    print(f"✓ Loaded {len(mtf_data)} timeframes")
    
    # Get setup timeframe (15m)
    setup_tf = '15m'
    if setup_tf not in mtf_data:
        print(f"❌ Setup timeframe {setup_tf} not found")
        return {'status': 'error', 'reason': 'setup_tf_missing'}
    
    setup_data = mtf_data[setup_tf]
    
    # Initialize autopsy
    print(f"\n🔬 Initializing SMC Autopsy...")
    autopsy = SMCAutopsy(config)
    
    print(f"Configuration:")
    print(f"  OB range threshold:  {autopsy.ob_threshold:.3f} ({autopsy.ob_threshold*100:.1f}%)")
    print(f"  FVG min gap:         {autopsy.fvg_min_gap:.3f} ({autopsy.fvg_min_gap*100:.1f}%)")
    
    # Run analysis
    print(f"\n🔬 Analyzing {len(setup_data)} bars...")
    
    warmup = 50
    bars_to_analyze = min(100, len(setup_data) - warmup)
    
    patterns_detected = 0
    setup_passed = 0
    
    for i in range(warmup, warmup + bars_to_analyze):
        # Get data up to this bar
        current_data = setup_data.iloc[:i+1]
        
        # Analyze
        patterns = autopsy.analyze_bar(current_data)
        
        # Check if any pattern detected
        has_any_pattern = any(patterns.values())
        
        if has_any_pattern:
            patterns_detected += 1
            # Would need to check alignment here for actual setup pass
    
    # Get summary
    summary = autopsy.get_summary()
    verdict = autopsy.determine_verdict(summary)
    
    # Print results
    _print_summary(summary, verdict, bars_to_analyze, patterns_detected, setup_passed)
    
    # Save JSON
    output_dir_fractal = Path(output_dir) / 'fractal'
    output_dir_fractal.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir_fractal / f"{pair}_smc_autopsy.json"
    
    report = {
        'pair': pair,
        'reference_date': current_date.isoformat(),
        'bars_analyzed': bars_to_analyze,
        'code_path': summary['code_path'],
        'fvg_audit': summary['fvg_audit'],
        'ob_audit': summary['ob_audit'],
        'configuration': summary['configuration'],
        'final_output': {
            'patterns_detected': patterns_detected,
            'setup_passed': setup_passed
        },
        'verdict': verdict
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Autopsy report saved: {report_path}")
    
    # Run sensitivity analysis if requested
    if sensitivity:
        print(f"\n{'='*80}")
        print(f"SENSITIVITY ANALYSIS")
        print(f"{'='*80}")
        
        sensitivity_results = _run_sensitivity_analysis(
            setup_data, config, warmup, bars_to_analyze
        )
        
        # Add to report
        report['sensitivity_analysis'] = sensitivity_results
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Sensitivity results added to report")
    
    return {
        'status': 'completed',
        'pair': pair,
        'verdict': verdict['category'],
        'output': str(report_path)
    }


def _print_summary(summary: dict, verdict: dict, bars_analyzed: int, 
                   patterns_detected: int, setup_passed: int):
    """Print human-readable summary"""
    
    print(f"\n{'='*80}")
    print(f"SMC AUTOPSY RESULTS")
    print(f"{'='*80}")
    
    code_path = summary['code_path']
    fvg = summary['fvg_audit']
    ob = summary['ob_audit']
    
    print(f"\nCODE PATH")
    print(f"Detector called:           {'YES' if code_path['detector_called'] else 'NO'}")
    print(f"FVG function calls:        {code_path['fvg_function_called']}")
    print(f"OB function calls:         {code_path['ob_function_called']}")
    
    print(f"\nFVG AUDIT")
    print(f"Candidates seen:           {fvg['candidates_seen']}")
    print(f"Detected:                  {fvg['detected']}")
    if fvg['candidates_seen'] > 0:
        main_failure = 'gap threshold' if fvg['failed_gap_threshold'] > fvg['failed_structure_rules'] else 'structure rules'
        print(f"Main failure:              {main_failure} ({fvg['failed_gap_threshold']} gap, {fvg['failed_structure_rules']} structure)")
    else:
        print(f"Main failure:              none / no candidate structures")
    
    print(f"\nOB AUDIT")
    print(f"Candidates seen:           {ob['candidates_seen']}")
    print(f"Detected:                  {ob['detected']}")
    if ob['candidates_seen'] > 0:
        main_failure = 'range threshold' if ob['failed_range_threshold'] > ob['failed_followthrough'] else 'followthrough'
        print(f"Main failure:              {main_failure} ({ob['failed_range_threshold']} range, {ob['failed_followthrough']} followthrough)")
    else:
        print(f"Main failure:              none / no candidate structures")
    
    print(f"\nFINAL OUTPUT")
    print(f"Patterns detected:         {patterns_detected}")
    print(f"Setup passed:              {setup_passed}")
    
    print(f"\nVERDICT")
    print(f"Category: {verdict['category']}")
    print(f"{verdict['summary']}")
    
    print(f"{'='*80}\n")


def _run_sensitivity_analysis(data: pd.DataFrame, config: dict, 
                               warmup: int, bars_to_analyze: int) -> dict:
    """
    Run sensitivity analysis with alternative thresholds
    
    Args:
        data: Setup timeframe data
        config: System configuration
        warmup: Warmup bars
        bars_to_analyze: Bars to analyze
    
    Returns:
        Sensitivity results
    """
    
    # Test configurations
    test_configs = [
        {
            'name': 'baseline',
            'ob_range_threshold': 0.015,
            'fvg_min_gap': 0.005
        },
        {
            'name': 'relaxed_25pct',
            'ob_range_threshold': 0.0112,  # 25% lower
            'fvg_min_gap': 0.00375
        },
        {
            'name': 'relaxed_50pct',
            'ob_range_threshold': 0.0075,  # 50% lower
            'fvg_min_gap': 0.0025
        },
        {
            'name': 'very_relaxed',
            'ob_range_threshold': 0.005,  # Very permissive
            'fvg_min_gap': 0.001
        }
    ]
    
    results = []
    
    for test_config in test_configs:
        print(f"\nTesting: {test_config['name']}")
        print(f"  OB threshold: {test_config['ob_range_threshold']:.3f} ({test_config['ob_range_threshold']*100:.1f}%)")
        print(f"  FVG min gap:  {test_config['fvg_min_gap']:.3f} ({test_config['fvg_min_gap']*100:.1f}%)")
        
        # Create modified config
        test_sys_config = config.copy()
        test_sys_config['smc_detector'] = {
            'ob_range_threshold': test_config['ob_range_threshold'],
            'fvg_min_gap': test_config['fvg_min_gap'],
            'liquidity_lookback': config.get('smc_detector', {}).get('liquidity_lookback', 20)
        }
        
        # Run autopsy with this config
        autopsy = SMCAutopsy(test_sys_config)
        
        patterns_detected = 0
        
        for i in range(warmup, warmup + bars_to_analyze):
            current_data = data.iloc[:i+1]
            patterns = autopsy.analyze_bar(current_data)
            
            if any(patterns.values()):
                patterns_detected += 1
        
        summary = autopsy.get_summary()
        
        result = {
            'name': test_config['name'],
            'configuration': test_config,
            'fvg_detected': summary['fvg_audit']['detected'],
            'ob_detected': summary['ob_audit']['detected'],
            'total_patterns': patterns_detected,
            'detection_rate': patterns_detected / bars_to_analyze if bars_to_analyze > 0 else 0
        }
        
        results.append(result)
        
        print(f"  → Patterns detected: {patterns_detected} ({result['detection_rate']:.1%})")
    
    return {
        'test_configurations': results,
        'summary': _summarize_sensitivity(results)
    }


def _summarize_sensitivity(results: list) -> str:
    """Summarize sensitivity analysis"""
    
    baseline = results[0]
    best = max(results, key=lambda x: x['total_patterns'])
    
    if best['total_patterns'] == baseline['total_patterns']:
        return "Relaxing thresholds does not improve detection - detector may be fundamentally silent or broken."
    else:
        improvement = best['total_patterns'] - baseline['total_patterns']
        return f"Relaxing thresholds to '{best['name']}' improves detection by {improvement} patterns ({best['detection_rate']:.1%} vs {baseline['detection_rate']:.1%})."


if __name__ == "__main__":
    import yaml
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_smc_autopsy('BTCUSDT', config, Path('reports/validation'), sample_date='2023-12-15', sensitivity=True)
