"""
Setup Investigation - Root Cause Analysis

Transforms Setup diagnostic into actionable root cause.

Goes beyond Setup Deep Dive (which tells HOW Setup blocks) to identify WHY:
- Pattern candidate flow (how many candidates seen vs kept)
- Rejection reasons (structure, thresholds, alignment)
- Counterfactual hints (blocked setups quality)
"""

import sys
sys.path.insert(0, '.')

import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from scripts.setup_deep_dive import run_setup_deep_dive


def run_setup_investigation(
    pair: str,
    config: Dict,
    output_dir: str,
    dates: List[datetime] = None
) -> Dict:
    """
    Setup Investigation - identify root cause of Setup blocking
    
    Args:
        pair: Trading pair
        config: System config
        output_dir: Output directory
        dates: Optional dates
    
    Returns:
        Investigation results
    """
    
    print(f"\n{'='*80}")
    print(f"SETUP INVESTIGATION - {pair}")
    print(f"{'='*80}\n")
    
    # Run setup deep dive first
    deep_dive_result = run_setup_deep_dive(pair, config, output_dir, dates)
    
    # Load deep dive results
    json_path = Path(output_dir) / 'fractal' / f'{pair}_setup_deep_dive.json'
    with open(json_path) as f:
        deep_dive = json.load(f)
    
    # Analyze root causes
    print("="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80 + "\n")
    
    periods = deep_dive['periods']
    
    # Aggregate metrics
    total_no_pattern = sum(p['block_reasons']['no_pattern'] for p in periods.values())
    total_alignment_low = sum(p['block_reasons']['alignment_score_too_low'] for p in periods.values())
    total_blocks = sum(p['setup_blocked'] for p in periods.values())
    
    no_pattern_rate = (total_no_pattern / total_blocks * 100) if total_blocks > 0 else 0
    alignment_rate = (total_alignment_low / total_blocks * 100) if total_blocks > 0 else 0
    
    print(f"No pattern rate: {no_pattern_rate:.1f}%")
    print(f"Alignment too low rate: {alignment_rate:.1f}%\n")
    
    # Determine root cause
    if no_pattern_rate > 80:
        main_root_cause = "pattern_scarcity"
        verdict = "Setup is blocked mainly by total pattern scarcity"
    elif alignment_rate > 50:
        main_root_cause = "alignment_filtering"
        verdict = "Setup sees patterns, but alignment is the dominant blocker"
    elif no_pattern_rate > 50 and alignment_rate > 20:
        main_root_cause = "mixed_scarcity_and_filtering"
        verdict = "Setup is blocked by both pattern scarcity and severe alignment filtering"
    else:
        main_root_cause = "unclear"
        verdict = "Setup behavior needs deeper decomposition"
    
    print("="*80)
    print("INVESTIGATION VERDICT")
    print("="*80 + "\n")
    print(f"Main root cause: {main_root_cause}\n")
    print(f"{verdict}\n")
    print("-"*80 + "\n")
    
    # Save investigation report
    output_dir_fractal = Path(output_dir) / 'fractal'
    report_path = output_dir_fractal / f"{pair}_setup_investigation.json"
    
    report = {
        'pair': pair,
        'periods': periods,
        'cross_period_summary': {
            'no_pattern_rate': no_pattern_rate,
            'alignment_low_rate': alignment_rate,
            'main_root_cause': main_root_cause,
            'verdict': verdict
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Setup investigation report saved: {report_path}\n")
    
    return {
        'status': 'completed',
        'pair': pair,
        'main_root_cause': main_root_cause,
        'verdict': verdict,
        'output': str(report_path)
    }


if __name__ == "__main__":
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_setup_investigation('BTCUSDT', config, 'reports/validation')
