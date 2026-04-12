"""
Setup Deep Dive - Pattern Detection and Blocking Analysis

Measures precisely HOW Setup blocks the pipeline.

Purpose:
--------
Now that Regime is no longer the main blocker (75% trend+), we need to understand
why Setup blocks 97.7% of signals. This deep dive analyzes:
- How many bars are ready for Setup decision
- Setup pass rate per period
- Blocking reasons (no_pattern, misaligned, alignment_too_low, etc.)
- Patterns actually observed (FVG, OB)
- Alignment score distribution
- Sample blocked setups

Approach:
---------
Similar to HSMM Deep Dive and Fractal Bottleneck, but focused on Setup layer.
For each test period:
1. Load fractal context
2. Iterate through bars using orchestrator
3. Analyze Setup component of each decision
4. Categorize blocking reasons
5. Count patterns observed
6. Analyze alignment scores
7. Provide sample examples

Output:
-------
- JSON report with per-period details
- Cross-period summary
- Clear verdict on dominant blocking reason
"""

import sys
sys.path.insert(0, '.')

import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from src.data.mtf_loader import load_fractal_context, MTFLoader
from src.agents.orchestrator import Orchestrator


def _tf_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes"""
    mapping = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
    }
    return mapping.get(tf, 0)


def analyze_setup_period(
    pair: str,
    reference_date: datetime,
    config: Dict
) -> Dict[str, Any]:
    """
    Analyze Setup blocking for a single period
    
    Args:
        pair: Trading pair
        reference_date: Reference date
        config: System config
    
    Returns:
        Dict with period analysis
    """
    
    # Load data
    mtf_data = load_fractal_context(pair, reference_date, config)
    
    # Create orchestrator
    orchestrator = Orchestrator(config)
    
    # Get lowest TF for iteration
    lowest_tf = min(mtf_data.keys(), key=lambda x: _tf_to_minutes(x))
    warmup = 50
    n_signals = len(mtf_data[lowest_tf]) - warmup
    
    # Analyze Setup blocks
    bars_ready = 0
    setup_passed = 0
    setup_blocked = 0
    
    block_reasons = {
        'no_pattern': 0,
        'misaligned_pattern': 0,
        'alignment_score_too_low': 0,
        'direction_conflict': 0,
        'setup_not_ready': 0,
        'unknown': 0
    }
    
    patterns_observed = {
        'bullish_fvg': 0,
        'bearish_fvg': 0,
        'bullish_ob': 0,
        'bearish_ob': 0,
        'no_pattern': 0
    }
    
    alignment_scores = []
    blocked_examples = []
    
    # Process each bar
    loader = MTFLoader('data/raw/mtf')
    
    for i in range(warmup, warmup + n_signals):
        try:
            current_time = pd.Timestamp(mtf_data[lowest_tf].index[i])

            # Align data at this timestamp
            aligned_data = loader.align_at_timestamp(mtf_data, current_time, lowest_tf)
            
            # Get decision
            decision = orchestrator.decide(aligned_data)
            
            # Check if all agents ready
            all_agents_ready = all(comp.ready for comp in decision.components.values())
            
            if not all_agents_ready:
                # Skip if not all agents ready
                continue
            
            bars_ready += 1
            
            # Get Setup component
            setup_comp = decision.components.get('setup')
            if not setup_comp:
                continue
            
            setup_state = setup_comp.state
            setup_passed_flag = setup_comp.passed
            setup_score = setup_comp.score if hasattr(setup_comp, 'score') else 0.0
            
            # Track alignment score
            if setup_score > 0:
                alignment_scores.append(setup_score)
            
            # Count pass/block
            if setup_passed_flag:
                setup_passed += 1
            else:
                setup_blocked += 1
                
                # Categorize blocking reason
                # Check setup state for hints
                state_lower = setup_state.lower() if setup_state else ''
                
                if 'no_pattern' in state_lower or setup_state == 'misaligned':
                    # Check if it's truly no pattern or just misaligned
                    # If score is 0 or very low and state is misaligned, it's likely no pattern
                    if setup_score < 0.1:
                        block_reasons['no_pattern'] += 1
                        patterns_observed['no_pattern'] += 1
                    elif setup_score < 0.6:  # Below alignment threshold
                        block_reasons['alignment_score_too_low'] += 1
                    else:
                        block_reasons['misaligned_pattern'] += 1
                elif 'fvg' in state_lower:
                    # Pattern exists but blocked
                    if setup_score < 0.6:
                        block_reasons['alignment_score_too_low'] += 1
                    else:
                        block_reasons['misaligned_pattern'] += 1
                    
                    # Count pattern
                    if 'bullish' in state_lower:
                        patterns_observed['bullish_fvg'] += 1
                    else:
                        patterns_observed['bearish_fvg'] += 1
                elif 'ob' in state_lower or 'order' in state_lower:
                    # Pattern exists but blocked
                    if setup_score < 0.6:
                        block_reasons['alignment_score_too_low'] += 1
                    else:
                        block_reasons['misaligned_pattern'] += 1
                    
                    # Count pattern
                    if 'bullish' in state_lower:
                        patterns_observed['bullish_ob'] += 1
                    else:
                        patterns_observed['bearish_ob'] += 1
                else:
                    block_reasons['unknown'] += 1
                
                # Save example
                if len(blocked_examples) < 5:
                    blocked_examples.append({
                        'timestamp': str(current_time),
                        'state': setup_state,
                        'alignment_score': float(setup_score),
                        'context_state': decision.components['context'].state if 'context' in decision.components else 'unknown',
                        'regime_state': decision.components['regime'].state if 'regime' in decision.components else 'unknown'
                    })
        
        except Exception as e:
            # Skip errors
            continue
    
    # Calculate pass rate
    setup_pass_rate = (setup_passed / bars_ready * 100) if bars_ready > 0 else 0
    
    # Determine primary issue
    primary_issue = max(block_reasons.items(), key=lambda x: x[1])[0] if sum(block_reasons.values()) > 0 else 'unknown'
    
    # Alignment summary
    alignment_summary = {
        'mean': float(np.mean(alignment_scores)) if alignment_scores else 0.0,
        'median': float(np.median(alignment_scores)) if alignment_scores else 0.0,
        'min': float(np.min(alignment_scores)) if alignment_scores else 0.0,
        'max': float(np.max(alignment_scores)) if alignment_scores else 0.0,
        'threshold': 0.6
    }
    
    return {
        'bars_ready_for_decision': bars_ready,
        'setup_passed': setup_passed,
        'setup_blocked': setup_blocked,
        'setup_pass_rate': setup_pass_rate,
        'block_reasons': block_reasons,
        'patterns_observed': patterns_observed,
        'alignment_summary': alignment_summary,
        'primary_setup_issue': primary_issue,
        'sample_blocked_setups': blocked_examples
    }


def _determine_setup_verdict(period_results: List[Dict]) -> tuple:
    """
    Determine main Setup blocking reason across periods
    
    Returns:
        (main_issue, verdict) tuple
    """
    
    # Aggregate metrics
    avg_pass_rate = np.mean([p['setup_pass_rate'] for p in period_results])
    
    # Aggregate block reasons
    total_no_pattern = sum(p['block_reasons']['no_pattern'] for p in period_results)
    total_alignment_low = sum(p['block_reasons']['alignment_score_too_low'] for p in period_results)
    total_misaligned = sum(p['block_reasons']['misaligned_pattern'] for p in period_results)
    total_blocks = sum(p['setup_blocked'] for p in period_results)
    
    # Calculate proportions
    no_pattern_rate = (total_no_pattern / total_blocks) if total_blocks > 0 else 0
    alignment_low_rate = (total_alignment_low / total_blocks) if total_blocks > 0 else 0
    misaligned_rate = (total_misaligned / total_blocks) if total_blocks > 0 else 0
    
    # Determine verdict
    if no_pattern_rate > 0.80:
        return (
            "pattern_absence_dominant",
            "Setup is dominated by total absence of patterns"
        )
    elif alignment_low_rate > 0.50:
        return (
            "alignment_dominant",
            "Patterns exist, but alignment is the main blocker"
        )
    elif no_pattern_rate > 0.50 and alignment_low_rate > 0.20:
        return (
            "mixed_pattern_and_alignment",
            "Setup is mixed: both pattern scarcity and alignment matter"
        )
    elif avg_pass_rate < 5:
        return (
            "severe_blocking",
            "Setup blocks almost everything (<5% pass rate) - needs decomposition"
        )
    else:
        return (
            "inconclusive",
            "Setup diagnostics remain inconclusive and need deeper decomposition"
        )


def run_setup_deep_dive(
    pair: str,
    config: Dict,
    output_dir: str,
    dates: "List[datetime] | None" = None
) -> Dict:
    """
    Run Setup Deep Dive analysis
    
    Args:
        pair: Trading pair
        config: System config
        output_dir: Output directory
        dates: Optional list of dates (defaults to 3 test periods)
    
    Returns:
        Results dict
    """
    
    print(f"\n{'='*80}")
    print(f"SETUP DEEP DIVE - {pair}")
    print(f"{'='*80}\n")
    
    # Default test dates if not provided
    if dates is None:
        dates = [
            datetime(2022, 6, 15),   # Bear period
            datetime(2023, 3, 15),   # Bull period
            datetime(2023, 12, 15),  # Range period
        ]
    
    print(f"Test periods: {len(dates)}")
    for date in dates:
        print(f"  - {date.date()}")
    print()
    
    # Analyze each period
    print("="*80)
    print("ANALYZING PERIODS")
    print("="*80 + "\n")
    
    period_results = []
    
    for date in dates:
        print(f"Analyzing {date.date()}...")
        
        result = analyze_setup_period(pair, date, config)
        result['date'] = date.isoformat()
        
        period_results.append(result)
        
        # Print summary
        print(f"  Bars ready: {result['bars_ready_for_decision']}")
        print(f"  Setup pass rate: {result['setup_pass_rate']:.1f}%")
        print(f"  Primary issue: {result['primary_setup_issue']}")
        print()
    
    # Cross-period analysis
    print("="*80)
    print("CROSS-PERIOD ANALYSIS")
    print("="*80 + "\n")
    
    avg_pass_rate = np.mean([p['setup_pass_rate'] for p in period_results])
    
    print(f"Average setup pass rate: {avg_pass_rate:.1f}%\n")
    
    # Determine verdict
    main_issue, verdict = _determine_setup_verdict(period_results)
    
    print("="*80)
    print("VERDICT")
    print("="*80 + "\n")
    print(f"Main issue: {main_issue}\n")
    print(f"{verdict}\n")
    print("-"*80 + "\n")
    
    # Save report
    output_dir_fractal = Path(output_dir) / 'fractal'
    output_dir_fractal.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir_fractal / f"{pair}_setup_deep_dive.json"
    
    report = {
        'pair': pair,
        'periods': {p['date']: p for p in period_results},
        'cross_period_summary': {
            'average_pass_rate': avg_pass_rate,
            'main_issue': main_issue,
            'verdict': verdict
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Setup deep dive report saved: {report_path}\n")
    
    return {
        'status': 'completed',
        'pair': pair,
        'avg_pass_rate': avg_pass_rate,
        'main_issue': main_issue,
        'output': str(report_path)
    }


if __name__ == "__main__":
    # Quick test
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_setup_deep_dive(
        pair='BTCUSDT',
        config=config,
        output_dir='reports/validation'
    )
