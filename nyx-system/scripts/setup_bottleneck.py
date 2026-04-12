"""
Setup Bottleneck Diagnosis

Diagnose precisely why the Setup agent blocks ready decisions.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

sys.path.insert(0, '.')

from src.data.mtf_loader import load_fractal_context
from src.agents.orchestrator import Orchestrator


def run_setup_bottleneck(pair: str, config: dict, output_dir: Path, sample_date: str = ""):
    """
    Diagnose Setup Agent bottleneck
    
    Analyzes why Setup blocks decisions on ready bars.
    """
    
    print(f"\n{'='*80}")
    print(f"SETUP BOTTLENECK DIAGNOSIS - {pair}")
    print(f"{'='*80}")
    
    # Check fractal enabled
    fractal_config = config.get('fractal', {})
    if not fractal_config.get('enabled', False):
        print("❌ Fractal architecture not enabled in config")
        return {'status': 'error', 'reason': 'fractal not enabled'}
    
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
    
    # Initialize Orchestrator
    print(f"\n🚀 Initializing Fractal Orchestrator...")
    orchestrator = Orchestrator(config)
    
    # Run analysis
    print(f"\n🔬 Analyzing Setup bottleneck...")
    
    decisions = []
    warmup = 50
    
    # Track Setup-specific diagnostics
    setup_passed = 0
    setup_blocked = 0
    
    # Block reasons
    block_reasons = {
        'no_pattern': 0,
        'misaligned_pattern': 0,
        'alignment_score_too_low': 0,
        'direction_conflict': 0,
        'setup_not_ready': 0,
        'unknown': 0
    }
    
    # Pattern observations
    pattern_observed = {
        'bullish_fvg': 0,
        'bearish_fvg': 0,
        'bullish_ob': 0,
        'bearish_ob': 0,
        'no_pattern': 0
    }
    
    # Alignment scores
    alignment_scores = []
    
    # Context/Regime states
    context_states = []
    regime_states = []
    setup_directions = []
    
    # Sample blocked setups
    sample_blocked = []
    
    # Get lowest TF for iteration
    lowest_tf = min(mtf_data.keys(), key=lambda x: _tf_to_minutes(x))
    available_bars = len(mtf_data[lowest_tf])
    max_signals = min(100, available_bars - warmup)
    
    print(f"  Evaluating {max_signals} signals (after {warmup} bar warmup)...")
    
    for i in range(warmup, warmup + max_signals):
        try:
            current_time = mtf_data[lowest_tf].index[i]
            
            # Align data
            from src.data.mtf_loader import MTFLoader
            loader = MTFLoader('data/raw/mtf')
            aligned_data = loader.align_at_timestamp(mtf_data, current_time, lowest_tf)
            
            # Get decision
            decision = orchestrator.decide(aligned_data)
            decisions.append(decision)
            
            # Check if ready
            all_ready = all(comp.ready for comp in decision.components.values())
            
            if not all_ready:
                continue  # Skip non-ready bars
            
            # Analyze Setup component
            setup_comp = decision.components.get('setup')
            context_comp = decision.components.get('context')
            regime_comp = decision.components.get('regime')
            
            if not setup_comp:
                continue
            
            # Track context/regime states
            if context_comp:
                context_states.append(context_comp.state)
            if regime_comp:
                regime_states.append(regime_comp.state)
            
            # Setup result
            if setup_comp.passed:
                setup_passed += 1
            else:
                setup_blocked += 1
                
                # Analyze why blocked
                metadata = setup_comp.metadata if hasattr(setup_comp, 'metadata') else {}
                
                # Check patterns
                has_bullish_fvg = metadata.get('has_bullish_fvg', False)
                has_bearish_fvg = metadata.get('has_bearish_fvg', False)
                has_bullish_ob = metadata.get('has_bullish_ob', False)
                has_bearish_ob = metadata.get('has_bearish_ob', False)
                
                # Count patterns
                if has_bullish_fvg:
                    pattern_observed['bullish_fvg'] += 1
                if has_bearish_fvg:
                    pattern_observed['bearish_fvg'] += 1
                if has_bullish_ob:
                    pattern_observed['bullish_ob'] += 1
                if has_bearish_ob:
                    pattern_observed['bearish_ob'] += 1
                
                has_any_pattern = has_bullish_fvg or has_bearish_fvg or has_bullish_ob or has_bearish_ob
                
                if not has_any_pattern:
                    pattern_observed['no_pattern'] += 1
                
                # Get alignment score
                alignment_score = setup_comp.score
                alignment_scores.append(alignment_score)
                
                # Determine block reason
                if not has_any_pattern:
                    block_reasons['no_pattern'] += 1
                    reason = "No pattern detected"
                    setup_direction = 'none'
                else:
                    # Has pattern - check why rejected
                    setup_direction = 'bullish' if (has_bullish_fvg or has_bullish_ob) else 'bearish'
                    
                    # Check alignment threshold
                    alignment_threshold = config.get('smc_detector', {}).get('alignment_15m_min', 0.6)
                    
                    if alignment_score < alignment_threshold:
                        if alignment_score < (alignment_threshold - 0.2):
                            block_reasons['misaligned_pattern'] += 1
                            reason = f"Pattern misaligned (score {alignment_score:.2f} << {alignment_threshold})"
                        else:
                            block_reasons['alignment_score_too_low'] += 1
                            reason = f"Alignment score too low (score {alignment_score:.2f} < {alignment_threshold})"
                    else:
                        # Score OK but still blocked - direction conflict?
                        context_state = context_comp.state if context_comp else 'unknown'
                        if (setup_direction == 'bullish' and context_state == 'bearish') or \
                           (setup_direction == 'bearish' and context_state == 'bullish'):
                            block_reasons['direction_conflict'] += 1
                            reason = f"{setup_direction.capitalize()} pattern against {context_state} context"
                        else:
                            block_reasons['unknown'] += 1
                            reason = "Unknown rejection reason"
                
                setup_directions.append(setup_direction)
                
                # Sample for output
                if len(sample_blocked) < 10:
                    sample_blocked.append({
                        'timestamp': str(current_time),
                        'state': setup_comp.state,
                        'reason': reason,
                        'alignment_score': round(alignment_score, 2),
                        'patterns': {
                            'bullish_fvg': has_bullish_fvg,
                            'bearish_fvg': has_bearish_fvg,
                            'bullish_ob': has_bullish_ob,
                            'bearish_ob': has_bearish_ob
                        },
                        'context_state': context_comp.state if context_comp else 'unknown',
                        'regime_state': regime_comp.state if regime_comp else 'unknown'
                    })
        
        except Exception as e:
            continue
    
    # Calculate statistics
    bars_ready = setup_passed + setup_blocked
    pass_rate = (setup_passed / bars_ready * 100) if bars_ready > 0 else 0.0
    
    # Alignment statistics
    if alignment_scores:
        mean_alignment = np.mean(alignment_scores)
        median_alignment = np.median(alignment_scores)
        min_alignment = np.min(alignment_scores)
        max_alignment = np.max(alignment_scores)
    else:
        mean_alignment = median_alignment = min_alignment = max_alignment = 0.0
    
    alignment_threshold = config.get('smc_detector', {}).get('alignment_15m_min', 0.6)
    
    # Determine primary issue
    primary_issue = max(block_reasons.items(), key=lambda x: x[1])[0]
    
    # Verdict interpretation
    if primary_issue == 'no_pattern':
        interpretation = "Setup is blocked mostly by total absence of patterns"
    elif primary_issue == 'misaligned_pattern':
        interpretation = "Setup is blocked mostly by misalignment, not by pattern absence"
    elif primary_issue == 'alignment_score_too_low':
        interpretation = "Setup is blocked mostly by scores close but below threshold"
    elif primary_issue == 'direction_conflict':
        interpretation = "Setup is blocked mostly by directional conflicts with context"
    else:
        interpretation = "Setup blocks for mixed reasons and needs deeper decomposition"
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SETUP BOTTLENECK RESULTS")
    print(f"{'='*60}")
    print(f"Bars ready for decision:     {bars_ready}")
    print(f"Setup passed:                {setup_passed}")
    print(f"Setup blocked:               {setup_blocked}")
    print(f"Setup pass rate:             {pass_rate:.1f}%")
    
    print(f"\nBLOCK REASONS")
    print(f"No pattern:                  {block_reasons['no_pattern']}")
    print(f"Misaligned pattern:          {block_reasons['misaligned_pattern']}")
    print(f"Alignment too low:           {block_reasons['alignment_score_too_low']}")
    print(f"Direction conflict:          {block_reasons['direction_conflict']}")
    print(f"Setup not ready:             {block_reasons['setup_not_ready']}")
    print(f"Unknown:                     {block_reasons['unknown']}")
    
    print(f"\nPATTERNS OBSERVED")
    print(f"Bullish FVG:                 {pattern_observed['bullish_fvg']}")
    print(f"Bearish FVG:                 {pattern_observed['bearish_fvg']}")
    print(f"Bullish OB:                  {pattern_observed['bullish_ob']}")
    print(f"Bearish OB:                  {pattern_observed['bearish_ob']}")
    print(f"No pattern:                  {pattern_observed['no_pattern']}")
    
    print(f"\nALIGNMENT")
    print(f"Mean alignment score:        {mean_alignment:.2f}")
    print(f"Median alignment score:      {median_alignment:.2f}")
    print(f"Min alignment score:         {min_alignment:.2f}")
    print(f"Max alignment score:         {max_alignment:.2f}")
    print(f"Threshold required:          {alignment_threshold:.2f}")
    
    print(f"\nConclusion:")
    print(f"{interpretation}")
    print(f"{'='*60}")
    
    # Save JSON
    output_dir = Path(output_dir) / 'fractal'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{pair}_setup_bottleneck.json"
    
    # Count regime states
    regime_distribution = dict(Counter(regime_states))
    setup_direction_distribution = dict(Counter(setup_directions))
    context_state = context_states[0] if context_states else 'unknown'
    
    report = {
        'pair': pair,
        'reference_date': current_date.isoformat(),
        'bars_ready_for_decision': bars_ready,
        'setup_summary': {
            'passed': setup_passed,
            'blocked': setup_blocked,
            'pass_rate': round(pass_rate, 1)
        },
        'block_reasons': block_reasons,
        'pattern_observed': pattern_observed,
        'alignment_summary': {
            'mean_alignment_score': round(mean_alignment, 2),
            'median_alignment_score': round(median_alignment, 2),
            'min_alignment_score': round(min_alignment, 2),
            'max_alignment_score': round(max_alignment, 2),
            'threshold': alignment_threshold
        },
        'directional_summary': {
            'context_state': context_state,
            'regime_state_distribution': regime_distribution,
            'setup_direction_distribution': setup_direction_distribution
        },
        'sample_blocked_setups': sample_blocked,
        'verdict': {
            'primary_setup_issue': primary_issue,
            'interpretation': interpretation
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    
    return {
        'status': 'completed',
        'pair': pair,
        'primary_issue': primary_issue,
        'output': str(report_path)
    }


def _tf_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes"""
    mapping = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    return mapping.get(tf, 1440)


if __name__ == "__main__":
    import yaml
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_setup_bottleneck('BTCUSDT', config, Path('reports/validation'), sample_date='2023-12-15')
