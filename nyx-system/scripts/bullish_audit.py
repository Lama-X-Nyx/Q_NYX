"""
Bullish Context/Regime Attribution Audit

Understand why NYX doesn't express bullish bias on BTC during bullish periods.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

from src.data.mtf_loader import load_fractal_context
from src.agents.context_agent import ContextAgent
from src.agents.regime_agent import RegimeAgent


# Default test dates for BTC bullish periods
DEFAULT_DATES = [
    ('2023-01-15', 'early_bull'),
    ('2023-03-15', 'mid_bull'),
    ('2023-10-15', 'late_bull'),
    ('2023-12-15', 'range_comparison')
]


def run_bullish_audit(pair: str, config: dict, output_dir: Path, dates: list = None):
    """
    Audit Context and Regime attribution on BTC bullish periods
    
    Args:
        pair: Trading pair
        config: System configuration
        output_dir: Output directory
        dates: List of date strings to test
    """
    
    print(f"\n{'='*80}")
    print(f"BULLISH ATTRIBUTION AUDIT - {pair}")
    print(f"{'='*80}")
    
    # Use default dates if not provided
    if dates is None:
        test_dates = DEFAULT_DATES
    else:
        # Parse custom dates (assume all are unknown regime)
        test_dates = [(d, 'unknown') for d in dates]
    
    print(f"\nTesting {len(test_dates)} periods:")
    for date, label in test_dates:
        print(f"  - {date} ({label})")
    
    # Initialize agents
    context_agent = ContextAgent(config)
    regime_agent = RegimeAgent(config)
    
    # Get timeframes from config
    fractal_config = config.get('fractal', {})
    context_tf = fractal_config.get('context_tf', '1d')
    regime_tf = fractal_config.get('regime_tf', '1h')
    
    # Analyze each period
    periods = {}
    
    for date_str, period_label in test_dates:
        print(f"\n{'='*60}")
        print(f"Analyzing: {date_str} ({period_label})")
        print(f"{'='*60}")
        
        try:
            # Parse date
            current_date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Load fractal context
            print(f"📊 Loading fractal context...")
            mtf_data = load_fractal_context(pair, current_date, config)
            print(f"✓ Loaded {len(mtf_data)} timeframes")
            
            # Get Context and Regime data
            context_data = mtf_data.get(context_tf)
            regime_data = mtf_data.get(regime_tf)
            
            if context_data is None or regime_data is None:
                print(f"❌ Missing required timeframe data")
                continue
            
            # Analyze Context
            print(f"🔬 Analyzing Context Agent ({context_tf})...")
            context_result = context_agent.analyze(context_data)
            
            # Analyze Regime (with context)
            print(f"🔬 Analyzing Regime Agent ({regime_tf})...")
            regime_result = regime_agent.analyze(regime_data, context_state=context_result.state)
            
            # Summarize results
            context_summary = {
                'state': context_result.state,
                'score': round(context_result.score, 2),
                'passed': context_result.passed,
                'reason': context_result.reason
            }
            
            regime_summary = {
                'state': regime_result.state,
                'score': round(regime_result.score, 2),
                'passed': regime_result.passed,
                'reason': regime_result.reason,
                'sdc': regime_result.metadata.get('sdc', 0.0) if hasattr(regime_result, 'metadata') else 0.0,
                'stability': regime_result.metadata.get('stability', 0.0) if hasattr(regime_result, 'metadata') else 0.0
            }
            
            # Determine joint interpretation
            joint_interpretation = _determine_joint_interpretation(
                context_summary, regime_summary
            )
            
            # Store results
            periods[date_str] = {
                'period_label': period_label,
                'context': context_summary,
                'regime': regime_summary,
                'joint_interpretation': joint_interpretation
            }
            
            # Print summary
            _print_period_summary(date_str, period_label, context_summary, 
                                 regime_summary, joint_interpretation)
            
        except Exception as e:
            print(f"❌ Failed to analyze {date_str}: {e}")
            periods[date_str] = {
                'period_label': period_label,
                'error': str(e)
            }
    
    # Compute cross-period summary
    print(f"\n{'='*80}")
    print(f"CROSS-PERIOD ANALYSIS")
    print(f"{'='*80}")
    
    cross_summary = _compute_cross_period_summary(periods, test_dates)
    
    # Print final summary
    _print_final_summary(cross_summary)
    
    # Save report
    output_dir_fractal = Path(output_dir) / 'fractal'
    output_dir_fractal.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir_fractal / f"{pair}_bullish_audit.json"
    
    report = {
        'pair': pair,
        'periods': periods,
        'cross_period_summary': cross_summary
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Bullish audit report saved: {report_path}")
    
    return {
        'status': 'completed',
        'pair': pair,
        'periods_tested': len(periods),
        'verdict': cross_summary['verdict'],
        'output': str(report_path)
    }


def _summarize_agent_results(results: list, agent_name: str) -> dict:
    """Summarize agent results across multiple bars"""
    
    if not results or len(results) == 0:
        return {
            'state': 'unknown',
            'score': 0.0,
            'passed': False,
            'reason': 'No data',
            'passed_rate': 0.0
        }
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        return {
            'state': 'unknown',
            'score': 0.0,
            'passed': False,
            'reason': 'No valid results',
            'passed_rate': 0.0
        }
    
    # Count states
    states = [r.state for r in valid_results]
    state_counts = {}
    for state in states:
        state_counts[state] = state_counts.get(state, 0) + 1
    
    # Most common state
    dominant_state = max(state_counts.items(), key=lambda x: x[1])[0]
    
    # Average score
    avg_score = sum(r.score for r in valid_results) / len(valid_results)
    
    # Pass rate
    passed_count = sum(1 for r in valid_results if r.passed)
    passed_rate = passed_count / len(valid_results) * 100
    
    # Get representative reason
    representative = next((r for r in valid_results if r.state == dominant_state), valid_results[0])
    
    summary = {
        'state': dominant_state,
        'score': round(avg_score, 2),
        'passed': passed_rate >= 50.0,
        'reason': representative.reason if representative else 'Unknown',
        'passed_rate': round(passed_rate, 1),
        'state_distribution': state_counts
    }
    
    # Add regime-specific fields
    if agent_name == 'regime' and representative:
        summary['sdc'] = getattr(representative, 'metadata', {}).get('sdc', 0.0)
        summary['stability'] = getattr(representative, 'metadata', {}).get('stability', 0.0)
    
    return summary


def _determine_joint_interpretation(context_summary: dict, regime_summary: dict) -> str:
    """Determine joint interpretation of context + regime"""
    
    context_state = context_summary.get('state', 'unknown')
    regime_state = regime_summary.get('state', 'unknown')
    
    # Bullish coherent: both bullish
    if context_state == 'bullish' and regime_state == 'trend_plus':
        return 'bullish_coherent'
    
    # Bullish not expressed: one or both neutral/range
    elif context_state == 'bullish' and regime_state in ['range', 'neutral']:
        return 'regime_too_conservative'
    
    elif context_state in ['neutral', 'bearish'] and regime_state == 'trend_plus':
        return 'context_too_neutral'
    
    elif context_state in ['neutral', 'bearish'] and regime_state in ['range', 'neutral']:
        return 'bullish_not_expressed'
    
    # Bearish (wrong for bullish test)
    elif context_state == 'bearish' or regime_state == 'trend_minus':
        return 'bearish_detected'
    
    else:
        return 'mixed_signals'


def _extract_sample_decisions(context_results: list, regime_results: list, n: int = 5) -> list:
    """Extract sample decisions for inspection"""
    
    samples = []
    
    valid_pairs = [(c, r) for c, r in zip(context_results, regime_results) 
                   if c is not None and r is not None]
    
    # Take first n samples
    for i, (context, regime) in enumerate(valid_pairs[:n]):
        samples.append({
            'index': i,
            'context_state': context.state,
            'context_score': round(context.score, 2),
            'regime_state': regime.state,
            'regime_score': round(regime.score, 2),
            'regime_sdc': regime.metadata.get('sdc', 0.0) if hasattr(regime, 'metadata') else 0.0,
            'interpretation': _determine_joint_interpretation(
                {'state': context.state},
                {'state': regime.state}
            )
        })
    
    return samples


def _compute_cross_period_summary(periods: dict, test_dates: list) -> dict:
    """Compute cross-period statistics and verdict"""
    
    # Count Context states
    context_bullish = 0
    context_neutral = 0
    context_bearish = 0
    
    # Count Regime states
    regime_trend_plus = 0
    regime_range = 0
    regime_trend_minus = 0
    
    # Count joint interpretations
    interpretations = []
    
    total_valid = 0
    
    for date_str, _ in test_dates:
        if date_str not in periods or 'error' in periods[date_str]:
            continue
        
        period = periods[date_str]
        total_valid += 1
        
        # Context
        context_state = period['context'].get('state')
        if context_state == 'bullish':
            context_bullish += 1
        elif context_state == 'neutral':
            context_neutral += 1
        elif context_state == 'bearish':
            context_bearish += 1
        
        # Regime
        regime_state = period['regime'].get('state')
        if regime_state == 'trend_plus':
            regime_trend_plus += 1
        elif regime_state == 'range':
            regime_range += 1
        elif regime_state == 'trend_minus':
            regime_trend_minus += 1
        
        # Interpretation
        interpretations.append(period['joint_interpretation'])
    
    if total_valid == 0:
        return {
            'context_bullish_rate': 0.0,
            'regime_trend_plus_rate': 0.0,
            'main_issue': 'insufficient_data',
            'verdict': 'Insufficient valid periods to determine verdict'
        }
    
    # Calculate rates
    context_bullish_rate = context_bullish / total_valid * 100
    regime_trend_plus_rate = regime_trend_plus / total_valid * 100
    
    # Determine main issue
    main_issue = _determine_main_issue(
        context_bullish_rate, regime_trend_plus_rate, interpretations
    )
    
    # Determine verdict
    verdict = _determine_verdict(context_bullish_rate, regime_trend_plus_rate, main_issue)
    
    return {
        'context_bullish_rate': round(context_bullish_rate, 1),
        'context_neutral_rate': round(context_neutral / total_valid * 100, 1) if total_valid > 0 else 0.0,
        'regime_trend_plus_rate': round(regime_trend_plus_rate, 1),
        'regime_range_rate': round(regime_range / total_valid * 100, 1) if total_valid > 0 else 0.0,
        'main_issue': main_issue,
        'verdict': verdict,
        'interpretation_distribution': dict(
            (k, interpretations.count(k)) for k in set(interpretations)
        )
    }


def _determine_main_issue(context_bullish_rate: float, regime_trend_plus_rate: float,
                          interpretations: list) -> str:
    """Determine main issue from rates"""
    
    # Count interpretation types
    regime_conservative = interpretations.count('regime_too_conservative')
    context_neutral = interpretations.count('context_too_neutral')
    not_expressed = interpretations.count('bullish_not_expressed')
    
    # Determine main issue
    if regime_conservative > len(interpretations) * 0.4:
        return 'regime_too_conservative'
    elif context_neutral > len(interpretations) * 0.4:
        return 'context_too_neutral'
    elif not_expressed > len(interpretations) * 0.4:
        return 'both_too_neutral'
    elif context_bullish_rate >= 60 and regime_trend_plus_rate < 40:
        return 'regime_too_conservative'
    elif context_bullish_rate < 40 and regime_trend_plus_rate >= 60:
        return 'context_too_neutral'
    else:
        return 'sample_dependent'


def _determine_verdict(context_bullish_rate: float, regime_trend_plus_rate: float,
                       main_issue: str) -> str:
    """Determine cross-period verdict"""
    
    # Cas A: Both work well
    if context_bullish_rate >= 75 and regime_trend_plus_rate >= 75:
        return "Context and regime both express bullish BTC correctly."
    
    # Cas B: Context good, Regime conservative
    elif context_bullish_rate >= 60 and regime_trend_plus_rate < 40:
        return "Context is often bullish, but regime collapses too often into range."
    
    # Cas C: Context neutral, Regime variable
    elif context_bullish_rate < 40:
        return "Context itself is too neutral during bullish BTC periods."
    
    # Cas D: Inconsistent
    else:
        return "Bullish expression is inconsistent and depends strongly on the chosen sample."


def _print_period_summary(date: str, label: str, context: dict, regime: dict, joint: str):
    """Print period summary"""
    
    context_check = "✅" if context['passed'] else "❌"
    regime_check = "✅" if regime['passed'] else "❌"
    
    print(f"\n{date} ({label})")
    print(f"  Context: {context['state']:10s} ({context['score']:.2f}) {context_check}")
    print(f"  Regime:  {regime['state']:10s} ({regime['score']:.2f}) {regime_check}")
    print(f"  Verdict: {joint}")


def _print_final_summary(summary: dict):
    """Print final summary"""
    
    print(f"\n{'='*80}")
    print(f"CROSS-PERIOD SUMMARY")
    print(f"{'='*80}")
    print(f"Context bullish rate:     {summary['context_bullish_rate']:.1f}%")
    print(f"Context neutral rate:     {summary['context_neutral_rate']:.1f}%")
    print(f"Regime trend_plus rate:   {summary['regime_trend_plus_rate']:.1f}%")
    print(f"Regime range rate:        {summary['regime_range_rate']:.1f}%")
    print(f"\nMain issue: {summary['main_issue']}")
    print(f"\nVERDICT:")
    print(f"{summary['verdict']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import yaml
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_bullish_audit('BTCUSDT', config, Path('reports/validation'))
