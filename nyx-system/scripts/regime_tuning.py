"""
Regime Sensitivity Tuning

Tests multiple Regime parameter scenarios to find the best balance between
expressing bullish trends and avoiding over-aggressive trend detection.

Purpose:
--------
Diagnostics showed that Regime collapses to 'range' 100% of the time on BTC,
even during known bullish periods. This script tunes Regime sensitivity
without touching Context, fractal geometry, cache, or SMC.

Approach:
---------
1. Define 4-8 scenarios varying:
   - trend_plus_threshold
   - stability_min
   - sdc_min

2. Test each scenario on multiple periods

3. Compare trend_plus_rate vs range_rate

4. Identify best compromise

Output:
-------
- JSON report with per-scenario results
- CLI summary with verdict
- Best scenario recommendation
"""

import sys
sys.path.insert(0, '.')

import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import copy

from src.data.mtf_loader import load_fractal_context
from src.agents.regime_agent import RegimeAgent


# ============================================================================
# TUNING SCENARIOS
# ============================================================================

def get_tuning_scenarios(config: Dict) -> List[Dict[str, Any]]:
    """
    Define Regime tuning scenarios
    
    Args:
        config: Base system config
    
    Returns:
        List of scenario configs
    """
    
    scenarios = []
    
    # Get baseline values
    baseline_sdc = config['strategy']['mtf_conditions']['sdc_min']
    baseline_stability = config['strategy']['mtf_conditions']['stability_4h_min']
    
    # ========================================================================
    # SCENARIO A: BASELINE (current settings)
    # ========================================================================
    baseline_config = copy.deepcopy(config)
    
    scenarios.append({
        'name': 'baseline',
        'label': 'Baseline (current)',
        'config': baseline_config,
        'description': f'Current production settings - sdc_min={baseline_sdc}, stability_min={baseline_stability}'
    })
    
    # ========================================================================
    # SCENARIO B: SLIGHTLY RELAXED
    # ========================================================================
    slightly_relaxed = copy.deepcopy(config)
    
    # Relax sdc from 5.0 to 4.0
    slightly_relaxed['strategy']['mtf_conditions']['sdc_min'] = 4.0
    
    # Relax stability from 0.60 to 0.50
    slightly_relaxed['strategy']['mtf_conditions']['stability_4h_min'] = 0.50
    
    scenarios.append({
        'name': 'slightly_relaxed',
        'label': 'Slightly Relaxed',
        'config': slightly_relaxed,
        'description': 'Modest relaxation: sdc_min=4.0, stability_min=0.50'
    })
    
    # ========================================================================
    # SCENARIO C: MODERATELY RELAXED
    # ========================================================================
    moderately_relaxed = copy.deepcopy(config)
    
    # Relax sdc to 3.0
    moderately_relaxed['strategy']['mtf_conditions']['sdc_min'] = 3.0
    
    # Relax stability to 0.40
    moderately_relaxed['strategy']['mtf_conditions']['stability_4h_min'] = 0.40
    
    scenarios.append({
        'name': 'moderately_relaxed',
        'label': 'Moderately Relaxed',
        'config': moderately_relaxed,
        'description': 'Moderate relaxation: sdc_min=3.0, stability_min=0.40'
    })
    
    # ========================================================================
    # SCENARIO D: AGGRESSIVE
    # ========================================================================
    aggressive = copy.deepcopy(config)
    
    # Relax sdc to 2.0
    aggressive['strategy']['mtf_conditions']['sdc_min'] = 2.0
    
    # Relax stability to 0.30
    aggressive['strategy']['mtf_conditions']['stability_4h_min'] = 0.30
    
    scenarios.append({
        'name': 'aggressive',
        'label': 'Aggressive',
        'config': aggressive,
        'description': 'Aggressive relaxation: sdc_min=2.0, stability_min=0.30'
    })
    
    return scenarios


# ============================================================================
# SCENARIO TESTING
# ============================================================================

def test_scenario_on_period(
    scenario: Dict[str, Any],
    pair: str,
    date: datetime,
    base_config: Dict
) -> Dict[str, Any]:
    """
    Test a scenario on a specific period
    
    Args:
        scenario: Scenario config (contains full system config)
        pair: Trading pair
        date: Reference date
        base_config: Base system config (unused, scenario has full config)
    
    Returns:
        Result dict with regime state and metrics
    """
    
    # Use scenario's modified config
    config = scenario['config']
    
    # Load data
    mtf_data = load_fractal_context(pair, date, config)
    
    # Get regime timeframe
    regime_tf = config['fractal']['timeframes']['regime_tf']
    
    # Create Regime agent with full config
    regime_agent = RegimeAgent(config)
    
    # Analyze
    result = regime_agent.analyze(mtf_data[regime_tf])
    
    # Extract metrics
    return {
        'date': date.isoformat(),
        'state': result.state,
        'score': round(result.score, 4),
        'passed': result.passed,
        'ready': result.ready,
        'metadata': {
            'sdc': round(result.metadata.get('sdc', 0), 4),
            'stability': round(result.metadata.get('stability', 0), 4),
            'trend_plus': result.state == 'trend_plus'
        }
    }


def run_scenario_across_periods(
    scenario: Dict[str, Any],
    pair: str,
    dates: List[datetime],
    base_config: Dict
) -> Dict[str, Any]:
    """
    Run a scenario across multiple periods
    
    Args:
        scenario: Scenario config
        pair: Trading pair
        dates: List of reference dates
        base_config: Base system config
    
    Returns:
        Aggregated results
    """
    
    results = []
    
    for date in dates:
        try:
            result = test_scenario_on_period(scenario, pair, date, base_config)
            results.append(result)
        except Exception as e:
            print(f"  ⚠️  Failed on {date.date()}: {e}")
            results.append({
                'date': date.isoformat(),
                'state': 'error',
                'score': 0,
                'passed': False,
                'ready': False,
                'error': str(e)
            })
    
    # Calculate aggregate metrics
    total = len(results)
    trend_plus_count = sum(1 for r in results if r.get('metadata', {}).get('trend_plus', False))
    range_count = sum(1 for r in results if r.get('state') == 'range')
    
    trend_plus_rate = (trend_plus_count / total * 100) if total > 0 else 0
    range_rate = (range_count / total * 100) if total > 0 else 0
    
    # Extract only the modified parameters (not full config)
    params = {
        'sdc_min': scenario['config']['strategy']['mtf_conditions']['sdc_min'],
        'stability_min': scenario['config']['strategy']['mtf_conditions']['stability_4h_min']
    }
    
    return {
        'scenario': scenario['name'],
        'label': scenario['label'],
        'description': scenario['description'],
        'params': params,
        'results': results,
        'summary': {
            'total_periods': total,
            'trend_plus_count': trend_plus_count,
            'range_count': range_count,
            'trend_plus_rate': round(trend_plus_rate, 1),
            'range_rate': round(range_rate, 1)
        }
    }


# ============================================================================
# VERDICT
# ============================================================================

def determine_verdict(scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Determine overall verdict from scenario results
    
    Args:
        scenario_results: List of scenario result dicts
    
    Returns:
        Verdict dict
    """
    
    # Find baseline
    baseline = next((s for s in scenario_results if s['scenario'] == 'baseline'), None)
    
    if not baseline:
        return {
            'conclusion': 'error',
            'message': 'Baseline scenario not found'
        }
    
    baseline_trend_rate = baseline['summary']['trend_plus_rate']
    
    # Find best scenario (highest trend_plus_rate without going to 100%)
    valid_scenarios = [
        s for s in scenario_results 
        if s['summary']['trend_plus_rate'] < 95  # Not over-aggressive
    ]
    
    if not valid_scenarios:
        return {
            'conclusion': 'Regime tuning alone is not enough',
            'message': 'All relaxation scenarios produce over-aggressive trend detection',
            'best_scenario': None,
            'recommendation': 'Consider deeper structural changes to Regime or investigate Context/Setup interaction'
        }
    
    best = max(valid_scenarios, key=lambda s: s['summary']['trend_plus_rate'])
    best_trend_rate = best['summary']['trend_plus_rate']
    
    improvement = best_trend_rate - baseline_trend_rate
    
    # Determine conclusion
    if improvement < 10:
        conclusion = 'Regime remains too conservative'
        message = f'Best scenario improves trend_plus_rate by only {improvement:.1f}%'
        recommendation = 'Regime tuning alone insufficient - investigate Context alignment or deeper HSMM issues'
    
    elif improvement < 30:
        conclusion = 'Regime becomes usable with mild relaxation'
        message = f'Best scenario ({best["label"]}) improves trend_plus_rate by {improvement:.1f}%'
        recommendation = f'Deploy {best["scenario"]} scenario and monitor for over-detection'
    
    elif improvement < 60:
        conclusion = 'Regime only improves with aggressive relaxation'
        message = f'Best scenario ({best["label"]}) improves trend_plus_rate by {improvement:.1f}%'
        recommendation = 'Consider aggressive relaxation but validate carefully on broader dataset'
    
    else:
        conclusion = 'Regime becomes usable with mild relaxation'
        message = f'Strong improvement ({improvement:.1f}%) with {best["label"]}'
        recommendation = f'Deploy {best["scenario"]} scenario - significant unlock achieved'
    
    return {
        'conclusion': conclusion,
        'message': message,
        'best_scenario': best['scenario'],
        'best_label': best['label'],
        'baseline_trend_rate': baseline_trend_rate,
        'best_trend_rate': best_trend_rate,
        'improvement': round(improvement, 1),
        'recommendation': recommendation
    }


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_regime_tuning(
    pair: str,
    config: Dict,
    output_dir: str,
    dates: "List[datetime] | None" = None
) -> Dict:
    """
    Run regime sensitivity tuning
    
    Args:
        pair: Trading pair
        config: Base config
        output_dir: Output directory
        dates: Optional list of dates (defaults to 4 test periods)
    
    Returns:
        Results dict
    """
    
    print(f"\n{'='*80}")
    print(f"REGIME SENSITIVITY TUNING - {pair}")
    print(f"{'='*80}\n")
    
    # Default test dates if not provided
    if dates is None:
        dates = [
            datetime(2023, 1, 15),
            datetime(2023, 3, 15),
            datetime(2023, 10, 15),
            datetime(2023, 12, 15)
        ]
    
    print(f"Test periods: {len(dates)}")
    for d in dates:
        print(f"  - {d.date()}")
    
    # Get scenarios
    scenarios = get_tuning_scenarios(config)
    print(f"\nScenarios to test: {len(scenarios)}")
    for s in scenarios:
        print(f"  - {s['label']}: {s['description']}")
    
    # Test each scenario
    print(f"\n{'='*80}")
    print("RUNNING SCENARIOS")
    print(f"{'='*80}\n")
    
    scenario_results = []
    
    for scenario in scenarios:
        print(f"Testing {scenario['label']}...")
        
        results = run_scenario_across_periods(scenario, pair, dates, config)
        scenario_results.append(results)
        
        print(f"  Trend+: {results['summary']['trend_plus_rate']:.1f}%")
        print(f"  Range:  {results['summary']['range_rate']:.1f}%\n")
    
    # Determine verdict
    print(f"{'='*80}")
    print("CROSS-SCENARIO ANALYSIS")
    print(f"{'='*80}\n")
    
    verdict = determine_verdict(scenario_results)
    
    # Print summary
    for result in scenario_results:
        status = ""
        if result['scenario'] == 'baseline':
            status = "(current)"
        elif result['scenario'] == verdict.get('best_scenario'):
            status = "⭐ BEST"
        
        print(f"{result['label']:25s} {status}")
        print(f"  Trend+ rate: {result['summary']['trend_plus_rate']:5.1f}%")
        print(f"  Range rate:  {result['summary']['range_rate']:5.1f}%")
        print()
    
    print(f"{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    print(f"\nConclusion: {verdict['conclusion']}")
    print(f"\n{verdict['message']}")
    print(f"\nRecommendation:")
    print(f"  {verdict['recommendation']}")
    print(f"\n{'-'*80}\n")
    
    # Save report
    output_dir_fractal = Path(output_dir) / 'fractal'
    output_dir_fractal.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir_fractal / f"{pair}_regime_tuning.json"
    
    report = {
        'pair': pair,
        'test_dates': [d.isoformat() for d in dates],
        'scenarios': scenario_results,
        'verdict': verdict,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_scenarios': len(scenarios),
            'total_periods': len(dates)
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Tuning report saved: {report_path}\n")
    
    return {
        'status': 'completed',
        'pair': pair,
        'best_scenario': verdict.get('best_scenario'),
        'conclusion': verdict['conclusion'],
        'output': str(report_path)
    }


if __name__ == "__main__":
    # Quick test
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_regime_tuning(
        pair='BTCUSDT',
        config=config,
        output_dir='reports/validation'
    )
