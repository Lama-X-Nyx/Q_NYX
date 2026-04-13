"""
HSMM Deep Dive - Feature, State, and Mapping Audit

Diagnoses why the HSMM never produces trend_plus on tested BTC periods.

Purpose:
--------
After Regime Sensitivity Tuning showed that even aggressive threshold relaxation
doesn't unlock bullish expression, we need to understand if the problem is:
- Weak/non-trending features fed to HSMM
- Structural dominance of range state probabilities
- Rigid mapping from HSMM states to final states
- Biased calibration/initialization

Approach:
---------
1. Extract features actually fed to HSMM (returns, ATR)
2. Inspect HSMM state probabilities (Trend+, Range, Trend-)
3. Examine mapping from HSMM output to final state
4. Produce clear diagnosis

Output:
-------
- JSON report with per-period details
- Cross-period summary
- Clear verdict on root cause
"""

import sys
sys.path.insert(0, '.')

import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.data.mtf_loader import load_fractal_context
from src.agents.regime_agent import RegimeAgent


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract features that are fed to HSMM
    
    Args:
        df: DataFrame for regime timeframe
    
    Returns:
        Dict with feature statistics
    """
    
    # Compute returns if not present
    if 'returns' not in df.columns:
        returns = df['close'].pct_change()
    else:
        returns = df['returns']
    
    # Compute ATR if not present
    if 'atr_14' not in df.columns:
        high = np.asarray(df['high'].values)
        low = np.asarray(df['low'].values)
        close = np.asarray(df['close'].values)

        tr = np.maximum(high - low,
                       np.maximum(np.abs(high - np.roll(close, 1)),
                                 np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]

        atr: pd.Series = pd.Series(tr).rolling(14).mean()
    else:
        atr: pd.Series = df['atr_14']
    
    # Get recent window (last 50 bars as used by RegimeAgent)
    window_size = min(50, len(df))
    recent_returns = returns.tail(window_size).dropna()
    recent_atr = atr.tail(window_size).dropna()
    
    # Compute statistics
    features = {
        'returns_mean': float(recent_returns.mean(axis=0)) if len(recent_returns) > 0 else 0.0,
        'returns_std': float(recent_returns.std(axis=0)) if len(recent_returns) > 0 else 0.0,
        'returns_median': float(recent_returns.median(axis=0)) if len(recent_returns) > 0 else 0.0,
        'returns_skew': float(recent_returns.skew(axis=0)) if len(recent_returns) > 2 else 0.0,
        'atr_mean': float(recent_atr.mean(axis=0)) if len(recent_atr) > 0 else 0.0,
        'atr_std': float(recent_atr.std(axis=0)) if len(recent_atr) > 0 else 0.0,
        'price_slope': float(np.polyfit(range(len(recent_returns)), recent_returns.cumsum(), 1)[0]) if len(recent_returns) > 1 else 0.0,
        'window_size': window_size
    }
    
    return features


# ============================================================================
# HSMM STATE INSPECTION
# ============================================================================

def inspect_hsmm_states(
    regime_agent: RegimeAgent,
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Inspect HSMM state probabilities and mapping
    
    Args:
        regime_agent: RegimeAgent instance
        df: DataFrame for regime timeframe
    
    Returns:
        Dict with state probabilities and mapping details
    """
    
    # Prepare data
    df_prepared = regime_agent._prepare_data(df)
    
    # Initialize HSMM
    regime_agent.hsmm.initialize_parameters(df_prepared)
    
    # Build observations (same as RegimeAgent)
    window_size = min(50, len(df_prepared))
    window = df_prepared.tail(window_size)
    
    observations = []
    for idx in range(len(window)):
        obs = {
            'price': window.iloc[idx]['returns'] if 'returns' in window.columns else 0.0,
            'atr': window.iloc[idx]['atr_14'] if 'atr_14' in window.columns else window.iloc[idx]['close'] * 0.02
        }
        observations.append(obs)
    
    # Run HSMM forward-backward
    state_probs = regime_agent.hsmm.forward_backward(observations)
    
    if len(state_probs) == 0:
        return {
            'error': 'HSMM returned empty state probabilities',
            'state_probabilities': {},
            'selected_state': 'error'
        }
    
    # Current state probabilities
    current_probs = state_probs[-1]
    
    # Map to state names
    state_probabilities = {
        'trend_plus': float(current_probs[0]),
        'range': float(current_probs[1]),
        'trend_minus': float(current_probs[2])
    }
    
    # Get most likely state
    state_idx = np.argmax(current_probs)
    state_names = ['Trend+', 'Range', 'Trend-']
    raw_state = state_names[state_idx]
    
    # Map to standardized names
    state_mapping = {
        'Trend+': 'trend_plus',
        'Range': 'range',
        'Trend-': 'trend_minus'
    }
    selected_state = state_mapping[raw_state]
    
    # Compute SdC and stability
    sdc = 10 * current_probs[state_idx]
    trans_matrix = regime_agent.hsmm.transition_matrix
    if trans_matrix is not None:
        stability = trans_matrix[state_idx, state_idx]
    else:
        stability = 0.0
    
    return {
        'state_probabilities': state_probabilities,
        'selected_state': selected_state,
        'raw_state': raw_state,
        'sdc': float(sdc),
        'stability': float(stability),
        'state_prob_max': float(current_probs[state_idx]),
        'window_size': window_size
    }


# ============================================================================
# PERIOD ANALYSIS
# ============================================================================

def analyze_period(
    pair: str,
    date: datetime,
    config: Dict
) -> Dict[str, Any]:
    """
    Deep dive analysis for a single period
    
    Args:
        pair: Trading pair
        date: Reference date
        config: System config
    
    Returns:
        Dict with complete analysis
    """
    
    # Load data
    mtf_data = load_fractal_context(pair, date, config)
    
    # Get regime timeframe
    regime_tf = config['fractal']['timeframes']['regime_tf']
    df = mtf_data[regime_tf]
    
    # Create regime agent
    regime_agent = RegimeAgent(config)
    
    # Extract features
    features = extract_features(df)
    
    # Inspect HSMM states
    hsmm_inspection = inspect_hsmm_states(regime_agent, df)
    
    # Get full agent result for comparison
    agent_result = regime_agent.analyze(df)
    
    # Determine diagnostic
    diagnostic = _diagnose_period(features, hsmm_inspection, agent_result)
    
    return {
        'date': date.isoformat(),
        'input_features': features,
        'state_probabilities': hsmm_inspection['state_probabilities'],
        'selected_state': hsmm_inspection['selected_state'],
        'sdc': hsmm_inspection['sdc'],
        'stability': hsmm_inspection['stability'],
        'mapping_details': {
            'raw_state': hsmm_inspection['raw_state'],
            'mapped_state': hsmm_inspection['selected_state'],
            'mapping_reason': 'highest probability state',
            'agent_final_state': agent_result.state,
            'agent_passed': agent_result.passed
        },
        'diagnostic': diagnostic
    }


def _diagnose_period(
    features: Dict,
    hsmm_inspection: Dict,
    agent_result
) -> str:
    """
    Diagnose why trend_plus doesn't emerge
    
    Args:
        features: Input features
        hsmm_inspection: HSMM state inspection
        agent_result: Final agent result
    
    Returns:
        Diagnostic string
    """
    
    probs = hsmm_inspection['state_probabilities']
    selected = hsmm_inspection['selected_state']
    
    # Check if range dominates structurally
    if probs['range'] > 0.80:
        return "range dominates structurally (>80% probability)"
    
    # Check if trend_plus is close but loses
    if probs['trend_plus'] > 0.30 and selected == 'range':
        return f"trend_plus competitive ({probs['trend_plus']:.2f}) but range wins"
    
    # Check if features are weak
    if abs(features['returns_mean']) < 0.0005:
        return "weak directional signal in features (returns_mean near zero)"
    
    # Check if trend_plus is very weak
    if probs['trend_plus'] < 0.10:
        return f"trend_plus probability very low ({probs['trend_plus']:.2f})"
    
    # Default
    return f"{selected} selected with {probs[selected]:.2f} probability"


# ============================================================================
# CROSS-PERIOD ANALYSIS
# ============================================================================

def cross_period_summary(period_results: List[Dict]) -> Dict[str, Any]:
    """
    Summarize findings across all periods
    
    Args:
        period_results: List of per-period results
    
    Returns:
        Cross-period summary
    """
    
    # Aggregate probabilities
    trend_plus_probs = [p['state_probabilities']['trend_plus'] for p in period_results]
    range_probs = [p['state_probabilities']['range'] for p in period_results]
    trend_minus_probs = [p['state_probabilities']['trend_minus'] for p in period_results]
    
    # Aggregate features
    returns_means = [p['input_features']['returns_mean'] for p in period_results]
    returns_stds = [p['input_features']['returns_std'] for p in period_results]
    
    # Count states
    states = [p['selected_state'] for p in period_results]
    state_counts = {
        'trend_plus': states.count('trend_plus'),
        'range': states.count('range'),
        'trend_minus': states.count('trend_minus')
    }
    
    # Determine main issue (pass selected states for better verdict)
    main_issue, verdict = _determine_verdict(
        trend_plus_probs,
        range_probs,
        returns_means,
        returns_stds,
        selected_states=states
    )
    
    return {
        'total_periods': len(period_results),
        'state_counts': state_counts,
        'probability_means': {
            'trend_plus': float(np.mean(trend_plus_probs)),
            'range': float(np.mean(range_probs)),
            'trend_minus': float(np.mean(trend_minus_probs))
        },
        'probability_stds': {
            'trend_plus': float(np.std(trend_plus_probs)),
            'range': float(np.std(range_probs)),
            'trend_minus': float(np.std(trend_minus_probs))
        },
        'feature_summary': {
            'returns_mean_avg': float(np.mean(returns_means)),
            'returns_std_avg': float(np.mean(returns_stds))
        },
        'main_issue': main_issue,
        'verdict': verdict
    }


def _determine_verdict(
    trend_plus_probs: List[float],
    range_probs: List[float],
    returns_means: List[float],
    returns_stds: List[float],
    selected_states: Optional[List[str]] = None
) -> tuple:
    """
    Determine main issue and verdict

    Args:
        trend_plus_probs: List of trend+ probabilities per period
        range_probs: List of range probabilities per period
        returns_means: List of returns means per period
        returns_stds: List of returns stds per period
        selected_states: List of selected states per period

    Returns:
        (main_issue, verdict) tuple
    """
    selected_states = selected_states or []
    
    avg_trend_plus = np.mean(trend_plus_probs)
    avg_range = np.mean(range_probs)
    avg_returns = np.mean([abs(r) for r in returns_means])
    
    # Count selected states if provided
    if selected_states:
        trend_plus_count = sum(1 for s in selected_states if s == 'trend_plus')
        range_count = sum(1 for s in selected_states if s == 'range')
        total_count = len(selected_states)
        trend_plus_rate = trend_plus_count / total_count if total_count > 0 else 0
    else:
        trend_plus_count = 0
        trend_plus_rate = 0
    
    # NEW CASE A: Strong improvement - feature alignment fix worked
    # After fix: avg_trend_plus should be high (>60%), range low (<40%)
    if avg_trend_plus > 0.60 and avg_range < 0.40:
        if trend_plus_rate >= 0.5:  # At least half the periods select trend+
            return (
                "feature_alignment_fixed",
                "HSMM feature alignment fix successfully unlocked trend states"
            )
        else:
            return (
                "improved_but_mapping_conservative",
                "HSMM probabilities improved significantly, but final state selection remains conservative"
            )
    
    # NEW CASE B: Partial improvement - trend+ visible but not dominant
    # After fix: avg_trend_plus moderate (30-60%), better than before but not fully unlocked
    if 0.30 < avg_trend_plus <= 0.60 and avg_range < 0.70:
        return (
            "partial_improvement",
            "HSMM improved partially, but range still dominates some periods"
        )
    
    # EXISTING CASE: Weak features
    if avg_returns < 0.0005:
        return (
            "weak_features",
            "The HSMM receives weak / non-trending features, so range is the natural output"
        )
    
    # EXISTING CASE: Structural range dominance
    if avg_range > 0.80:
        return (
            "range_state_dominance",
            "HSMM remains structurally dominated by range"
        )
    
    # EXISTING CASE: Competitive but mapping issue
    if avg_trend_plus > 0.20 and avg_range > avg_trend_plus:
        return (
            "mapping_rigidity",
            "The HSMM emits useful variation, but the final mapping consistently favors range"
        )
    
    # EXISTING CASE: Very weak trend_plus
    if avg_trend_plus < 0.10:
        return (
            "initialization_bias",
            "The HSMM initialization / calibration biases the model toward range, making trend states effectively unreachable"
        )
    
    # Default - mixed results
    return (
        "mixed_results",
        "HSMM results are mixed and need further investigation"
    )


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_hsmm_deep_dive(
    pair: str,
    config: Dict,
    output_dir: str,
    dates: Optional[List[datetime]] = None
) -> Dict:
    """
    Run HSMM deep dive analysis

    Args:
        pair: Trading pair
        config: System config
        output_dir: Output directory
        dates: Optional list of dates (defaults to 4 test periods)

    Returns:
        Results dict
    """

    print(f"\n{'='*80}")
    print(f"HSMM DEEP DIVE - {pair}")
    print(f"{'='*80}\n")

    # Default test dates if not provided
    if dates is None or len(dates) == 0:
        dates = [
            datetime(2023, 1, 15),
            datetime(2023, 3, 15),
            datetime(2023, 10, 15),
            datetime(2023, 12, 15)
        ]
    
    print(f"Test periods: {len(dates)}")
    for d in dates:
        print(f"  - {d.date()}")
    
    # Analyze each period
    print(f"\n{'='*80}")
    print("ANALYZING PERIODS")
    print(f"{'='*80}\n")
    
    period_results = []
    
    for date in dates:
        print(f"Analyzing {date.date()}...")
        
        try:
            result = analyze_period(pair, date, config)
            period_results.append(result)
            
            # Print summary
            print(f"  Features:")
            print(f"    returns_mean: {result['input_features']['returns_mean']:.6f}")
            print(f"    returns_std:  {result['input_features']['returns_std']:.6f}")
            print(f"    atr_mean:     {result['input_features']['atr_mean']:.6f}")
            print(f"  State probabilities:")
            print(f"    Trend+:  {result['state_probabilities']['trend_plus']:.4f}")
            print(f"    Range:   {result['state_probabilities']['range']:.4f}")
            print(f"    Trend-:  {result['state_probabilities']['trend_minus']:.4f}")
            print(f"  Selected: {result['selected_state']}")
            print(f"  SDC: {result['sdc']:.2f}")
            print(f"  Stability: {result['stability']:.2f}")
            print(f"  Diagnostic: {result['diagnostic']}")
            print()
            
        except Exception as e:
            print(f"  ⚠️  Failed: {e}\n")
            period_results.append({
                'date': date.isoformat(),
                'error': str(e)
            })
    
    # Cross-period summary
    print(f"{'='*80}")
    print("CROSS-PERIOD ANALYSIS")
    print(f"{'='*80}\n")
    
    summary = cross_period_summary([r for r in period_results if 'error' not in r])
    
    print(f"Probability means:")
    print(f"  Trend+: {summary['probability_means']['trend_plus']:.4f}")
    print(f"  Range:  {summary['probability_means']['range']:.4f}")
    print(f"  Trend-: {summary['probability_means']['trend_minus']:.4f}")
    
    print(f"\nState counts:")
    print(f"  Trend+: {summary['state_counts']['trend_plus']}")
    print(f"  Range:  {summary['state_counts']['range']}")
    print(f"  Trend-: {summary['state_counts']['trend_minus']}")
    
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    print(f"\nMain issue: {summary['main_issue']}")
    print(f"\n{summary['verdict']}")
    print(f"\n{'-'*80}\n")
    
    # Save report
    output_dir_fractal = Path(output_dir) / 'fractal'
    output_dir_fractal.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir_fractal / f"{pair}_hsmm_deep_dive.json"
    
    report = {
        'pair': pair,
        'test_dates': [d.isoformat() for d in dates],
        'periods': {r['date']: r for r in period_results},
        'cross_period_summary': summary,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_periods': len(dates)
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Deep dive report saved: {report_path}\n")
    
    return {
        'status': 'completed',
        'pair': pair,
        'verdict': summary['verdict'],
        'main_issue': summary['main_issue'],
        'output': str(report_path)
    }


if __name__ == "__main__":
    # Quick test
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_hsmm_deep_dive(
        pair='BTCUSDT',
        config=config,
        output_dir='reports/validation'
    )
