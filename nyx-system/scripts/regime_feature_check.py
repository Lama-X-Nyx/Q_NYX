"""
Regime Feature Alignment Check

Validates that RegimeAgent prepares all features required by HSMM.

This check was created after discovering that HSMM._label_states_heuristic()
requires sma_20 and sma_50, but RegimeAgent._prepare_data() was not providing
them, causing initialization to default to Range state.
"""

import sys
sys.path.insert(0, '.')

import json
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from src.data.mtf_loader import load_fractal_context
from src.agents.regime_agent import RegimeAgent
from src.core.hsmm import SemiMarkovHMM


def check_feature_alignment(
    pair: str,
    sample_date: datetime,
    config: Dict
) -> Dict:
    """
    Check alignment between Regime features and HSMM requirements
    
    Args:
        pair: Trading pair
        sample_date: Reference date for testing
        config: System config
    
    Returns:
        Dict with alignment check results
    """
    
    # Load data
    mtf_data = load_fractal_context(pair, sample_date, config)
    
    # Get regime timeframe
    regime_tf = config['fractal']['timeframes']['regime_tf']
    df = mtf_data[regime_tf]
    
    # Create regime agent
    regime_agent = RegimeAgent(config)
    
    # Prepare data (this should add required features)
    df_prepared = regime_agent._prepare_data(df)
    
    # Get prepared columns
    prepared_columns = df_prepared.columns.tolist()
    
    # HSMM required features (from _label_states_heuristic inspection)
    required_by_hsmm = ['sma_20', 'sma_50']
    
    # Also recommended (used by emission parameters)
    recommended = ['returns', 'atr_14']
    
    # Check what's missing
    missing_required = [f for f in required_by_hsmm if f not in prepared_columns]
    missing_recommended = [f for f in recommended if f not in prepared_columns]
    
    # Determine if alignment is correct
    alignment_ok = len(missing_required) == 0 and len(missing_recommended) == 0
    
    # Generate verdict
    if alignment_ok:
        verdict = "All required HSMM features are present"
    elif missing_required:
        verdict = f"CRITICAL: Missing required features: {missing_required}. HSMM will fail or default to Range."
    else:
        verdict = f"WARNING: Missing recommended features: {missing_recommended}"
    
    # Try to initialize HSMM to verify
    hsmm_init_success = False
    hsmm_error = None
    
    try:
        hsmm = SemiMarkovHMM()
        hsmm.initialize_parameters(df_prepared)
        hsmm_init_success = True
    except Exception as e:
        hsmm_error = str(e)
    
    return {
        'pair': pair,
        'sample_date': sample_date.isoformat(),
        'regime_timeframe': regime_tf,
        'prepared_columns': prepared_columns,
        'required_by_hsmm': required_by_hsmm,
        'recommended_features': recommended,
        'missing_required': missing_required,
        'missing_recommended': missing_recommended,
        'feature_alignment_ok': alignment_ok,
        'hsmm_initialization_success': hsmm_init_success,
        'hsmm_error': hsmm_error,
        'verdict': verdict
    }


def run_regime_feature_check(
    pair: str,
    config: Dict,
    output_dir: str,
    sample_date: datetime = None
) -> Dict:
    """
    Run regime feature alignment check
    
    Args:
        pair: Trading pair
        config: System config
        output_dir: Output directory
        sample_date: Optional sample date (defaults to 2023-12-15)
    
    Returns:
        Results dict
    """
    
    print(f"\n{'='*80}")
    print(f"REGIME FEATURE ALIGNMENT CHECK - {pair}")
    print(f"{'='*80}\n")
    
    # Default sample date
    if sample_date is None:
        sample_date = datetime(2023, 12, 15)
    
    print(f"Reference date: {sample_date.date()}\n")
    
    # Run check
    result = check_feature_alignment(pair, sample_date, config)
    
    # Print results
    print(f"Prepared columns:")
    for col in result['prepared_columns']:
        print(f"  {col}")
    
    print(f"\nRequired by HSMM heuristic:")
    for feat in result['required_by_hsmm']:
        status = "✓" if feat in result['prepared_columns'] else "✗"
        print(f"  {status} {feat}")
    
    print(f"\nRecommended features:")
    for feat in result['recommended_features']:
        status = "✓" if feat in result['prepared_columns'] else "✗"
        print(f"  {status} {feat}")
    
    if result['missing_required']:
        print(f"\nMissing required:")
        for feat in result['missing_required']:
            print(f"  ✗ {feat}")
    
    if result['missing_recommended']:
        print(f"\nMissing recommended:")
        for feat in result['missing_recommended']:
            print(f"  ⚠ {feat}")
    
    print(f"\nHSMM initialization: {'✓ Success' if result['hsmm_initialization_success'] else '✗ Failed'}")
    if result['hsmm_error']:
        print(f"Error: {result['hsmm_error']}")
    
    print(f"\nVerdict:")
    if result['feature_alignment_ok']:
        print(f"  ✅ {result['verdict']}")
    else:
        print(f"  ❌ {result['verdict']}")
    
    print(f"\n{'-'*80}\n")
    
    # Save report
    output_dir_fractal = Path(output_dir) / 'fractal'
    output_dir_fractal.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir_fractal / f"{pair}_regime_feature_check.json"
    
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Feature check report saved: {report_path}\n")
    
    return {
        'status': 'completed',
        'pair': pair,
        'alignment_ok': result['feature_alignment_ok'],
        'verdict': result['verdict'],
        'output': str(report_path)
    }


if __name__ == "__main__":
    # Quick test
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_regime_feature_check(
        pair='BTCUSDT',
        config=config,
        output_dir='reports/validation'
    )
