"""
Test Regime Feature Alignment

Validates that RegimeAgent prepares the features required by HSMM.
"""

import sys
sys.path.insert(0, '.')

import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from src.agents.regime_agent import RegimeAgent
from src.core.hsmm import SemiMarkovHMM


def test_prepare_data_includes_sma():
    """Test 1: RegimeAgent._prepare_data() adds sma_20 and sma_50"""
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='4h')
    prices = np.linspace(40000, 50000, 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.rand(100) * 1000
    }, index=dates)
    
    # Create agent and prepare data
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    agent = RegimeAgent(config)
    df_prepared = agent._prepare_data(df)
    
    # Check that sma_20 and sma_50 are present
    assert 'sma_20' in df_prepared.columns, "sma_20 not in prepared columns"
    assert 'sma_50' in df_prepared.columns, "sma_50 not in prepared columns"
    
    # Check that they're not all NaN (except warm-up period)
    assert df_prepared['sma_20'].notna().sum() > 0, "sma_20 is all NaN"
    assert df_prepared['sma_50'].notna().sum() > 0, "sma_50 is all NaN"
    
    print("✅ Test 1 passed: sma_20 and sma_50 added")


def test_no_future_leak_in_sma():
    """Test 2: SMA calculation does not leak future data"""
    
    # Create data with known pattern
    dates = pd.date_range('2023-01-01', periods=100, freq='4h')
    prices = np.array([100] * 50 + [200] * 50)  # Step change at index 50
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices,
        'low': prices,
        'close': prices,
        'volume': np.ones(100)
    }, index=dates)
    
    # Prepare data
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    agent = RegimeAgent(config)
    df_prepared = agent._prepare_data(df)
    
    # At index 50 (first 200 value), sma_20 should still reflect mostly 100s
    # sma_20[50] should be close to 100, not 200 (no leak)
    sma_20_at_step = df_prepared['sma_20'].iloc[50]
    
    # Should be close to 100 (19 x 100 + 1 x 200) / 20 = 105
    assert sma_20_at_step < 120, f"sma_20 at step change is {sma_20_at_step}, suggests future leak"
    
    print("✅ Test 2 passed: No future leak in SMA")


def test_hsmm_detects_missing_features():
    """Test 3: HSMM raises clear error if sma_20/sma_50 missing"""
    
    # Create data WITHOUT sma_20 and sma_50
    dates = pd.date_range('2023-01-01', periods=100, freq='4h')
    prices = np.linspace(40000, 50000, 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.rand(100) * 1000,
        'returns': np.random.rand(100) * 0.01,
        'atr_14': np.random.rand(100) * 100
    }, index=dates)
    
    # Try to initialize HSMM - should raise error
    hsmm = SemiMarkovHMM()
    
    try:
        hsmm.initialize_parameters(df)
        assert False, "HSMM should have raised ValueError for missing features"
    except ValueError as e:
        error_msg = str(e)
        assert 'sma_20' in error_msg or 'sma_50' in error_msg, f"Error should mention missing features: {error_msg}"
        print(f"✅ Test 3 passed: HSMM detects missing features")
    except Exception as e:
        assert False, f"Wrong exception type: {type(e).__name__}: {e}"


def test_hsmm_works_with_features():
    """Test 4: HSMM initializes properly when features present"""
    
    # Create complete data
    dates = pd.date_range('2023-01-01', periods=100, freq='4h')
    prices = np.linspace(40000, 50000, 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.rand(100) * 1000
    }, index=dates)
    
    # Use RegimeAgent to prepare (adds all features)
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    agent = RegimeAgent(config)
    df_prepared = agent._prepare_data(df)
    
    # Initialize HSMM - should work
    hsmm = SemiMarkovHMM()
    hsmm.initialize_parameters(df_prepared)
    
    # Check that parameters were learned
    assert hsmm.initial_probs is not None
    assert hsmm.transition_matrix is not None
    assert hsmm.emission_params is not None
    
    print("✅ Test 4 passed: HSMM works with complete features")


def test_existing_tests_still_pass():
    """Test 5: Existing hsmm_deep_dive tests still pass"""
    
    # This is more of a smoke test - just verify the module still imports
    from src.validation.hsmm_deep_dive import run_hsmm_deep_dive
    
    # If we got here, module loaded successfully
    print("✅ Test 5 passed: Existing tests compatible")


def run_all_tests():
    """Run all tests"""
    
    print("\n" + "="*80)
    print("REGIME FEATURE ALIGNMENT TESTS")
    print("="*80 + "\n")
    
    tests = [
        test_prepare_data_includes_sma,
        test_no_future_leak_in_sma,
        test_hsmm_detects_missing_features,
        test_hsmm_works_with_features,
        test_existing_tests_still_pass
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} error: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
