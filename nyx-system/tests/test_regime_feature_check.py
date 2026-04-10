"""
Test Regime Feature Check

Validates regime_feature_check mode and output.
"""

import sys
sys.path.insert(0, '.')

import json
import yaml
from pathlib import Path
from datetime import datetime


def test_feature_check_runs():
    """Test 1: regime_feature_check mode runs without errors"""
    
    from scripts.regime_feature_check import run_regime_feature_check
    
    # Load config
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    # Run check
    result = run_regime_feature_check(
        pair='BTCUSDT',
        config=config,
        output_dir='reports/validation',
        sample_date=datetime(2023, 12, 15)
    )
    
    assert result['status'] == 'completed'
    assert result['pair'] == 'BTCUSDT'
    assert 'alignment_ok' in result
    
    print("✅ Test 1 passed: feature_check runs")


def test_feature_check_json_valid():
    """Test 2: JSON output is valid and complete"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_regime_feature_check.json')
    
    assert report_path.exists(), f"Report not found: {report_path}"
    
    with open(report_path) as f:
        report = json.load(f)
    
    # Check structure
    assert 'pair' in report
    assert 'prepared_columns' in report
    assert 'required_by_hsmm' in report
    assert 'missing_required' in report
    assert 'feature_alignment_ok' in report
    assert 'verdict' in report
    
    print("✅ Test 2 passed: JSON output valid")


def test_feature_check_detects_alignment():
    """Test 3: Correctly identifies if features are aligned"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_regime_feature_check.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    # After the fix, alignment should be OK
    assert report['feature_alignment_ok'] == True, "Features should be aligned after fix"
    
    # Check that required features are present
    assert 'sma_20' in report['prepared_columns']
    assert 'sma_50' in report['prepared_columns']
    
    # Check that nothing is missing
    assert len(report['missing_required']) == 0
    
    print("✅ Test 3 passed: Correctly detects alignment")


def test_hsmm_init_success():
    """Test 4: HSMM initialization succeeds after fix"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_regime_feature_check.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    # HSMM init should succeed
    assert report['hsmm_initialization_success'] == True
    assert report['hsmm_error'] is None
    
    print("✅ Test 4 passed: HSMM initialization succeeds")


def run_all_tests():
    """Run all tests"""
    
    print("\n" + "="*80)
    print("REGIME FEATURE CHECK TESTS")
    print("="*80 + "\n")
    
    tests = [
        test_feature_check_runs,
        test_feature_check_json_valid,
        test_feature_check_detects_alignment,
        test_hsmm_init_success
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
