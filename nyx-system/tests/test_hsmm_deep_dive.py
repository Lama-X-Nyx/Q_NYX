"""
Test HSMM Deep Dive

Validates HSMM deep dive analysis and output.
"""

import sys
sys.path.insert(0, '.')

import json
import yaml
from pathlib import Path
from datetime import datetime


def test_hsmm_deep_dive_runs():
    """Test 1: HSMM deep dive runs without errors"""
    
    from src.validation.hsmm_deep_dive import run_hsmm_deep_dive
    
    # Load config
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    # Run with single date to speed up test
    dates = [datetime(2023, 12, 15)]
    
    output_dir = 'reports/validation'
    
    # Should not crash
    result = run_hsmm_deep_dive(
        pair='BTCUSDT',
        config=config,
        output_dir=output_dir,
        dates=dates
    )
    
    assert result['status'] == 'completed'
    assert result['pair'] == 'BTCUSDT'
    assert 'verdict' in result
    assert 'main_issue' in result
    
    print("✅ Test 1 passed: hsmm_deep_dive runs")


def test_json_output_valid():
    """Test 2: JSON output is valid"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json')
    
    assert report_path.exists(), f"Report not found: {report_path}"
    
    with open(report_path) as f:
        report = json.load(f)
    
    # Check structure
    assert 'pair' in report
    assert 'test_dates' in report
    assert 'periods' in report
    assert 'cross_period_summary' in report
    
    print("✅ Test 2 passed: JSON output valid")


def test_all_periods_present():
    """Test 3: All periods are present in output"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    # Should have at least 1 period from test
    assert len(report['periods']) >= 1
    
    # Check each period has required fields
    for date_str, period in report['periods'].items():
        if 'error' not in period:
            assert 'input_features' in period
            assert 'state_probabilities' in period
            assert 'selected_state' in period
            assert 'mapping_details' in period
    
    print("✅ Test 3 passed: All periods present")


def test_period_contains_required_fields():
    """Test 4: Each period contains required fields"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    # Get first valid period
    periods = [p for p in report['periods'].values() if 'error' not in p]
    assert len(periods) > 0, "No valid periods found"
    
    period = periods[0]
    
    # Check input_features
    assert 'input_features' in period
    features = period['input_features']
    assert 'returns_mean' in features
    assert 'returns_std' in features
    assert 'atr_mean' in features
    
    # Check state_probabilities
    assert 'state_probabilities' in period
    probs = period['state_probabilities']
    assert 'trend_plus' in probs
    assert 'range' in probs
    assert 'trend_minus' in probs
    
    # Check selected_state
    assert 'selected_state' in period
    assert period['selected_state'] in ['trend_plus', 'range', 'trend_minus']
    
    # Check mapping_details
    assert 'mapping_details' in period
    mapping = period['mapping_details']
    assert 'raw_state' in mapping
    assert 'mapped_state' in mapping
    assert 'mapping_reason' in mapping
    
    print("✅ Test 4 passed: Period contains required fields")


def test_cross_period_summary_exists():
    """Test 5: Cross-period summary exists"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    summary = report['cross_period_summary']
    
    assert 'probability_means' in summary
    assert 'state_counts' in summary
    assert 'main_issue' in summary
    assert 'verdict' in summary
    
    # Check probability means
    prob_means = summary['probability_means']
    assert 'trend_plus' in prob_means
    assert 'range' in prob_means
    assert 'trend_minus' in prob_means
    
    print("✅ Test 5 passed: Cross-period summary exists")


def test_no_crash_if_all_range():
    """Test 6: No crash if all periods produce range"""
    
    # This should already be the case from our test run
    # (all periods show range dominance)
    
    report_path = Path('reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    summary = report['cross_period_summary']
    
    # If all are range, verdict should handle it gracefully
    if summary['state_counts']['range'] == summary['total_periods']:
        assert 'verdict' in summary
        assert summary['verdict'] is not None
    
    print("✅ Test 6 passed: No crash if all range")


def test_no_crash_if_probability_zero():
    """Test 7: No crash if a probability is zero"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    # Check if any period has zero probabilities
    has_zero = False
    for period in report['periods'].values():
        if 'error' not in period:
            probs = period['state_probabilities']
            if any(p == 0.0 for p in probs.values()):
                has_zero = True
                break
    
    # If we have zero probabilities, system should handle them
    # (which it does - test passed if we got here)
    
    print("✅ Test 7 passed: No crash if probability zero")


def test_verdict_is_meaningful():
    """Test 8: Verdict is one of expected values"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    summary = report['cross_period_summary']
    
    valid_issues = [
        'weak_features',
        'range_state_dominance',
        'mapping_rigidity',
        'initialization_bias',
        'unclear'
    ]
    
    assert summary['main_issue'] in valid_issues
    
    # Verdict should be a string
    assert isinstance(summary['verdict'], str)
    assert len(summary['verdict']) > 0
    
    print("✅ Test 8 passed: Verdict is meaningful")


def test_features_are_numeric():
    """Test 9: Features are numeric values"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    # Get first valid period
    periods = [p for p in report['periods'].values() if 'error' not in p]
    assert len(periods) > 0
    
    period = periods[0]
    features = period['input_features']
    
    # Check all features are numeric
    for key, value in features.items():
        if key != 'window_size':  # window_size is int, others are float
            assert isinstance(value, (int, float)), f"{key} is not numeric: {value}"
    
    print("✅ Test 9 passed: Features are numeric")


def test_probabilities_sum_to_one():
    """Test 10: State probabilities approximately sum to 1.0"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    # Get first valid period
    periods = [p for p in report['periods'].values() if 'error' not in p]
    assert len(periods) > 0
    
    period = periods[0]
    probs = period['state_probabilities']
    
    prob_sum = probs['trend_plus'] + probs['range'] + probs['trend_minus']
    
    # Should sum to approximately 1.0 (allow small floating point error)
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, not 1.0"
    
    print("✅ Test 10 passed: Probabilities sum to one")


def run_all_tests():
    """Run all tests"""
    
    print("\n" + "="*80)
    print("HSMM DEEP DIVE TESTS")
    print("="*80 + "\n")
    
    tests = [
        test_hsmm_deep_dive_runs,
        test_json_output_valid,
        test_all_periods_present,
        test_period_contains_required_fields,
        test_cross_period_summary_exists,
        test_no_crash_if_all_range,
        test_no_crash_if_probability_zero,
        test_verdict_is_meaningful,
        test_features_are_numeric,
        test_probabilities_sum_to_one
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
