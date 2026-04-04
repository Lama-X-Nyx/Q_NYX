"""
Test Regime Tuning

Validates regime_tuning script and output.
"""

import sys
sys.path.insert(0, '.')

import json
import yaml
from pathlib import Path
from datetime import datetime


def test_regime_tuning_runs():
    """Test 1: regime_tuning runs without errors"""
    
    from scripts.regime_tuning import run_regime_tuning
    
    # Load config
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    # Run with single date to speed up test
    dates = [datetime(2023, 12, 15)]
    
    output_dir = 'reports/validation'
    
    # Should not crash
    result = run_regime_tuning(
        pair='BTCUSDT',
        config=config,
        output_dir=output_dir,
        dates=dates
    )
    
    assert result['status'] == 'completed'
    assert result['pair'] == 'BTCUSDT'
    assert 'conclusion' in result
    
    print("✅ Test 1 passed: regime_tuning runs")


def test_json_output_valid():
    """Test 2: JSON output is valid"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_regime_tuning.json')
    
    assert report_path.exists(), f"Report not found: {report_path}"
    
    with open(report_path) as f:
        report = json.load(f)
    
    # Check structure
    assert 'pair' in report
    assert 'test_dates' in report
    assert 'scenarios' in report
    assert 'verdict' in report
    
    # Check scenarios
    assert len(report['scenarios']) >= 4
    
    for scenario in report['scenarios']:
        assert 'scenario' in scenario
        assert 'label' in scenario
        assert 'description' in scenario
        assert 'params' in scenario
        assert 'results' in scenario
        assert 'summary' in scenario
        
        # Check summary
        summary = scenario['summary']
        assert 'trend_plus_rate' in summary
        assert 'range_rate' in summary
        assert 'total_periods' in summary
    
    print("✅ Test 2 passed: JSON output valid")


def test_multiple_scenarios_present():
    """Test 3: Multiple scenarios are tested"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_regime_tuning.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    scenarios = report['scenarios']
    
    # Should have baseline + relaxed variants
    scenario_names = [s['scenario'] for s in scenarios]
    
    assert 'baseline' in scenario_names
    assert len(scenario_names) >= 3  # At least baseline + 2 variants
    
    # Check params vary
    params_set = set()
    for scenario in scenarios:
        params = scenario['params']
        params_tuple = (params['sdc_min'], params['stability_min'])
        params_set.add(params_tuple)
    
    assert len(params_set) >= 3, "Scenarios should have different parameters"
    
    print("✅ Test 3 passed: Multiple scenarios present")


def test_verdict_present():
    """Test 4: Verdict is produced"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_regime_tuning.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    verdict = report['verdict']
    
    assert 'conclusion' in verdict
    assert 'message' in verdict
    assert 'recommendation' in verdict
    
    # Conclusion should be one of expected values
    valid_conclusions = [
        'Regime remains too conservative',
        'Regime becomes usable with mild relaxation',
        'Regime only improves with aggressive relaxation',
        'Regime tuning alone is not enough'
    ]
    
    assert verdict['conclusion'] in valid_conclusions
    
    print("✅ Test 4 passed: Verdict present")


def test_no_crash_if_no_improvement():
    """Test 5: No crash if no scenario improves"""
    
    # This should already be the case from our test run
    # (all scenarios show 0% trend_plus)
    
    report_path = Path('reports/validation/fractal/BTCUSDT_regime_tuning.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    scenarios = report['scenarios']
    
    # Check if all scenarios have same or similar trend_plus_rate
    rates = [s['summary']['trend_plus_rate'] for s in scenarios]
    
    # If all rates are 0, verdict should handle it gracefully
    if all(r == 0 for r in rates):
        verdict = report['verdict']
        assert 'conclusion' in verdict
        assert verdict['conclusion'] in [
            'Regime remains too conservative',
            'Regime tuning alone is not enough'
        ]
    
    print("✅ Test 5 passed: No crash if no improvement")


def test_no_crash_if_all_trend_plus():
    """Test 6: No crash if scenario makes everything trend_plus"""
    
    # This test verifies the verdict logic handles over-aggressive scenarios
    
    from scripts.regime_tuning import determine_verdict
    
    # Mock scenario results - all trend_plus
    mock_results = [
        {
            'scenario': 'baseline',
            'label': 'Baseline',
            'summary': {
                'trend_plus_rate': 10.0,
                'range_rate': 90.0
            }
        },
        {
            'scenario': 'aggressive',
            'label': 'Aggressive',
            'summary': {
                'trend_plus_rate': 100.0,  # Over-aggressive
                'range_rate': 0.0
            }
        }
    ]
    
    # Should not crash
    verdict = determine_verdict(mock_results)
    
    assert 'conclusion' in verdict
    
    print("✅ Test 6 passed: No crash if all trend_plus")


def test_scenario_params_differ():
    """Test 7: Scenario parameters actually differ"""
    
    report_path = Path('reports/validation/fractal/BTCUSDT_regime_tuning.json')
    
    with open(report_path) as f:
        report = json.load(f)
    
    scenarios = report['scenarios']
    
    # Get baseline params
    baseline = next(s for s in scenarios if s['scenario'] == 'baseline')
    baseline_sdc = baseline['params']['sdc_min']
    baseline_stability = baseline['params']['stability_min']
    
    # Check at least one scenario has different params
    has_different = False
    for scenario in scenarios:
        if scenario['scenario'] != 'baseline':
            if (scenario['params']['sdc_min'] != baseline_sdc or 
                scenario['params']['stability_min'] != baseline_stability):
                has_different = True
                break
    
    assert has_different, "At least one scenario should have different params from baseline"
    
    print("✅ Test 7 passed: Scenario params differ")


def run_all_tests():
    """Run all tests"""
    
    print("\n" + "="*80)
    print("REGIME TUNING TESTS")
    print("="*80 + "\n")
    
    tests = [
        test_regime_tuning_runs,
        test_json_output_valid,
        test_multiple_scenarios_present,
        test_verdict_present,
        test_no_crash_if_no_improvement,
        test_no_crash_if_all_trend_plus,
        test_scenario_params_differ
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
