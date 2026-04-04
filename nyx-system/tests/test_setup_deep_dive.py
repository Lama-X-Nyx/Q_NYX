"""
Test Setup Deep Dive

Tests that setup_deep_dive correctly:
- Produces valid JSON output
- Categorizes blocking reasons
- Calculates pass rates
- Determines verdicts
"""

import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from datetime import datetime
import yaml

from scripts.setup_deep_dive import run_setup_deep_dive, _determine_setup_verdict


def test_setup_deep_dive_produces_valid_json():
    """Test that setup_deep_dive produces valid JSON"""
    print("Testing setup_deep_dive JSON output...")
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    # Run with single date for speed
    dates = [datetime(2023, 12, 15)]
    result = run_setup_deep_dive('BTCUSDT', config, 'reports/validation', dates=dates)
    
    # Check result
    assert result['status'] == 'completed'
    assert 'avg_pass_rate' in result
    assert 'main_issue' in result
    
    # Check JSON file exists and is valid
    json_path = Path('reports/validation/fractal/BTCUSDT_setup_deep_dive.json')
    assert json_path.exists()
    
    with open(json_path) as f:
        data = json.load(f)
    
    assert 'pair' in data
    assert 'periods' in data
    assert 'cross_period_summary' in data
    
    print("✅ Valid JSON output produced")


def test_setup_verdict_logic():
    """Test setup verdict determination logic"""
    print("Testing setup verdict logic...")
    
    # Case: Pattern absence dominant
    period_results = [
        {
            'setup_pass_rate': 1.0,
            'setup_blocked': 100,
            'block_reasons': {'no_pattern': 85, 'alignment_score_too_low': 10, 'misaligned_pattern': 5}
        }
    ]
    
    issue, verdict = _determine_setup_verdict(period_results)
    assert issue == "pattern_absence_dominant"
    print("✅ Detects pattern absence dominance")
    
    # Case: Alignment dominant
    period_results = [
        {
            'setup_pass_rate': 2.0,
            'setup_blocked': 100,
            'block_reasons': {'no_pattern': 30, 'alignment_score_too_low': 60, 'misaligned_pattern': 10}
        }
    ]
    
    issue, verdict = _determine_setup_verdict(period_results)
    assert issue == "alignment_dominant"
    print("✅ Detects alignment dominance")


def test_no_crash_on_zero_patterns():
    """Test that setup_deep_dive handles zero patterns gracefully"""
    print("Testing zero pattern handling...")
    
    # This should not crash even if no patterns found
    period_results = [
        {
            'setup_pass_rate': 0.0,
            'setup_blocked': 50,
            'block_reasons': {'no_pattern': 50, 'alignment_score_too_low': 0, 'misaligned_pattern': 0}
        }
    ]
    
    issue, verdict = _determine_setup_verdict(period_results)
    assert issue == "pattern_absence_dominant"
    print("✅ Handles zero patterns gracefully")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING SETUP DEEP DIVE")
    print("="*60 + "\n")
    
    test_setup_verdict_logic()
    test_no_crash_on_zero_patterns()
    test_setup_deep_dive_produces_valid_json()
    
    print("\n✅ ALL SETUP DEEP DIVE TESTS PASSED\n")
