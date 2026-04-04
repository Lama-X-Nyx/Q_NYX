"""
Test Setup Investigation

Tests that setup_investigation correctly:
- Runs setup_deep_dive first
- Aggregates metrics
- Determines root causes
- Produces valid JSON
"""

import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from datetime import datetime
import yaml

from scripts.setup_investigation import run_setup_investigation


def test_setup_investigation_produces_valid_json():
    """Test that setup_investigation produces valid JSON"""
    print("Testing setup_investigation JSON output...")
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    # Run with single date for speed
    dates = [datetime(2023, 12, 15)]
    result = run_setup_investigation('BTCUSDT', config, 'reports/validation', dates=dates)
    
    # Check result
    assert result['status'] == 'completed'
    assert 'main_root_cause' in result
    assert 'verdict' in result
    
    # Check JSON file exists and is valid
    json_path = Path('reports/validation/fractal/BTCUSDT_setup_investigation.json')
    assert json_path.exists()
    
    with open(json_path) as f:
        data = json.load(f)
    
    assert 'pair' in data
    assert 'periods' in data
    assert 'cross_period_summary' in data
    
    summary = data['cross_period_summary']
    assert 'no_pattern_rate' in summary
    assert 'alignment_low_rate' in summary
    assert 'main_root_cause' in summary
    assert 'verdict' in summary
    
    print("✅ Valid JSON output with root cause analysis")


def test_root_cause_identification():
    """Test that root cause is correctly identified"""
    print("Testing root cause identification...")
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    dates = [datetime(2023, 12, 15)]
    result = run_setup_investigation('BTCUSDT', config, 'reports/validation', dates=dates)
    
    # Root cause should be one of the expected values
    valid_root_causes = [
        'pattern_scarcity',
        'alignment_filtering',
        'mixed_scarcity_and_filtering',
        'unclear'
    ]
    
    assert result['main_root_cause'] in valid_root_causes
    print(f"✅ Root cause identified: {result['main_root_cause']}")


def test_verdict_is_present():
    """Test that investigation provides actionable verdict"""
    print("Testing verdict presence...")
    
    json_path = Path('reports/validation/fractal/BTCUSDT_setup_investigation.json')
    
    with open(json_path) as f:
        data = json.load(f)
    
    verdict = data['cross_period_summary']['verdict']
    
    # Verdict should be non-empty and actionable
    assert len(verdict) > 0
    assert isinstance(verdict, str)
    
    # Should mention the nature of the problem
    should_contain_one_of = ['scarcity', 'filtering', 'alignment', 'pattern', 'decomposition']
    assert any(keyword in verdict.lower() for keyword in should_contain_one_of)
    
    print(f"✅ Verdict is actionable: {verdict[:60]}...")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING SETUP INVESTIGATION")
    print("="*60 + "\n")
    
    test_setup_investigation_produces_valid_json()
    test_root_cause_identification()
    test_verdict_is_present()
    
    print("\n✅ ALL SETUP INVESTIGATION TESTS PASSED\n")
