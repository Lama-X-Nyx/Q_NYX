"""
Test Bullish Audit

Validates that bullish audit correctly identifies Context vs Regime issues.
"""

import sys
sys.path.insert(0, '.')

import pytest
import json
from pathlib import Path
import tempfile
import yaml


class TestBullishAudit:
    """Test suite for bullish audit"""
    
    @pytest.fixture
    def config(self):
        """Load real config"""
        with open('config/validation_baseline.yaml') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_produces_valid_json(self, config, temp_output_dir):
        """Test 1: Mode produces valid JSON"""
        from scripts.bullish_audit import run_bullish_audit
        
        result = run_bullish_audit(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        # Check JSON file exists
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_bullish_audit.json'
        assert json_path.exists()
        
        # Load and validate JSON
        with open(json_path) as f:
            report = json.load(f)
        
        # Check required fields
        assert 'pair' in report
        assert 'periods' in report
        assert 'cross_period_summary' in report
    
    def test_all_required_periods_present(self, config, temp_output_dir):
        """Test 2: All required periods are present"""
        from scripts.bullish_audit import run_bullish_audit
        
        run_bullish_audit(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_bullish_audit.json'
        with open(json_path) as f:
            report = json.load(f)
        
        periods = report['periods']
        
        # Check all 4 default periods
        assert '2023-01-15' in periods
        assert '2023-03-15' in periods
        assert '2023-10-15' in periods
        assert '2023-12-15' in periods
    
    def test_period_structure_complete(self, config, temp_output_dir):
        """Test 3: Each period has context, regime, joint_interpretation"""
        from scripts.bullish_audit import run_bullish_audit
        
        run_bullish_audit(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_bullish_audit.json'
        with open(json_path) as f:
            report = json.load(f)
        
        for date, period in report['periods'].items():
            # Skip if error
            if 'error' in period:
                continue
            
            # Check required fields
            assert 'context' in period
            assert 'regime' in period
            assert 'joint_interpretation' in period
            
            # Check context fields
            context = period['context']
            assert 'state' in context
            assert 'score' in context
            assert 'passed' in context
            assert 'reason' in context
            
            # Check regime fields
            regime = period['regime']
            assert 'state' in regime
            assert 'score' in regime
            assert 'passed' in regime
            assert 'reason' in regime
    
    def test_cross_period_summary_exists(self, config, temp_output_dir):
        """Test 4: cross_period_summary exists and is complete"""
        from scripts.bullish_audit import run_bullish_audit
        
        run_bullish_audit(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_bullish_audit.json'
        with open(json_path) as f:
            report = json.load(f)
        
        summary = report['cross_period_summary']
        
        # Check required fields
        assert 'context_bullish_rate' in summary
        assert 'regime_trend_plus_rate' in summary
        assert 'main_issue' in summary
        assert 'verdict' in summary
    
    def test_verdict_automatic_present(self, config, temp_output_dir):
        """Test 5: Automatic verdict is present and valid"""
        from scripts.bullish_audit import run_bullish_audit
        
        run_bullish_audit(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_bullish_audit.json'
        with open(json_path) as f:
            report = json.load(f)
        
        verdict = report['cross_period_summary']['verdict']
        
        # Should be one of the expected verdicts
        valid_verdicts = [
            "Context and regime both express bullish BTC correctly.",
            "Context is often bullish, but regime collapses too often into range.",
            "Context itself is too neutral during bullish BTC periods.",
            "Bullish expression is inconsistent and depends strongly on the chosen sample."
        ]
        
        assert verdict in valid_verdicts
    
    def test_handles_neutral_range_periods(self, config, temp_output_dir):
        """Test 6: No crash if periods remain neutral/range"""
        from scripts.bullish_audit import run_bullish_audit
        
        # Should complete without error even if all periods are neutral/range
        result = run_bullish_audit(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        assert result['status'] == 'completed'
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_bullish_audit.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should still produce valid report
        assert report['cross_period_summary']['verdict'] is not None
    
    def test_custom_dates_work(self, config, temp_output_dir):
        """Test custom date list works"""
        from scripts.bullish_audit import run_bullish_audit
        
        # Test with custom dates
        custom_dates = ['2023-01-15', '2023-03-15']
        
        result = run_bullish_audit(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            dates=custom_dates
        )
        
        assert result['status'] == 'completed'
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_bullish_audit.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should have 2 periods
        assert len(report['periods']) == 2
        assert '2023-01-15' in report['periods']
        assert '2023-03-15' in report['periods']
    
    def test_main_issue_identified(self, config, temp_output_dir):
        """Test that main issue is correctly identified"""
        from scripts.bullish_audit import run_bullish_audit
        
        run_bullish_audit(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_bullish_audit.json'
        with open(json_path) as f:
            report = json.load(f)
        
        main_issue = report['cross_period_summary']['main_issue']
        
        # Should be one of the expected issues
        valid_issues = [
            'regime_too_conservative',
            'context_too_neutral',
            'both_too_neutral',
            'sample_dependent',
            'insufficient_data'
        ]
        
        assert main_issue in valid_issues
    
    def test_joint_interpretations_valid(self, config, temp_output_dir):
        """Test that joint interpretations are valid"""
        from scripts.bullish_audit import run_bullish_audit
        
        run_bullish_audit(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_bullish_audit.json'
        with open(json_path) as f:
            report = json.load(f)
        
        valid_interpretations = [
            'bullish_coherent',
            'regime_too_conservative',
            'context_too_neutral',
            'bullish_not_expressed',
            'bearish_detected',
            'mixed_signals'
        ]
        
        for period in report['periods'].values():
            if 'joint_interpretation' in period:
                assert period['joint_interpretation'] in valid_interpretations
