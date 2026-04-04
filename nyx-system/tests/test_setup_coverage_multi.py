"""
Test Multi-Period Setup Coverage

Validates that multi-period comparison works correctly.
"""

import sys
sys.path.insert(0, '.')

import pytest
import json
from pathlib import Path
import tempfile
import yaml


class TestSetupCoverageMulti:
    """Test suite for multi-period setup coverage"""
    
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
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        
        result = run_setup_coverage_multi(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        # Check JSON file exists
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_coverage_multi.json'
        assert json_path.exists()
        
        # Load and validate JSON
        with open(json_path) as f:
            report = json.load(f)
        
        # Check required fields
        assert 'pair' in report
        assert 'periods' in report
        assert 'cross_period_summary' in report
    
    def test_all_required_periods_present(self, config, temp_output_dir):
        """Test 2: All 3 required periods are present"""
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        
        run_setup_coverage_multi(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_coverage_multi.json'
        with open(json_path) as f:
            report = json.load(f)
        
        periods = report['periods']
        
        # Check all 3 required periods
        assert '2022-06-15' in periods  # Bear
        assert '2023-03-15' in periods  # Bull
        assert '2023-12-15' in periods  # Range
    
    def test_period_structure_complete(self, config, temp_output_dir):
        """Test 3: Each period has required fields"""
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        
        run_setup_coverage_multi(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_coverage_multi.json'
        with open(json_path) as f:
            report = json.load(f)
        
        for date, period in report['periods'].items():
            # Skip if error
            if 'error' in period:
                continue
            
            # Check required fields
            assert 'bars_ready_for_decision' in period
            assert 'setup_pass_rate' in period
            assert 'block_reasons' in period
            assert 'pattern_observed' in period
            assert 'primary_setup_issue' in period
            assert 'sample_blocked_setups' in period
    
    def test_cross_period_summary_exists(self, config, temp_output_dir):
        """Test 4: cross_period_summary exists and is complete"""
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        
        run_setup_coverage_multi(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_coverage_multi.json'
        with open(json_path) as f:
            report = json.load(f)
        
        summary = report['cross_period_summary']
        
        # Check required fields
        assert 'average_setup_pass_rate' in summary
        assert 'bull_behavior' in summary
        assert 'bear_behavior' in summary
        assert 'range_behavior' in summary
        assert 'global_interpretation' in summary
    
    def test_handles_zero_patterns(self, config, temp_output_dir):
        """Test 5: No crash if period has zero patterns"""
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        
        # Should complete without error even if all periods have 0 patterns
        result = run_setup_coverage_multi(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        assert result['status'] == 'completed'
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_coverage_multi.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should still produce valid report
        assert report['cross_period_summary']['global_interpretation'] is not None
    
    def test_handles_zero_pass_rate(self, config, temp_output_dir):
        """Test 6: No crash if period has zero pass rate"""
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        
        run_setup_coverage_multi(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_coverage_multi.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should handle 0% pass rate gracefully
        for period in report['periods'].values():
            if 'setup_pass_rate' in period:
                assert isinstance(period['setup_pass_rate'], (int, float))
                assert period['setup_pass_rate'] >= 0
    
    def test_verdict_is_valid(self, config, temp_output_dir):
        """Test verdict is one of the 4 expected cases"""
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        
        run_setup_coverage_multi(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_coverage_multi.json'
        with open(json_path) as f:
            report = json.load(f)
        
        verdict = report['cross_period_summary']['global_interpretation']
        
        # Should be one of the 4 expected verdicts
        valid_verdicts = [
            "Detector silence is regime-dependent and strongest in range markets",  # Cas A
            "Detector is globally too silent across all tested regimes",  # Cas B
            "Patterns exist across periods, but alignment is the dominant blocker",  # Cas C
            "Setup behavior is mixed and needs deeper decomposition"  # Cas D
        ]
        
        assert verdict in valid_verdicts
    
    def test_custom_dates_work(self, config, temp_output_dir):
        """Test custom date list works"""
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        
        # Test with custom dates
        custom_dates = ['2023-12-15', '2023-03-15']
        
        result = run_setup_coverage_multi(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            dates=custom_dates
        )
        
        assert result['status'] == 'completed'
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_coverage_multi.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should have 2 periods
        assert len(report['periods']) == 2
        assert '2023-12-15' in report['periods']
        assert '2023-03-15' in report['periods']
    
    def test_sample_setups_limited(self, config, temp_output_dir):
        """Test that sample setups are limited to 5 per period"""
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        
        run_setup_coverage_multi(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_coverage_multi.json'
        with open(json_path) as f:
            report = json.load(f)
        
        for period in report['periods'].values():
            if 'sample_blocked_setups' in period:
                # Should be limited to 5 samples
                assert len(period['sample_blocked_setups']) <= 5
    
    def test_regime_behaviors_present(self, config, temp_output_dir):
        """Test that regime behaviors are categorized"""
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        
        run_setup_coverage_multi(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_coverage_multi.json'
        with open(json_path) as f:
            report = json.load(f)
        
        summary = report['cross_period_summary']
        
        # Check behaviors are strings
        assert isinstance(summary['bull_behavior'], str)
        assert isinstance(summary['bear_behavior'], str)
        assert isinstance(summary['range_behavior'], str)
        
        # Should not be empty
        assert len(summary['bull_behavior']) > 0
        assert len(summary['bear_behavior']) > 0
        assert len(summary['range_behavior']) > 0
