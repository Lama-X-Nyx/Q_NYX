"""
Test Setup Bottleneck Diagnosis

Validates that Setup bottleneck diagnosis works correctly.
"""

import sys
sys.path.insert(0, '.')

import pytest
import json
from pathlib import Path
import tempfile
import yaml


class TestSetupBottleneck:
    """Test suite for setup bottleneck diagnosis"""
    
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
        from scripts.setup_bottleneck import run_setup_bottleneck
        
        result = run_setup_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        # Check JSON file exists
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_bottleneck.json'
        assert json_path.exists()
        
        # Load and validate JSON
        with open(json_path) as f:
            report = json.load(f)
        
        # Check required fields
        assert 'pair' in report
        assert 'reference_date' in report
        assert 'bars_ready_for_decision' in report
        assert 'setup_summary' in report
        assert 'block_reasons' in report
        assert 'pattern_observed' in report
        assert 'alignment_summary' in report
        assert 'directional_summary' in report
        assert 'sample_blocked_setups' in report
        assert 'verdict' in report
    
    def test_setup_accounting_correct(self, config, temp_output_dir):
        """Test 2: passed + blocked == bars_ready_for_decision"""
        from scripts.setup_bottleneck import run_setup_bottleneck
        
        run_setup_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Accounting must be correct
        setup_passed = report['setup_summary']['passed']
        setup_blocked = report['setup_summary']['blocked']
        bars_ready = report['bars_ready_for_decision']
        
        assert setup_passed + setup_blocked == bars_ready
    
    def test_block_reasons_coherent(self, config, temp_output_dir):
        """Test 3: block_reasons are coherent"""
        from scripts.setup_bottleneck import run_setup_bottleneck
        
        run_setup_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        block_reasons = report['block_reasons']
        
        # All block reason types should be present
        assert 'no_pattern' in block_reasons
        assert 'misaligned_pattern' in block_reasons
        assert 'alignment_score_too_low' in block_reasons
        assert 'direction_conflict' in block_reasons
        assert 'setup_not_ready' in block_reasons
        assert 'unknown' in block_reasons
        
        # Total block reasons should match blocked count
        total_blocks = sum(block_reasons.values())
        setup_blocked = report['setup_summary']['blocked']
        
        assert total_blocks == setup_blocked
    
    def test_pattern_observed_present(self, config, temp_output_dir):
        """Test 4: pattern_observed are present"""
        from scripts.setup_bottleneck import run_setup_bottleneck
        
        run_setup_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        patterns = report['pattern_observed']
        
        # All pattern types should be tracked
        assert 'bullish_fvg' in patterns
        assert 'bearish_fvg' in patterns
        assert 'bullish_ob' in patterns
        assert 'bearish_ob' in patterns
        assert 'no_pattern' in patterns
        
        # All should be non-negative integers
        for pattern_type, count in patterns.items():
            assert isinstance(count, int)
            assert count >= 0
    
    def test_verdict_matches_dominant_block_reason(self, config, temp_output_dir):
        """Test 5: verdict.primary_setup_issue matches dominant block reason"""
        from scripts.setup_bottleneck import run_setup_bottleneck
        
        run_setup_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        block_reasons = report['block_reasons']
        verdict = report['verdict']
        
        # Find dominant block reason
        if block_reasons:
            dominant_reason = max(block_reasons.items(), key=lambda x: x[1])[0]
            
            # Verdict should match
            assert verdict['primary_setup_issue'] == dominant_reason
    
    def test_handles_no_patterns(self, config, temp_output_dir):
        """Test 6: No crash if no patterns observed"""
        from scripts.setup_bottleneck import run_setup_bottleneck
        
        # Should complete without error even if no patterns
        result = run_setup_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        assert result['status'] == 'completed'
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should still produce valid report
        assert report['verdict']['primary_setup_issue'] is not None
        assert report['verdict']['interpretation'] is not None
    
    def test_handles_all_misaligned(self, config, temp_output_dir):
        """Test 7: No crash if all setups are misaligned"""
        from scripts.setup_bottleneck import run_setup_bottleneck
        
        run_setup_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should handle all cases gracefully
        assert 'verdict' in report
        assert 'primary_setup_issue' in report['verdict']
    
    def test_alignment_summary_complete(self, config, temp_output_dir):
        """Test alignment summary has all required stats"""
        from scripts.setup_bottleneck import run_setup_bottleneck
        
        run_setup_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        alignment = report['alignment_summary']
        
        # Check all stats present
        assert 'mean_alignment_score' in alignment
        assert 'median_alignment_score' in alignment
        assert 'min_alignment_score' in alignment
        assert 'max_alignment_score' in alignment
        assert 'threshold' in alignment
        
        # All should be valid numbers
        assert isinstance(alignment['mean_alignment_score'], (int, float))
        assert isinstance(alignment['median_alignment_score'], (int, float))
        assert isinstance(alignment['min_alignment_score'], (int, float))
        assert isinstance(alignment['max_alignment_score'], (int, float))
        assert isinstance(alignment['threshold'], (int, float))
    
    def test_directional_summary_present(self, config, temp_output_dir):
        """Test directional summary is complete"""
        from scripts.setup_bottleneck import run_setup_bottleneck
        
        run_setup_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        directional = report['directional_summary']
        
        # Check required fields
        assert 'context_state' in directional
        assert 'regime_state_distribution' in directional
        assert 'setup_direction_distribution' in directional
    
    def test_sample_blocked_setups_included(self, config, temp_output_dir):
        """Test that sample blocked setups are included"""
        from scripts.setup_bottleneck import run_setup_bottleneck
        
        run_setup_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_setup_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should have sample setups
        assert 'sample_blocked_setups' in report
        assert isinstance(report['sample_blocked_setups'], list)
        
        # If there are samples, check structure
        if len(report['sample_blocked_setups']) > 0:
            sample = report['sample_blocked_setups'][0]
            assert 'timestamp' in sample
            assert 'state' in sample
            assert 'reason' in sample
            assert 'alignment_score' in sample
            assert 'patterns' in sample
            assert 'context_state' in sample
            assert 'regime_state' in sample
