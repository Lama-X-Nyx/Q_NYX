"""
Test SMC Autopsy

Validates that SMC detector autopsy works correctly.
"""

import sys
sys.path.insert(0, '.')

import pytest
import json
from pathlib import Path
import tempfile
import yaml


class TestSMCAutopsy:
    """Test suite for SMC autopsy"""
    
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
        from scripts.smc_autopsy import run_smc_autopsy
        
        result = run_smc_autopsy(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15',
            sensitivity=False
        )
        
        # Check JSON file exists
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_smc_autopsy.json'
        assert json_path.exists()
        
        # Load and validate JSON
        with open(json_path) as f:
            report = json.load(f)
        
        # Check required fields
        assert 'pair' in report
        assert 'reference_date' in report
        assert 'bars_analyzed' in report
        assert 'code_path' in report
        assert 'fvg_audit' in report
        assert 'ob_audit' in report
        assert 'verdict' in report
    
    def test_detector_called_present(self, config, temp_output_dir):
        """Test 2: detector_called field is present"""
        from scripts.smc_autopsy import run_smc_autopsy
        
        run_smc_autopsy(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15',
            sensitivity=False
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_smc_autopsy.json'
        with open(json_path) as f:
            report = json.load(f)
        
        code_path = report['code_path']
        
        # Check detector_called
        assert 'detector_called' in code_path
        assert isinstance(code_path['detector_called'], int)
        assert code_path['detector_called'] > 0
    
    def test_function_call_counters_exist(self, config, temp_output_dir):
        """Test 3: Function call counters exist"""
        from scripts.smc_autopsy import run_smc_autopsy
        
        run_smc_autopsy(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15',
            sensitivity=False
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_smc_autopsy.json'
        with open(json_path) as f:
            report = json.load(f)
        
        code_path = report['code_path']
        
        # Check counters
        assert 'fvg_function_called' in code_path
        assert 'ob_function_called' in code_path
        assert isinstance(code_path['fvg_function_called'], int)
        assert isinstance(code_path['ob_function_called'], int)
    
    def test_fvg_ob_audits_exist(self, config, temp_output_dir):
        """Test 4: FVG/OB audit fields exist"""
        from scripts.smc_autopsy import run_smc_autopsy
        
        run_smc_autopsy(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15',
            sensitivity=False
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_smc_autopsy.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Check FVG audit
        fvg = report['fvg_audit']
        assert 'candidates_seen' in fvg
        assert 'detected' in fvg
        assert 'failed_gap_threshold' in fvg
        assert 'failed_structure_rules' in fvg
        
        # Check OB audit
        ob = report['ob_audit']
        assert 'candidates_seen' in ob
        assert 'detected' in ob
        assert 'failed_range_threshold' in ob
        assert 'failed_followthrough' in ob
    
    def test_verdict_present(self, config, temp_output_dir):
        """Test 5: Verdict is present"""
        from scripts.smc_autopsy import run_smc_autopsy
        
        run_smc_autopsy(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15',
            sensitivity=False
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_smc_autopsy.json'
        with open(json_path) as f:
            report = json.load(f)
        
        verdict = report['verdict']
        
        # Check verdict structure
        assert 'category' in verdict
        assert 'summary' in verdict
        
        # Category should be one of A/B/C/D/OK
        assert verdict['category'] in ['A', 'B', 'C', 'D', 'OK']
    
    def test_handles_zero_patterns(self, config, temp_output_dir):
        """Test 6: No crash if zero patterns"""
        from scripts.smc_autopsy import run_smc_autopsy
        
        # Should complete without error even if 0 patterns
        result = run_smc_autopsy(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15',
            sensitivity=False
        )
        
        assert result['status'] == 'completed'
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_smc_autopsy.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should still produce valid report
        assert report['verdict']['category'] is not None
    
    def test_sensitivity_mode_works(self, config, temp_output_dir):
        """Test 7: Sensitivity mode produces multiple scenarios"""
        from scripts.smc_autopsy import run_smc_autopsy
        
        run_smc_autopsy(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15',
            sensitivity=True
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_smc_autopsy.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Check sensitivity analysis exists
        assert 'sensitivity_analysis' in report
        
        sensitivity = report['sensitivity_analysis']
        assert 'test_configurations' in sensitivity
        assert 'summary' in sensitivity
        
        # Should have multiple test configs
        configs = sensitivity['test_configurations']
        assert len(configs) >= 2  # At least baseline + 1 variant
        
        # Each config should have required fields
        for test_config in configs:
            assert 'name' in test_config
            assert 'configuration' in test_config
            assert 'fvg_detected' in test_config
            assert 'ob_detected' in test_config
            assert 'total_patterns' in test_config
            assert 'detection_rate' in test_config
    
    def test_no_persistent_config_mutation(self, config, temp_output_dir):
        """Test 8: No persistent config mutation"""
        from scripts.smc_autopsy import run_smc_autopsy
        
        # Get original threshold
        original_ob_threshold = config.get('smc_detector', {}).get('ob_range_threshold', 0.015)
        
        # Run with sensitivity
        run_smc_autopsy(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15',
            sensitivity=True
        )
        
        # Check config wasn't mutated
        current_ob_threshold = config.get('smc_detector', {}).get('ob_range_threshold', 0.015)
        assert current_ob_threshold == original_ob_threshold
    
    def test_configuration_recorded(self, config, temp_output_dir):
        """Test configuration is recorded in output"""
        from scripts.smc_autopsy import run_smc_autopsy
        
        run_smc_autopsy(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15',
            sensitivity=False
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_smc_autopsy.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Check configuration recorded
        assert 'configuration' in report
        cfg = report['configuration']
        
        assert 'ob_range_threshold' in cfg
        assert 'fvg_min_gap' in cfg
        assert 'liquidity_lookback' in cfg
    
    def test_candidates_vs_detected_coherent(self, config, temp_output_dir):
        """Test candidates vs detected counts are coherent"""
        from scripts.smc_autopsy import run_smc_autopsy
        
        run_smc_autopsy(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15',
            sensitivity=False
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_smc_autopsy.json'
        with open(json_path) as f:
            report = json.load(f)
        
        fvg = report['fvg_audit']
        ob = report['ob_audit']
        
        # Detected should never exceed candidates
        assert fvg['detected'] <= fvg['candidates_seen']
        assert ob['detected'] <= ob['candidates_seen']
        
        # All counts should be non-negative
        assert fvg['candidates_seen'] >= 0
        assert fvg['detected'] >= 0
        assert ob['candidates_seen'] >= 0
        assert ob['detected'] >= 0
