"""
Test Fractal Pipeline End-to-End
"""

import sys
sys.path.insert(0, '.')

import pytest
import json
from pathlib import Path
import tempfile
import yaml


class TestFractalPipeline:
    """Test suite for complete fractal pipeline"""
    
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
    
    def test_fractal_check_runs(self, config, temp_output_dir):
        """Test that fractal_check mode runs without errors"""
        from scripts.fractal_check import run_fractal_check
        
        # Run with specific date
        result = run_fractal_check(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        # Should complete
        assert result['status'] == 'completed'
        assert 'signals' in result
    
    def test_fractal_check_produces_json(self, config, temp_output_dir):
        """Test that fractal_check produces valid JSON"""
        from scripts.fractal_check import run_fractal_check
        
        result = run_fractal_check(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        # Check JSON file exists
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_check.json'
        assert json_path.exists()
        
        # Load and validate JSON
        with open(json_path) as f:
            report = json.load(f)
        
        assert 'pair' in report
        assert 'signals_evaluated' in report
        assert 'diagnostics' in report
        assert 'active_tfs' in report
    
    def test_fractal_check_diagnostics(self, config, temp_output_dir):
        """Test that diagnostics are present with new schema"""
        from scripts.fractal_check import run_fractal_check
        
        result = run_fractal_check(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_check.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Check new structured diagnostics
        assert 'diagnostics' in report
        diag = report['diagnostics']
        
        # Check readiness_blocks structure
        assert 'readiness_blocks' in diag
        readiness = diag['readiness_blocks']
        assert 'context' in readiness
        assert 'regime' in readiness
        assert 'setup' in readiness
        
        # Check logic_blocks structure
        assert 'logic_blocks' in diag
        logic = diag['logic_blocks']
        assert 'context' in logic
        assert 'regime' in logic
        assert 'setup' in logic
        assert 'risk' in logic
        
        # Check actions structure
        assert 'actions' in diag
        actions = diag['actions']
        assert 'approved_buy' in actions
        assert 'approved_sell' in actions
        assert 'wait_neutral' in actions
        
        # Check top-level counters
        assert 'bars_ready_for_decision' in report
        assert 'bars_skipped_due_to_readiness' in report
    
    def test_no_5m_usage(self, config, temp_output_dir):
        """Test that 5m is not used (deferred)"""
        from scripts.fractal_check import run_fractal_check
        
        result = run_fractal_check(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_check.json'
        with open(json_path) as f:
            report = json.load(f)
        
        active_tfs = report['active_tfs']
        
        # Entry should be marked as DEFERRED
        assert active_tfs['entry'] == 'DEFERRED'
        
        # Context, regime, setup should be active
        assert active_tfs['context'] in ['1d', '4h']
        assert active_tfs['regime'] in ['1h', '4h']
        assert active_tfs['setup'] == '15m'
    
    def test_sample_decisions_included(self, config, temp_output_dir):
        """Test that sample decisions are included in output"""
        from scripts.fractal_check import run_fractal_check
        
        result = run_fractal_check(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_check.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should have sample decisions
        assert 'sample_decisions' in report
        assert len(report['sample_decisions']) > 0
        
        # Each decision should have required fields
        for decision in report['sample_decisions']:
            assert 'action' in decision
            assert 'components' in decision
            assert 'blocked_by' in decision


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
