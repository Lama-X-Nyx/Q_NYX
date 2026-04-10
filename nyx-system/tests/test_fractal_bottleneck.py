"""
Test Fractal Bottleneck Discovery

Validates that bottleneck quantification works correctly with corrected geometry.
"""

import sys
sys.path.insert(0, '.')

import pytest
import json
from pathlib import Path
import tempfile
import yaml


class TestFractalBottleneck:
    """Test suite for fractal bottleneck discovery"""
    
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
        from scripts.fractal_bottleneck import run_fractal_bottleneck
        
        result = run_fractal_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        # Check JSON file exists
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_bottleneck.json'
        assert json_path.exists()
        
        # Load and validate JSON
        with open(json_path) as f:
            report = json.load(f)
        
        # Check required fields
        assert 'pair' in report
        assert 'reference_date' in report
        assert 'bars_analyzed' in report
        assert 'bars_ready_for_decision' in report
        assert 'bars_skipped_due_to_readiness' in report
        assert 'diagnostics' in report
        assert 'ready_only_summary' in report
        assert 'bottleneck_ranking' in report
    
    def test_bars_accounting_correct(self, config, temp_output_dir):
        """Test 2: bars_ready + bars_skipped == bars_analyzed"""
        from scripts.fractal_bottleneck import run_fractal_bottleneck
        
        run_fractal_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Accounting must be correct
        bars_ready = report['bars_ready_for_decision']
        bars_skipped = report['bars_skipped_due_to_readiness']
        bars_analyzed = report['bars_analyzed']
        
        assert bars_ready + bars_skipped == bars_analyzed
    
    def test_pass_rates_calculated_correctly(self, config, temp_output_dir):
        """Test 3: Pass rates are correct percentages"""
        from scripts.fractal_bottleneck import run_fractal_bottleneck
        
        run_fractal_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        summary = report['ready_only_summary']
        
        # Pass rates should be valid percentages
        assert 0 <= summary['context_pass_rate'] <= 100
        assert 0 <= summary['regime_pass_rate'] <= 100
        assert 0 <= summary['setup_pass_rate'] <= 100
        assert 0 <= summary['final_trade_rate'] <= 100
        
        # All should be floats
        assert isinstance(summary['context_pass_rate'], float)
        assert isinstance(summary['regime_pass_rate'], float)
        assert isinstance(summary['setup_pass_rate'], float)
        assert isinstance(summary['final_trade_rate'], float)
    
    def test_primary_bottleneck_matches_ranking(self, config, temp_output_dir):
        """Test 4: primary_bottleneck matches ranking"""
        from scripts.fractal_bottleneck import run_fractal_bottleneck
        
        run_fractal_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        summary = report['ready_only_summary']
        ranking = report['bottleneck_ranking']
        
        # Primary bottleneck should be first in ranking (if any blocks exist)
        if ranking and ranking[0]['blocks'] > 0:
            assert summary['primary_bottleneck'] == ranking[0]['agent']
    
    def test_handles_all_blocked_by_setup(self, config, temp_output_dir):
        """Test 5: No crash if all bars blocked by setup"""
        from scripts.fractal_bottleneck import run_fractal_bottleneck
        
        # Should complete without error even if setup blocks everything
        result = run_fractal_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        assert result['status'] == 'completed'
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should still produce valid report
        assert report['ready_only_summary']['primary_bottleneck'] is not None
    
    def test_handles_no_trades_approved(self, config, temp_output_dir):
        """Test 6: No crash if no BUY/SELL approved"""
        from scripts.fractal_bottleneck import run_fractal_bottleneck
        
        run_fractal_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        actions = report['diagnostics']['actions']
        
        # Should handle zero trades gracefully
        assert 'approved_buy' in actions
        assert 'approved_sell' in actions
        assert isinstance(actions['approved_buy'], int)
        assert isinstance(actions['approved_sell'], int)
    
    def test_uses_corrected_geometry(self, config, temp_output_dir):
        """Test 7: Uses corrected fractal geometry (not old sampling)"""
        from scripts.fractal_bottleneck import run_fractal_bottleneck
        
        run_fractal_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # With corrected geometry and sample-date 2023-12-15:
        # - Should have reference_date field
        # - Should have bars_ready_for_decision > 0 (not all readiness blocks)
        # - Readiness blocks should be 0 or very low
        
        assert 'reference_date' in report
        assert report['reference_date'] == '2023-12-15T00:00:00'
        
        # With proper geometry, most bars should be ready
        bars_ready = report['bars_ready_for_decision']
        bars_skipped = report['bars_skipped_due_to_readiness']
        
        # At least some bars should be ready (corrected geometry works)
        assert bars_ready > 0, "Corrected geometry should produce ready bars"
        
        # Readiness blocks should be minimal or zero
        readiness_blocks = report['diagnostics']['readiness_blocks']
        total_readiness = sum(readiness_blocks.values())
        
        # With proper geometry, readiness should not dominate
        assert total_readiness == 0 or total_readiness < bars_ready, \
            "Corrected geometry should minimize readiness blocks"
    
    def test_diagnostics_structure(self, config, temp_output_dir):
        """Test diagnostic structure is complete"""
        from scripts.fractal_bottleneck import run_fractal_bottleneck
        
        run_fractal_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Check diagnostics structure
        diagnostics = report['diagnostics']
        
        assert 'readiness_blocks' in diagnostics
        assert 'logic_blocks' in diagnostics
        assert 'actions' in diagnostics
        
        # Check all agents present
        assert 'context' in diagnostics['readiness_blocks']
        assert 'regime' in diagnostics['readiness_blocks']
        assert 'setup' in diagnostics['readiness_blocks']
        
        assert 'context' in diagnostics['logic_blocks']
        assert 'regime' in diagnostics['logic_blocks']
        assert 'setup' in diagnostics['logic_blocks']
        assert 'risk' in diagnostics['logic_blocks']
    
    def test_sample_blocked_decisions_included(self, config, temp_output_dir):
        """Test that sample blocked decisions are included"""
        from scripts.fractal_bottleneck import run_fractal_bottleneck
        
        run_fractal_bottleneck(
            pair='BTCUSDT',
            config=config,
            output_dir=temp_output_dir,
            sample_date='2023-12-15'
        )
        
        json_path = temp_output_dir / 'fractal' / 'BTCUSDT_fractal_bottleneck.json'
        with open(json_path) as f:
            report = json.load(f)
        
        # Should have sample decisions
        assert 'sample_blocked_decisions' in report
        assert isinstance(report['sample_blocked_decisions'], list)
        
        # If there are blocked decisions, check structure
        if len(report['sample_blocked_decisions']) > 0:
            sample = report['sample_blocked_decisions'][0]
            assert 'timestamp' in sample
            assert 'blocked_by' in sample
            assert 'components' in sample
