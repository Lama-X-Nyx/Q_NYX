"""
Tests for Aggregated Reporting

Run:
    pytest tests/test_reporting.py -v
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.reporting import (
    _generate_final_verdict,
    _generate_markdown_report,
    generate_summary_report
)


class TestVerdictGeneration:
    """Test verdict logic"""
    
    def test_verdict_with_no_data(self):
        """Should handle missing data gracefully"""
        summary = {
            'benchmarks': None,
            'walk_forward': None,
            'out_of_sample': None
        }
        
        verdict = _generate_final_verdict(summary)
        
        assert verdict['questions']['beats_benchmarks']['answer'] == 'UNKNOWN'
        assert verdict['questions']['holds_oos']['answer'] == 'UNKNOWN'
        assert verdict['overall_status'] == 'INCOMPLETE'
    
    def test_verdict_oos_stable(self):
        """OOS STABLE should give PROMISING verdict"""
        summary = {
            'benchmarks': {'results': {}},
            'walk_forward': None,
            'out_of_sample': {
                'verdict': {
                    'status': 'STABLE',
                    'reason': 'Degradation within range'
                }
            }
        }
        
        verdict = _generate_final_verdict(summary)
        
        assert verdict['questions']['holds_oos']['answer'] == 'YES'
        assert verdict['questions']['holds_oos']['status'] == 'STABLE'
        assert verdict['overall_status'] == 'PROMISING'
    
    def test_verdict_oos_failed(self):
        """OOS FAILED should give CONCERNING verdict"""
        summary = {
            'benchmarks': None,
            'walk_forward': None,
            'out_of_sample': {
                'verdict': {
                    'status': 'FAILED',
                    'reason': 'Performance degraded >50%'
                }
            }
        }
        
        verdict = _generate_final_verdict(summary)
        
        assert verdict['questions']['holds_oos']['answer'] == 'NO'
        assert verdict['overall_status'] == 'CONCERNING'
    
    def test_pending_questions_sprint_3(self):
        """Sprint 3 questions should be PENDING"""
        summary = {
            'benchmarks': None,
            'walk_forward': None,
            'out_of_sample': None
        }
        
        verdict = _generate_final_verdict(summary)
        
        assert verdict['questions']['survives_monte_carlo']['answer'] == 'PENDING'
        assert verdict['questions']['parameter_sensitivity']['answer'] == 'PENDING'
        assert verdict['questions']['regime_edge']['answer'] == 'PENDING'


class TestMarkdownGeneration:
    """Test markdown report generation"""
    
    def test_markdown_contains_key_sections(self):
        """Markdown should have all key sections"""
        summary = {
            'pair': 'BTCUSDT',
            'final_verdict': {
                'overall_status': 'PROMISING',
                'recommendation': 'Continue to Sprint 3',
                'questions': {
                    'beats_benchmarks': {'answer': 'YES', 'detail': 'NYX outperforms'},
                    'holds_oos': {'answer': 'YES', 'detail': 'Stable'},
                    'survives_monte_carlo': {'answer': 'PENDING', 'detail': 'Sprint 3'},
                    'parameter_sensitivity': {'answer': 'PENDING', 'detail': 'Sprint 3'},
                    'regime_edge': {'answer': 'PENDING', 'detail': 'Sprint 3'},
                    'stop_criteria': {'answer': 'PENDING', 'detail': 'Sprint 4'}
                }
            },
            'benchmarks': {'results': {}},
            'walk_forward': None,
            'out_of_sample': {
                'verdict': {'status': 'STABLE', 'reason': 'Within range'}
            }
        }
        
        md = _generate_markdown_report(summary)
        
        # Check key sections present
        assert '# NYX Phase 1 Validation Summary - BTCUSDT' in md
        assert 'Overall Status: PROMISING' in md
        assert 'Does NYX beat simple benchmarks?' in md
        assert 'Does NYX hold in out-of-sample?' in md
        assert 'Does NYX survive Monte Carlo?' in md


class TestSummaryReportIntegration:
    """Test full summary report generation"""
    
    def test_summary_report_creates_files(self, tmp_path):
        """Summary should create JSON and Markdown files"""
        
        # Create mock validation files
        val_dir = tmp_path / 'validation'
        
        benchmarks_dir = val_dir / 'benchmarks'
        benchmarks_dir.mkdir(parents=True)
        
        benchmark_data = {
            'pair': 'BTCUSDT',
            'results': {
                'buy_hold': {'metrics': {'total_return': 50.0}},
                'sma_crossover': {'metrics': {'total_return': -20.0}}
            }
        }
        
        with open(benchmarks_dir / 'BTCUSDT_benchmarks.json', 'w') as f:
            json.dump(benchmark_data, f)
        
        # Run summary generation
        summary = generate_summary_report(pair='BTCUSDT', validation_dir=str(val_dir))
        
        # Check outputs created
        summary_json = val_dir / 'summary' / 'BTCUSDT_summary.json'
        summary_md = val_dir / 'summary' / 'BTCUSDT_summary.md'
        
        assert summary_json.exists()
        assert summary_md.exists()
        
        # Check content
        assert summary['pair'] == 'BTCUSDT'
        assert summary['benchmarks'] is not None
        assert 'final_verdict' in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
