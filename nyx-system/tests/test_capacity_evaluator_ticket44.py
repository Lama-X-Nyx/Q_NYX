"""
TDD Tests — Ticket 44 — Capacity & Liquidity Stress Evaluator.

Plugs into T42.1 framework. Runs capital ladder + execution stress
scenarios through CanonicalOOSEngine. Classifies deployability.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _make_oos_result_data():
    return {
        'run_id': 'cap-test-001',
        'config': {
            'assets': ['BTCUSDT'], 'start_date': '2023-01-01',
            'end_date': '2023-12-31', 'initial_capital': 10_000.0,
            'mode': 'idealized', 'train_end': '2022-12-31',
        },
        'per_asset': [{
            'symbol': 'BTCUSDT', 'capital': 10_000.0, 'mode': 'idealized',
            'performance': {'n_trades': 63, 'win_rate': 0.65, 'sharpe': 7.0,
                            'total_pnl': 2171.0, 'max_drawdown_pct': 1.5},
        }],
        'portfolio': {'total_pnl': 2171.0, 'total_trades': 63},
    }


class TestCapacityStressConfig:

    def test_config_creation(self):
        from src.live.evaluators.capacity import CapacityStressConfig
        cfg = CapacityStressConfig(
            assets=['BTCUSDT'],
            capital_ladder=[10_000, 100_000, 1_000_000],
        )
        assert cfg.assets == ['BTCUSDT']
        assert len(cfg.capital_ladder) == 3

    def test_config_defaults(self):
        from src.live.evaluators.capacity import CapacityStressConfig
        cfg = CapacityStressConfig(assets=['BTCUSDT'])
        assert len(cfg.capital_ladder) > 0
        assert cfg.mode == 'idealized'

    def test_config_serializable(self):
        from src.live.evaluators.capacity import CapacityStressConfig
        cfg = CapacityStressConfig(
            assets=['BTCUSDT', 'ETHUSDT'],
            capital_ladder=[1000, 10_000, 100_000],
        )
        d = cfg.to_dict()
        s = json.dumps(d)
        assert 'capital_ladder' in s

    def test_config_validation_empty_ladder(self):
        from src.live.evaluators.capacity import CapacityStressConfig
        with pytest.raises(ValueError):
            CapacityStressConfig(assets=['BTCUSDT'], capital_ladder=[])


class TestScenarioGenerator:

    def test_generate_scenarios(self):
        from src.live.evaluators.capacity import CapacityStressConfig, generate_scenarios
        cfg = CapacityStressConfig(
            assets=['BTCUSDT'],
            capital_ladder=[10_000, 100_000, 1_000_000],
        )
        scenarios = generate_scenarios(cfg)
        assert len(scenarios) == 3
        assert scenarios[0]['capital'] == 10_000
        assert scenarios[2]['capital'] == 1_000_000

    def test_scenarios_have_ids(self):
        from src.live.evaluators.capacity import CapacityStressConfig, generate_scenarios
        cfg = CapacityStressConfig(assets=['BTCUSDT'], capital_ladder=[10_000, 100_000])
        scenarios = generate_scenarios(cfg)
        ids = [s['scenario_id'] for s in scenarios]
        assert len(set(ids)) == 2

    def test_multi_asset_scenarios(self):
        from src.live.evaluators.capacity import CapacityStressConfig, generate_scenarios
        cfg = CapacityStressConfig(
            assets=['BTCUSDT', 'ETHUSDT'],
            capital_ladder=[10_000, 100_000],
        )
        scenarios = generate_scenarios(cfg)
        assert len(scenarios) == 2
        assert all(set(s['assets']) == {'BTCUSDT', 'ETHUSDT'} for s in scenarios)


class TestCapacityMetrics:

    def test_compute_edge_retention(self):
        from src.live.evaluators.capacity import compute_edge_retention
        baseline = {'sharpe': 7.0, 'total_pnl': 2171.0}
        stressed = {'sharpe': 5.0, 'total_pnl': 1500.0}
        ratio = compute_edge_retention(baseline, stressed)
        assert 0.0 <= ratio <= 1.5
        assert abs(ratio - 5.0 / 7.0) < 0.01

    def test_edge_retention_zero_baseline(self):
        from src.live.evaluators.capacity import compute_edge_retention
        ratio = compute_edge_retention({'sharpe': 0.0}, {'sharpe': 2.0})
        assert ratio >= 0.0


class TestBreakpointDetection:

    def test_detect_breakpoint(self):
        from src.live.evaluators.capacity import detect_breakpoint
        results = [
            {'capital': 10_000, 'sharpe': 7.0, 'edge_retention': 1.0},
            {'capital': 100_000, 'sharpe': 5.0, 'edge_retention': 0.71},
            {'capital': 1_000_000, 'sharpe': 3.0, 'edge_retention': 0.43},
            {'capital': 10_000_000, 'sharpe': 0.5, 'edge_retention': 0.07},
        ]
        bp = detect_breakpoint(results, sharpe_threshold=1.0, retention_threshold=0.5)
        assert bp is not None
        assert bp['capital'] == 1_000_000

    def test_no_breakpoint(self):
        from src.live.evaluators.capacity import detect_breakpoint
        results = [
            {'capital': 10_000, 'sharpe': 7.0, 'edge_retention': 1.0},
            {'capital': 100_000, 'sharpe': 6.5, 'edge_retention': 0.93},
        ]
        bp = detect_breakpoint(results, sharpe_threshold=1.0, retention_threshold=0.5)
        assert bp is None


class TestDeployabilityClassification:

    def test_classify_core_scalable(self):
        from src.live.evaluators.capacity import classify_deployability
        result = classify_deployability(
            max_capital_before_break=100_000_000,
            baseline_sharpe=7.0,
            edge_retention_at_10M=0.9,
        )
        assert result['class'] == 'core_scalable'

    def test_classify_not_deployable(self):
        from src.live.evaluators.capacity import classify_deployability
        result = classify_deployability(
            max_capital_before_break=50_000,
            baseline_sharpe=1.0,
            edge_retention_at_10M=0.1,
        )
        assert result['class'] in ('opportunistic_small_only', 'not_deployable')

    def test_classify_has_max_capital(self):
        from src.live.evaluators.capacity import classify_deployability
        result = classify_deployability(
            max_capital_before_break=10_000_000,
            baseline_sharpe=4.0,
            edge_retention_at_10M=0.7,
        )
        assert 'max_recommended_capital' in result


class TestCapacityFlags:

    def test_flags_generated(self):
        from src.live.evaluators.capacity import generate_capacity_flags
        flags = generate_capacity_flags(
            breakpoint_capital=100_000,
            baseline_sharpe=7.0,
            edge_retention_at_max=0.3,
            miss_rate_at_max=0.5,
        )
        assert isinstance(flags, list)
        assert 'capital_limited_edge' in flags

    def test_core_scalable_flags(self):
        from src.live.evaluators.capacity import generate_capacity_flags
        flags = generate_capacity_flags(
            breakpoint_capital=None,
            baseline_sharpe=7.0,
            edge_retention_at_max=0.9,
            miss_rate_at_max=0.3,
        )
        assert 'core_scalable' in flags


class TestCapacityEvaluatorIntegration:

    def test_evaluator_has_correct_name(self):
        from src.live.evaluators.capacity import CapacityEvaluator
        ev = CapacityEvaluator()
        assert ev.evaluator_name == 'capacity'

    def test_evaluator_produces_output(self):
        from src.live.evaluators.capacity import CapacityEvaluator
        from src.live.oos_engine import OOSResult
        ev = CapacityEvaluator()
        result = OOSResult(_make_oos_result_data())
        output = ev.evaluate(result)
        assert isinstance(output, dict)
        assert 'per_asset' in output
        assert 'BTCUSDT' in output['per_asset']
        btc = output['per_asset']['BTCUSDT']
        assert 'scenario_results' in btc
        assert 'deployability' in btc
        assert 'flags' in btc

    def test_evaluator_output_serializable(self):
        from src.live.evaluators.capacity import CapacityEvaluator
        from src.live.oos_engine import OOSResult
        ev = CapacityEvaluator()
        output = ev.evaluate(OOSResult(_make_oos_result_data()))
        s = json.dumps(output, default=str)
        assert len(s) > 10

    def test_evaluator_deterministic(self):
        from src.live.evaluators.capacity import CapacityEvaluator
        from src.live.oos_engine import OOSResult
        ev = CapacityEvaluator()
        data = _make_oos_result_data()
        r1 = ev.evaluate(OOSResult(data))
        r2 = ev.evaluate(OOSResult(data))
        assert json.dumps(r1, default=str) == json.dumps(r2, default=str)
