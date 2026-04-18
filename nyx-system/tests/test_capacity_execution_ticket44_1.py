"""
TDD Tests — Ticket 44.1 — Execution Reality Stress Evaluator.

Extends T44 with execution friction dimensions: offset, timeout,
fill degradation, fees, combined scenarios.
"""
from __future__ import annotations

import json

import pytest


class TestExecutionStressConfig:

    def test_config_creation(self):
        from src.live.evaluators.capacity_execution import ExecutionStressConfig
        cfg = ExecutionStressConfig(
            assets=['BTCUSDT'],
            capital_ladder=[10_000, 1_000_000],
            offset_ladder_bps=[5.0, 10.0, 20.0],
        )
        assert len(cfg.offset_ladder_bps) == 3

    def test_config_defaults(self):
        from src.live.evaluators.capacity_execution import ExecutionStressConfig
        cfg = ExecutionStressConfig(assets=['BTCUSDT'])
        assert len(cfg.capital_ladder) > 0
        assert len(cfg.offset_ladder_bps) > 0
        assert len(cfg.fill_degradation_ladder) > 0
        assert len(cfg.fee_multiplier_ladder) > 0

    def test_config_serializable(self):
        from src.live.evaluators.capacity_execution import ExecutionStressConfig
        cfg = ExecutionStressConfig(assets=['BTCUSDT'])
        d = cfg.to_dict()
        s = json.dumps(d)
        assert 'offset_ladder_bps' in s

    def test_config_validation_empty_assets(self):
        from src.live.evaluators.capacity_execution import ExecutionStressConfig
        with pytest.raises(ValueError):
            ExecutionStressConfig(assets=[])


class TestExecutionScenarioGenerator:

    def test_generate_execution_scenarios(self):
        from src.live.evaluators.capacity_execution import (
            ExecutionStressConfig, generate_execution_scenarios,
        )
        cfg = ExecutionStressConfig(
            assets=['BTCUSDT'],
            capital_ladder=[10_000],
            offset_ladder_bps=[5.0, 10.0],
            fill_degradation_ladder=[1.0, 0.75],
            fee_multiplier_ladder=[1.0],
        )
        scenarios = generate_execution_scenarios(cfg)
        assert len(scenarios) == 4  # 1 capital × 2 offsets × 2 fills × 1 fee

    def test_scenarios_have_execution_params(self):
        from src.live.evaluators.capacity_execution import (
            ExecutionStressConfig, generate_execution_scenarios,
        )
        cfg = ExecutionStressConfig(assets=['BTCUSDT'], capital_ladder=[10_000])
        scenarios = generate_execution_scenarios(cfg)
        for s in scenarios:
            assert 'offset_bps' in s
            assert 'fill_degradation' in s
            assert 'fee_multiplier' in s
            assert 'scenario_id' in s

    def test_scenario_ids_unique(self):
        from src.live.evaluators.capacity_execution import (
            ExecutionStressConfig, generate_execution_scenarios,
        )
        cfg = ExecutionStressConfig(
            assets=['BTCUSDT'],
            capital_ladder=[10_000, 100_000],
            offset_ladder_bps=[5.0, 10.0],
        )
        scenarios = generate_execution_scenarios(cfg)
        ids = [s['scenario_id'] for s in scenarios]
        assert len(set(ids)) == len(ids)


class TestCompositeEdgeRetention:

    def test_composite_retention(self):
        from src.live.evaluators.capacity_execution import compute_composite_retention
        result = compute_composite_retention(
            sharpe_retention=0.8,
            miss_rate_penalty=0.1,
            drawdown_penalty=0.05,
            profit_factor_retention=0.9,
        )
        assert 0.0 <= result <= 1.0

    def test_worse_with_penalties(self):
        from src.live.evaluators.capacity_execution import compute_composite_retention
        clean = compute_composite_retention(
            sharpe_retention=0.9, miss_rate_penalty=0.0,
            drawdown_penalty=0.0, profit_factor_retention=0.95,
        )
        stressed = compute_composite_retention(
            sharpe_retention=0.5, miss_rate_penalty=0.3,
            drawdown_penalty=0.2, profit_factor_retention=0.5,
        )
        assert clean > stressed


class TestExecutionBreakpoint:

    def test_detect_execution_breakpoint(self):
        from src.live.evaluators.capacity_execution import detect_execution_breakpoint
        results = [
            {'scenario_id': 's1', 'composite_retention': 0.9, 'miss_rate': 0.2},
            {'scenario_id': 's2', 'composite_retention': 0.6, 'miss_rate': 0.35},
            {'scenario_id': 's3', 'composite_retention': 0.3, 'miss_rate': 0.6},
        ]
        bp = detect_execution_breakpoint(results, retention_threshold=0.5)
        assert bp is not None
        assert bp['scenario_id'] == 's3'

    def test_no_breakpoint(self):
        from src.live.evaluators.capacity_execution import detect_execution_breakpoint
        results = [
            {'scenario_id': 's1', 'composite_retention': 0.9, 'miss_rate': 0.2},
        ]
        bp = detect_execution_breakpoint(results, retention_threshold=0.5)
        assert bp is None


class TestExecutionFlags:

    def test_offset_sensitive_flag(self):
        from src.live.evaluators.capacity_execution import generate_execution_flags
        flags = generate_execution_flags(
            offset_sensitivity=0.7, timeout_sensitivity=0.2,
            fill_fragility=0.3, fee_fragility=0.1,
            has_breakpoint=True,
        )
        assert 'offset_sensitive' in flags
        assert 'execution_breakpoint_reached' in flags

    def test_resilient_no_flags(self):
        from src.live.evaluators.capacity_execution import generate_execution_flags
        flags = generate_execution_flags(
            offset_sensitivity=0.1, timeout_sensitivity=0.1,
            fill_fragility=0.1, fee_fragility=0.1,
            has_breakpoint=False,
        )
        assert 'execution_resilient' in flags
        assert 'offset_sensitive' not in flags


class TestCapacityExecutionEvaluatorIntegration:

    def test_evaluator_name(self):
        from src.live.evaluators.capacity_execution import CapacityExecutionEvaluator
        ev = CapacityExecutionEvaluator()
        assert ev.evaluator_name == 'capacity_execution'

    def test_evaluator_produces_output(self):
        from src.live.evaluators.capacity_execution import CapacityExecutionEvaluator
        from src.live.oos_engine import OOSResult
        data = {
            'run_id': 'exec-test-001',
            'config': {
                'assets': ['BTCUSDT'], 'start_date': '2023-01-01',
                'end_date': '2023-12-31', 'initial_capital': 10_000.0,
                'mode': 'realistic', 'train_end': '2022-12-31',
            },
            'per_asset': [{
                'symbol': 'BTCUSDT', 'capital': 10_000.0, 'mode': 'realistic',
                'performance': {'n_trades': 39, 'win_rate': 0.769, 'sharpe': 4.2,
                                'total_pnl': 1150.0, 'max_drawdown_pct': 0.93},
                'execution': {'idealized_signals': 63, 'placed': 59, 'filled': 39,
                              'timed_out': 20, 'fill_rate': 0.66, 'miss_rate': 0.34,
                              'skipped_quality': 0, 'blocked_risk': 4},
            }],
            'portfolio': {'total_pnl': 1150.0, 'total_trades': 39},
        }
        ev = CapacityExecutionEvaluator()
        output = ev.evaluate(OOSResult(data))
        assert 'per_asset' in output
        assert 'BTCUSDT' in output['per_asset']
        btc = output['per_asset']['BTCUSDT']
        assert 'scenario_grid' in btc
        assert 'execution_breakpoint' in btc
        assert 'execution_flags' in btc
        assert 'deployability' in btc

    def test_evaluator_deterministic(self):
        from src.live.evaluators.capacity_execution import CapacityExecutionEvaluator
        from src.live.oos_engine import OOSResult
        data = {
            'run_id': 'exec-det',
            'config': {'assets': ['BTCUSDT'], 'start_date': '2023-01-01',
                       'end_date': '2023-12-31', 'mode': 'realistic',
                       'initial_capital': 10_000.0, 'train_end': '2022-12-31'},
            'per_asset': [{
                'symbol': 'BTCUSDT', 'capital': 10_000.0, 'mode': 'realistic',
                'performance': {'n_trades': 39, 'win_rate': 0.769, 'sharpe': 4.2,
                                'total_pnl': 1150.0, 'max_drawdown_pct': 0.93},
                'execution': {'idealized_signals': 63, 'placed': 59, 'filled': 39,
                              'timed_out': 20, 'fill_rate': 0.66, 'miss_rate': 0.34,
                              'skipped_quality': 0, 'blocked_risk': 4},
            }],
            'portfolio': {'total_pnl': 1150.0, 'total_trades': 39},
        }
        ev = CapacityExecutionEvaluator()
        r1 = ev.evaluate(OOSResult(data))
        r2 = ev.evaluate(OOSResult(data))
        assert json.dumps(r1, default=str) == json.dumps(r2, default=str)
