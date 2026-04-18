"""
TDD Tests — Ticket 42.1 — OOS Evaluator Framework.

Pluggable post-OOS analysis layer. Evaluators consume OOSResult,
produce serializable analysis outputs, persist results.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent


def _make_oos_result_data() -> dict:
    return {
        'run_id': 'test-run-001',
        'config': {
            'assets': ['BTCUSDT'], 'start_date': '2023-01-01',
            'end_date': '2023-12-31', 'initial_capital': 10_000.0,
            'mode': 'realistic',
        },
        'per_asset': [{
            'symbol': 'BTCUSDT', 'capital': 10_000.0, 'mode': 'realistic',
            'performance': {
                'n_trades': 39, 'win_rate': 0.769, 'sharpe': 4.20,
                'total_pnl': 1150.42, 'max_drawdown_pct': 0.93,
            },
            'execution': {
                'idealized_signals': 63, 'placed': 59, 'filled': 39,
                'timed_out': 20, 'fill_rate': 0.66, 'miss_rate': 0.34,
                'skipped_quality': 0, 'blocked_risk': 4,
            },
        }],
        'portfolio': {
            'total_pnl': 1150.42, 'return_pct': 11.50,
            'final_equity': 11_150.42, 'total_trades': 39,
            'aggregate_win_rate': 0.769,
        },
    }


class TestOOSResult:

    def test_create_from_dict(self):
        from src.live.oos_engine import OOSResult
        data = _make_oos_result_data()
        result = OOSResult(data)
        assert result.run_id == 'test-run-001'
        assert len(result.per_asset) == 1

    def test_result_is_immutable_view(self):
        from src.live.oos_engine import OOSResult
        data = _make_oos_result_data()
        result = OOSResult(data)
        assert result.config['assets'] == ['BTCUSDT']

    def test_evaluate_with_evaluator(self):
        from src.live.oos_engine import OOSResult, OOSResultEvaluator
        class DummyEval(OOSResultEvaluator):
            evaluator_name = 'dummy'
            def evaluate(self, oos_result):
                return {'answer': 42}
        data = _make_oos_result_data()
        result = OOSResult(data)
        result.register_evaluator(DummyEval())
        analysis = result.evaluate(['dummy'])
        assert analysis['dummy']['answer'] == 42

    def test_evaluate_unknown_evaluator_fails(self):
        from src.live.oos_engine import OOSResult
        result = OOSResult(_make_oos_result_data())
        with pytest.raises(KeyError):
            result.evaluate(['nonexistent'])

    def test_evaluate_multiple(self):
        from src.live.oos_engine import OOSResult, OOSResultEvaluator
        class EvalA(OOSResultEvaluator):
            evaluator_name = 'eval_a'
            def evaluate(self, oos_result):
                return {'a': 1}
        class EvalB(OOSResultEvaluator):
            evaluator_name = 'eval_b'
            def evaluate(self, oos_result):
                return {'b': 2}
        result = OOSResult(_make_oos_result_data())
        result.register_evaluator(EvalA())
        result.register_evaluator(EvalB())
        analysis = result.evaluate(['eval_a', 'eval_b'])
        assert analysis['eval_a']['a'] == 1
        assert analysis['eval_b']['b'] == 2

    def test_to_dict_roundtrip(self):
        from src.live.oos_engine import OOSResult
        data = _make_oos_result_data()
        result = OOSResult(data)
        d = result.to_dict()
        s = json.dumps(d)
        loaded = json.loads(s)
        assert loaded['run_id'] == 'test-run-001'


class TestOOSResultEvaluator:

    def test_base_class_interface(self):
        from src.live.oos_engine import OOSResultEvaluator
        assert hasattr(OOSResultEvaluator, 'evaluator_name')
        assert hasattr(OOSResultEvaluator, 'evaluate')

    def test_evaluator_must_have_name(self):
        from src.live.oos_engine import OOSResultEvaluator
        class BadEval(OOSResultEvaluator):
            pass
        with pytest.raises(TypeError):
            BadEval()

    def test_evaluator_output_serializable(self):
        from src.live.oos_engine import OOSResult, OOSResultEvaluator
        class SerEval(OOSResultEvaluator):
            evaluator_name = 'ser_test'
            def evaluate(self, oos_result):
                return {'sharpe': oos_result.per_asset[0]['performance']['sharpe']}
        result = OOSResult(_make_oos_result_data())
        result.register_evaluator(SerEval())
        analysis = result.evaluate(['ser_test'])
        s = json.dumps(analysis)
        assert 'sharpe' in s


class TestEvaluatorRegistry:

    def test_registry_register_and_resolve(self):
        from src.live.oos_engine import EvaluatorRegistry, OOSResultEvaluator
        class MyEval(OOSResultEvaluator):
            evaluator_name = 'my_eval'
            def evaluate(self, oos_result):
                return {}
        reg = EvaluatorRegistry()
        reg.register(MyEval())
        assert reg.get('my_eval') is not None

    def test_registry_unknown_raises(self):
        from src.live.oos_engine import EvaluatorRegistry
        reg = EvaluatorRegistry()
        with pytest.raises(KeyError):
            reg.get('nonexistent')

    def test_registry_list_evaluators(self):
        from src.live.oos_engine import EvaluatorRegistry, OOSResultEvaluator
        class E1(OOSResultEvaluator):
            evaluator_name = 'e1'
            def evaluate(self, r): return {}
        class E2(OOSResultEvaluator):
            evaluator_name = 'e2'
            def evaluate(self, r): return {}
        reg = EvaluatorRegistry()
        reg.register(E1())
        reg.register(E2())
        assert set(reg.list()) == {'e1', 'e2'}


class TestEvaluationPersistence:

    def test_evaluation_persisted(self):
        from src.live.oos_engine import OOSResult, OOSResultEvaluator
        class PersistEval(OOSResultEvaluator):
            evaluator_name = 'persist_test'
            def evaluate(self, oos_result):
                return {'value': 99}
        reports_dir = Path(tempfile.mkdtemp())
        result = OOSResult(_make_oos_result_data(), reports_dir=reports_dir)
        result.register_evaluator(PersistEval())
        result.evaluate(['persist_test'])
        path = reports_dir / 'eval_test-run-001_persist_test.json'
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded['value'] == 99


class TestEngineIntegration:

    def test_engine_run_returns_oos_result(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine, OOSResult
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31', mode='idealized',
        )
        engine = CanonicalOOSEngine()
        result = engine.run_oos(cfg)
        assert isinstance(result, OOSResult)
        assert result.run_id is not None

    def test_engine_evaluate_run(self):
        from src.live.oos_engine import (
            OOSConfig, CanonicalOOSEngine, OOSResultEvaluator,
        )
        class SummaryEval(OOSResultEvaluator):
            evaluator_name = 'summary'
            def evaluate(self, oos_result):
                return {'n_assets': len(oos_result.per_asset)}
        engine = CanonicalOOSEngine()
        engine.register_evaluator(SummaryEval())
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31', mode='idealized',
        )
        result = engine.run_oos(cfg)
        analysis = result.evaluate(['summary'])
        assert analysis['summary']['n_assets'] == 1
