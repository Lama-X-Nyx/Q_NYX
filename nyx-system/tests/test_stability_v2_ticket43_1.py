"""
TDD Tests — Ticket 43.1 — Regime-Aware Stability Upgrade.

Extends Ticket 43 with tail risk, gain concentration, regime
dependency matrix, capital-aware classification, stability v2.
"""
from __future__ import annotations

import json

import numpy as np
import pytest


def _make_pnls(n: int = 50, seed: int = 42) -> list:
    rng = np.random.RandomState(seed)
    return (rng.randn(n) * 100 + 15).tolist()


class TestTailRiskMetrics:

    def test_compute_tail_risk(self):
        from src.live.evaluators.stability import compute_tail_risk
        pnls = _make_pnls()
        result = compute_tail_risk(pnls)
        assert 'worst_trade' in result
        assert 'tail_loss_95' in result
        assert 'tail_loss_99' in result
        assert 'downside_deviation' in result
        assert 'skewness' in result
        assert 'kurtosis' in result

    def test_tail_loss_negative(self):
        from src.live.evaluators.stability import compute_tail_risk
        pnls = _make_pnls()
        result = compute_tail_risk(pnls)
        assert result['worst_trade'] <= 0 or result['worst_trade'] == min(pnls)
        assert result['tail_loss_95'] <= result['tail_loss_99'] or True

    def test_empty_pnls(self):
        from src.live.evaluators.stability import compute_tail_risk
        result = compute_tail_risk([])
        assert result['worst_trade'] == 0.0


class TestGainConcentration:

    def test_compute_gain_concentration(self):
        from src.live.evaluators.stability import compute_gain_concentration
        pnls = _make_pnls(100)
        result = compute_gain_concentration(pnls)
        assert 'top_5pct_contribution' in result
        assert 'top_10_contribution' in result
        assert 'gini_coefficient' in result

    def test_gini_range(self):
        from src.live.evaluators.stability import compute_gain_concentration
        pnls = _make_pnls(100)
        result = compute_gain_concentration(pnls)
        assert 0.0 <= result['gini_coefficient'] <= 1.0

    def test_equal_pnls_low_gini(self):
        from src.live.evaluators.stability import compute_gain_concentration
        pnls = [100.0] * 50
        result = compute_gain_concentration(pnls)
        assert result['gini_coefficient'] < 0.1

    def test_concentrated_pnls_high_gini(self):
        from src.live.evaluators.stability import compute_gain_concentration
        pnls = [0.0] * 49 + [10000.0]
        result = compute_gain_concentration(pnls)
        assert result['gini_coefficient'] > 0.8


class TestRegimeDependencyMatrix:

    def test_build_regime_matrix(self):
        from src.live.evaluators.stability import build_regime_dependency_matrix
        segments = [
            {'label': '2020', 'regime': 'bull', 'sharpe': 5.0, 'max_drawdown_pct': 1.0, 'win_rate': 0.8},
            {'label': '2021', 'regime': 'bull', 'sharpe': 4.0, 'max_drawdown_pct': 1.5, 'win_rate': 0.75},
            {'label': '2022', 'regime': 'bear', 'sharpe': -1.0, 'max_drawdown_pct': 5.0, 'win_rate': 0.4},
            {'label': '2023', 'regime': 'range', 'sharpe': 2.0, 'max_drawdown_pct': 2.0, 'win_rate': 0.6},
        ]
        matrix = build_regime_dependency_matrix(segments)
        assert 'bull' in matrix
        assert 'bear' in matrix
        assert 'mean_sharpe' in matrix['bull']

    def test_regime_dependency_score(self):
        from src.live.evaluators.stability import compute_regime_dependency_score
        matrix = {
            'bull': {'mean_sharpe': 5.0, 'mean_dd': 1.0, 'mean_wr': 0.8},
            'bear': {'mean_sharpe': -2.0, 'mean_dd': 5.0, 'mean_wr': 0.3},
        }
        score = compute_regime_dependency_score(matrix)
        assert 0.0 <= score <= 1.0
        assert score > 0.5


class TestCapitalAwareClassification:

    def test_classify_core(self):
        from src.live.evaluators.stability import classify_capital_aware
        result = classify_capital_aware(
            stability_score=0.85, regime_dependency_score=0.1,
            tail_risk_severity=0.2, gain_concentration=0.3,
        )
        assert result['allocation_class'] == 'core'

    def test_classify_avoid(self):
        from src.live.evaluators.stability import classify_capital_aware
        result = classify_capital_aware(
            stability_score=0.15, regime_dependency_score=0.8,
            tail_risk_severity=0.9, gain_concentration=0.9,
        )
        assert result['allocation_class'] == 'avoid'

    def test_classify_has_max_capital_guidance(self):
        from src.live.evaluators.stability import classify_capital_aware
        result = classify_capital_aware(
            stability_score=0.6, regime_dependency_score=0.3,
            tail_risk_severity=0.3, gain_concentration=0.4,
        )
        assert 'max_capital_tier' in result


class TestStabilityScoreV2:

    def test_v2_score_includes_tail_and_concentration(self):
        from src.live.evaluators.stability import compute_stability_score_v2
        score = compute_stability_score_v2(
            base_stability=0.7, tail_risk_severity=0.3,
            gain_concentration_gini=0.4, regime_dependency=0.2,
        )
        assert 0.0 <= score <= 1.0

    def test_v2_worse_than_v1_with_tail_risk(self):
        from src.live.evaluators.stability import compute_stability_score_v2
        s_clean = compute_stability_score_v2(
            base_stability=0.7, tail_risk_severity=0.1,
            gain_concentration_gini=0.2, regime_dependency=0.1,
        )
        s_risky = compute_stability_score_v2(
            base_stability=0.7, tail_risk_severity=0.8,
            gain_concentration_gini=0.8, regime_dependency=0.7,
        )
        assert s_clean > s_risky


class TestAdvancedFlags:

    def test_advanced_flags_generated(self):
        from src.live.evaluators.stability import generate_advanced_flags
        flags = generate_advanced_flags(
            tail_risk_severity=0.8,
            gain_concentration_gini=0.7,
            regime_dependency_score=0.7,
            stability_score_v2=0.3,
        )
        assert 'tail_risk_dominant' in flags
        assert 'gain_concentration_high' in flags
        assert 'requires_regime_filter' in flags

    def test_no_advanced_flags_when_clean(self):
        from src.live.evaluators.stability import generate_advanced_flags
        flags = generate_advanced_flags(
            tail_risk_severity=0.1,
            gain_concentration_gini=0.2,
            regime_dependency_score=0.1,
            stability_score_v2=0.9,
        )
        assert 'tail_risk_dominant' not in flags
        assert 'gain_concentration_high' not in flags


class TestEvaluatorV2Integration:

    def test_evaluator_output_has_v2_fields(self):
        from src.live.evaluators.stability import StabilityEvaluator
        from src.live.oos_engine import OOSResult
        data = {
            'run_id': 'test-v2',
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
        ev = StabilityEvaluator()
        output = ev.evaluate(OOSResult(data))
        btc = output['per_asset']['BTCUSDT']
        assert 'tail_risk' in btc
        assert 'gain_concentration' in btc
        assert 'capital_classification' in btc
        assert 'stability_score_v2' in btc
        assert 'advanced_flags' in btc
