"""
TDD Tests — Ticket 46 v1 Hardening.

Fixes: double-counted signals, no smoothing, hard cliffs.
Backward compatible with T46 API.
"""
from __future__ import annotations

import pytest


class TestIndependentSignals:

    def test_health_v2_uses_three_independent_signals(self):
        from src.live.execution_monitor import compute_health_score_v2
        score = compute_health_score_v2(
            fill_rate=0.9, execution_cost_factor=0.8, timeout_quality=0.9,
        )
        assert 0.0 <= score <= 1.0

    def test_weights_sum_to_one(self):
        from src.live.execution_monitor import HEALTH_V2_WEIGHTS
        total = sum(HEALTH_V2_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_fill_dominant_weight(self):
        from src.live.execution_monitor import HEALTH_V2_WEIGHTS
        assert HEALTH_V2_WEIGHTS['fill_rate'] >= 0.5

    def test_perfect_signals(self):
        from src.live.execution_monitor import compute_health_score_v2
        score = compute_health_score_v2(
            fill_rate=1.0, execution_cost_factor=1.0, timeout_quality=1.0,
        )
        assert score >= 0.99

    def test_zero_fill_kills_score(self):
        from src.live.execution_monitor import compute_health_score_v2
        score = compute_health_score_v2(
            fill_rate=0.0, execution_cost_factor=1.0, timeout_quality=1.0,
        )
        assert score <= 0.45


class TestExecutionCostFactor:

    def test_cost_factor_at_baseline(self):
        from src.live.execution_monitor import compute_execution_cost_factor
        assert compute_execution_cost_factor(avg_fee=1.0, baseline_fee=1.0) == 1.0

    def test_cost_factor_decreases_with_fees(self):
        from src.live.execution_monitor import compute_execution_cost_factor
        c1 = compute_execution_cost_factor(avg_fee=1.5, baseline_fee=1.0)
        c2 = compute_execution_cost_factor(avg_fee=2.0, baseline_fee=1.0)
        assert c1 > c2
        assert c1 < 1.0

    def test_cost_factor_bounded(self):
        from src.live.execution_monitor import compute_execution_cost_factor
        assert 0.0 <= compute_execution_cost_factor(10.0, 1.0) <= 1.0


class TestTimeoutQuality:

    def test_timeout_quality_perfect(self):
        from src.live.execution_monitor import compute_timeout_quality
        assert compute_timeout_quality(timeout_rate=0.0) == 1.0

    def test_timeout_quality_degrades(self):
        from src.live.execution_monitor import compute_timeout_quality
        q = compute_timeout_quality(timeout_rate=0.5)
        assert q < 1.0
        assert q > 0.0


class TestEMASmoothing:

    def test_ema_initial(self):
        from src.live.execution_monitor import EMASmoother
        s = EMASmoother(alpha=0.2)
        s.update(0.8)
        assert abs(s.value - 0.8) < 1e-9

    def test_ema_smooths_noise(self):
        from src.live.execution_monitor import EMASmoother
        s = EMASmoother(alpha=0.2)
        s.update(0.9)
        s.update(0.1)
        s.update(0.9)
        s.update(0.1)
        assert 0.3 < s.value < 0.9

    def test_ema_alpha_controls_reactivity(self):
        from src.live.execution_monitor import EMASmoother
        fast = EMASmoother(alpha=0.8)
        slow = EMASmoother(alpha=0.05)
        fast.update(0.9)
        slow.update(0.9)
        fast.update(0.1)
        slow.update(0.1)
        assert fast.value < slow.value


class TestContinuousMultiplier:

    def test_multiplier_healthy_is_one(self):
        from src.live.execution_monitor import continuous_risk_multiplier
        m = continuous_risk_multiplier(
            health=0.9, critical=0.35, normal=0.70,
        )
        assert m == 1.0

    def test_multiplier_critical_is_zero(self):
        from src.live.execution_monitor import continuous_risk_multiplier
        m = continuous_risk_multiplier(
            health=0.2, critical=0.35, normal=0.70,
        )
        assert m == 0.0

    def test_multiplier_interpolates(self):
        from src.live.execution_monitor import continuous_risk_multiplier
        m = continuous_risk_multiplier(
            health=0.525, critical=0.35, normal=0.70,
        )
        assert abs(m - 0.5) < 0.01

    def test_multiplier_monotonic(self):
        from src.live.execution_monitor import continuous_risk_multiplier
        ms = [
            continuous_risk_multiplier(h, critical=0.35, normal=0.70)
            for h in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        ]
        for a, b in zip(ms[:-1], ms[1:]):
            assert a <= b


class TestNoNoiseOverreaction:

    def test_single_bad_fill_does_not_collapse(self):
        from src.live.execution_monitor import ExecutionMonitor
        em = ExecutionMonitor(
            assets=['BTCUSDT'], use_v2=True, smoothing_alpha=0.15,
        )
        for _ in range(30):
            em.record_order('BTCUSDT', placed=True)
            em.record_fill('BTCUSDT', filled=True, fee=1.0)
        em.record_order('BTCUSDT', placed=True)
        em.record_fill('BTCUSDT', filled=False)
        multiplier = em.get_risk_multiplier('BTCUSDT')
        assert multiplier > 0.7

    def test_sustained_degradation_reduces_multiplier(self):
        from src.live.execution_monitor import ExecutionMonitor
        em = ExecutionMonitor(
            assets=['BTCUSDT'], use_v2=True, smoothing_alpha=0.3,
            window_size=30,
        )
        for _ in range(30):
            em.record_order('BTCUSDT', placed=True)
            em.record_fill('BTCUSDT', filled=True, fee=1.0)
        em.get_smoothed_health('BTCUSDT')
        for _ in range(30):
            em.record_order('BTCUSDT', placed=True)
            em.record_fill('BTCUSDT', filled=False, fee=1.0)
            em.get_smoothed_health('BTCUSDT')
        multiplier = em.get_risk_multiplier('BTCUSDT')
        assert multiplier < 0.8


class TestBackwardCompatibility:

    def test_v1_api_still_works(self):
        from src.live.execution_monitor import ExecutionMonitor
        em = ExecutionMonitor(assets=['BTCUSDT'])
        em.record_order('BTCUSDT', placed=True)
        em.record_fill('BTCUSDT', filled=True, fee=1.0)
        m = em.get_multiplier('BTCUSDT')
        assert 0.0 <= m <= 1.0

    def test_get_health_still_works(self):
        from src.live.execution_monitor import ExecutionMonitor
        em = ExecutionMonitor(assets=['BTCUSDT'])
        for _ in range(20):
            em.record_order('BTCUSDT', placed=True)
            em.record_fill('BTCUSDT', filled=True, fee=1.0)
        h = em.get_health('BTCUSDT')
        assert 0.0 <= h <= 1.0

    def test_v1_compute_health_score_still_works(self):
        from src.live.execution_monitor import compute_health_score
        s = compute_health_score(
            fill_rate=0.9, miss_rate=0.1, avg_fee=1.0, baseline_fee=1.0,
        )
        assert 0.0 <= s <= 1.0


class TestDeterminism:

    def test_same_inputs_same_outputs(self):
        from src.live.execution_monitor import ExecutionMonitor
        em1 = ExecutionMonitor(assets=['BTCUSDT'], use_v2=True)
        em2 = ExecutionMonitor(assets=['BTCUSDT'], use_v2=True)
        events = [(True, 1.0), (True, 1.0), (False, 0.0),
                  (True, 1.5), (True, 1.0), (False, 0.0)]
        for e in [em1, em2]:
            for filled, fee in events:
                e.record_order('BTCUSDT', placed=True)
                e.record_fill('BTCUSDT', filled=filled, fee=fee)
        assert em1.get_health('BTCUSDT') == em2.get_health('BTCUSDT')
        assert em1.get_risk_multiplier('BTCUSDT') == em2.get_risk_multiplier('BTCUSDT')
