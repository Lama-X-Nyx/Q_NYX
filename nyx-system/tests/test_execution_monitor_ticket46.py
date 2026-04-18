"""
TDD Tests — Ticket 46 — Real-Time Execution Monitoring & Adaptive Risk.

ExecutionMonitor tracks real-time fill/fee/miss metrics.
RiskController converts health into risk_multiplier (0.0-1.0).
Integration with NYXRuntime applies the multiplier to size_multiplier.
"""
from __future__ import annotations

import pytest


class TestExecutionMetricsCollector:

    def test_record_fill(self):
        from src.live.execution_monitor import ExecutionMetricsCollector
        c = ExecutionMetricsCollector(window_size=100)
        c.record_order('BTCUSDT', placed=True)
        c.record_fill('BTCUSDT', filled=True, fee=2.0)
        snap = c.snapshot('BTCUSDT')
        assert snap['placed'] == 1
        assert snap['filled'] == 1
        assert snap['fill_rate'] == 1.0

    def test_record_miss(self):
        from src.live.execution_monitor import ExecutionMetricsCollector
        c = ExecutionMetricsCollector(window_size=100)
        c.record_order('BTCUSDT', placed=True)
        c.record_fill('BTCUSDT', filled=False)
        snap = c.snapshot('BTCUSDT')
        assert snap['placed'] == 1
        assert snap['filled'] == 0
        assert snap['miss_rate'] == 1.0

    def test_rolling_window(self):
        from src.live.execution_monitor import ExecutionMetricsCollector
        c = ExecutionMetricsCollector(window_size=5)
        for i in range(10):
            c.record_order('BTCUSDT', placed=True)
            c.record_fill('BTCUSDT', filled=(i >= 5))
        snap = c.snapshot('BTCUSDT')
        assert snap['placed'] == 5
        assert snap['fill_rate'] == 1.0

    def test_multi_asset(self):
        from src.live.execution_monitor import ExecutionMetricsCollector
        c = ExecutionMetricsCollector()
        c.record_order('BTCUSDT', placed=True)
        c.record_order('ETHUSDT', placed=True)
        c.record_fill('BTCUSDT', filled=True)
        c.record_fill('ETHUSDT', filled=False)
        assert c.snapshot('BTCUSDT')['fill_rate'] == 1.0
        assert c.snapshot('ETHUSDT')['fill_rate'] == 0.0

    def test_average_fee(self):
        from src.live.execution_monitor import ExecutionMetricsCollector
        c = ExecutionMetricsCollector()
        for fee in [1.0, 2.0, 3.0]:
            c.record_order('BTCUSDT', placed=True)
            c.record_fill('BTCUSDT', filled=True, fee=fee)
        snap = c.snapshot('BTCUSDT')
        assert snap['avg_fee'] == 2.0


class TestExecutionHealthScore:

    def test_perfect_health(self):
        from src.live.execution_monitor import compute_health_score
        score = compute_health_score(
            fill_rate=1.0, miss_rate=0.0,
            avg_fee=1.0, baseline_fee=1.0,
        )
        assert score >= 0.95

    def test_poor_health(self):
        from src.live.execution_monitor import compute_health_score
        score = compute_health_score(
            fill_rate=0.3, miss_rate=0.7,
            avg_fee=3.0, baseline_fee=1.0,
        )
        assert score < 0.3

    def test_score_range(self):
        from src.live.execution_monitor import compute_health_score
        score = compute_health_score(0.5, 0.5, 2.0, 1.0)
        assert 0.0 <= score <= 1.0


class TestRiskController:

    def test_normal_state(self):
        from src.live.execution_monitor import RiskController
        rc = RiskController()
        state = rc.classify(health_score=0.9)
        assert state == 'normal'

    def test_degraded_state(self):
        from src.live.execution_monitor import RiskController
        rc = RiskController()
        state = rc.classify(health_score=0.5)
        assert state == 'degraded'

    def test_critical_state(self):
        from src.live.execution_monitor import RiskController
        rc = RiskController()
        state = rc.classify(health_score=0.2)
        assert state == 'critical'

    def test_normal_multiplier(self):
        from src.live.execution_monitor import RiskController
        rc = RiskController()
        m = rc.get_multiplier(health_score=0.9)
        assert m == 1.0

    def test_degraded_multiplier(self):
        from src.live.execution_monitor import RiskController
        rc = RiskController()
        m = rc.get_multiplier(health_score=0.5)
        assert 0.0 < m < 1.0

    def test_critical_multiplier_zero(self):
        from src.live.execution_monitor import RiskController
        rc = RiskController()
        m = rc.get_multiplier(health_score=0.2)
        assert m == 0.0


class TestThresholdFlags:

    def test_fill_collapse_detected(self):
        from src.live.execution_monitor import detect_flags
        flags = detect_flags(
            fill_rate=0.3, miss_rate=0.7,
            avg_fee=1.0, baseline_fee=1.0,
            health_score=0.25,
        )
        assert 'fill_collapse_detected' in flags
        assert 'execution_critical' in flags

    def test_fee_spike_detected(self):
        from src.live.execution_monitor import detect_flags
        flags = detect_flags(
            fill_rate=0.9, miss_rate=0.1,
            avg_fee=2.0, baseline_fee=1.0,
            health_score=0.6,
        )
        assert 'fee_spike_detected' in flags

    def test_healthy_flag(self):
        from src.live.execution_monitor import detect_flags
        flags = detect_flags(
            fill_rate=0.95, miss_rate=0.05,
            avg_fee=1.0, baseline_fee=1.0,
            health_score=0.95,
        )
        assert 'execution_healthy' in flags


class TestGlobalVsAssetControl:

    def test_systemic_detection(self):
        from src.live.execution_monitor import ExecutionMonitor
        em = ExecutionMonitor(assets=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
            for _ in range(20):
                em.record_order(sym, placed=True)
                em.record_fill(sym, filled=False)
        assert em.is_systemic_degradation()

    def test_isolated_degradation(self):
        from src.live.execution_monitor import ExecutionMonitor
        em = ExecutionMonitor(assets=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        for _ in range(20):
            em.record_order('SOLUSDT', placed=True)
            em.record_fill('SOLUSDT', filled=False)
            em.record_order('BTCUSDT', placed=True)
            em.record_fill('BTCUSDT', filled=True, fee=1.0)
            em.record_order('ETHUSDT', placed=True)
            em.record_fill('ETHUSDT', filled=True, fee=1.0)
        assert not em.is_systemic_degradation()

    def test_per_asset_multiplier(self):
        from src.live.execution_monitor import ExecutionMonitor
        em = ExecutionMonitor(assets=['BTCUSDT', 'ETHUSDT'])
        for _ in range(20):
            em.record_order('BTCUSDT', placed=True)
            em.record_fill('BTCUSDT', filled=True, fee=1.0)
            em.record_order('ETHUSDT', placed=True)
            em.record_fill('ETHUSDT', filled=False)
        m_btc = em.get_multiplier('BTCUSDT')
        m_eth = em.get_multiplier('ETHUSDT')
        assert m_btc > m_eth
        assert m_btc >= 0.9


class TestHysteresis:

    def test_hysteresis_prevents_oscillation(self):
        from src.live.execution_monitor import RiskController
        rc = RiskController(hysteresis_margin=0.05)
        rc.update_state(health_score=0.9)
        assert rc.current_state == 'normal'
        rc.update_state(health_score=0.71)
        assert rc.current_state == 'normal'
        rc.update_state(health_score=0.5)
        assert rc.current_state == 'degraded'


class TestNYXRuntimeIntegration:

    def test_runtime_accepts_execution_monitor(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        from src.live.execution_monitor import ExecutionMonitor
        HERE = Path(__file__).resolve().parent.parent
        em = ExecutionMonitor(assets=['BTCUSDT'])
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
            execution_monitor=em,
        )
        assert rt.execution_monitor is em

    def test_runtime_without_monitor_still_works(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        HERE = Path(__file__).resolve().parent.parent
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        assert rt.execution_monitor is None
