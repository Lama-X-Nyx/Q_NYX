"""
TDD Tests — Ticket 27 — Monitoring & Observability.

MetricsCollector tracks trading + system metrics in real time.
AlertManager triggers on threshold breaches. No blind trading.
"""
from __future__ import annotations

import pytest


# ===========================================================================
class TestMetricsCollector:

    def test_record_trade(self):
        from src.live.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.record_trade(pnl=50.0, side='buy', filled=True)
        assert mc.total_trades == 1
        assert mc.total_pnl == pytest.approx(50.0)

    def test_win_rate(self):
        from src.live.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.record_trade(pnl=100.0, side='buy', filled=True)
        mc.record_trade(pnl=-30.0, side='sell', filled=True)
        mc.record_trade(pnl=50.0, side='buy', filled=True)
        assert mc.win_rate == pytest.approx(2 / 3)

    def test_fill_and_miss_rate(self):
        from src.live.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.record_order_attempt(filled=True)
        mc.record_order_attempt(filled=True)
        mc.record_order_attempt(filled=False)
        assert mc.fill_rate == pytest.approx(2 / 3)
        assert mc.miss_rate == pytest.approx(1 / 3)

    def test_drawdown_tracking(self):
        from src.live.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.update_equity(10_000.0)
        mc.update_equity(9_800.0)
        assert mc.max_drawdown_pct == pytest.approx(2.0)

    def test_snapshot(self):
        from src.live.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.record_trade(pnl=50.0, side='buy', filled=True)
        mc.update_equity(10_050.0)
        snap = mc.snapshot()
        assert 'total_trades' in snap
        assert 'total_pnl' in snap
        assert 'max_drawdown_pct' in snap


# ===========================================================================
class TestSystemMetrics:

    def test_ws_latency(self):
        from src.live.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.record_ws_latency_ms(15.0)
        mc.record_ws_latency_ms(25.0)
        assert mc.avg_ws_latency_ms == pytest.approx(20.0)

    def test_reconnect_count(self):
        from src.live.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.record_reconnect()
        mc.record_reconnect()
        assert mc.reconnect_count == 2

    def test_reject_rate(self):
        from src.live.monitoring import MetricsCollector
        mc = MetricsCollector()
        mc.record_order_attempt(filled=True)
        mc.record_order_reject()
        assert mc.reject_rate == pytest.approx(0.5)


# ===========================================================================
class TestAlertManager:

    def test_loss_alert_triggers(self):
        from src.live.monitoring import AlertManager
        am = AlertManager(max_daily_loss_pct=1.0)
        alerts = am.check(daily_loss_pct=1.5, equity=10_000.0)
        assert any('daily_loss' in a['type'] for a in alerts)

    def test_disconnect_alert(self):
        from src.live.monitoring import AlertManager
        am = AlertManager()
        alerts = am.check(ws_connected=False, equity=10_000.0)
        assert any('disconnect' in a['type'] for a in alerts)

    def test_no_alert_when_healthy(self):
        from src.live.monitoring import AlertManager
        am = AlertManager()
        alerts = am.check(
            daily_loss_pct=0.1, ws_connected=True, equity=10_000.0,
        )
        assert len(alerts) == 0

    def test_drawdown_alert(self):
        from src.live.monitoring import AlertManager
        am = AlertManager(max_drawdown_alert_pct=3.0)
        alerts = am.check(drawdown_pct=4.0, equity=10_000.0)
        assert any('drawdown' in a['type'] for a in alerts)
