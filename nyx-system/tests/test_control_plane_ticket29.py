"""
TDD Tests — Ticket 29 — Control Plane (Operator Layer).

Commands live ON NYXRuntime (natural owner per CLAUDE.md Rule 1).
No separate controller module. Operator controls the system
WITHOUT modifying code.
"""
from __future__ import annotations

from pathlib import Path

import pytest


MODELS_DIR = Path(__file__).parent.parent / 'models'


@pytest.fixture(scope='module')
def runtime():
    from src.live.nyx_runtime import NYXRuntime
    return NYXRuntime(
        symbol='BTCUSDT',
        models_dir=MODELS_DIR / 'BTCUSDT',
        state_dir=Path('/tmp/nyx_test_control'),
        initial_capital=10_000.0,
    )


# ===========================================================================
class TestStartStop:

    def test_runtime_starts_running(self, runtime):
        runtime.start()
        assert runtime.is_running is True

    def test_runtime_stops(self, runtime):
        runtime.start()
        runtime.stop()
        assert runtime.is_running is False

    def test_on_bar_rejected_when_stopped(self, runtime):
        runtime.stop()
        result = runtime.on_bar({
            'timestamp': '2023-06-15T10:00:00',
            'open': 16500, 'high': 16510, 'low': 16490,
            'close': 16505, 'volume': 500,
        })
        assert result['action'] == 'SYSTEM_STOPPED'


# ===========================================================================
class TestPauseResume:

    def test_pause_blocks_trading(self, runtime):
        runtime.start()
        runtime.pause()
        assert runtime.is_paused is True
        result = runtime.on_bar({
            'timestamp': '2023-06-15T10:15:00',
            'open': 16500, 'high': 16510, 'low': 16490,
            'close': 16505, 'volume': 500,
        })
        assert result['action'] == 'PAUSED'

    def test_resume_unblocks(self, runtime):
        runtime.start()
        runtime.pause()
        runtime.resume()
        assert runtime.is_paused is False


# ===========================================================================
class TestEmergencyStop:

    def test_emergency_stop_kills_and_stops(self, runtime):
        runtime.start()
        runtime.emergency_stop(reason='operator panic')
        assert runtime.is_running is False
        assert runtime.risk_engine.is_killed is True


# ===========================================================================
class TestCancelAll:

    def test_cancel_all_open_orders(self, runtime):
        runtime.start()
        runtime.risk_engine.reset_kill_switch()
        # Submit 2 orders.
        runtime.oms.submit_order('ctrl-1', 'BTCUSDT', 'buy', 0.01, 16500.0)
        runtime.oms.submit_order('ctrl-2', 'BTCUSDT', 'sell', 0.01, 17000.0)
        n = runtime.cancel_all_orders()
        assert n >= 2
        # All open orders must be CANCELLED.
        for o in runtime.oms._orders.values():
            assert o.status in ('CANCELLED', 'FILLED', 'REJECTED')


# ===========================================================================
class TestFlattenAll:

    def test_flatten_reports_positions(self, runtime):
        """flatten_all must return the list of positions that need
        closing. Actual order placement is the runtime's responsibility
        (it would call OMS in live)."""
        runtime.start()
        runtime.risk_engine.reset_kill_switch()
        # Simulate a position.
        runtime.portfolio.on_fill('BTCUSDT', 'buy', 0.5, 16500.0)
        to_close = runtime.flatten_all()
        assert len(to_close) >= 1
        assert to_close[0]['symbol'] == 'BTCUSDT'
        assert to_close[0]['side'] == 'sell'  # opposite to close long
