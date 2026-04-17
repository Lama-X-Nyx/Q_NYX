"""
TDD Tests — Ticket 23 — Order Management System (OMS).

The OMS wraps PostOnlyPaperBroker as an execution CONTROLLER.
It owns order state, enforces valid transitions, prevents
duplicates, tracks partial fills, and logs every mutation.

NYXEngine expresses intent via `oms.submit_order()`.
OMS is the SINGLE SOURCE OF TRUTH for order state.
"""
from __future__ import annotations

import pytest


# ===========================================================================
class TestOMSOrderLifecycle:
    """Valid and invalid state transitions."""

    def test_submit_creates_order(self):
        from src.live.oms import OMS
        oms = OMS()
        oid = oms.submit_order(
            client_order_id='test-001', symbol='BTCUSDT',
            side='buy', quantity=0.1, price=16500.0,
            order_type='post_only_limit',
        )
        order = oms.get_order(oid)
        assert order is not None
        assert order.status == 'SUBMITTED'

    def test_fill_transitions_to_filled(self):
        from src.live.oms import OMS
        oms = OMS()
        oid = oms.submit_order(
            client_order_id='test-002', symbol='BTCUSDT',
            side='buy', quantity=0.1, price=16500.0,
        )
        oms.handle_fill(oid, fill_qty=0.1, fill_price=16500.0)
        assert oms.get_order(oid).status == 'FILLED'

    def test_partial_fill_transitions_correctly(self):
        from src.live.oms import OMS
        oms = OMS()
        oid = oms.submit_order(
            client_order_id='test-003', symbol='BTCUSDT',
            side='buy', quantity=1.0, price=16500.0,
        )
        oms.handle_fill(oid, fill_qty=0.3, fill_price=16500.0)
        order = oms.get_order(oid)
        assert order.status == 'PARTIALLY_FILLED'
        assert order.filled_qty == pytest.approx(0.3)

        oms.handle_fill(oid, fill_qty=0.7, fill_price=16501.0)
        order = oms.get_order(oid)
        assert order.status == 'FILLED'
        assert order.filled_qty == pytest.approx(1.0)

    def test_reject_transitions(self):
        from src.live.oms import OMS
        oms = OMS()
        oid = oms.submit_order(
            client_order_id='test-004', symbol='BTCUSDT',
            side='buy', quantity=0.1, price=16500.0,
        )
        oms.handle_reject(oid, reason='post-only would cross')
        order = oms.get_order(oid)
        assert order.status == 'REJECTED'
        assert order.reject_reason == 'post-only would cross'

    def test_cancel_transitions(self):
        from src.live.oms import OMS
        oms = OMS()
        oid = oms.submit_order(
            client_order_id='test-005', symbol='BTCUSDT',
            side='buy', quantity=0.1, price=16500.0,
        )
        oms.cancel_order(oid, reason='timeout')
        assert oms.get_order(oid).status == 'CANCELLED'

    def test_invalid_transition_raises(self):
        """Terminal orders (FILLED/REJECTED/CANCELLED) cannot be
        mutated further."""
        from src.live.oms import OMS
        oms = OMS()
        oid = oms.submit_order(
            client_order_id='test-006', symbol='BTCUSDT',
            side='buy', quantity=0.1, price=16500.0,
        )
        oms.handle_reject(oid, reason='test')
        with pytest.raises((ValueError, RuntimeError)):
            oms.handle_fill(oid, fill_qty=0.1, fill_price=16500.0)


# ===========================================================================
class TestDuplicateProtection:

    def test_duplicate_client_order_id_blocked(self):
        from src.live.oms import OMS
        oms = OMS()
        oms.submit_order(
            client_order_id='dup-001', symbol='BTCUSDT',
            side='buy', quantity=0.1, price=16500.0,
        )
        with pytest.raises((ValueError, RuntimeError)):
            oms.submit_order(
                client_order_id='dup-001', symbol='BTCUSDT',
                side='buy', quantity=0.1, price=16500.0,
            )

    def test_cancel_on_terminal_is_noop(self):
        """Cancelling an already-filled order must not crash."""
        from src.live.oms import OMS
        oms = OMS()
        oid = oms.submit_order(
            client_order_id='term-001', symbol='BTCUSDT',
            side='buy', quantity=0.1, price=16500.0,
        )
        oms.handle_fill(oid, fill_qty=0.1, fill_price=16500.0)
        # Should not raise — just a no-op or warning.
        oms.cancel_order(oid, reason='late cancel')
        assert oms.get_order(oid).status == 'FILLED'  # unchanged


# ===========================================================================
class TestFillReconciliation:

    def test_avg_fill_price_computed(self):
        from src.live.oms import OMS
        oms = OMS()
        oid = oms.submit_order(
            client_order_id='fill-001', symbol='BTCUSDT',
            side='buy', quantity=1.0, price=16500.0,
        )
        oms.handle_fill(oid, fill_qty=0.5, fill_price=16500.0)
        oms.handle_fill(oid, fill_qty=0.5, fill_price=16510.0)
        order = oms.get_order(oid)
        assert order.avg_fill_price == pytest.approx(16505.0)
        assert order.filled_qty == pytest.approx(1.0)

    def test_overfill_prevented(self):
        """Filling more than ordered quantity must raise."""
        from src.live.oms import OMS
        oms = OMS()
        oid = oms.submit_order(
            client_order_id='over-001', symbol='BTCUSDT',
            side='buy', quantity=0.5, price=16500.0,
        )
        with pytest.raises((ValueError, RuntimeError)):
            oms.handle_fill(oid, fill_qty=1.0, fill_price=16500.0)


# ===========================================================================
class TestExecutionLog:

    def test_events_logged(self):
        from src.live.oms import OMS
        oms = OMS()
        oid = oms.submit_order(
            client_order_id='log-001', symbol='BTCUSDT',
            side='buy', quantity=0.1, price=16500.0,
        )
        oms.handle_fill(oid, fill_qty=0.1, fill_price=16500.0)
        events = oms.get_events(oid)
        assert len(events) >= 2  # SUBMITTED + FILLED
        assert events[0]['event'] == 'SUBMITTED'
        assert events[-1]['event'] == 'FILLED'
