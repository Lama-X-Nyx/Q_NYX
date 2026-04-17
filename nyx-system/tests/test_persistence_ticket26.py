"""
TDD Tests — Ticket 26 — Persistence & Recovery.

All critical state (OMS orders, Portfolio positions, event log)
must survive a crash. On restart, state is LOADED — not
reconstructed from guesswork.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / 'state'


# ===========================================================================
class TestOMSPersistence:

    def test_save_and_load_orders(self, state_dir):
        from src.live.oms import OMS
        from src.live.state_store import StateStore

        oms = OMS()
        oms.submit_order('c1', 'BTCUSDT', 'buy', 0.1, 16500.0)
        oms.submit_order('c2', 'ETHUSDT', 'sell', 1.0, 1200.0)
        oms.handle_fill(1, fill_qty=0.1, fill_price=16500.0)

        store = StateStore(state_dir)
        store.save_oms(oms)

        oms2 = OMS()
        store.load_oms(oms2)
        assert oms2.get_order(1).status == 'FILLED'
        assert oms2.get_order(2).status == 'SUBMITTED'
        assert 'c1' in oms2._by_client_id

    def test_no_duplicate_after_reload(self, state_dir):
        from src.live.oms import OMS
        from src.live.state_store import StateStore

        oms = OMS()
        oms.submit_order('dup-test', 'BTCUSDT', 'buy', 0.1, 16500.0)
        store = StateStore(state_dir)
        store.save_oms(oms)

        oms2 = OMS()
        store.load_oms(oms2)
        with pytest.raises((ValueError, RuntimeError)):
            oms2.submit_order('dup-test', 'BTCUSDT', 'buy', 0.1, 16500.0)


# ===========================================================================
class TestPortfolioPersistence:

    def test_save_and_load_portfolio(self, state_dir):
        from src.live.portfolio_state import Portfolio
        from src.live.state_store import StateStore

        pf = Portfolio(initial_capital=10_000.0)
        pf.on_fill('BTCUSDT', 'buy', 0.5, 16500.0, fee=1.65)

        store = StateStore(state_dir)
        store.save_portfolio(pf)

        pf2 = Portfolio(initial_capital=10_000.0)
        store.load_portfolio(pf2)
        pos = pf2.get_position('BTCUSDT')
        assert pos is not None
        assert pos.quantity == pytest.approx(0.5)
        assert pos.avg_entry_price == pytest.approx(16500.0)
        assert pf2.total_fees == pytest.approx(1.65)

    def test_no_lost_positions(self, state_dir):
        from src.live.portfolio_state import Portfolio
        from src.live.state_store import StateStore

        pf = Portfolio(initial_capital=50_000.0)
        pf.on_fill('BTCUSDT', 'buy', 0.5, 16500.0)
        pf.on_fill('ETHUSDT', 'buy', 5.0, 1200.0)

        store = StateStore(state_dir)
        store.save_portfolio(pf)

        pf2 = Portfolio(initial_capital=50_000.0)
        store.load_portfolio(pf2)
        assert 'BTCUSDT' in pf2.positions
        assert 'ETHUSDT' in pf2.positions


# ===========================================================================
class TestEventLogPersistence:

    def test_event_log_saved(self, state_dir):
        from src.live.oms import OMS
        from src.live.state_store import StateStore

        oms = OMS()
        oms.submit_order('ev1', 'BTCUSDT', 'buy', 0.1, 16500.0)
        oms.handle_fill(1, fill_qty=0.1, fill_price=16500.0)

        store = StateStore(state_dir)
        store.save_event_log(oms)

        log_file = state_dir / 'event_log.jsonl'
        assert log_file.exists()
        lines = log_file.read_text().strip().split('\n')
        assert len(lines) >= 2


# ===========================================================================
class TestAtomicWrite:

    def test_partial_write_does_not_corrupt(self, state_dir):
        """StateStore must use atomic writes (tmp → rename) so a
        crash mid-write doesn't leave a corrupt file."""
        from src.live.state_store import StateStore
        store = StateStore(state_dir)
        store._atomic_write(state_dir / 'test.json', {'ok': True})
        data = json.loads((state_dir / 'test.json').read_text())
        assert data == {'ok': True}
