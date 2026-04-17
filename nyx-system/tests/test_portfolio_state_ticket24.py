"""
TDD Tests — Ticket 24 — Position & Portfolio State.

Single source of truth for positions, exposure, PnL. Derived ONLY
from OMS fill events — no parallel state system, no NYXEngine
reconstruction.

Position lifecycle :
  fill(buy)  → open long position (or close short)
  fill(sell) → open short position (or close long)
  partial close → reduce qty, realize partial PnL
  full close → qty=0, realize full PnL
  flip → close + open opposite in one step
"""
from __future__ import annotations

import pytest


# ===========================================================================
class TestPositionModel:

    def test_open_long(self):
        from src.live.portfolio_state import Position
        p = Position('BTCUSDT')
        p.on_fill(side='buy', qty=0.5, price=16500.0)
        assert p.side == 'long'
        assert p.quantity == pytest.approx(0.5)
        assert p.avg_entry_price == pytest.approx(16500.0)
        assert p.realized_pnl == pytest.approx(0.0)

    def test_open_short(self):
        from src.live.portfolio_state import Position
        p = Position('BTCUSDT')
        p.on_fill(side='sell', qty=0.3, price=17000.0)
        assert p.side == 'short'
        assert p.quantity == pytest.approx(0.3)

    def test_add_to_long(self):
        from src.live.portfolio_state import Position
        p = Position('BTCUSDT')
        p.on_fill(side='buy', qty=0.5, price=16500.0)
        p.on_fill(side='buy', qty=0.5, price=16600.0)
        assert p.quantity == pytest.approx(1.0)
        assert p.avg_entry_price == pytest.approx(16550.0)

    def test_partial_close_long(self):
        from src.live.portfolio_state import Position
        p = Position('BTCUSDT')
        p.on_fill(side='buy', qty=1.0, price=16500.0)
        p.on_fill(side='sell', qty=0.5, price=17000.0)
        assert p.quantity == pytest.approx(0.5)
        assert p.side == 'long'
        # Realized PnL = 0.5 × (17000 - 16500) = 250
        assert p.realized_pnl == pytest.approx(250.0)

    def test_full_close_long(self):
        from src.live.portfolio_state import Position
        p = Position('BTCUSDT')
        p.on_fill(side='buy', qty=1.0, price=16500.0)
        p.on_fill(side='sell', qty=1.0, price=17000.0)
        assert p.quantity == pytest.approx(0.0)
        assert p.side == 'flat'
        assert p.realized_pnl == pytest.approx(500.0)

    def test_flip_long_to_short(self):
        from src.live.portfolio_state import Position
        p = Position('BTCUSDT')
        p.on_fill(side='buy', qty=1.0, price=16500.0)
        p.on_fill(side='sell', qty=1.5, price=17000.0)
        # Close 1.0 long → realize PnL, then open 0.5 short
        assert p.side == 'short'
        assert p.quantity == pytest.approx(0.5)
        assert p.realized_pnl == pytest.approx(500.0)
        assert p.avg_entry_price == pytest.approx(17000.0)

    def test_unrealized_pnl_long(self):
        from src.live.portfolio_state import Position
        p = Position('BTCUSDT')
        p.on_fill(side='buy', qty=1.0, price=16500.0)
        upnl = p.unrealized_pnl(mark_price=17000.0)
        assert upnl == pytest.approx(500.0)

    def test_unrealized_pnl_short(self):
        from src.live.portfolio_state import Position
        p = Position('BTCUSDT')
        p.on_fill(side='sell', qty=1.0, price=17000.0)
        upnl = p.unrealized_pnl(mark_price=16500.0)
        assert upnl == pytest.approx(500.0)

    def test_flat_unrealized_is_zero(self):
        from src.live.portfolio_state import Position
        p = Position('BTCUSDT')
        assert p.unrealized_pnl(mark_price=17000.0) == pytest.approx(0.0)


# ===========================================================================
class TestPortfolioModel:

    def test_initial_state(self):
        from src.live.portfolio_state import Portfolio
        pf = Portfolio(initial_capital=10_000.0)
        assert pf.total_equity == pytest.approx(10_000.0)
        assert pf.available_balance == pytest.approx(10_000.0)
        assert pf.total_exposure == pytest.approx(0.0)

    def test_fill_updates_portfolio(self):
        from src.live.portfolio_state import Portfolio
        pf = Portfolio(initial_capital=10_000.0)
        pf.on_fill(symbol='BTCUSDT', side='buy', qty=0.1,
                    price=16500.0, fee=0.33)
        pos = pf.get_position('BTCUSDT')
        assert pos is not None
        assert pos.quantity == pytest.approx(0.1)
        assert pf.total_exposure > 0

    def test_close_updates_realized_pnl(self):
        from src.live.portfolio_state import Portfolio
        pf = Portfolio(initial_capital=10_000.0)
        pf.on_fill(symbol='BTCUSDT', side='buy', qty=0.1,
                    price=16500.0, fee=0.33)
        pf.on_fill(symbol='BTCUSDT', side='sell', qty=0.1,
                    price=17000.0, fee=0.34)
        assert pf.realized_pnl_total == pytest.approx(50.0)
        # Fees deducted from available balance
        assert pf.available_balance < 10_000.0 + 50.0

    def test_exposure_per_symbol(self):
        from src.live.portfolio_state import Portfolio
        pf = Portfolio(initial_capital=10_000.0)
        pf.on_fill(symbol='BTCUSDT', side='buy', qty=0.5,
                    price=16500.0, fee=1.65)
        exp = pf.exposure('BTCUSDT', mark_price=16500.0)
        assert exp == pytest.approx(0.5 * 16500.0)

    def test_total_exposure_multiple_symbols(self):
        from src.live.portfolio_state import Portfolio
        pf = Portfolio(initial_capital=50_000.0)
        pf.on_fill(symbol='BTCUSDT', side='buy', qty=0.5,
                    price=16500.0, fee=1.65)
        pf.on_fill(symbol='ETHUSDT', side='buy', qty=5.0,
                    price=1200.0, fee=1.20)
        marks = {'BTCUSDT': 16500.0, 'ETHUSDT': 1200.0}
        total = pf.total_exposure_at(marks)
        assert total == pytest.approx(0.5 * 16500.0 + 5.0 * 1200.0)

    def test_equity_includes_unrealized(self):
        from src.live.portfolio_state import Portfolio
        pf = Portfolio(initial_capital=10_000.0)
        pf.on_fill(symbol='BTCUSDT', side='buy', qty=1.0,
                    price=16500.0, fee=3.30)
        eq = pf.equity_at({'BTCUSDT': 17000.0})
        # equity = initial + realized + unrealized - fees
        expected_upnl = 1.0 * (17000.0 - 16500.0)
        assert eq == pytest.approx(10_000.0 - 3.30 + expected_upnl)


# ===========================================================================
class TestNoParallelState:
    """Portfolio must be the ONLY position tracker. No reconstruction."""

    def test_portfolio_positions_dict_is_authoritative(self):
        from src.live.portfolio_state import Portfolio
        pf = Portfolio(initial_capital=10_000.0)
        pf.on_fill(symbol='BTCUSDT', side='buy', qty=0.1,
                    price=16500.0, fee=0.33)
        assert 'BTCUSDT' in pf.positions
        assert len(pf.positions) == 1
