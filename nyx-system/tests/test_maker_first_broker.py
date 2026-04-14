"""
TDD Tests — MakerFirstBroker

Paper execution that tries to fill as maker first, falls back to taker
after `max_wait` bars, and charges the correct fees on each path.

Contract:
  - place_order(side, qty, limit_price): returns open OrderId.
  - on_bar(ohlcv): update open orders; a limit is "filled as maker" if
    the bar traded through `limit_price`.
  - After `max_wait` bars without a maker fill, order converts to taker
    (filled at close + slippage).
  - Fee model: 0.02% maker, 0.04% taker + 0.03% slippage on taker fills.
  - Supports cancel() and returns fills history.
"""
import pytest


def _bar(high: float, low: float, close: float) -> dict:
    return {'open': low, 'high': high, 'low': low, 'close': close, 'volume': 100.0}


# ===========================================================================
# TEST 1: maker fill when price crosses the limit
# ===========================================================================
class TestMakerFill:

    def test_buy_fills_as_maker_when_bar_low_below_limit(self):
        """Buy limit at 45000. Bar low=44950 → maker fill."""
        from src.paper_live.broker import MakerFirstBroker
        b = MakerFirstBroker()
        oid = b.place_limit_buy(qty=0.01, limit_price=45000.0)
        b.on_bar('BTCUSDT', _bar(high=45100, low=44950, close=45050))
        fills = b.fills(oid)
        assert len(fills) == 1
        assert fills[0]['role'] == 'maker'
        assert fills[0]['price'] == 45000.0
        assert fills[0]['qty'] == 0.01

    def test_sell_fills_as_maker_when_bar_high_above_limit(self):
        from src.paper_live.broker import MakerFirstBroker
        b = MakerFirstBroker()
        oid = b.place_limit_sell(qty=0.01, limit_price=46000.0)
        b.on_bar('BTCUSDT', _bar(high=46100, low=45900, close=45950))
        fills = b.fills(oid)
        assert len(fills) == 1
        assert fills[0]['role'] == 'maker'
        assert fills[0]['price'] == 46000.0

    def test_no_fill_if_price_does_not_cross(self):
        from src.paper_live.broker import MakerFirstBroker
        b = MakerFirstBroker()
        oid = b.place_limit_buy(qty=0.01, limit_price=44000.0)
        b.on_bar('BTCUSDT', _bar(high=45100, low=45000, close=45050))
        assert b.fills(oid) == []
        assert b.is_open(oid) is True


# ===========================================================================
# TEST 2: maker fee charged correctly
# ===========================================================================
class TestMakerFee:

    def test_maker_fee_is_0p02pct(self):
        from src.paper_live.broker import MakerFirstBroker
        b = MakerFirstBroker()
        oid = b.place_limit_buy(qty=0.01, limit_price=45000.0)
        b.on_bar('BTCUSDT', _bar(high=45100, low=44950, close=45050))
        fills = b.fills(oid)
        expected_fee = 45000.0 * 0.01 * 0.0002
        assert abs(fills[0]['fee'] - expected_fee) < 1e-9
        assert fills[0]['role'] == 'maker'


# ===========================================================================
# TEST 3: taker fallback after max_wait
# ===========================================================================
class TestTakerFallback:

    def test_converts_to_taker_after_max_wait(self):
        """After max_wait=2 bars unfilled, next bar closes at market as taker."""
        from src.paper_live.broker import MakerFirstBroker
        b = MakerFirstBroker(max_wait_bars=2)
        oid = b.place_limit_buy(qty=0.01, limit_price=44000.0)
        # 3 bars without crossing the limit
        b.on_bar('BTCUSDT', _bar(high=45100, low=45000, close=45050))
        b.on_bar('BTCUSDT', _bar(high=45200, low=45100, close=45150))
        b.on_bar('BTCUSDT', _bar(high=45300, low=45150, close=45200))
        fills = b.fills(oid)
        assert len(fills) == 1
        assert fills[0]['role'] == 'taker'
        # taker fill price = close + slippage
        assert fills[0]['price'] > 45200.0   # slippage pushes price up for BUY

    def test_taker_fee_and_slippage(self):
        from src.paper_live.broker import MakerFirstBroker
        b = MakerFirstBroker(max_wait_bars=1, maker_fee=0.0002,
                             taker_fee=0.0004, taker_slippage=0.0003)
        oid = b.place_limit_buy(qty=0.01, limit_price=44000.0)
        b.on_bar('BTCUSDT', _bar(high=45100, low=45000, close=45050))
        b.on_bar('BTCUSDT', _bar(high=45300, low=45150, close=45200))
        fills = b.fills(oid)
        assert len(fills) == 1
        f = fills[0]
        assert f['role'] == 'taker'
        # price includes slippage on buy
        assert abs(f['price'] - 45200.0 * (1 + 0.0003)) < 1e-6
        # fee = taker_fee * notional
        assert abs(f['fee'] - (f['price'] * 0.01 * 0.0004)) < 1e-9


# ===========================================================================
# TEST 4: cancel
# ===========================================================================
class TestCancel:

    def test_cancel_removes_open_order(self):
        from src.paper_live.broker import MakerFirstBroker
        b = MakerFirstBroker()
        oid = b.place_limit_buy(qty=0.01, limit_price=44000.0)
        b.cancel(oid)
        # Even a crossing bar does not fill a cancelled order.
        b.on_bar('BTCUSDT', _bar(high=45000, low=43500, close=44500))
        assert b.fills(oid) == []
        assert b.is_open(oid) is False


# ===========================================================================
# TEST 5: multiple orders
# ===========================================================================
class TestMultipleOrders:

    def test_two_orders_both_fill(self):
        from src.paper_live.broker import MakerFirstBroker
        b = MakerFirstBroker()
        o1 = b.place_limit_buy(qty=0.01, limit_price=45000.0)
        o2 = b.place_limit_buy(qty=0.02, limit_price=44800.0)
        b.on_bar('BTCUSDT', _bar(high=45100, low=44700, close=44900))
        assert len(b.fills(o1)) == 1
        assert len(b.fills(o2)) == 1
        assert b.fills(o2)[0]['qty'] == 0.02
