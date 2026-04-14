"""
TDD Tests — PostOnlyPaperBroker

Honest maker-first simulator:
  - No silent taker fallback.
  - Post-only: limit that would cross immediately is REJECTED.
  - Timeout: order that doesn't fill within max_wait_bars is TIMED_OUT
    (= missed trade).
  - Each order carries a state machine:
      NEW → POSTED → FILLED / CANCELLED / TIMED_OUT / REJECTED.

Events:
  - on_bar advances POSTED orders; fills as maker when price crosses.
  - on_timeout event is observable (caller logs / alerts).
"""
import pytest


# ------- helpers --------------------------------------------------------------

def _bar(high: float, low: float, close: float) -> dict:
    return {'open': low, 'high': high, 'low': low, 'close': close, 'volume': 100.0}


def _ts(h: int) -> str:
    return f"2025-01-01T{h:02d}:00:00+00:00"


# ==============================================================================
# TEST 1: post-only rejection semantics
# ==============================================================================
class TestPostOnlyRejection:

    def test_buy_rejected_if_limit_at_or_above_mark(self):
        """Buy limit = mark → would cross the ask → post-only reject."""
        from src.paper_live.post_only_broker import PostOnlyPaperBroker, OrderState
        b = PostOnlyPaperBroker()
        oid = b.place_post_only(
            pair='BTCUSDT', side='buy', qty=0.01,
            limit_price=45000.0, mark_price=44999.0,
            placed_at=_ts(0),
        )
        o = b.get(oid)
        assert o.state == OrderState.REJECTED
        assert 'post_only' in (o.reject_reason or '').lower() or \
               'cross' in (o.reject_reason or '').lower()

    def test_sell_rejected_if_limit_at_or_below_mark(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker, OrderState
        b = PostOnlyPaperBroker()
        oid = b.place_post_only(
            pair='BTCUSDT', side='sell', qty=0.01,
            limit_price=45000.0, mark_price=45001.0,
            placed_at=_ts(0),
        )
        o = b.get(oid)
        assert o.state == OrderState.REJECTED

    def test_buy_posted_if_limit_below_mark(self):
        """Buy limit = 44900, market = 45000 → posts as maker."""
        from src.paper_live.post_only_broker import PostOnlyPaperBroker, OrderState
        b = PostOnlyPaperBroker()
        oid = b.place_post_only(
            pair='BTCUSDT', side='buy', qty=0.01,
            limit_price=44900.0, mark_price=45000.0,
            placed_at=_ts(0),
        )
        assert b.get(oid).state == OrderState.POSTED

    def test_sell_posted_if_limit_above_mark(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker, OrderState
        b = PostOnlyPaperBroker()
        oid = b.place_post_only(
            pair='BTCUSDT', side='sell', qty=0.01,
            limit_price=45200.0, mark_price=45000.0,
            placed_at=_ts(0),
        )
        assert b.get(oid).state == OrderState.POSTED


# ==============================================================================
# TEST 2: maker fill when bar crosses
# ==============================================================================
class TestMakerFill:

    def test_buy_fills_as_maker_when_low_crosses(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker, OrderState
        b = PostOnlyPaperBroker()
        oid = b.place_post_only(
            pair='BTCUSDT', side='buy', qty=0.01,
            limit_price=44900.0, mark_price=45000.0, placed_at=_ts(0),
        )
        b.on_bar('BTCUSDT', _bar(high=45100, low=44850, close=45050), bar_ts=_ts(1))
        o = b.get(oid)
        assert o.state == OrderState.FILLED
        assert len(o.fills) == 1
        assert o.fills[0]['role'] == 'maker'
        assert o.fills[0]['price'] == 44900.0

    def test_sell_fills_as_maker_when_high_crosses(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker, OrderState
        b = PostOnlyPaperBroker()
        oid = b.place_post_only(
            pair='BTCUSDT', side='sell', qty=0.01,
            limit_price=45200.0, mark_price=45000.0, placed_at=_ts(0),
        )
        b.on_bar('BTCUSDT', _bar(high=45300, low=44950, close=45100), bar_ts=_ts(1))
        o = b.get(oid)
        assert o.state == OrderState.FILLED

    def test_maker_fee_applied(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker
        b = PostOnlyPaperBroker(maker_fee=0.0002)
        oid = b.place_post_only(
            pair='BTCUSDT', side='buy', qty=0.01,
            limit_price=44900.0, mark_price=45000.0, placed_at=_ts(0),
        )
        b.on_bar('BTCUSDT', _bar(high=45100, low=44850, close=45050), bar_ts=_ts(1))
        fill = b.get(oid).fills[0]
        assert abs(fill['fee'] - 44900.0 * 0.01 * 0.0002) < 1e-9


# ==============================================================================
# TEST 3: timeout = missed trade (NO taker fallback)
# ==============================================================================
class TestTimeout:

    def test_times_out_after_max_wait_bars(self):
        """Order that never crosses must become TIMED_OUT (not filled)."""
        from src.paper_live.post_only_broker import PostOnlyPaperBroker, OrderState
        b = PostOnlyPaperBroker(max_wait_bars=3)
        oid = b.place_post_only(
            pair='BTCUSDT', side='buy', qty=0.01,
            limit_price=40000.0, mark_price=45000.0, placed_at=_ts(0),
        )
        for i in range(4):
            b.on_bar('BTCUSDT', _bar(high=45300, low=45100, close=45200),
                     bar_ts=_ts(i + 1))
        assert b.get(oid).state == OrderState.TIMED_OUT
        assert b.get(oid).fills == []

    def test_timed_out_has_timeout_reason(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker
        b = PostOnlyPaperBroker(max_wait_bars=1)
        oid = b.place_post_only(
            pair='BTCUSDT', side='buy', qty=0.01,
            limit_price=40000.0, mark_price=45000.0, placed_at=_ts(0),
        )
        b.on_bar('BTCUSDT', _bar(high=45300, low=45100, close=45200), bar_ts=_ts(1))
        b.on_bar('BTCUSDT', _bar(high=45300, low=45100, close=45200), bar_ts=_ts(2))
        o = b.get(oid)
        assert o.timeout_reason is not None
        assert 'max_wait' in o.timeout_reason.lower() or \
               'timeout' in o.timeout_reason.lower()

    def test_no_taker_fallback(self):
        """Crucially: after timeout, order must NOT have a taker fill."""
        from src.paper_live.post_only_broker import PostOnlyPaperBroker, OrderState
        b = PostOnlyPaperBroker(max_wait_bars=2)
        oid = b.place_post_only(
            pair='BTCUSDT', side='buy', qty=0.01,
            limit_price=40000.0, mark_price=45000.0, placed_at=_ts(0),
        )
        for i in range(5):
            b.on_bar('BTCUSDT', _bar(high=45300, low=45100, close=45200),
                     bar_ts=_ts(i + 1))
        o = b.get(oid)
        assert o.state == OrderState.TIMED_OUT
        assert not any(f.get('role') == 'taker' for f in o.fills)


# ==============================================================================
# TEST 4: cancel
# ==============================================================================
class TestCancel:

    def test_cancel_posted_order(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker, OrderState
        b = PostOnlyPaperBroker()
        oid = b.place_post_only(
            pair='BTCUSDT', side='buy', qty=0.01,
            limit_price=44900.0, mark_price=45000.0, placed_at=_ts(0),
        )
        b.cancel(oid)
        assert b.get(oid).state == OrderState.CANCELLED

    def test_cancelled_does_not_fill_on_next_bar(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker, OrderState
        b = PostOnlyPaperBroker()
        oid = b.place_post_only(
            pair='BTCUSDT', side='buy', qty=0.01,
            limit_price=44900.0, mark_price=45000.0, placed_at=_ts(0),
        )
        b.cancel(oid)
        b.on_bar('BTCUSDT', _bar(high=45100, low=44850, close=44950), bar_ts=_ts(1))
        o = b.get(oid)
        assert o.state == OrderState.CANCELLED
        assert o.fills == []


# ==============================================================================
# TEST 5: queries
# ==============================================================================
class TestQueries:

    def test_missed_trades_returns_timed_out_orders(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker
        b = PostOnlyPaperBroker(max_wait_bars=1)
        # one will fill
        a = b.place_post_only('X', 'buy', 0.01, 44900.0, 45000.0, placed_at=_ts(0))
        # one will miss
        c = b.place_post_only('X', 'buy', 0.01, 40000.0, 45000.0, placed_at=_ts(0))
        b.on_bar('X', _bar(high=45100, low=44850, close=45050), bar_ts=_ts(1))
        b.on_bar('X', _bar(high=45300, low=45200, close=45250), bar_ts=_ts(2))
        missed = b.missed_trades()
        assert [o.oid for o in missed] == [c]

    def test_filled_orders_returns_filled(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker
        b = PostOnlyPaperBroker(max_wait_bars=1)
        a = b.place_post_only('X', 'buy', 0.01, 44900.0, 45000.0, placed_at=_ts(0))
        c = b.place_post_only('X', 'buy', 0.01, 40000.0, 45000.0, placed_at=_ts(0))
        b.on_bar('X', _bar(high=45100, low=44850, close=45050), bar_ts=_ts(1))
        b.on_bar('X', _bar(high=45300, low=45200, close=45250), bar_ts=_ts(2))
        assert [o.oid for o in b.filled_orders()] == [a]

    def test_rejected_orders(self):
        from src.paper_live.post_only_broker import PostOnlyPaperBroker
        b = PostOnlyPaperBroker()
        r = b.place_post_only('X', 'buy', 0.01, 45001.0, 45000.0, placed_at=_ts(0))
        p = b.place_post_only('X', 'buy', 0.01, 44900.0, 45000.0, placed_at=_ts(0))
        assert [o.oid for o in b.rejected_orders()] == [r]


# ==============================================================================
# TEST 6: miss rate metric
# ==============================================================================
class TestMissRate:

    def test_miss_rate_computation(self):
        """miss_rate = TIMED_OUT / (TIMED_OUT + FILLED). Rejected excluded."""
        from src.paper_live.post_only_broker import PostOnlyPaperBroker
        b = PostOnlyPaperBroker(max_wait_bars=1)
        # 3 will fill, 2 will miss
        for _ in range(3):
            b.place_post_only('X', 'buy', 0.01, 44900.0, 45000.0, placed_at=_ts(0))
        for _ in range(2):
            b.place_post_only('X', 'buy', 0.01, 40000.0, 45000.0, placed_at=_ts(0))
        b.on_bar('X', _bar(high=45100, low=44850, close=45050), bar_ts=_ts(1))
        b.on_bar('X', _bar(high=45300, low=45200, close=45250), bar_ts=_ts(2))
        assert abs(b.miss_rate() - 2/5) < 1e-9
