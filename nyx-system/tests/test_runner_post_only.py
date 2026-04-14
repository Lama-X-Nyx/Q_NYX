"""
TDD Tests — PaperLiveRunner integration with post-only broker + missed log.

Flow we're asserting:
  1. Strategy emits BUY with (limit_price, mark_price) → runner places post-only.
  2. If post-only is rejected, MissedTradeLogger records it + event fires.
  3. If order fills within max_wait, recorded as filled (no missed log).
  4. If order times out, MissedTradeLogger records + order_unfilled event fires.
  5. miss_rate() reported through runner.
"""
from pathlib import Path

import pytest


class _FakeAlerter:
    def __init__(self):
        self.calls = []
    def info(self, t):     self.calls.append(('info', t));     return True
    def warn(self, t):     self.calls.append(('warn', t));     return True
    def error(self, t):    self.calls.append(('error', t));    return True
    def critical(self, t): self.calls.append(('critical', t)); return True
    def send(self, t):     self.calls.append(('send', t));     return True


class _Strategy:
    """Emits one BUY with given limit_price and mark_price on first bar,
    then WAIT forever."""
    def __init__(self, limit_price: float, mark_price: float):
        self.limit_price = limit_price
        self.mark_price = mark_price
        self.fired = False

    def decide(self, pair, bar):
        ar = {k: {'state': '', 'score': 0.0, 'passed': False}
              for k in ('context', 'regime', 'setup', 'entry')}
        if not self.fired:
            self.fired = True
            return {
                'action': 'BUY',
                'orch_score': 0.8,
                'size_factor': 1.0,
                'blocked_by': [],
                'features': {},
                'trade_size': 0.01,
                'trade_direction': 1,
                'trade_entry_price': self.limit_price,
                'trade_stop': self.limit_price * 0.97,
                'trade_tp': self.limit_price * 1.04,
                'order_intent': {
                    'side': 'buy',
                    'qty': 0.01,
                    'limit_price': self.limit_price,
                    'mark_price': self.mark_price,
                },
                'agent_results': ar,
            }
        return {
            'action': 'WAIT', 'orch_score': 0.0, 'size_factor': 0.0,
            'blocked_by': ['wait'], 'features': {},
            'trade_size': 0.0, 'agent_results': ar,
        }


def _bar(ts, high, low, close):
    return {'timestamp': ts, 'open': low, 'high': high,
            'low': low, 'close': close, 'volume': 100.0}


def _build(tmp_path, strategy, fake: _FakeAlerter):
    from src.paper_live.runner import PaperLiveRunner
    from src.paper_live.event_alerter import EventAlerter
    r = PaperLiveRunner(
        strategy=strategy,
        db_path=tmp_path / "decisions.db",
        state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        missed_db_path=tmp_path / "missed.db",
        telegram_token=None, telegram_chat=None,
        max_wait_bars=2,
    )
    r.events = EventAlerter(fake, cooldown_s=0.0)
    return r


# ==============================================================================
class TestPostOnlyRejection:

    def test_rejected_order_logged_to_missed(self, tmp_path):
        """Strategy intent crosses market → post-only rejects → MissedTradeLogger row."""
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        fake = _FakeAlerter()
        # limit=45001, mark=45000 → buy at/above ask → REJECTED.
        strat = _Strategy(limit_price=45001.0, mark_price=45000.0)
        r = _build(tmp_path, strat, fake)

        r.on_bar('BTCUSDT', _bar('2025-01-01T00:00:00+00:00',
                                 high=45010, low=44990, close=45000))
        r.shutdown()

        with MissedTradeLogger(tmp_path / "missed.db") as mtl:
            assert mtl.count_by_state('REJECTED') == 1

    def test_rejected_emits_order_unfilled(self, tmp_path):
        fake = _FakeAlerter()
        strat = _Strategy(limit_price=45001.0, mark_price=45000.0)
        r = _build(tmp_path, strat, fake)
        r.on_bar('BTCUSDT', _bar('2025-01-01T00:00:00+00:00',
                                 high=45010, low=44990, close=45000))
        r.shutdown()
        unfilled = [t for lvl, t in fake.calls
                    if lvl == 'warn' and 'order_unfilled' in t.lower()]
        assert len(unfilled) >= 1


# ==============================================================================
class TestTimeoutMissed:

    def test_timed_out_logged(self, tmp_path):
        """Limit too far from market → times out after max_wait=2 bars."""
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        fake = _FakeAlerter()
        strat = _Strategy(limit_price=40000.0, mark_price=45000.0)  # posts OK
        r = _build(tmp_path, strat, fake)

        # bar 1: BUY intent → posts. Bar trades 44990-45010, does not cross 40000.
        r.on_bar('BTCUSDT', _bar('2025-01-01T00:00:00+00:00',
                                 high=45010, low=44990, close=45000))
        # bar 2: still no fill.
        r.on_bar('BTCUSDT', _bar('2025-01-01T00:15:00+00:00',
                                 high=45020, low=44995, close=45010))
        # bar 3: hits max_wait → TIMED_OUT.
        r.on_bar('BTCUSDT', _bar('2025-01-01T00:30:00+00:00',
                                 high=45030, low=45005, close=45020))
        r.shutdown()

        with MissedTradeLogger(tmp_path / "missed.db") as mtl:
            assert mtl.count_by_state('TIMED_OUT') == 1

    def test_timed_out_emits_order_unfilled(self, tmp_path):
        fake = _FakeAlerter()
        strat = _Strategy(limit_price=40000.0, mark_price=45000.0)
        r = _build(tmp_path, strat, fake)
        for i in range(3):
            r.on_bar('BTCUSDT', _bar(
                f'2025-01-01T00:{i*15:02d}:00+00:00',
                high=45030, low=45005, close=45020))
        r.shutdown()
        unfilled = [t for lvl, t in fake.calls
                    if lvl == 'warn' and 'order_unfilled' in t.lower()]
        assert len(unfilled) >= 1


# ==============================================================================
class TestHappyPathFill:

    def test_order_fills_no_missed_row(self, tmp_path):
        """Bar low crosses limit → fills as maker, no missed log."""
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        fake = _FakeAlerter()
        strat = _Strategy(limit_price=44990.0, mark_price=45000.0)  # valid post-only
        r = _build(tmp_path, strat, fake)

        # First bar: strategy emits BUY, broker posts.
        # Same bar: low=44980 → would cross on NEXT bar ideally.
        # Actually: runner.broker.on_bar is called BEFORE strategy.decide.
        # On bar 1: no orders yet. strategy fires, posts.
        # On bar 2: broker advances → low=44980 crosses 44990 → FILLED.
        r.on_bar('BTCUSDT', _bar('2025-01-01T00:00:00+00:00',
                                 high=45010, low=44995, close=45000))
        r.on_bar('BTCUSDT', _bar('2025-01-01T00:15:00+00:00',
                                 high=45010, low=44980, close=45000))
        r.shutdown()

        with MissedTradeLogger(tmp_path / "missed.db") as mtl:
            assert mtl.count() == 0
