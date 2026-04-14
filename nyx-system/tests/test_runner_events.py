"""
TDD Tests — PaperLiveRunner integration with EventAlerter.

The runner must emit semantic events through `self.events` at the right
moments:
  - restart         : on construction (always, even fresh state)
  - trade_decision  : every BUY / SELL decision
  - api_error       : when strategy.decide() raises
  - order_unfilled  : (checked via helper; tested in broker integration)
  - service_down    : check helper (caller calls it when stale)
  - reconnect_exchange : helper present

The implementation uses a fake alerter to capture calls.
"""
from pathlib import Path

import pytest


class _StubStrategy:
    """Alternates WAIT / BUY / ERROR."""
    def __init__(self, mode='normal'):
        self.mode = mode
        self.i = 0

    def decide(self, pair, bar):
        self.i += 1
        if self.mode == 'crash':
            raise RuntimeError("boom")
        if self.i % 2 == 0:
            return {
                'action': 'BUY',
                'orch_score': 0.72,
                'size_factor': 1.0,
                'blocked_by': [],
                'features': {},
                'trade_size': 0.01,
                'agent_results': {
                    'context': {'state': 'bullish', 'score': 0.8, 'passed': True},
                    'regime':  {'state': 'trend_plus', 'score': 0.7, 'passed': True},
                    'setup':   {'state': 'valid_setup', 'score': 0.65, 'passed': True},
                    'entry':   {'state': 'ready', 'score': 0.72, 'passed': True,
                                'direction': 1, 'p_up': 0.6, 'p_down': 0.2},
                },
            }
        return {
            'action': 'WAIT',
            'orch_score': 0.3,
            'size_factor': 0.0,
            'blocked_by': ['regime'],
            'features': {},
            'agent_results': {
                k: {'state': '', 'score': 0.0, 'passed': False}
                for k in ('context', 'regime', 'setup', 'entry')
            },
        }


class _FakeAlerter:
    """Recording alerter capturing every underlying call."""
    def __init__(self):
        self.calls = []

    def info(self, text):     self.calls.append(('info', text));     return True
    def warn(self, text):     self.calls.append(('warn', text));     return True
    def error(self, text):    self.calls.append(('error', text));    return True
    def critical(self, text): self.calls.append(('critical', text)); return True
    def send(self, text):     self.calls.append(('send', text));     return True


def _bar(ts="2025-01-01T00:00:00+00:00", price=45000.0):
    return {
        'timestamp': ts, 'open': price, 'high': price*1.001,
        'low': price*0.999, 'close': price*1.0005, 'volume': 100.0,
    }


def _build_runner(tmp_path, fake: _FakeAlerter, mode='normal'):
    from src.paper_live.runner import PaperLiveRunner
    from src.paper_live.event_alerter import EventAlerter
    runner = PaperLiveRunner(
        strategy=_StubStrategy(mode=mode),
        db_path=tmp_path / "decisions.db",
        state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        telegram_token=None, telegram_chat=None,
    )
    # Replace the auto-built events alerter with one wired to our fake.
    runner.events = EventAlerter(fake, cooldown_s=0.0)
    return runner


# ===========================================================================
class TestRestartEvent:

    def test_emit_restart_on_startup(self, tmp_path):
        from src.paper_live.runner import PaperLiveRunner
        from src.paper_live.event_alerter import EventAlerter
        fake = _FakeAlerter()
        r = PaperLiveRunner(
            strategy=_StubStrategy(),
            db_path=tmp_path / "decisions.db",
            state_path=tmp_path / "state.json",
            heartbeat_path=tmp_path / "hb.json",
            telegram_token=None, telegram_chat=None,
        )
        # Replace events + emit restart now (mirrors main.py flow).
        r.events = EventAlerter(fake)
        r.notify_restart()
        r.shutdown()
        assert any('restart' in t.lower() for _lvl, t in fake.calls)
        # Restart is 'info' level.
        assert any(lvl == 'info' and 'restart' in t.lower() for lvl, t in fake.calls)


# ===========================================================================
class TestTradeDecisionEvent:

    def test_buy_decision_emits_trade_event(self, tmp_path):
        fake = _FakeAlerter()
        r = _build_runner(tmp_path, fake)
        # First call is WAIT (no alert), second is BUY (alert).
        r.on_bar('BTCUSDT', _bar(ts="2025-01-01T00:00:00+00:00"))
        r.on_bar('BTCUSDT', _bar(ts="2025-01-01T00:15:00+00:00", price=45010))
        r.shutdown()

        trade_alerts = [t for lvl, t in fake.calls
                        if lvl == 'info' and 'trade_decision' in t.lower()]
        assert len(trade_alerts) == 1
        assert 'BUY' in trade_alerts[0]
        assert 'BTCUSDT' in trade_alerts[0]

    def test_wait_does_not_emit_trade_event(self, tmp_path):
        fake = _FakeAlerter()
        r = _build_runner(tmp_path, fake)
        r.on_bar('BTCUSDT', _bar(ts="2025-01-01T00:00:00+00:00"))
        r.shutdown()
        trade_alerts = [t for lvl, t in fake.calls
                        if 'trade_decision' in t.lower()]
        assert trade_alerts == []


# ===========================================================================
class TestApiErrorEvent:

    def test_strategy_crash_emits_api_error(self, tmp_path):
        fake = _FakeAlerter()
        r = _build_runner(tmp_path, fake, mode='crash')
        with pytest.raises(RuntimeError):
            r.on_bar('BTCUSDT', _bar())
        r.shutdown()
        err_alerts = [t for lvl, t in fake.calls if lvl == 'error']
        assert len(err_alerts) >= 1
        assert any('api_error' in t.lower() or 'strategy' in t.lower()
                   for t in err_alerts)


# ===========================================================================
class TestServiceDownHelper:

    def test_notify_service_down_is_critical(self, tmp_path):
        fake = _FakeAlerter()
        r = _build_runner(tmp_path, fake)
        r.notify_service_down("heartbeat stale 600s")
        r.shutdown()
        assert any(lvl == 'critical' and 'service_down' in t.lower()
                   for lvl, t in fake.calls)


# ===========================================================================
class TestReconnectHelper:

    def test_notify_reconnect_is_warn(self, tmp_path):
        fake = _FakeAlerter()
        r = _build_runner(tmp_path, fake)
        r.notify_reconnect_exchange("Binance WS came back after 2 retries")
        r.shutdown()
        assert any(lvl == 'warn' and 'reconnect_exchange' in t.lower()
                   for lvl, t in fake.calls)


# ===========================================================================
class TestOrderUnfilledIntegration:

    def test_runner_emits_unfilled_when_order_ages_out(self, tmp_path):
        """After max_wait+1 bars with an open order, broker converts to taker;
        runner must emit an order_unfilled event."""
        fake = _FakeAlerter()
        r = _build_runner(tmp_path, fake)
        # Place an order well below market that will NEVER fill as maker.
        oid = r.broker.place_limit_buy(qty=0.01, limit_price=1.0)
        for i in range(5):
            # Bars keep price way above the 1.0 limit.
            r.on_bar('BTCUSDT', _bar(
                ts=f"2025-01-01T{i:02d}:00:00+00:00",
                price=45000.0 + i,
            ))
        r.shutdown()

        unfilled = [t for lvl, t in fake.calls
                    if lvl == 'warn' and 'order_unfilled' in t.lower()]
        # We sent 5 bars, max_wait defaults to 3 → order converted to taker
        # on bar #4. Runner must have alerted once.
        assert len(unfilled) >= 1
