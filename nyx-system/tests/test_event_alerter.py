"""
TDD Tests — EventAlerter

Semantic events with per-event severity + per-event cooldown.

Events (from user spec):
  1. service_down         → critical
  2. reconnect_exchange   → warn
  3. trade_decision       → info (BUY/SELL)
  4. order_unfilled       → warn
  5. api_error            → error
  6. restart              → info (coming back online)

Contract:
  - Each event is a method on EventAlerter.
  - Delegates to an underlying `alerter` (Telegram / Discord / Multi).
  - Messages include a consistent prefix: `[NYX] <event_name>: <details>`.
  - Each event has an independent rate-limit cooldown (e.g. don't spam
    api_error every bar — dedup by message key).
  - Returns True iff the underlying alerter returned True.
"""
import pytest


class _Recorder:
    def __init__(self, return_value: bool = True):
        self.return_value = return_value
        self.calls = []

    def send(self, text):        self.calls.append(('send', text));     return self.return_value
    def info(self, text):        self.calls.append(('info', text));     return self.return_value
    def warn(self, text):        self.calls.append(('warn', text));     return self.return_value
    def error(self, text):       self.calls.append(('error', text));    return self.return_value
    def critical(self, text):    self.calls.append(('critical', text)); return self.return_value


# ===========================================================================
# TEST 1: each event routes to correct severity
# ===========================================================================
class TestEventSeverity:

    def test_service_down_is_critical(self):
        from src.paper_live.event_alerter import EventAlerter
        rec = _Recorder()
        ea = EventAlerter(rec)
        ea.service_down("heartbeat stale 600s")
        assert rec.calls[0][0] == 'critical'
        assert 'service_down' in rec.calls[0][1].lower() or 'down' in rec.calls[0][1].lower()

    def test_reconnect_exchange_is_warn(self):
        from src.paper_live.event_alerter import EventAlerter
        rec = _Recorder()
        ea = EventAlerter(rec)
        ea.reconnect_exchange("Binance WS reconnected after 3 retries")
        assert rec.calls[0][0] == 'warn'

    def test_trade_decision_is_info(self):
        from src.paper_live.event_alerter import EventAlerter
        rec = _Recorder()
        ea = EventAlerter(rec)
        ea.trade_decision(pair="BTCUSDT", action="BUY",
                          price=45000.0, score=0.72, size=0.01)
        assert rec.calls[0][0] == 'info'
        text = rec.calls[0][1]
        assert "BUY" in text
        assert "BTCUSDT" in text
        assert "45000" in text or "45,000" in text

    def test_order_unfilled_is_warn(self):
        from src.paper_live.event_alerter import EventAlerter
        rec = _Recorder()
        ea = EventAlerter(rec)
        ea.order_unfilled(oid=42, pair="BTCUSDT", side="buy",
                          price=44800.0, waited_bars=4)
        assert rec.calls[0][0] == 'warn'
        assert "44800" in rec.calls[0][1] or "44,800" in rec.calls[0][1]

    def test_api_error_is_error(self):
        from src.paper_live.event_alerter import EventAlerter
        rec = _Recorder()
        ea = EventAlerter(rec)
        ea.api_error("Binance 418 rate limited")
        assert rec.calls[0][0] == 'error'
        assert 'Binance' in rec.calls[0][1]

    def test_restart_is_info(self):
        from src.paper_live.event_alerter import EventAlerter
        rec = _Recorder()
        ea = EventAlerter(rec)
        ea.restart(version="v0.3.2", last_ts="2025-01-01T00:00:00+00:00")
        assert rec.calls[0][0] == 'info'
        assert 'v0.3.2' in rec.calls[0][1]


# ===========================================================================
# TEST 2: message format
# ===========================================================================
class TestMessageFormat:

    def test_messages_start_with_nyx_prefix(self):
        from src.paper_live.event_alerter import EventAlerter
        rec = _Recorder()
        ea = EventAlerter(rec)
        ea.service_down("x")
        ea.reconnect_exchange("x")
        ea.trade_decision(pair="X", action="BUY", price=1.0, score=0.5, size=0.1)
        ea.api_error("x")
        ea.restart(version="v", last_ts="t")
        ea.order_unfilled(oid=1, pair="X", side="buy", price=1.0, waited_bars=2)
        for _method, text in rec.calls:
            assert text.startswith("[NYX]"), f"bad prefix in: {text!r}"


# ===========================================================================
# TEST 3: per-event cooldown (prevent spam)
# ===========================================================================
class TestCooldown:

    def test_api_error_dedups_same_message(self):
        """Two back-to-back identical api_error calls → only 1 alert."""
        from src.paper_live.event_alerter import EventAlerter
        rec = _Recorder()
        ea = EventAlerter(rec, cooldown_s=60.0)
        ea.api_error("Binance 418 rate limited")
        ea.api_error("Binance 418 rate limited")
        assert len(rec.calls) == 1

    def test_different_api_errors_both_sent(self):
        from src.paper_live.event_alerter import EventAlerter
        rec = _Recorder()
        ea = EventAlerter(rec, cooldown_s=60.0)
        ea.api_error("A error")
        ea.api_error("B error")
        assert len(rec.calls) == 2

    def test_trade_decision_not_deduped_even_if_similar(self):
        """Each new trade must always alert — no silent drops."""
        from src.paper_live.event_alerter import EventAlerter
        rec = _Recorder()
        ea = EventAlerter(rec, cooldown_s=60.0)
        ea.trade_decision(pair="BTC", action="BUY", price=1.0, score=0.5, size=0.1)
        ea.trade_decision(pair="BTC", action="BUY", price=1.0, score=0.5, size=0.1)
        assert len(rec.calls) == 2


# ===========================================================================
# TEST 4: return value plumbing
# ===========================================================================
class TestReturnValue:

    def test_returns_true_when_underlying_alerter_succeeds(self):
        from src.paper_live.event_alerter import EventAlerter
        ea = EventAlerter(_Recorder(return_value=True))
        assert ea.service_down("x") is True

    def test_returns_false_when_underlying_alerter_fails(self):
        from src.paper_live.event_alerter import EventAlerter
        ea = EventAlerter(_Recorder(return_value=False))
        assert ea.service_down("x") is False
