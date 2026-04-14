"""
TDD Tests — TelegramAlerter

Contract:
  - send(text) calls Telegram Bot API via injected `http_post` callable.
  - Rate limited: identical messages within `cooldown_s` are suppressed.
  - Robust: network errors do not crash the caller (logged + returned as False).
  - Disabled mode: if no token, send() is a no-op returning False.
  - Levels (INFO, WARN, ERROR, CRITICAL) prefix the message.
"""
import pytest


class _FakeHTTP:
    """Injected HTTP double."""
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list = []

    def __call__(self, url, data=None, timeout=None):
        self.calls.append({'url': url, 'data': data, 'timeout': timeout})
        if self.fail:
            raise RuntimeError("simulated network error")
        return {'ok': True}


# ===========================================================================
# TEST 1: basic send
# ===========================================================================
class TestBasic:

    def test_send_calls_http(self):
        from src.paper_live.telegram_alerter import TelegramAlerter
        http = _FakeHTTP()
        a = TelegramAlerter(token="t", chat_id="c", http_post=http)
        ok = a.send("hello")
        assert ok is True
        assert len(http.calls) == 1
        assert "api.telegram.org" in http.calls[0]['url']
        assert http.calls[0]['data']['chat_id'] == "c"
        assert "hello" in http.calls[0]['data']['text']

    def test_disabled_without_token(self):
        from src.paper_live.telegram_alerter import TelegramAlerter
        http = _FakeHTTP()
        a = TelegramAlerter(token=None, chat_id=None, http_post=http)
        ok = a.send("hello")
        assert ok is False
        assert http.calls == []

    def test_network_error_returns_false(self):
        from src.paper_live.telegram_alerter import TelegramAlerter
        http = _FakeHTTP(fail=True)
        a = TelegramAlerter(token="t", chat_id="c", http_post=http)
        ok = a.send("hello")
        assert ok is False


# ===========================================================================
# TEST 2: levels
# ===========================================================================
class TestLevels:

    def test_warn_prefixes_text(self):
        from src.paper_live.telegram_alerter import TelegramAlerter
        http = _FakeHTTP()
        a = TelegramAlerter(token="t", chat_id="c", http_post=http)
        a.warn("high drawdown")
        text = http.calls[0]['data']['text']
        assert 'WARN' in text.upper() or '⚠' in text

    def test_error_prefixes_text(self):
        from src.paper_live.telegram_alerter import TelegramAlerter
        http = _FakeHTTP()
        a = TelegramAlerter(token="t", chat_id="c", http_post=http)
        a.error("broker disconnected")
        text = http.calls[0]['data']['text']
        assert 'ERROR' in text.upper() or 'CRITICAL' in text.upper() or '🚨' in text


# ===========================================================================
# TEST 3: rate limiting
# ===========================================================================
class TestRateLimit:

    def test_duplicate_within_cooldown_suppressed(self):
        from src.paper_live.telegram_alerter import TelegramAlerter
        http = _FakeHTTP()
        a = TelegramAlerter(token="t", chat_id="c", http_post=http, cooldown_s=60)
        a.send("same message")
        a.send("same message")
        assert len(http.calls) == 1

    def test_different_messages_not_suppressed(self):
        from src.paper_live.telegram_alerter import TelegramAlerter
        http = _FakeHTTP()
        a = TelegramAlerter(token="t", chat_id="c", http_post=http, cooldown_s=60)
        a.send("msg A")
        a.send("msg B")
        assert len(http.calls) == 2
