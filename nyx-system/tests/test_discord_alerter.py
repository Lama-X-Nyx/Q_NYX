"""
TDD Tests — DiscordAlerter

Mirror of TelegramAlerter but for a Discord webhook URL.

Contract:
  - send(text) POSTs `{"content": text}` to the webhook URL.
  - Missing webhook URL → no-op returning False.
  - Network errors → False (caller never crashes).
  - Cooldown: identical messages within cooldown_s are suppressed.
  - Levels info/warn/error/critical prefix the message.
"""
import pytest


class _FakeHTTP:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    def __call__(self, url, data=None, json=None, timeout=None):
        self.calls.append({'url': url, 'data': data, 'json': json, 'timeout': timeout})
        if self.fail:
            raise RuntimeError("simulated network error")
        return {'ok': True}


class TestBasic:

    def test_send_posts_to_webhook(self):
        from src.paper_live.discord_alerter import DiscordAlerter
        http = _FakeHTTP()
        a = DiscordAlerter(webhook_url="https://discord/hook/x", http_post=http)
        ok = a.send("hello")
        assert ok is True
        assert len(http.calls) == 1
        assert http.calls[0]['url'] == "https://discord/hook/x"
        payload = http.calls[0].get('json') or http.calls[0].get('data')
        assert payload['content'] == "hello"

    def test_disabled_without_url(self):
        from src.paper_live.discord_alerter import DiscordAlerter
        http = _FakeHTTP()
        a = DiscordAlerter(webhook_url=None, http_post=http)
        assert a.send("hello") is False
        assert http.calls == []

    def test_network_error_returns_false(self):
        from src.paper_live.discord_alerter import DiscordAlerter
        http = _FakeHTTP(fail=True)
        a = DiscordAlerter(webhook_url="https://discord/hook/x", http_post=http)
        assert a.send("hello") is False


class TestLevels:

    def test_warn_prefix(self):
        from src.paper_live.discord_alerter import DiscordAlerter
        http = _FakeHTTP()
        a = DiscordAlerter(webhook_url="x", http_post=http)
        a.warn("high dd")
        text = (http.calls[0].get('json') or http.calls[0].get('data'))['content']
        assert 'WARN' in text.upper() or '⚠' in text

    def test_error_prefix(self):
        from src.paper_live.discord_alerter import DiscordAlerter
        http = _FakeHTTP()
        a = DiscordAlerter(webhook_url="x", http_post=http)
        a.error("boom")
        text = (http.calls[0].get('json') or http.calls[0].get('data'))['content']
        assert 'ERROR' in text.upper() or 'CRITICAL' in text.upper() or '🚨' in text


class TestRateLimit:

    def test_duplicate_suppressed_within_cooldown(self):
        from src.paper_live.discord_alerter import DiscordAlerter
        http = _FakeHTTP()
        a = DiscordAlerter(webhook_url="x", http_post=http, cooldown_s=60)
        a.send("dup")
        a.send("dup")
        assert len(http.calls) == 1

    def test_distinct_messages_not_suppressed(self):
        from src.paper_live.discord_alerter import DiscordAlerter
        http = _FakeHTTP()
        a = DiscordAlerter(webhook_url="x", http_post=http, cooldown_s=60)
        a.send("a")
        a.send("b")
        assert len(http.calls) == 2
