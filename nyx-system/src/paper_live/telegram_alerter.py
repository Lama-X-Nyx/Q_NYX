"""
TelegramAlerter — minimal notification layer with rate limiting.

All network I/O goes through an injected `http_post` callable so unit tests
never talk to the real Telegram API.

If you pass `http_post=None`, a default `requests.post`-style wrapper is used.
Missing token/chat_id → alerter is a no-op (useful for local dev).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional


def _default_http_post(url: str, data: Dict, timeout: float) -> Dict:
    import urllib.parse
    import urllib.request

    encoded = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=encoded)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {'ok': 200 <= resp.status < 300}


class TelegramAlerter:
    """Telegram bot notifications with simple cooldown."""

    def __init__(
        self,
        token: Optional[str],
        chat_id: Optional[str],
        http_post: Optional[Callable] = None,
        cooldown_s: float = 0.0,
        timeout: float = 5.0,
    ):
        self.token = token
        self.chat_id = chat_id
        self.http_post = http_post or _default_http_post
        self.cooldown_s = float(cooldown_s)
        self.timeout = float(timeout)
        self._last_send: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------
    def send(self, text: str) -> bool:
        if not self.token or not self.chat_id:
            return False

        # Cooldown: skip identical message within cooldown window.
        now = time.time()
        last = self._last_send.get(text, 0.0)
        if self.cooldown_s > 0 and (now - last) < self.cooldown_s:
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {'chat_id': self.chat_id, 'text': text}
        try:
            self.http_post(url, data=data, timeout=self.timeout)
        except Exception:
            return False

        self._last_send[text] = now
        return True

    # ------------------------------------------------------------------
    # Level helpers
    # ------------------------------------------------------------------
    def info(self, text: str) -> bool:
        return self.send(f"ℹ️ INFO — {text}")

    def warn(self, text: str) -> bool:
        return self.send(f"⚠️ WARN — {text}")

    def error(self, text: str) -> bool:
        return self.send(f"🚨 ERROR — {text}")

    def critical(self, text: str) -> bool:
        return self.send(f"🚨🚨 CRITICAL — {text}")
