"""
DiscordAlerter — send notifications to a Discord webhook URL.

Same semantic contract as TelegramAlerter (send/info/warn/error/critical,
cooldown dedup, injectable http_post, no-op without URL).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional


def _default_http_post(url: str, data=None, json=None, timeout: float = 5.0) -> Dict:
    import urllib.parse
    import urllib.request
    if json is not None:
        import json as _json
        body = _json.dumps(json).encode()
        headers = {'Content-Type': 'application/json'}
    else:
        body = urllib.parse.urlencode(data or {}).encode()
        headers = {}
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {'ok': 200 <= resp.status < 300}


class DiscordAlerter:
    """Discord webhook notifier with cooldown."""

    def __init__(
        self,
        webhook_url: Optional[str],
        http_post: Optional[Callable] = None,
        cooldown_s: float = 0.0,
        timeout: float = 5.0,
    ):
        self.webhook_url = webhook_url
        self.http_post = http_post or _default_http_post
        self.cooldown_s = float(cooldown_s)
        self.timeout = float(timeout)
        self._last_send: Dict[str, float] = {}

    # -- core --------------------------------------------------------------

    def send(self, text: str) -> bool:
        if not self.webhook_url:
            return False

        now = time.time()
        if self.cooldown_s > 0:
            last = self._last_send.get(text, 0.0)
            if (now - last) < self.cooldown_s:
                return False

        try:
            self.http_post(
                self.webhook_url,
                json={'content': text},
                timeout=self.timeout,
            )
        except Exception:
            return False

        self._last_send[text] = now
        return True

    # -- levels ------------------------------------------------------------

    def info(self, text: str) -> bool:
        return self.send(f"ℹ️ INFO — {text}")

    def warn(self, text: str) -> bool:
        return self.send(f"⚠️ WARN — {text}")

    def error(self, text: str) -> bool:
        return self.send(f"🚨 ERROR — {text}")

    def critical(self, text: str) -> bool:
        return self.send(f"🚨🚨 CRITICAL — {text}")
