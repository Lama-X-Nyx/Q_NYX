"""
EventAlerter — semantic events routed to the right severity.

Wraps any alerter (TelegramAlerter / DiscordAlerter / MultiAlerter).
Six events, each with its own severity + cooldown-key.

+-----------------------+------------+--------------------------------------+
| event                 | severity   | rationale                            |
+-----------------------+------------+--------------------------------------+
| service_down          | critical   | heartbeat stale / watchdog trip      |
| reconnect_exchange    | warn       | WS dropped + came back               |
| trade_decision        | info       | BUY/SELL actually taken              |
| order_unfilled        | warn       | limit aged out / partial             |
| api_error             | error      | HTTP 4xx/5xx from exchange           |
| restart               | info       | process started, model v, last_ts    |
+-----------------------+------------+--------------------------------------+

Cooldown: identical event messages within `cooldown_s` are suppressed —
except for `trade_decision`, which must NEVER be deduped (every trade
must produce an alert).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Protocol


class _AlerterLike(Protocol):
    def info(self, text: str) -> bool: ...
    def warn(self, text: str) -> bool: ...
    def error(self, text: str) -> bool: ...
    def critical(self, text: str) -> bool: ...


_PREFIX = "[NYX]"


class EventAlerter:
    """Semantic event → (severity, message) → underlying alerter."""

    # Events that must NEVER be rate-limited (single-trade visibility).
    _NEVER_DEDUP = frozenset({"trade_decision"})

    def __init__(self, alerter: _AlerterLike, cooldown_s: float = 0.0):
        self.alerter = alerter
        self.cooldown_s = float(cooldown_s)
        self._last_seen: Dict[str, float] = {}

    # ---------------------------------------------------------------------
    # internal dispatch
    # ---------------------------------------------------------------------
    def _emit(self, event: str, severity: str, text: str) -> bool:
        # Dedup key = (event, text). Some events opt out.
        if self.cooldown_s > 0 and event not in self._NEVER_DEDUP:
            key = f"{event}|{text}"
            now = time.time()
            last = self._last_seen.get(key, 0.0)
            if (now - last) < self.cooldown_s:
                return False
            self._last_seen[key] = now

        full = f"{_PREFIX} {event}: {text}"
        fn = getattr(self.alerter, severity)
        return bool(fn(full))

    # ---------------------------------------------------------------------
    # public events
    # ---------------------------------------------------------------------
    def service_down(self, reason: str) -> bool:
        return self._emit("service_down", "critical", reason)

    def reconnect_exchange(self, reason: str) -> bool:
        return self._emit("reconnect_exchange", "warn", reason)

    def trade_decision(
        self,
        pair: str,
        action: str,
        price: float,
        score: float,
        size: float,
    ) -> bool:
        msg = (
            f"{action} {pair} @ {price:,.2f} "
            f"score={score:.2f} size={size:g}"
        )
        return self._emit("trade_decision", "info", msg)

    def order_unfilled(
        self,
        oid: int,
        pair: str,
        side: str,
        price: float,
        waited_bars: int,
    ) -> bool:
        msg = (
            f"oid={oid} {side.upper()} {pair} limit={price:,.2f} "
            f"waited={waited_bars} bars"
        )
        return self._emit("order_unfilled", "warn", msg)

    def api_error(self, details: str) -> bool:
        return self._emit("api_error", "error", details)

    def restart(self, version: str, last_ts: str) -> bool:
        return self._emit("restart", "info",
                          f"model={version} last_ts={last_ts}")
