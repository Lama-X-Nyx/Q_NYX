"""
MultiAlerter — fan-out a single alert to N backends.

Failure isolation: a backend that raises never prevents other backends
from being called. The return value is True if ANY backend returned True.
"""
from __future__ import annotations

import logging
from typing import List, Protocol


log = logging.getLogger(__name__)


class _AlerterLike(Protocol):
    def send(self, text: str) -> bool: ...
    def info(self, text: str) -> bool: ...
    def warn(self, text: str) -> bool: ...
    def error(self, text: str) -> bool: ...
    def critical(self, text: str) -> bool: ...


class MultiAlerter:
    """Broadcast alerts to every underlying alerter."""

    def __init__(self, alerters: List[_AlerterLike]):
        self.alerters = list(alerters)

    # ---- dispatch helper --------------------------------------------------
    def _dispatch(self, method: str, text: str) -> bool:
        any_ok = False
        for a in self.alerters:
            try:
                fn = getattr(a, method)
                if fn(text):
                    any_ok = True
            except Exception as e:
                log.warning("alerter %r .%s raised: %s", a, method, e)
        return any_ok

    # ---- public API -------------------------------------------------------
    def send(self, text: str) -> bool:
        return self._dispatch('send', text)

    def info(self, text: str) -> bool:
        return self._dispatch('info', text)

    def warn(self, text: str) -> bool:
        return self._dispatch('warn', text)

    def error(self, text: str) -> bool:
        return self._dispatch('error', text)

    def critical(self, text: str) -> bool:
        return self._dispatch('critical', text)
