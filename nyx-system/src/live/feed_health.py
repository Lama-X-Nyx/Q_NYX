"""
FeedHealth — staleness + monotonicity + gap detection (Ticket 22).

Monitors the health of the live Binance kline feed so NYX can
refuse to act when data quality is degraded.

Checks :
  - Staleness : no event received within `stale_seconds`
  - Monotonicity : bar timestamps must be strictly increasing
  - Missing bars : gap > `expected_interval_ms` between consecutive
    bar timestamps
"""
from __future__ import annotations

import time
from typing import Optional


class FeedHealth:
    """Live feed health monitor."""

    def __init__(
        self,
        stale_seconds: float = 120.0,
        expected_interval_ms: int = 900_000,  # 15m = 900,000 ms
    ) -> None:
        self._stale_seconds = float(stale_seconds)
        self._expected_interval_ms = int(expected_interval_ms)
        self._last_event_time: Optional[float] = None
        self._last_bar_ts_ms: Optional[int] = None
        self._monotonicity_violated = False
        self._missing_bars = 0

    def on_event(self, wall_clock: float) -> None:
        """Called on every WS event with `time.time()` timestamp."""
        self._last_event_time = float(wall_clock)

    def on_bar_timestamp(self, bar_start_ms: int) -> None:
        """Called when a CLOSED bar is promoted. Checks monotonicity
        and gap detection."""
        ms = int(bar_start_ms)
        if self._last_bar_ts_ms is not None:
            if ms <= self._last_bar_ts_ms:
                self._monotonicity_violated = True
            else:
                gap = ms - self._last_bar_ts_ms
                if gap > self._expected_interval_ms * 1.5:
                    expected_bars = gap // self._expected_interval_ms
                    self._missing_bars += int(expected_bars - 1)
        self._last_bar_ts_ms = ms

    def is_healthy(self) -> bool:
        """True if a recent event was received (not stale)."""
        if self._last_event_time is None:
            return False
        return (time.time() - self._last_event_time) < self._stale_seconds

    def has_monotonicity_violation(self) -> bool:
        return self._monotonicity_violated

    def missing_bar_count(self) -> int:
        return self._missing_bars

    def status(self) -> dict:
        return {
            'healthy': self.is_healthy(),
            'last_event_age_s': (
                round(time.time() - self._last_event_time, 1)
                if self._last_event_time else None
            ),
            'monotonicity_ok': not self._monotonicity_violated,
            'missing_bars': self._missing_bars,
        }
