"""
ReSyncManager — replay missed bars after a restart.

Given the last timestamp the strategy saw before crashing and the current
wall-clock time, compute which bars need to be fetched and run them
through the strategy in order.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional


_TF_DELTA = {
    '15m': timedelta(minutes=15),
    '1h':  timedelta(hours=1),
    '4h':  timedelta(hours=4),
    '1d':  timedelta(days=1),
}


def _parse(ts: str) -> datetime:
    # Accept ISO 8601 with timezone.
    return datetime.fromisoformat(ts)


def _fmt(dt: datetime) -> str:
    # Keep +00:00 style (python fromisoformat→isoformat roundtrip works since 3.11).
    return dt.isoformat()


class ReSyncManager:
    """Gap detector + replay driver."""

    def needs_resync(
        self,
        last_seen: Optional[str],
        current: str,
        timeframe: str,
    ) -> bool:
        if last_seen is None:
            return True
        delta = _TF_DELTA[timeframe]
        missed = (_parse(current) - _parse(last_seen)).total_seconds()
        return missed > delta.total_seconds()

    def missing_bars(
        self,
        last_seen: str,
        current: str,
        timeframe: str,
    ) -> List[str]:
        delta = _TF_DELTA[timeframe]
        t = _parse(last_seen) + delta
        end = _parse(current)
        out: List[str] = []
        while t <= end:
            out.append(_fmt(t))
            t = t + delta
        return out

    def replay(
        self,
        last_seen: str,
        current: str,
        timeframe: str,
        fetch_fn: Callable[[str], dict],
    ) -> List[dict]:
        """Fetch + return each missing bar in order."""
        bars: List[dict] = []
        for ts in self.missing_bars(last_seen, current, timeframe):
            bars.append(fetch_fn(ts))
        return bars
