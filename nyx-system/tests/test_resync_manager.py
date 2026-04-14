"""
TDD Tests — ReSyncManager

On restart, determine which bars were missed and replay them before
going back online.

Contract:
  - needs_resync(last_seen_ts, now_ts, tf) → True if gap > 1 bar.
  - missing_bars(last_seen, current) returns list of ISO timestamps to fetch.
  - replay(fetch_fn) iterates missing bars through a callback.
  - Supports 15m, 1h, 1d timeframes.
"""
import pytest
from datetime import datetime, timedelta, timezone


class TestNeedsResync:

    def test_no_resync_when_current_equals_last(self):
        from src.paper_live.resync_manager import ReSyncManager
        m = ReSyncManager()
        assert m.needs_resync(
            last_seen='2025-01-01T00:00:00+00:00',
            current='2025-01-01T00:00:00+00:00',
            timeframe='15m',
        ) is False

    def test_no_resync_when_exactly_one_bar_ahead(self):
        from src.paper_live.resync_manager import ReSyncManager
        m = ReSyncManager()
        assert m.needs_resync(
            last_seen='2025-01-01T00:00:00+00:00',
            current='2025-01-01T00:15:00+00:00',
            timeframe='15m',
        ) is False

    def test_resync_when_multiple_bars_ahead(self):
        from src.paper_live.resync_manager import ReSyncManager
        m = ReSyncManager()
        assert m.needs_resync(
            last_seen='2025-01-01T00:00:00+00:00',
            current='2025-01-01T02:00:00+00:00',
            timeframe='15m',
        ) is True

    def test_resync_when_last_is_none(self):
        """Never seen a bar → must resync (cold start)."""
        from src.paper_live.resync_manager import ReSyncManager
        m = ReSyncManager()
        assert m.needs_resync(
            last_seen=None,
            current='2025-01-01T02:00:00+00:00',
            timeframe='15m',
        ) is True


class TestMissingBars:

    def test_missing_bars_15m(self):
        from src.paper_live.resync_manager import ReSyncManager
        m = ReSyncManager()
        bars = m.missing_bars(
            last_seen='2025-01-01T00:00:00+00:00',
            current='2025-01-01T01:00:00+00:00',
            timeframe='15m',
        )
        # Should include 00:15, 00:30, 00:45, 01:00 (4 bars after last_seen)
        assert len(bars) == 4
        assert bars[0].endswith('00:15:00+00:00')
        assert bars[-1].endswith('01:00:00+00:00')

    def test_missing_bars_1h(self):
        from src.paper_live.resync_manager import ReSyncManager
        m = ReSyncManager()
        bars = m.missing_bars(
            last_seen='2025-01-01T00:00:00+00:00',
            current='2025-01-01T03:00:00+00:00',
            timeframe='1h',
        )
        assert len(bars) == 3

    def test_missing_bars_1d(self):
        from src.paper_live.resync_manager import ReSyncManager
        m = ReSyncManager()
        bars = m.missing_bars(
            last_seen='2025-01-01T00:00:00+00:00',
            current='2025-01-05T00:00:00+00:00',
            timeframe='1d',
        )
        assert len(bars) == 4

    def test_empty_when_no_gap(self):
        from src.paper_live.resync_manager import ReSyncManager
        m = ReSyncManager()
        bars = m.missing_bars(
            last_seen='2025-01-01T00:00:00+00:00',
            current='2025-01-01T00:00:00+00:00',
            timeframe='15m',
        )
        assert bars == []


class TestReplay:

    def test_replay_calls_back_for_each_missing_bar(self):
        from src.paper_live.resync_manager import ReSyncManager
        m = ReSyncManager()
        seen = []

        def fetch_fn(ts: str) -> dict:
            bar = {
                'timestamp': ts, 'open': 1.0, 'high': 2.0,
                'low': 0.5, 'close': 1.5, 'volume': 10.0,
            }
            seen.append(ts)
            return bar

        bars = m.replay(
            last_seen='2025-01-01T00:00:00+00:00',
            current='2025-01-01T01:00:00+00:00',
            timeframe='15m',
            fetch_fn=fetch_fn,
        )
        assert len(bars) == 4
        assert len(seen) == 4
