"""
TDD Tests — Ticket 22 — Binance Market Connectivity Layer.

Tests use MOCKED WebSocket payloads (no real Binance connection).
The adapter, bar builder, and feed health modules are tested in
isolation with deterministic inputs.

The integration test verifies that a closed kline event flows
through the canonical path: WS event → BarBuilder → normalized
bar dict → compatible with NYXLiveDecider.on_15m_bar().
"""
from __future__ import annotations

import time
from typing import Any, Dict

import pytest


# Canonical Binance kline WS event (simplified).
def _kline_event(
    symbol: str = 'BTCUSDT',
    interval: str = '15m',
    open_: float = 16500.0,
    high: float = 16510.0,
    low: float = 16490.0,
    close: float = 16505.0,
    volume: float = 1234.5,
    start_ms: int = 1672515000000,
    close_ms: int = 1672515899999,
    event_ms: int = 1672515900000,
    is_closed: bool = True,
) -> Dict[str, Any]:
    return {
        'e': 'kline',
        'E': event_ms,
        's': symbol,
        'k': {
            't': start_ms,
            'T': close_ms,
            's': symbol,
            'i': interval,
            'o': str(open_),
            'h': str(high),
            'l': str(low),
            'c': str(close),
            'v': str(volume),
            'x': is_closed,
        },
    }


# ===========================================================================
class TestBarBuilderNormalization:
    """BarBuilder normalizes a raw Binance kline event into the
    canonical NYX bar dict format."""

    def test_module_exists(self):
        from src.live.bar_builder import BarBuilder
        assert BarBuilder is not None

    def test_normalize_closed_kline(self):
        from src.live.bar_builder import BarBuilder
        bb = BarBuilder()
        event = _kline_event(is_closed=True)
        bar = bb.on_event(event)
        assert bar is not None, 'closed kline must produce a bar'
        for key in ('timestamp', 'open', 'high', 'low', 'close', 'volume'):
            assert key in bar, f'bar missing {key!r}'
        assert bar['open'] == pytest.approx(16500.0)
        assert bar['close'] == pytest.approx(16505.0)

    def test_unclosed_kline_returns_none(self):
        from src.live.bar_builder import BarBuilder
        bb = BarBuilder()
        event = _kline_event(is_closed=False)
        bar = bb.on_event(event)
        assert bar is None, 'unclosed kline must NOT produce a bar'

    def test_duplicate_closed_kline_rejected(self):
        from src.live.bar_builder import BarBuilder
        bb = BarBuilder()
        event = _kline_event(is_closed=True, start_ms=1000000)
        bar1 = bb.on_event(event)
        bar2 = bb.on_event(event)  # same start_ms = duplicate
        assert bar1 is not None
        assert bar2 is None, 'duplicate closed kline must be rejected'

    def test_timestamp_is_iso_string(self):
        from src.live.bar_builder import BarBuilder
        bb = BarBuilder()
        bar = bb.on_event(_kline_event(is_closed=True))
        assert isinstance(bar['timestamp'], str)


# ===========================================================================
class TestFeedHealth:

    def test_module_exists(self):
        from src.live.feed_health import FeedHealth
        assert FeedHealth is not None

    def test_fresh_feed_is_healthy(self):
        from src.live.feed_health import FeedHealth
        fh = FeedHealth(stale_seconds=60)
        fh.on_event(time.time())
        assert fh.is_healthy()

    def test_stale_feed_detected(self):
        from src.live.feed_health import FeedHealth
        fh = FeedHealth(stale_seconds=5)
        fh.on_event(time.time() - 10)  # 10s ago
        assert not fh.is_healthy()

    def test_monotonicity_violation_detected(self):
        from src.live.feed_health import FeedHealth
        fh = FeedHealth(stale_seconds=60)
        fh.on_bar_timestamp(1000)
        fh.on_bar_timestamp(999)  # goes backwards
        assert fh.has_monotonicity_violation()

    def test_missing_bar_detected(self):
        """If bar timestamps jump by > expected_interval_ms, a gap
        is detected."""
        from src.live.feed_health import FeedHealth
        fh = FeedHealth(stale_seconds=60, expected_interval_ms=900_000)
        fh.on_bar_timestamp(1000)
        fh.on_bar_timestamp(1000 + 900_000 * 3)  # 3× expected
        assert fh.missing_bar_count() >= 1


# ===========================================================================
class TestBarFormatCompatibility:
    """The normalized bar dict must be directly consumable by
    NYXLiveDecider.on_15m_bar() — same keys, same types."""

    def test_bar_keys_match_decider_input(self):
        from src.live.bar_builder import BarBuilder
        bb = BarBuilder()
        bar = bb.on_event(_kline_event(is_closed=True))
        required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        assert required.issubset(bar.keys())
        assert isinstance(bar['open'], float)
        assert isinstance(bar['volume'], float)


# ===========================================================================
class TestNoExchangeLeakage:
    """Exchange-specific payloads must NOT leak into the bar dict.
    The bar dict must only contain canonical OHLCV fields."""

    def test_no_binance_specific_keys(self):
        from src.live.bar_builder import BarBuilder
        bb = BarBuilder()
        bar = bb.on_event(_kline_event(is_closed=True))
        binance_keys = {'e', 'E', 's', 'k', 'x', 'i', 't', 'T'}
        leaked = binance_keys & set(bar.keys())
        assert not leaked, (
            f'Exchange-specific keys leaked into bar dict: {leaked}'
        )
