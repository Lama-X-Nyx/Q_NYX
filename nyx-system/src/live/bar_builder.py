"""
BarBuilder — normalizes Binance kline WS events into the canonical
NYX bar dict format (Ticket 22).

Only CLOSED klines (event['k']['x'] == True) are promoted to bars.
Duplicates (same kline start time) are rejected.

Output format (compatible with NYXLiveDecider.on_15m_bar) :
  {
      'timestamp': ISO8601 string,
      'open':   float,
      'high':   float,
      'low':    float,
      'close':  float,
      'volume': float,
  }

No exchange-specific keys leak into the output.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set


class BarBuilder:
    """Normalizes Binance kline WS events → canonical OHLCV bars."""

    def __init__(self) -> None:
        self._seen_start_times: Set[int] = set()

    def on_event(self, event: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Process a raw Binance kline event.

        Returns a canonical bar dict if the kline is CLOSED and not
        a duplicate. Returns None otherwise.
        """
        if event.get('e') != 'kline':
            return None
        k = event.get('k', {})
        if not k.get('x', False):
            return None

        start_ms = int(k.get('t', 0))
        if start_ms in self._seen_start_times:
            return None
        self._seen_start_times.add(start_ms)

        ts = datetime.fromtimestamp(
            start_ms / 1000.0, tz=timezone.utc,
        ).isoformat()

        return {
            'timestamp': ts,
            'open':   float(k.get('o', 0)),
            'high':   float(k.get('h', 0)),
            'low':    float(k.get('l', 0)),
            'close':  float(k.get('c', 0)),
            'volume': float(k.get('v', 0)),
        }
