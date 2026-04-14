"""
MTFFeatureStack — multi-timeframe feature buffer for live inference.

Task (b.B) layer 2.

Maintains 4 `IncrementalFeatureBuffer` (15m / 1h / 4h / 1d) and
handles 15m → 1h/4h/1d aggregation on close boundaries :

- 15m bar at minute 45 closes the current 1h candle
- 15m bar at minute 45 AND hour ∈ {3, 7, 11, 15, 19, 23} closes the
  4h candle
- 15m bar at minute 45 AND hour == 23 closes the 1d candle

Each newly closed higher-TF candle is OHLCV-aggregated from the
corresponding 15m bars (open=first, high=max, low=min, close=last,
volume=sum) and appended to the matching buffer.

`latest_features_dict()` returns a flat dict with :
  - 15m features (no prefix)
  - 1h features prefixed `h1_`
  - 4h features prefixed `h4_`
  - 1d features prefixed `d1_`

Format matches `NYXPipeline._generate_candidates()` feature blocks.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import pandas as pd

from src.ml.feature_buffer import IncrementalFeatureBuffer


_H4_CLOSE_HOURS = {3, 7, 11, 15, 19, 23}
_D1_CLOSE_HOUR = 23


class MTFFeatureStack:
    """4-TF sliding buffer stack + aggregation on close boundaries."""

    def __init__(self, symbol: str, window_size: int = 300):
        self.symbol = symbol
        self.buf_15m = IncrementalFeatureBuffer('15m', window_size)
        self.buf_1h  = IncrementalFeatureBuffer('1h',  window_size)
        self.buf_4h  = IncrementalFeatureBuffer('4h',  window_size)
        self.buf_1d  = IncrementalFeatureBuffer('1d',  window_size)

        # Rolling raw-15m buffers for aggregation (by boundary).
        # 4 bars for 1h, 16 for 4h, 96 for 1d. Using deques so memory
        # stays bounded independently of window_size.
        self._last_for_1h: Deque[Dict[str, Any]] = deque(maxlen=4)
        self._last_for_4h: Deque[Dict[str, Any]] = deque(maxlen=16)
        self._last_for_1d: Deque[Dict[str, Any]] = deque(maxlen=96)

    # ------------------------------------------------------------------
    def on_15m_bar(self, bar: Dict[str, Any]) -> None:
        """Feed a new 15m bar. Triggers higher-TF aggregation on
        applicable close boundaries."""
        # 1. Append 15m bar.
        self.buf_15m.append(bar)

        # 2. Keep a copy in each raw-agg buffer.
        self._last_for_1h.append(dict(bar))
        self._last_for_4h.append(dict(bar))
        self._last_for_1d.append(dict(bar))

        # 3. Close-boundary detection.
        ts = pd.Timestamp(bar['timestamp'])
        if ts.minute != 45:
            return

        # 1h candle close: 4 most recent 15m bars → one 1h bar.
        if len(self._last_for_1h) == 4:
            self.buf_1h.append(self._aggregate(list(self._last_for_1h),
                                                 hour_open_ts=ts.replace(
                                                     minute=0, second=0,
                                                     microsecond=0)))
            # Leave the raw 1h queue intact — it naturally rolls.

        # 4h candle close: hour ∈ 3/7/11/15/19/23 AND we have 16 bars.
        if ts.hour in _H4_CLOSE_HOURS and len(self._last_for_4h) == 16:
            h4_open = ts.replace(
                hour=(ts.hour // 4) * 4, minute=0, second=0, microsecond=0
            )
            self.buf_4h.append(self._aggregate(list(self._last_for_4h),
                                                 hour_open_ts=h4_open))

        # 1d candle close: hour == 23 AND we have 96 bars.
        if ts.hour == _D1_CLOSE_HOUR and len(self._last_for_1d) == 96:
            d1_open = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            self.buf_1d.append(self._aggregate(list(self._last_for_1d),
                                                 hour_open_ts=d1_open))

    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate(bars, hour_open_ts) -> Dict[str, Any]:
        """OHLCV aggregation of a list of 15m bars into a higher-TF bar."""
        return {
            'timestamp': hour_open_ts.isoformat(),
            'open':   float(bars[0]['open']),
            'high':   float(max(b['high']  for b in bars)),
            'low':    float(min(b['low']   for b in bars)),
            'close':  float(bars[-1]['close']),
            'volume': float(sum(b['volume'] for b in bars)),
        }

    # ------------------------------------------------------------------
    def latest_features_dict(self) -> Dict[str, float]:
        """Return flattened dict with prefixed feature names."""
        out: Dict[str, float] = {}

        f15 = self.buf_15m.latest_features()
        if len(f15):
            for col, val in f15.items():
                out[col] = float(val)

        f1h = self.buf_1h.latest_features()
        if len(f1h):
            for col, val in f1h.items():
                out[f'h1_{col}'] = float(val)

        f4h = self.buf_4h.latest_features()
        if len(f4h):
            for col, val in f4h.items():
                out[f'h4_{col}'] = float(val)

        f1d = self.buf_1d.latest_features()
        if len(f1d):
            for col, val in f1d.items():
                out[f'd1_{col}'] = float(val)

        return out
