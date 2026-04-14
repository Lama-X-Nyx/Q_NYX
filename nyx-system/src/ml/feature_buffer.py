"""
IncrementalFeatureBuffer — rolling OHLCV buffer + stationary features
for live multi-TF inference.

Task (b.B) layer 1.

Design
------

The live decider cannot keep appending to an unbounded history (memory
blow-up) but must produce features IDENTICAL to the batch pipeline that
trained the model. Otherwise live inference would differ from backtest
and the edge would be lost.

The buffer maintains a sliding window of OHLCV bars (window_size bars)
and, on each `append(bar)`, recomputes the full feature matrix on that
window (same `compute_stationary_features` used at training time).
`latest_features()` returns the last row.

Window size must be large enough so that the longest-lookback feature
(EMA200, z-score over 20, etc.) is equivalent to the batch value at the
same position. Default 300 is safe for EMA200; go higher if you ever
add longer-period indicators.

The per-append cost is O(window_size) vs batch O(N) where N = full
history. For window_size=300 and a 15m bar every 15 min, cost is
negligible.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import pandas as pd

from src.ml.jesse_features import compute_stationary_features


class IncrementalFeatureBuffer:
    """Sliding OHLCV buffer + stationary feature computation."""

    def __init__(self, tf: str, window_size: int = 300):
        if window_size < 50:
            raise ValueError("window_size must be >= 50 (EMA200 + slack)")
        self.tf = tf
        self.window_size = int(window_size)
        self._bars: Deque[Dict[str, Any]] = deque(maxlen=self.window_size)
        self._cached_features: Optional[pd.DataFrame] = None
        self._cache_at_n: int = -1

    # ------------------------------------------------------------------
    @property
    def n_bars(self) -> int:
        return len(self._bars)

    # ------------------------------------------------------------------
    def append(self, bar: Dict[str, Any]) -> None:
        """Append a bar. Drops oldest if window is full."""
        self._bars.append({
            'timestamp': bar['timestamp'],
            'open':   float(bar['open']),
            'high':   float(bar['high']),
            'low':    float(bar['low']),
            'close':  float(bar['close']),
            'volume': float(bar['volume']),
        })
        # Invalidate feature cache on every append.
        self._cached_features = None
        self._cache_at_n = -1

    # ------------------------------------------------------------------
    def _as_dataframe(self) -> pd.DataFrame:
        if not self._bars:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close',
                                          'volume'])
        df = pd.DataFrame(list(self._bars))
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('datetime').drop(columns='timestamp')
        return df

    # ------------------------------------------------------------------
    def features(self) -> pd.DataFrame:
        """Full feature DataFrame over the current window. Cached."""
        if (self._cached_features is not None
                and self._cache_at_n == self.n_bars):
            return self._cached_features
        df = self._as_dataframe()
        if df.empty:
            self._cached_features = pd.DataFrame()
        else:
            self._cached_features = compute_stationary_features(
                df, feature_set='full'
            )
        self._cache_at_n = self.n_bars
        return self._cached_features

    # ------------------------------------------------------------------
    def latest_features(self) -> pd.Series:
        """Return the last row of features. Empty series if no bars yet."""
        feats = self.features()
        if feats.empty:
            return pd.Series(dtype=float)
        return feats.iloc[-1].copy()
