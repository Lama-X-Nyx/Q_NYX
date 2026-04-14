"""
DataValidator — reject corrupt / stale OHLCV bars before they reach the strategy.
"""
from __future__ import annotations

import math
from typing import Optional


class DataValidationError(ValueError):
    """Raised when an incoming bar violates invariants."""


class DataValidator:
    """Stateful OHLCV validator (remembers last accepted timestamp)."""

    def __init__(self, require_volume: bool = False):
        self.require_volume = bool(require_volume)
        self._last_ts: Optional[str] = None

    def validate(self, bar: dict) -> bool:
        open_ = float(bar['open'])
        high = float(bar['high'])
        low = float(bar['low'])
        close = float(bar['close'])
        volume = float(bar['volume'])

        # NaN / inf
        for name, v in (('open', open_), ('high', high), ('low', low),
                        ('close', close), ('volume', volume)):
            if math.isnan(v) or math.isinf(v):
                raise DataValidationError(f"{name}={v!r} is NaN/inf")

        # Price positivity
        if open_ <= 0 or high <= 0 or low <= 0 or close <= 0:
            raise DataValidationError(
                f"non-positive price: o={open_} h={high} l={low} c={close}"
            )

        # Geometry
        if high < low:
            raise DataValidationError(f"high {high} < low {low}")
        if close > high or close < low:
            raise DataValidationError(
                f"close {close} outside [low={low}, high={high}]"
            )
        if open_ > high or open_ < low:
            raise DataValidationError(
                f"open {open_} outside [low={low}, high={high}]"
            )

        # Volume
        if self.require_volume and volume <= 0:
            raise DataValidationError(f"volume={volume} required > 0")
        if volume < 0:
            raise DataValidationError(f"volume={volume} negative")

        # Timestamp ordering (string compare works for ISO 8601)
        ts = bar.get('timestamp')
        if ts is not None:
            if self._last_ts is not None and str(ts) <= str(self._last_ts):
                raise DataValidationError(
                    f"stale bar: ts={ts} <= last_ts={self._last_ts}"
                )
            self._last_ts = str(ts)

        return True
