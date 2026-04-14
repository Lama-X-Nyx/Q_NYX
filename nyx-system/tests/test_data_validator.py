"""
TDD Tests — DataValidator

Sanity-checks incoming OHLCV bars before they reach the strategy.

Contract:
  - validate(bar) returns True or raises DataValidationError with a reason.
  - Rejects: negative/zero prices, high < low, close outside [low, high],
    zero volume (configurable), NaN/inf values, wrong timezone.
  - Also detects stale bars (timestamp older than last accepted).
"""
import math

import pytest


def _good_bar(**over) -> dict:
    base = {
        'timestamp': '2025-01-01T00:00:00+00:00',
        'open': 45000.0,
        'high': 45100.0,
        'low': 44900.0,
        'close': 45050.0,
        'volume': 100.0,
    }
    base.update(over)
    return base


class TestHappyPath:

    def test_good_bar_passes(self):
        from src.paper_live.data_validator import DataValidator
        v = DataValidator()
        assert v.validate(_good_bar()) is True


class TestPriceInvariants:

    def test_rejects_negative_price(self):
        from src.paper_live.data_validator import DataValidator, DataValidationError
        v = DataValidator()
        with pytest.raises(DataValidationError):
            v.validate(_good_bar(close=-1.0))

    def test_rejects_zero_price(self):
        from src.paper_live.data_validator import DataValidator, DataValidationError
        v = DataValidator()
        with pytest.raises(DataValidationError):
            v.validate(_good_bar(open=0.0))

    def test_rejects_high_below_low(self):
        from src.paper_live.data_validator import DataValidator, DataValidationError
        v = DataValidator()
        with pytest.raises(DataValidationError):
            v.validate(_good_bar(high=100, low=200))

    def test_rejects_close_above_high(self):
        from src.paper_live.data_validator import DataValidator, DataValidationError
        v = DataValidator()
        with pytest.raises(DataValidationError):
            v.validate(_good_bar(high=100, low=50, open=75, close=200))

    def test_rejects_close_below_low(self):
        from src.paper_live.data_validator import DataValidator, DataValidationError
        v = DataValidator()
        with pytest.raises(DataValidationError):
            v.validate(_good_bar(high=100, low=50, open=75, close=10))


class TestNaNAndInf:

    def test_rejects_nan(self):
        from src.paper_live.data_validator import DataValidator, DataValidationError
        v = DataValidator()
        with pytest.raises(DataValidationError):
            v.validate(_good_bar(close=float('nan')))

    def test_rejects_inf(self):
        from src.paper_live.data_validator import DataValidator, DataValidationError
        v = DataValidator()
        with pytest.raises(DataValidationError):
            v.validate(_good_bar(high=math.inf))


class TestStale:

    def test_rejects_stale_timestamp(self):
        from src.paper_live.data_validator import DataValidator, DataValidationError
        v = DataValidator()
        v.validate(_good_bar(timestamp='2025-01-02T00:00:00+00:00'))
        with pytest.raises(DataValidationError):
            v.validate(_good_bar(timestamp='2025-01-01T00:00:00+00:00'))


class TestVolume:

    def test_allows_zero_volume_by_default(self):
        from src.paper_live.data_validator import DataValidator
        v = DataValidator()
        assert v.validate(_good_bar(volume=0.0)) is True

    def test_strict_volume_rejects_zero(self):
        from src.paper_live.data_validator import DataValidator, DataValidationError
        v = DataValidator(require_volume=True)
        with pytest.raises(DataValidationError):
            v.validate(_good_bar(volume=0.0))
