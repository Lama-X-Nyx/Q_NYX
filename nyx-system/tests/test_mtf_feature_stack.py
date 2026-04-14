"""
TDD Tests — `MTFFeatureStack`.

Task (b.B) layer 2. Maintains 4 `IncrementalFeatureBuffer` instances
(15m / 1h / 4h / 1d) and handles 15m → 1h/4h/1d aggregation on close
boundaries (15m bar at minute 45 closes the 1h candle, etc.).

Key invariants:
  - On every 15m bar at minute 45, the 1h buffer gains exactly one bar
  - At minute 45 AND hour in {3,7,11,15,19,23}, the 4h buffer gains 1
  - At minute 45 AND hour == 23, the 1d buffer gains 1
  - `latest_features_dict()` returns a dict with the 4 blocks:
      15m (no prefix), h1_*, h4_*, d1_*
  - Equivalence with batch: features produced by the stack at time T
    match `compute_stationary_features` applied to the proper slice.
"""
import pandas as pd
import pytest


def _bar(ts: str, price: float = 100.0, vol: float = 500.0) -> dict:
    return {
        'timestamp': ts,
        'open':   price,
        'high':   price * 1.002,
        'low':    price * 0.998,
        'close':  price * 1.001,
        'volume': vol,
    }


def _feed_15m_sequence(stack, start: str, n_bars: int, base_price: float = 100.0):
    """Feed n_bars consecutive 15m bars starting at `start` (ISO)."""
    ts = pd.Timestamp(start)
    for i in range(n_bars):
        stack.on_15m_bar(_bar(
            ts=ts.isoformat(),
            price=base_price + i * 0.1,
            vol=500.0 + i,
        ))
        ts = ts + pd.Timedelta(minutes=15)


# ===========================================================================
class TestConstruction:

    def test_build_stack_for_symbol(self):
        from src.ml.mtf_feature_stack import MTFFeatureStack
        s = MTFFeatureStack(symbol='ETHUSDT', window_size=300)
        assert s.symbol == 'ETHUSDT'
        assert s.buf_15m.n_bars == 0
        assert s.buf_1h.n_bars == 0
        assert s.buf_4h.n_bars == 0
        assert s.buf_1d.n_bars == 0


class TestAggregationBoundaries:

    def test_1h_candle_created_on_minute_45(self):
        """15m bars at 00:00, 00:15, 00:30, 00:45 → exactly 1 bar in 1h."""
        from src.ml.mtf_feature_stack import MTFFeatureStack
        s = MTFFeatureStack(symbol='ETHUSDT')

        # 00:00, 00:15, 00:30 — no 1h bar yet
        for m in (0, 15, 30):
            s.on_15m_bar(_bar(f'2023-01-01T00:{m:02d}:00'))
            assert s.buf_1h.n_bars == 0, \
                f"1h bar created prematurely at minute {m}"

        # 00:45 — 1h boundary closes
        s.on_15m_bar(_bar('2023-01-01T00:45:00'))
        assert s.buf_1h.n_bars == 1, \
            f"1h bar missing after minute 45, n_bars={s.buf_1h.n_bars}"

    def test_4h_candle_on_minute_45_hour_3(self):
        """16 15m bars ending at 03:45 → exactly 1 bar in 4h buffer."""
        from src.ml.mtf_feature_stack import MTFFeatureStack
        s = MTFFeatureStack(symbol='ETHUSDT')
        _feed_15m_sequence(s, '2023-01-01T00:00:00', 16)
        assert s.buf_4h.n_bars == 1, (
            f"4h bar missing after 16 bars 00:00 → 03:45, "
            f"n_bars={s.buf_4h.n_bars}"
        )

    def test_1d_candle_on_minute_45_hour_23(self):
        """96 bars ending at 23:45 → exactly 1 bar in 1d buffer."""
        from src.ml.mtf_feature_stack import MTFFeatureStack
        s = MTFFeatureStack(symbol='ETHUSDT')
        _feed_15m_sequence(s, '2023-01-01T00:00:00', 96)
        assert s.buf_1d.n_bars == 1, \
            f"1d bar missing after 96 bars full day, n_bars={s.buf_1d.n_bars}"

    def test_no_1h_before_minute_45_of_second_hour(self):
        """After bar 00:00, no 1h candle (we closed nothing)."""
        from src.ml.mtf_feature_stack import MTFFeatureStack
        s = MTFFeatureStack(symbol='ETHUSDT')
        s.on_15m_bar(_bar('2023-01-01T00:00:00'))
        assert s.buf_1h.n_bars == 0

    def test_multiple_hours_produce_multiple_1h_bars(self):
        """3 full hours → 3 × 1h bars."""
        from src.ml.mtf_feature_stack import MTFFeatureStack
        s = MTFFeatureStack(symbol='ETHUSDT')
        _feed_15m_sequence(s, '2023-01-01T00:00:00', 12)   # 3 hours
        assert s.buf_1h.n_bars == 3


class TestAggregationOHLCV:

    def test_1h_ohlcv_values_correct(self):
        """Aggregation rules: open=first, high=max, low=min, close=last,
        volume=sum."""
        from src.ml.mtf_feature_stack import MTFFeatureStack
        s = MTFFeatureStack(symbol='ETHUSDT')
        # Four bars with distinctive O/H/L/C/V
        custom = [
            {'timestamp': '2023-01-01T00:00:00', 'open': 100.0, 'high': 105.0,
             'low':  99.0, 'close': 102.0, 'volume': 100.0},
            {'timestamp': '2023-01-01T00:15:00', 'open': 102.0, 'high': 108.0,
             'low': 101.0, 'close': 104.0, 'volume': 150.0},
            {'timestamp': '2023-01-01T00:30:00', 'open': 104.0, 'high': 106.0,
             'low':  97.0, 'close':  98.0, 'volume': 200.0},
            {'timestamp': '2023-01-01T00:45:00', 'open':  98.0, 'high': 103.0,
             'low':  95.0, 'close': 101.0, 'volume': 120.0},
        ]
        for b in custom:
            s.on_15m_bar(b)
        # The single 1h bar should aggregate exactly:
        assert s.buf_1h.n_bars == 1
        df = s.buf_1h._as_dataframe()
        row = df.iloc[0]
        assert row['open'] == 100.0, row['open']
        assert row['high'] == 108.0, row['high']
        assert row['low']  == 95.0, row['low']
        assert row['close'] == 101.0, row['close']
        assert row['volume'] == 570.0, row['volume']


class TestLatestFeaturesDict:

    def test_empty_dict_before_bars(self):
        from src.ml.mtf_feature_stack import MTFFeatureStack
        s = MTFFeatureStack(symbol='ETHUSDT')
        d = s.latest_features_dict()
        assert isinstance(d, dict)
        # Empty or all-NaN; we just require it doesn't crash.

    def test_has_prefixes_after_enough_bars(self):
        """After one full day (96 × 15m), the dict must have entries
        from all 4 TF blocks (15m + h1_ + h4_ + d1_)."""
        from src.ml.mtf_feature_stack import MTFFeatureStack
        s = MTFFeatureStack(symbol='ETHUSDT', window_size=200)
        # Seed with 3 days to give the 1d buffer enough bars for EMA-based
        # features to have non-NaN values on at least the most-recent row.
        _feed_15m_sequence(s, '2023-01-01T00:00:00', 3 * 96)
        d = s.latest_features_dict()

        has_15m = any(not k.startswith(('h1_', 'h4_', 'd1_')) for k in d)
        has_h1 = any(k.startswith('h1_') for k in d)
        has_h4 = any(k.startswith('h4_') for k in d)
        has_d1 = any(k.startswith('d1_') for k in d)
        assert has_15m, "no 15m feature"
        assert has_h1, "no h1_* feature"
        assert has_h4, "no h4_* feature"
        assert has_d1, "no d1_* feature"
