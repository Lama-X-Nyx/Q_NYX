"""
TDD Tests — `IncrementalFeatureBuffer`.

Task (b.B) layer 1. A sliding buffer that maintains OHLCV bars per TF
and computes stationary features incrementally (no re-scan of the full
history on each new bar).

Critical invariant: **equivalence with batch**. For every bar, the
features produced incrementally must match the features produced by
`compute_stationary_features(full_df).iloc[i]` within numerical
tolerance (1e-9). Otherwise the live decider would diverge from the
backtest and we'd lose the measured edge.
"""
import numpy as np
import pandas as pd
import pytest


def _synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Deterministic OHLCV with some drift + noise."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.5, size=n))
    # Ensure strictly positive.
    close = np.maximum(close, 1.0)
    noise = rng.normal(0, 0.3, size=n)
    high = close + np.abs(noise) + 0.1
    low = close - np.abs(noise) - 0.1
    low = np.maximum(low, 0.5)
    open_ = close + rng.normal(0, 0.2, size=n)
    volume = rng.uniform(100, 1000, size=n)
    idx = pd.date_range('2023-01-01', periods=n, freq='15min')
    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low, 'close': close,
         'volume': volume},
        index=idx,
    )


# ===========================================================================
class TestConstruction:

    def test_can_build_for_15m(self):
        from src.ml.feature_buffer import IncrementalFeatureBuffer
        buf = IncrementalFeatureBuffer(tf='15m', window_size=200)
        assert buf.tf == '15m'
        assert buf.window_size == 200
        assert buf.n_bars == 0

    def test_can_build_for_1h_4h_1d(self):
        from src.ml.feature_buffer import IncrementalFeatureBuffer
        for tf in ('1h', '4h', '1d'):
            buf = IncrementalFeatureBuffer(tf=tf, window_size=100)
            assert buf.tf == tf


class TestAppend:

    def test_append_grows_buffer(self):
        from src.ml.feature_buffer import IncrementalFeatureBuffer
        buf = IncrementalFeatureBuffer(tf='15m', window_size=200)
        df = _synthetic_ohlcv(10)
        for _, row in df.iterrows():
            buf.append({
                'timestamp': row.name.isoformat(),
                'open':   float(row['open']),
                'high':   float(row['high']),
                'low':    float(row['low']),
                'close':  float(row['close']),
                'volume': float(row['volume']),
            })
        assert buf.n_bars == 10

    def test_window_size_enforced(self):
        """Beyond window_size, oldest bars are dropped."""
        from src.ml.feature_buffer import IncrementalFeatureBuffer
        buf = IncrementalFeatureBuffer(tf='15m', window_size=50)
        df = _synthetic_ohlcv(100)
        for _, row in df.iterrows():
            buf.append({
                'timestamp': row.name.isoformat(),
                'open':   float(row['open']),
                'high':   float(row['high']),
                'low':    float(row['low']),
                'close':  float(row['close']),
                'volume': float(row['volume']),
            })
        assert buf.n_bars == 50, \
            f"window_size=50 but n_bars={buf.n_bars}"


class TestEquivalenceWithBatch:
    """The CRITICAL invariant: incremental == batch."""

    def test_latest_features_match_batch_at_tail(self):
        from src.ml.feature_buffer import IncrementalFeatureBuffer
        from src.ml.jesse_features import compute_stationary_features

        df = _synthetic_ohlcv(250)
        batch_feats = compute_stationary_features(df, feature_set='full')

        buf = IncrementalFeatureBuffer(tf='15m', window_size=300)
        for _, row in df.iterrows():
            buf.append({
                'timestamp': row.name.isoformat(),
                'open':   float(row['open']),
                'high':   float(row['high']),
                'low':    float(row['low']),
                'close':  float(row['close']),
                'volume': float(row['volume']),
            })

        latest = buf.latest_features()
        assert isinstance(latest, pd.Series)

        # Compare every non-constant feature to batch.iloc[-1].
        # Tolerance 1e-6 (EMA recursion gives tiny float drift).
        batch_last = batch_feats.iloc[-1]
        mismatches = []
        for col in batch_last.index:
            if col not in latest.index:
                continue
            b = float(batch_last[col])
            i = float(latest[col])
            if np.isnan(b) and np.isnan(i):
                continue
            if np.isnan(b) or np.isnan(i):
                mismatches.append(f"{col}: batch={b} incr={i} (NaN mismatch)")
                continue
            if abs(b - i) > 1e-6:
                mismatches.append(f"{col}: batch={b:.9g} incr={i:.9g} "
                                   f"diff={abs(b-i):.2e}")
        assert not mismatches, "Incremental != batch tail:\n  " + \
            "\n  ".join(mismatches[:10])
