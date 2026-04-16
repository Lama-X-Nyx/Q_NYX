"""
TDD Tests — liquidity-hunter feature family (Ticket 09).

NYX moves away from a pure trend-following bias by adding feature
families oriented toward liquidity hunting + microstructure proxies :

Primary :
- vwap_dist           : (close − VWAP_n) / VWAP_n        → direction + strength of vwap pull
- ad_slope            : Accum/Distrib slope vs its EMA   → buying-vs-selling pressure direction
- adosc_norm          : Chaikin A/D Oscillator norm.     → short-vs-long pressure imbalance
- mfi_norm            : Money Flow Index, [-1, +1]       → already present pre-Ticket-09
- marketfi_ratio      : (high − low) / volume * close    → effort vs displacement (reformed)
- bop                 : (close − open) / (high − low)    → bar-level rejection / pressure
- sr_break_up_20      : binary, close breaks 20-bar high
- sr_break_dn_20      : binary, close breaks 20-bar low
- sr_dist_high_20     : (close − max(high, 20)) / close  → distance to recent supply
- sr_dist_low_20      : (close − min(low,  20)) / close  → distance to recent demand
- chop_norm           : Choppiness Index / 100           → compression vs expansion

Secondary :
- kvo_norm            : Klinger Volume Oscillator normed
- vwma_dist           : (close − VWMA_n) / VWMA_n        → vwma reversion
- bb_width_ratio      : already present pre-Ticket-09
- keltner_position    : already present pre-Ticket-09
- atr_ratio           : already present pre-Ticket-09
- minmax_pos_20       : (close − min20) / (max20 − min20) → structural position [0, 1]

All values must be :
  - **stationary** (scale-invariant under price multiplication)
  - **finite** post-warmup
  - **bounded** when the family is naturally bounded
  - **NOT correlated ~1.0 with raw close** (no leakage)
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest


NEW_FEATURES_PRIMARY = [
    'vwap_dist',
    'ad_slope',
    'adosc_norm',
    'marketfi_ratio',
    'bop',
    'sr_break_up_20',
    'sr_break_dn_20',
    'sr_dist_high_20',
    'sr_dist_low_20',
    'chop_norm',
]

NEW_FEATURES_SECONDARY = [
    'kvo_norm',
    'vwma_dist',
    'minmax_pos_20',
]

ALL_NEW_FEATURES = NEW_FEATURES_PRIMARY + NEW_FEATURES_SECONDARY


def _bullish_df(n: int = 400, start: float = 1000.0) -> pd.DataFrame:
    """Synthetic bullish 15m OHLCV with realistic noise."""
    np.random.seed(42)
    drift = 0.001
    pct = drift + np.random.randn(n) * 0.003
    close = start * np.exp(np.cumsum(pct))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.002 + 0.001)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.002 - 0.001)
    open_ = np.roll(close, 1)
    open_[0] = start
    volume = np.abs(np.random.randn(n)) * 1000.0 + 500.0
    idx = pd.date_range('2023-01-01', periods=n, freq='15min')
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }, index=idx)


# ===========================================================================
class TestNewFeaturesPresent:
    """Each new liquidity-hunter feature key must exist in the output."""

    @pytest.fixture(scope='class')
    def feats(self):
        from src.ml.jesse_features import compute_stationary_features
        return compute_stationary_features(_bullish_df(400), feature_set='full')

    @pytest.mark.parametrize('name', ALL_NEW_FEATURES)
    def test_feature_present(self, feats, name):
        assert name in feats.columns, (
            f'feature {name!r} missing — Ticket 09 added it to '
            'compute_stationary_features full set'
        )


# ===========================================================================
class TestNoNaNPostWarmup:
    """Past warmup (50 bars to be safe), every new feature must be finite."""

    @pytest.fixture(scope='class')
    def feats(self):
        from src.ml.jesse_features import compute_stationary_features
        return compute_stationary_features(_bullish_df(400), feature_set='full')

    @pytest.mark.parametrize('name', ALL_NEW_FEATURES)
    def test_no_nan_after_warmup(self, feats, name):
        tail = feats[name].iloc[100:].values
        n_bad = int(np.sum(~np.isfinite(tail)))
        assert n_bad == 0, (
            f'feature {name!r} has {n_bad} non-finite values past '
            f'warmup — failed warmup behaviour'
        )


# ===========================================================================
class TestScaleInvariance:
    """Stationarity invariant — multiplying OHLC by k must leave each
    feature unchanged (within tolerance). Volume is preserved (volume
    is its own scale)."""

    @pytest.fixture(scope='class')
    def base_feats(self):
        from src.ml.jesse_features import compute_stationary_features
        return compute_stationary_features(_bullish_df(400), feature_set='full')

    @pytest.fixture(scope='class')
    def scaled_feats(self):
        from src.ml.jesse_features import compute_stationary_features
        df = _bullish_df(400)
        for c in ('open', 'high', 'low', 'close'):
            df[c] *= 10.0
        return compute_stationary_features(df, feature_set='full')

    @pytest.mark.parametrize('name', ALL_NEW_FEATURES)
    def test_scale_invariant(self, base_feats, scaled_feats, name):
        a = base_feats[name].iloc[100:].values
        b = scaled_feats[name].iloc[100:].values
        # Compare with absolute tolerance (some features may be very
        # small near zero; we accept rel ≤ 1e-6 OR abs ≤ 1e-6).
        diff = np.abs(a - b)
        max_abs = float(diff.max())
        assert max_abs < 1e-6, (
            f'feature {name!r} not scale-invariant: max |Δ| = {max_abs} '
            '(stationarity broken — likely raw price leakage)'
        )


# ===========================================================================
class TestNoRawPriceLeakage:
    """For each new feature, |corr(feature, close)| must NOT be ~1.0 on
    a trending dataset. We allow up to 0.95 since some features (like
    minmax_pos_20) can be moderately correlated with price by design,
    but a feature that LEAKS raw price would correlate 0.99+."""

    @pytest.fixture(scope='class')
    def df_and_feats(self):
        from src.ml.jesse_features import compute_stationary_features
        df = _bullish_df(400)
        feats = compute_stationary_features(df, feature_set='full')
        return df, feats

    @pytest.mark.parametrize('name', ALL_NEW_FEATURES)
    def test_corr_with_close_below_threshold(self, df_and_feats, name):
        df, feats = df_and_feats
        close = df['close'].values
        f = feats[name].iloc[100:].values
        c = close[100:]
        if np.std(f) < 1e-12 or np.std(c) < 1e-12:
            pytest.skip(f'{name} or close has zero variance on this slice')
        corr = float(np.corrcoef(f, c)[0, 1])
        assert abs(corr) < 0.95, (
            f'feature {name!r} correlates {corr:.3f} with raw close — '
            'likely raw price leakage'
        )


# ===========================================================================
class TestBoundedness:
    """Features designed to be bounded must stay within their range."""

    @pytest.fixture(scope='class')
    def feats(self):
        from src.ml.jesse_features import compute_stationary_features
        return compute_stationary_features(_bullish_df(400), feature_set='full')

    def test_bop_in_unit_interval(self, feats):
        v = feats['bop'].iloc[100:].values
        assert v.min() >= -1.001
        assert v.max() <= 1.001

    def test_chop_norm_in_unit_interval(self, feats):
        v = feats['chop_norm'].iloc[100:].values
        assert v.min() >= -0.001
        assert v.max() <= 1.001

    def test_minmax_pos_in_unit_interval(self, feats):
        v = feats['minmax_pos_20'].iloc[100:].values
        assert v.min() >= -0.001
        assert v.max() <= 1.001

    def test_sr_breaks_are_binary(self, feats):
        for col in ('sr_break_up_20', 'sr_break_dn_20'):
            v = feats[col].iloc[100:].values
            uniq = set(np.unique(v).tolist())
            assert uniq.issubset({0.0, 1.0}), (
                f'{col} must be {{0, 1}}, got {uniq}'
            )


# ===========================================================================
class TestSupportResistanceSemantics:
    """sr_dist_high_20 ≤ 0 (close ≤ recent high) and sr_dist_low_20 ≥ 0
    (close ≥ recent low). When the close TIES the high, sr_dist_high
    is exactly 0 and sr_break_up is 1."""

    def test_dist_high_non_positive(self):
        from src.ml.jesse_features import compute_stationary_features
        feats = compute_stationary_features(_bullish_df(400), feature_set='full')
        v = feats['sr_dist_high_20'].iloc[100:].values
        assert v.max() <= 1e-9, (
            'sr_dist_high_20 should be <= 0: close - max(high, 20) ≤ 0 '
            'by definition'
        )

    def test_dist_low_non_negative(self):
        from src.ml.jesse_features import compute_stationary_features
        feats = compute_stationary_features(_bullish_df(400), feature_set='full')
        v = feats['sr_dist_low_20'].iloc[100:].values
        assert v.min() >= -1e-9, (
            'sr_dist_low_20 should be >= 0: close - min(low, 20) ≥ 0'
        )


# ===========================================================================
class TestRegressionExistingFeatures:
    """The existing 25+ features must still be present (no breakage)."""

    EXISTING_PRIMARY_FEATURES = [
        'ema_ratio_9_21', 'ema_ratio_21_50', 'close_vs_ema50',
        'rsi_14', 'atr_ratio', 'volume_ratio',
        'momentum_10', 'momentum_20',
        'high_low_ratio', 'close_position',
        'returns_1', 'returns_5', 'vol_change',
        # full-only:
        'adx_norm', 'macd_hist_ratio', 'bb_percent_b', 'bb_width_ratio',
        'mfi_norm', 'obv_slope', 'roc_10', 'keltner_position', 'squeeze',
        'ema_ratio_50_200', 'zscore_20',
    ]

    @pytest.mark.parametrize('name', EXISTING_PRIMARY_FEATURES)
    def test_existing_feature_still_present(self, name):
        from src.ml.jesse_features import compute_stationary_features
        feats = compute_stationary_features(_bullish_df(400), feature_set='full')
        assert name in feats.columns, (
            f'PRE-Ticket-09 feature {name!r} disappeared — regression'
        )
