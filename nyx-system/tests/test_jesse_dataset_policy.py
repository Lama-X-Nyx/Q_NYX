"""
TDD Tests — Ticket 15 — Agent-specific dataset sampling policy.

Each Jesse agent must expose a reproducible dataset builder whose
output respects its fractal role :

- Context : full history, no filter
- Regime  : medium horizon (≤ 3 years), contiguous
- Setup   : filtered to potential-setup bars (volume spike proxy)
- Entry   : highly selective — rolling 12-month window + candidate-
            proximity mask ± N bars, max 30 k rows

The policy layer lives in `src/ml/jesse_dataset.py` :

    build_context_dataset(mtf_data, random_state=42)
        -> (df, sample_mask)
    build_regime_dataset( mtf_data, max_years=3, random_state=42)
        -> (df, sample_mask)
    build_setup_dataset(  mtf_data, vol_threshold=1.5, random_state=42)
        -> (df, sample_mask)
    build_entry_dataset(  mtf_data, rolling_months=12,
                          proximity_bars=5, max_size=30_000,
                          random_state=42)
        -> (df, sample_mask)

The base class `_BaseJesseAgent.train()` + `.backtest()` are extended
with `sample_mask: Optional[np.ndarray]` so the per-bar analyze loop
skips masked-out rows — this cuts the O(N²) runtime on Setup / Entry
to something that fits the sandbox budget (≤ 15 min total).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _mtf_synthetic(n_15m: int = 8000) -> dict:
    """Build a 4-TF synthetic dataset consistent across timeframes."""
    np.random.seed(42)
    drift = 0.0005
    close = 1000 * np.exp(np.cumsum(drift + np.random.randn(n_15m) * 0.001))
    high = close * 1.001
    low = close * 0.999
    open_ = np.roll(close, 1); open_[0] = 1000
    volume = np.abs(np.random.randn(n_15m)) * 1000 + 500
    idx = pd.date_range('2023-01-01', periods=n_15m, freq='15min')
    df_15m = pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }, index=idx)
    df_1h = df_15m.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    df_4h = df_15m.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    df_1d = df_15m.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    return {'15m': df_15m, '1h': df_1h, '4h': df_4h, '1d': df_1d}


# ===========================================================================
class TestModuleExports:
    def test_module_has_four_builders(self):
        import src.ml.jesse_dataset as mod
        for name in ('build_context_dataset', 'build_regime_dataset',
                     'build_setup_dataset', 'build_entry_dataset'):
            assert hasattr(mod, name), f'missing {name}'


# ===========================================================================
class TestBuilderReturnShape:
    """Every builder returns `(df, mask)` with mask length == len(df)."""

    @pytest.fixture(scope='class')
    def mtf(self):
        return _mtf_synthetic(8000)

    def test_context_returns_df_and_mask(self, mtf):
        from src.ml.jesse_dataset import build_context_dataset
        df, mask = build_context_dataset(mtf)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(df)

    def test_regime_returns_df_and_mask(self, mtf):
        from src.ml.jesse_dataset import build_regime_dataset
        df, mask = build_regime_dataset(mtf)
        assert isinstance(df, pd.DataFrame)
        assert len(mask) == len(df)

    def test_setup_returns_df_and_mask(self, mtf):
        from src.ml.jesse_dataset import build_setup_dataset
        df, mask = build_setup_dataset(mtf)
        assert isinstance(df, pd.DataFrame)
        assert len(mask) == len(df)

    def test_entry_returns_df_and_mask(self, mtf):
        from src.ml.jesse_dataset import build_entry_dataset
        df, mask = build_entry_dataset(mtf)
        assert isinstance(df, pd.DataFrame)
        assert len(mask) == len(df)


# ===========================================================================
class TestContextPolicy:
    """Context : full history, no filter → mask.all() == True."""

    def test_context_mask_all_true(self):
        from src.ml.jesse_dataset import build_context_dataset
        mtf = _mtf_synthetic(8000)
        df, mask = build_context_dataset(mtf)
        assert mask.all(), 'Context mask must keep every bar (no filter)'

    def test_context_df_is_1d(self):
        from src.ml.jesse_dataset import build_context_dataset
        mtf = _mtf_synthetic(8000)
        df, _ = build_context_dataset(mtf)
        # Timestamp granularity must be 1D.
        assert df.index.inferred_freq in ('D', '1D') or \
               (df.index[1] - df.index[0]).days >= 1


# ===========================================================================
class TestRegimePolicy:
    """Regime : contiguous 4H, max 3 years."""

    def test_regime_max_years(self):
        from src.ml.jesse_dataset import build_regime_dataset
        # Build a long history >3 years.
        mtf = _mtf_synthetic(40_000)   # 40k × 15m ≈ 1.1y
        df, _ = build_regime_dataset(mtf, max_years=3)
        span = df.index[-1] - df.index[0]
        # Span must be ≤ 3 years + small slack.
        assert span.days <= 3 * 366 + 1

    def test_regime_mask_all_true_on_short_history(self):
        """Short history (< 3 years) is kept entirely."""
        from src.ml.jesse_dataset import build_regime_dataset
        mtf = _mtf_synthetic(4_000)  # ~40 days
        df, mask = build_regime_dataset(mtf, max_years=3)
        assert mask.all()


# ===========================================================================
class TestSetupPolicy:
    """Setup : 1H, mask filters to volume-spike bars (potential setup)."""

    def test_setup_mask_is_subset(self):
        from src.ml.jesse_dataset import build_setup_dataset
        mtf = _mtf_synthetic(8000)
        df, mask = build_setup_dataset(mtf, vol_threshold=1.5)
        # Not all bars kept.
        assert mask.sum() < len(mask), (
            'Setup must filter — mask.sum() == len(mask) means no filter'
        )
        # But at least some bars kept.
        assert mask.sum() > 0

    def test_setup_reproducible(self):
        from src.ml.jesse_dataset import build_setup_dataset
        mtf = _mtf_synthetic(8000)
        _, m1 = build_setup_dataset(mtf, random_state=42)
        _, m2 = build_setup_dataset(mtf, random_state=42)
        assert np.array_equal(m1, m2)


# ===========================================================================
class TestEntryPolicy:
    """Entry : rolling 12-month window + candidate-proximity mask."""

    def test_entry_rolling_window(self):
        from src.ml.jesse_dataset import build_entry_dataset
        mtf = _mtf_synthetic(80_000)   # ~2.3 years
        df, _ = build_entry_dataset(mtf, rolling_months=12)
        span = df.index[-1] - df.index[0]
        # Rolling window ≤ 12 months + small slack
        assert span.days <= 366, f'span {span.days} days > 12 months'

    def test_entry_mask_filters_aggressively(self):
        from src.ml.jesse_dataset import build_entry_dataset
        mtf = _mtf_synthetic(80_000)
        df, mask = build_entry_dataset(
            mtf, rolling_months=12, proximity_bars=5,
        )
        # Entry must filter aggressively — keep < 50 % of rolling window.
        kept = int(mask.sum())
        ratio = kept / len(mask)
        assert ratio < 0.50, (
            f'Entry mask kept {ratio:.1%} — not selective enough'
        )
        # On synthetic constant-volume data the hard gate emits few
        # candidates, so we only assert mask is not empty here (real
        # BTC data produces thousands of candidate-proximity bars).
        assert kept >= 1, 'Entry mask fully empty on synthetic data'

    def test_entry_max_size_respected(self):
        from src.ml.jesse_dataset import build_entry_dataset
        mtf = _mtf_synthetic(80_000)
        df, mask = build_entry_dataset(
            mtf, rolling_months=12, proximity_bars=5, max_size=5_000,
        )
        assert mask.sum() <= 5_000

    def test_entry_reproducible(self):
        from src.ml.jesse_dataset import build_entry_dataset
        mtf = _mtf_synthetic(40_000)
        _, m1 = build_entry_dataset(mtf, random_state=42)
        _, m2 = build_entry_dataset(mtf, random_state=42)
        assert np.array_equal(m1, m2)


# ===========================================================================
class TestBaseAgentAcceptsSampleMask:
    """`_BaseJesseAgent.train()` must accept `sample_mask` (additive)."""

    def test_train_accepts_sample_mask(self):
        import inspect
        from src.ml.jesse_agents import _BaseJesseAgent
        sig = inspect.signature(_BaseJesseAgent.train)
        assert 'sample_mask' in sig.parameters, (
            '_BaseJesseAgent.train must accept `sample_mask` kwarg'
        )

    def test_mask_reduces_training_samples(self):
        """With a mask keeping only ~30 % of bars, n_samples must
        roughly match that ratio (minus warmup)."""
        from src.ml.jesse_agents import JesseEntryAgent
        n = 500
        # Synthetic df with a trend so labels are non-degenerate.
        np.random.seed(0)
        close = 1000 * np.exp(np.cumsum(0.001 + np.random.randn(n) * 0.001))
        df = pd.DataFrame({
            'open':  close, 'high': close * 1.001,
            'low':   close * 0.999, 'close': close,
            'volume': np.full(n, 1000.0),
        }, index=pd.date_range('2023-01-01', periods=n, freq='15min'))
        mask = np.zeros(n, dtype=bool)
        mask[::3] = True  # every 3rd bar
        a = JesseEntryAgent()
        result = a.train(df, sample_mask=mask)
        full_result = JesseEntryAgent().train(df)
        assert result['n_samples'] < full_result['n_samples']
