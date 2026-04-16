"""
TDD Tests — `edge_strategy` wired as the canonical candidate-generator
layer of `NYXEngine` (Ticket 08).

Pre-Ticket-08, `EdgeStrategy.backtest()` existed as an OFFLINE
walk-forward validator, and `NYXEngine._generate_candidates()` had its
own copy of the same hard-gate logic (EMA alignment + volume > 3x MA +
hour window). DRY violation + "edge lives in a separate branch".

Post-Ticket-08, `EdgeStrategy` owns the candidate-generation role
(pure bar-index emission). NYXEngine delegates the hard-gate to
`self._edge.generate_candidate_bars(...)`. `EdgeStrategy` is NOT a
standalone strategy — it is a COMPONENT of the canonical runtime.

Acceptance guards :

- `EdgeStrategy` exposes `.generate_candidate_bars()`.
- `NYXEngine._edge` is an `EdgeStrategy` sharing the same params.
- `NYXEngine._generate_candidates()` internally calls
  `self._edge.generate_candidate_bars(...)`.
- Numbers preserved 1:1 (equivalence guard test stays GREEN).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest


def _make_df(n: int = 200, kind: str = 'bullish') -> pd.DataFrame:
    """Minimal 15m OHLCV dataframe where EMA9 > EMA21 > EMA50 (bull)
    or inverse (bear). Volume at 3x baseline to pass vol_min=3."""
    np.random.seed(42)
    start = 1000.0
    if kind == 'bullish':
        drift = 0.004       # strong uptrend so EMAs align
    elif kind == 'bearish':
        drift = -0.004
    else:
        drift = 0.0
    pct = drift + np.random.randn(n) * 0.0005
    close = start * np.exp(np.cumsum(pct))
    high = close * 1.002
    low = close * 0.998
    open_ = np.roll(close, 1)
    open_[0] = start
    volume = np.full(n, 3000.0)  # 3x baseline
    idx = pd.date_range('2023-01-01 08:00:00', periods=n, freq='15T')
    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low,
         'close': close, 'volume': volume},
        index=idx,
    )


# ===========================================================================
class TestEdgeStrategyGenerateCandidateBars:
    """EdgeStrategy exposes a pure bar-index emitter for the hard gate
    — the canonical candidate-generation role (Ticket 08)."""

    def test_has_method(self):
        from src.ml.edge_strategy import EdgeStrategy
        edge = EdgeStrategy()
        assert hasattr(edge, 'generate_candidate_bars'), (
            'EdgeStrategy must expose `.generate_candidate_bars()` '
            '(Ticket 08 candidate-generator role)'
        )

    def test_returns_list_of_ints(self):
        from src.ml.edge_strategy import EdgeStrategy
        edge = EdgeStrategy()
        bars = edge.generate_candidate_bars(_make_df(200))
        assert isinstance(bars, list)
        for b in bars:
            assert isinstance(b, int)

    def test_respects_ema_alignment_bull(self):
        """Bullish synthetic df → at least some bars pass the hard gate
        (EMA9 > EMA21 > EMA50 is true on uptrend)."""
        from src.ml.edge_strategy import EdgeStrategy
        edge = EdgeStrategy(vol_min=1.0)
        bars = edge.generate_candidate_bars(
            _make_df(200, 'bullish'),
            hour_window=(0, 23),
        )
        assert len(bars) > 0

    def test_respects_vol_filter(self):
        """Bars with volume < vol_min × MA should be excluded."""
        from src.ml.edge_strategy import EdgeStrategy
        df = _make_df(200, 'bullish')
        # Set volume below the vol_min threshold everywhere.
        df['volume'] = 1.0
        edge = EdgeStrategy(vol_min=3.0)
        bars = edge.generate_candidate_bars(
            df, hour_window=(0, 23),
        )
        assert bars == [], 'low-volume bars should be excluded'

    def test_respects_hour_window(self):
        """Bars outside hour_window should be excluded."""
        from src.ml.edge_strategy import EdgeStrategy
        df = _make_df(200, 'bullish')
        edge = EdgeStrategy(vol_min=1.0)
        # Narrow to a window that excludes the whole day (impossible).
        bars = edge.generate_candidate_bars(
            df, hour_window=(23, 23),  # only hour 23 accepted
        )
        # The synthetic df spans hour 8 onward, no bar at hour 23 fits.
        # Expect zero or few bars.
        hours_ok = {df.index[b].hour for b in bars}
        assert hours_ok.issubset({23})


# ===========================================================================
class TestNYXEngineEdgeAttribute:
    """NYXEngine wires an `EdgeStrategy` instance on `self._edge`
    (Ticket 08 delegation)."""

    def test_engine_has_edge_attribute(self):
        from src.core.nyx_engine import NYXEngine
        from src.ml.edge_strategy import EdgeStrategy
        engine = NYXEngine()
        assert hasattr(engine, '_edge')
        assert isinstance(engine._edge, EdgeStrategy)

    def test_edge_params_match_engine_params(self):
        """The shared hard-gate params (tp_mult, sl_mult, max_bars,
        vol_min) must match between NYXEngine and its _edge instance
        so the generated bar indices are consistent with the rest of
        the engine's computations."""
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine(
            tp_mult=1.7, sl_mult=1.2, max_bars=60, vol_min=2.5,
        )
        assert engine._edge.tp_mult == pytest.approx(1.7)
        assert engine._edge.sl_mult == pytest.approx(1.2)
        assert engine._edge.max_bars == 60
        assert engine._edge.vol_min == pytest.approx(2.5)


# ===========================================================================
class TestNYXEngineDelegatesCandidateGeneration:
    """The bar indices at which NYXEngine computes candidates must
    match those emitted by `self._edge.generate_candidate_bars()`
    using the same params."""

    @pytest.fixture(scope='class')
    def eth_candidates(self, eth_mtf_data, eth_mtf_features):
        """Call `_generate_candidates` directly on a narrow test
        window to get the bar_idx list for comparison."""
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()

        # Use the engine's own helpers to build ctx (mirrors what
        # run() does internally for _generate_candidates).
        ctx_1d = engine._build_1d_context(
            eth_mtf_data['1d'], eth_mtf_features.get('1d', pd.DataFrame()),
        )
        ctx_1h = engine._build_1h_context(
            eth_mtf_data['1h'], eth_mtf_features.get('1h', pd.DataFrame()),
        )
        df_15m_slice = eth_mtf_data['15m'].loc['2023-01-01':'2023-01-31']
        feat_15m_slice = eth_mtf_features['15m'].loc['2023-01-01':'2023-01-31']

        cands = engine._generate_candidates(
            df_15m=df_15m_slice, feat_15m=feat_15m_slice,
            ctx_1d=ctx_1d, ctx_1h=ctx_1h,
            feat_1h=eth_mtf_features.get('1h'),
            feat_1d=eth_mtf_features.get('1d'),
            feat_4h=eth_mtf_features.get('4h'),
        )
        return engine, df_15m_slice, cands

    def test_candidates_bar_indices_match_edge_emission(
        self, eth_candidates,
    ):
        engine, df_15m_slice, cands = eth_candidates
        # Re-run the edge layer directly with matching params.
        engine_bars = [int(c['bar_idx']) for c in cands]
        # Apply the same `i < n - max_bars` upper bound that
        # _generate_candidates uses (skip the last max_bars bars).
        edge_bars = engine._edge.generate_candidate_bars(
            df_15m_slice,
            hour_window=(6, 20),
            vol_min=engine.vol_min,
            max_bars_lookback=engine.max_bars,
        )
        # The bar indices produced by the engine's candidate-generation
        # must be a subset of the edge-layer emission (the engine may
        # skip a few due to NaN ATR which the edge layer also skips).
        assert set(engine_bars) == set(edge_bars), (
            f'engine candidate bars ({len(engine_bars)}) must match '
            f'EdgeStrategy.generate_candidate_bars emission '
            f'({len(edge_bars)}) — delegation invariant broken'
        )
