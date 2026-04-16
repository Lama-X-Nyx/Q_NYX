"""
Jesse training-dataset policy — Ticket 15.

Agent-specific dataset builders that return `(df, sample_mask)` where
`df` is a contiguous OHLCV slice (so feature computation stays valid)
and `sample_mask` is a boolean array of length `len(df)` marking
which bars count as actual training / evaluation samples.

Rationale :
- Context (1D) — macro bias on slow/structural signals → full history,
  no filter.
- Regime (4H) — market state on a medium horizon → contiguous,
  capped at `max_years=3` to stay within data freshness.
- Setup (1H) — opportunity on potential-setup bars → volume-spike
  filter (volume > `vol_threshold` × MA20) proxies "relevant bars".
- Entry (15M) — timing on candidate-proximity only → rolling
  `rolling_months=12` window + EdgeStrategy candidate-bar proximity
  ± `proximity_bars`, capped at `max_size`.

Reproducibility : every builder accepts `random_state` so
downsampling (when the candidate-proximity mask exceeds `max_size`)
is deterministic.

The mono-file `_BaseJesseAgent.train()` + `.backtest()` accept the
returned `sample_mask` to skip masked-out bars in the per-bar analyze
loop — which is the O(N²) bottleneck on Setup / Entry.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.ml.jesse_features import _ema


MTF = Dict[str, pd.DataFrame]


# ===========================================================================
def build_context_dataset(
    mtf_data: MTF,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Context (1D) — full history, no filter."""
    df = mtf_data['1d']
    mask = np.ones(len(df), dtype=bool)
    return df, mask


# ===========================================================================
def build_regime_dataset(
    mtf_data: MTF,
    max_years: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Regime (4H) — contiguous, last `max_years` years.

    Keeps the tail of the 4H dataframe up to `max_years` back from
    the last index. Mask keeps every bar in the returned slice.
    """
    df = mtf_data['4h']
    end = df.index[-1]
    start_cutoff = end - pd.DateOffset(years=max_years)
    df_slice = df.loc[df.index >= start_cutoff].copy()
    mask = np.ones(len(df_slice), dtype=bool)
    return df_slice, mask


# ===========================================================================
def build_setup_dataset(
    mtf_data: MTF,
    vol_threshold: float = 1.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Setup (1H) — contiguous 1H, mask = volume-spike proxy.

    Keeps only bars where `volume[i] ≥ vol_threshold × EMA20(volume)[i]`.
    This proxies "potential setup bars" without requiring full
    candidate-generation logic at this TF. Mask is deterministic
    given the input volume series.
    """
    df = mtf_data['1h'].copy()
    vol = df['volume'].values.astype(float)
    vol_ma = _ema(vol, 20)
    with np.errstate(invalid='ignore'):
        ratio = np.where(vol_ma > 0, vol / vol_ma, 0.0)
    mask = ratio >= vol_threshold
    return df, mask


# ===========================================================================
def build_entry_dataset(
    mtf_data: MTF,
    rolling_months: int = 12,
    proximity_bars: int = 5,
    max_size: int = 30_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Entry (15M) — rolling window + candidate-proximity mask.

    Steps :
      1. Take the last `rolling_months` months of 15m data.
      2. Use `EdgeStrategy.generate_candidate_bars` to identify
         hard-gate-passing bars (trend + volume + hour).
      3. Expand each candidate bar ± `proximity_bars` into a kept-mask.
      4. If `mask.sum() > max_size`, deterministic downsample the mask
         to `max_size` True rows (random_state-controlled).
    """
    df_full = mtf_data['15m']
    end = df_full.index[-1]
    start_cutoff = end - pd.DateOffset(months=rolling_months)
    df = df_full.loc[df_full.index >= start_cutoff].copy()
    n = len(df)

    from src.ml.edge_strategy import EdgeStrategy
    edge = EdgeStrategy()
    cand_bars = edge.generate_candidate_bars(
        df, hour_window=(6, 20), vol_min=3.0,
    )

    mask = np.zeros(n, dtype=bool)
    for c in cand_bars:
        lo = max(0, c - proximity_bars)
        hi = min(n, c + proximity_bars + 1)
        mask[lo:hi] = True

    if mask.sum() > max_size:
        rng = np.random.default_rng(random_state)
        kept_idx = np.flatnonzero(mask)
        sel = rng.choice(kept_idx, size=max_size, replace=False)
        new_mask = np.zeros(n, dtype=bool)
        new_mask[sel] = True
        mask = new_mask

    return df, mask
