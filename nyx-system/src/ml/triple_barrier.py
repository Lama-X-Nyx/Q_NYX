"""
Ticket 2 — Triple Barrier Labelling

Inspiré de Marcos López de Prado (AFML, ch. 3).
Pour chaque barre candidate, on pose 3 barrières :
  - Barrière haute (TP) : prix monte de pt_mult × ATR
  - Barrière basse (SL) : prix descend de sl_mult × ATR
  - Barrière temporelle : expiration après num_bars

Label :
  1  = TP atteint en premier (trade gagnant)
  0  = SL ou expiration (trade perdant / nul)

Direction-aware : pour contexte bearish, TP = barrière basse.

Implémentation vectorisée numpy O(N × num_bars) ≈ 5-10s pour 150K barres.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder's ATR on OHLCV DataFrame. Returns (T,) array."""
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    prev = np.roll(c, 1); prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = np.empty(len(tr))
    if len(tr) < period:
        atr[:] = tr.mean() if len(tr) > 0 else 1.0
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr[:period - 1] = atr[period - 1]
    return atr


# ---------------------------------------------------------------------------
# Core triple barrier (vectorized numpy)
# ---------------------------------------------------------------------------

def triple_barrier_labels(
    df: pd.DataFrame,
    context: str = 'bullish',
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    num_bars: int = 16,
    atr_period: int = 14,
    atr_arr: Optional[np.ndarray] = None,
    min_atr_pct: float = 0.003,   # floor ATR at 0.3% of price (prevents hairline stops)
) -> pd.Series:
    """
    Compute triple barrier labels for a single context direction.

    Parameters
    ----------
    df          : OHLCV DataFrame (index = timestamps)
    context     : 'bullish' | 'bearish'
    pt_mult     : TP barrier = pt_mult × ATR from entry
    sl_mult     : SL barrier = sl_mult × ATR from entry
    num_bars    : time barrier (max holding in bars)
    atr_period  : ATR smoothing period
    atr_arr     : pre-computed ATR array (skip recomputation if provided)
    min_atr_pct : floor stop distance at min_atr_pct × close

    Returns
    -------
    pd.Series of int (0 / 1), same index as df.
    NaN for last num_bars (not enough future data).
    """
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    close = df['close'].values.astype(float)
    n     = len(df)

    if atr_arr is None:
        atr_arr = compute_atr(df, atr_period)

    labels = np.full(n, np.nan)

    bullish = (context == 'bullish')

    for i in range(n - num_bars):
        c  = close[i]
        # Effective stop distance (floor to min_atr_pct × price)
        atr_eff = max(atr_arr[i], c * min_atr_pct)

        if bullish:
            tp_price = c + pt_mult * atr_eff   # upper barrier
            sl_price = c - sl_mult * atr_eff   # lower barrier
            fut_high = high[i + 1: i + 1 + num_bars]
            fut_low  = low[i + 1:  i + 1 + num_bars]

            tp_hit = np.argmax(fut_high >= tp_price)  if np.any(fut_high >= tp_price)  else num_bars
            sl_hit = np.argmax(fut_low  <= sl_price)  if np.any(fut_low  <= sl_price)  else num_bars
        else:
            # Bearish : TP = lower barrier, SL = upper barrier
            tp_price = c - pt_mult * atr_eff   # lower barrier = TP
            sl_price = c + sl_mult * atr_eff   # upper barrier = SL
            fut_high = high[i + 1: i + 1 + num_bars]
            fut_low  = low[i + 1:  i + 1 + num_bars]

            tp_hit = np.argmax(fut_low  <= tp_price)  if np.any(fut_low  <= tp_price)  else num_bars
            sl_hit = np.argmax(fut_high >= sl_price)  if np.any(fut_high >= sl_price)  else num_bars

        labels[i] = 1 if tp_hit < sl_hit else 0

    return pd.Series(labels, index=df.index)


# ---------------------------------------------------------------------------
# Multi-context labeller (for training datasets)
# ---------------------------------------------------------------------------

def label_with_context(
    df: pd.DataFrame,
    context_series: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    num_bars: int = 16,
    atr_period: int = 14,
) -> pd.Series:
    """
    Apply triple barrier labels using direction from context_series.

    Parameters
    ----------
    df             : OHLCV DataFrame (full history)
    context_series : pd.Series of 'bullish'/'bearish'/'neutral', same index as df
    pt_mult        : TP multiple of ATR
    sl_mult        : SL multiple of ATR
    num_bars       : time barrier (bars to expiration)
    atr_period     : ATR smoothing

    Returns
    -------
    pd.Series of int (0/1), NaN for neutral context and last num_bars.
    """
    atr_arr = compute_atr(df, atr_period)
    labels  = pd.Series(np.nan, index=df.index)

    for ctx in ('bullish', 'bearish'):
        mask = (context_series == ctx)
        if mask.sum() == 0:
            continue
        # Compute on full df (barriers look at future, not just masked subset)
        lbl_full = triple_barrier_labels(df, context=ctx,
                                         pt_mult=pt_mult, sl_mult=sl_mult,
                                         num_bars=num_bars, atr_period=atr_period,
                                         atr_arr=atr_arr)
        # Only keep labels for bars where context == ctx
        labels[mask] = lbl_full[mask]

    return labels


# ---------------------------------------------------------------------------
# Diagnostic stats
# ---------------------------------------------------------------------------

def label_stats(labels: pd.Series, context_series: Optional[pd.Series] = None) -> dict:
    """Print and return label distribution statistics."""
    valid = labels.dropna()
    n     = len(valid)
    pos   = int(valid.sum())
    neg   = n - pos

    stats = {
        'total':   n,
        'pos':     pos,
        'neg':     neg,
        'pos_rate': round(pos / max(n, 1), 4),
    }

    print(f'  Triple Barrier Labels:')
    print(f'    Total valid : {n}')
    print(f'    TP hit (1)  : {pos}  ({pos/max(n,1)*100:.1f}%)')
    print(f'    SL/time (0) : {neg}  ({neg/max(n,1)*100:.1f}%)')

    if context_series is not None:
        for ctx in ('bullish', 'bearish'):
            mask = (context_series == ctx) & labels.notna()
            sub  = valid[mask[valid.index]]
            if len(sub) > 0:
                print(f'    {ctx:>8} : {int(sub.sum())}/{len(sub)} = {sub.mean()*100:.1f}% TP rate')

    return stats
