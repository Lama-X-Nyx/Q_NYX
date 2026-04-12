"""
Ticket 3 — Walk-Forward Cross-Validation Splitter

Generates (train_idx, test_idx) pairs for time-series data.

Two modes:
  - expanding : train window grows each fold (classic purged CV)
  - rolling   : train window is fixed size, slides forward

Each fold is:
  - train : [train_start, test_start)  — no look-ahead
  - test  : [test_start, test_end)
  - gap   : embargo_bars between train end and test start (prevents leakage
            when labels use forward-looking data such as triple-barrier)

Typical usage for 15M bars (2019-2022, ~140K bars):
    splitter = WalkForwardSplitter(
        n_folds=5, test_months=3, embargo_bars=32,
        mode='expanding', min_train_bars=10_000,
    )
    for train_idx, test_idx in splitter.split(df_15m):
        ...
"""

import numpy as np
import pandas as pd
from typing import Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# WalkForwardSplitter
# ---------------------------------------------------------------------------

class WalkForwardSplitter:
    """
    Purged walk-forward splitter for time-series ML.

    Parameters
    ----------
    n_folds        : number of folds
    test_months    : test window size in calendar months (approximate via bars)
    bars_per_month : 15M bars / month ≈ 2880 (30d × 24h × 4)
    embargo_bars   : gap appended to train end to avoid label leakage
                     (set to num_bars used in triple-barrier, e.g. 16)
    mode           : 'expanding' (train grows) | 'rolling' (fixed train window)
    train_months   : for rolling mode — fixed train window in months
    min_train_bars : minimum required training bars (skip folds below this)
    """

    def __init__(
        self,
        n_folds: int = 5,
        test_months: int = 3,
        bars_per_month: int = 2_880,
        embargo_bars: int = 32,
        mode: str = 'expanding',
        train_months: Optional[int] = 12,
        min_train_bars: int = 10_000,
    ):
        self.n_folds        = n_folds
        self.test_months    = test_months
        self.bars_per_month = bars_per_month
        self.embargo_bars   = embargo_bars
        self.mode           = mode
        self.train_months   = train_months
        self.min_train_bars = min_train_bars

        if mode not in ('expanding', 'rolling'):
            raise ValueError(f"mode must be 'expanding' or 'rolling', got '{mode}'")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_indices, test_indices) pairs.

        Parameters
        ----------
        df : pd.DataFrame with a DatetimeIndex (any timeframe)

        Yields
        ------
        (train_idx, test_idx) as integer position arrays
        """
        n = len(df)
        test_size = self.test_months * self.bars_per_month

        # Compute test window boundaries
        # Last fold ends at n - embargo_bars (leave room for labels)
        end_total = n - self.embargo_bars
        # First test start: leave enough room for at least one train window
        if self.mode == 'rolling' and self.train_months is not None:
            train_size = self.train_months * self.bars_per_month
            first_test_start = train_size + self.embargo_bars
        else:
            # Expanding: first test after 1/2 of the data
            first_test_start = max(self.min_train_bars, end_total // (self.n_folds + 1))

        # Build fold starts from earliest to latest
        # Spread n_folds test windows evenly between first_test_start and end_total
        fold_test_starts = np.linspace(
            first_test_start,
            end_total - test_size,
            num=self.n_folds,
            dtype=int,
        )

        for fold_i, test_start in enumerate(fold_test_starts):
            test_end   = min(test_start + test_size, end_total)
            train_end  = test_start - self.embargo_bars  # purge gap

            if self.mode == 'expanding':
                train_start = 0
            else:
                # Rolling: fixed window
                train_start = max(0, train_end - (self.train_months or 12) * self.bars_per_month)

            if train_end - train_start < self.min_train_bars:
                continue  # not enough training data for this fold

            train_idx = np.arange(train_start, train_end)
            test_idx  = np.arange(test_start, test_end)
            yield train_idx, test_idx

    def get_folds(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return all folds as a list (eager evaluation)."""
        return list(self.split(df))

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Print a human-readable summary of each fold.

        Returns a DataFrame with columns:
            fold, train_start, train_end, test_start, test_end,
            train_bars, test_bars, train_date_start, test_date_end
        """
        rows = []
        has_dt_index = hasattr(df.index, 'to_pydatetime')
        for i, (tr, te) in enumerate(self.split(df)):
            row = {
                'fold':         i,
                'train_start':  tr[0],
                'train_end':    tr[-1],
                'test_start':   te[0],
                'test_end':     te[-1],
                'train_bars':   len(tr),
                'test_bars':    len(te),
            }
            if has_dt_index:
                row['train_date_start'] = str(pd.Timestamp(df.index[tr[0]]).strftime('%Y-%m-%d'))
                row['train_date_end']   = str(pd.Timestamp(df.index[tr[-1]]).strftime('%Y-%m-%d'))
                row['test_date_start']  = str(pd.Timestamp(df.index[te[0]]).strftime('%Y-%m-%d'))
                row['test_date_end']    = str(pd.Timestamp(df.index[te[-1]]).strftime('%Y-%m-%d'))
            rows.append(row)
        result = pd.DataFrame(rows)
        if not result.empty:
            print(result.to_string(index=False))
        return result


# ---------------------------------------------------------------------------
# Purge helper (López de Prado style)
# ---------------------------------------------------------------------------

def purge_overlap(
    train_idx: np.ndarray,
    labels: pd.Series,
    num_bars: int,
) -> np.ndarray:
    """
    Remove from train_idx any bar whose label window overlaps with test.

    After constructing train/test split, bars near the boundary have
    labels computed using future bars that fall in the test set. This
    function removes such contaminated bars.

    Parameters
    ----------
    train_idx : integer positions of training set
    labels    : pd.Series with same index as original df
    num_bars  : triple-barrier horizon (bars to expiration)

    Returns
    -------
    Purged train_idx (shorter array)
    """
    if len(train_idx) == 0:
        return train_idx
    cutoff = train_idx[-1] - num_bars  # last safe bar
    return train_idx[train_idx <= cutoff]


# ---------------------------------------------------------------------------
# Convenience: sklearn-compatible splitter
# ---------------------------------------------------------------------------

class SklearnWalkForwardCV:
    """
    Thin sklearn-compatible wrapper around WalkForwardSplitter.
    Use with cross_val_score / GridSearchCV etc.

    Parameters
    ----------
    df_ref : reference DataFrame to compute split boundaries
    **kwargs : forwarded to WalkForwardSplitter
    """

    def __init__(self, df_ref: pd.DataFrame, **kwargs):
        self._splitter = WalkForwardSplitter(**kwargs)
        self._folds    = self._splitter.get_folds(df_ref)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._folds)

    def split(self, X, y=None, groups=None):
        for tr, te in self._folds:
            yield tr, te
