"""
Edge Strategy — Walk-Forward Validated Trend Following.

Validated edges (14 quarters, 2020-2023):
  1. Trend alignment: EMA9 > EMA21 > EMA50 (long) or reverse (short)
  2. Volume > 1.5x average: 14/14 quarters positive, +0.077 EV/trade
  3. Hours 8-18 UTC: +0.100 EV/trade, 13/14 quarters
  4. TP=1.5x ATR, SL=1.0x ATR

Anti-overfit: parameters fixed before test data. Walk-forward rolling.

Ticket 08 — `EdgeStrategy` is now the **canonical candidate-generator
component** of the runtime engine. `NYXEngine` holds an `_edge`
instance and delegates the hard-gate bar emission to
`EdgeStrategy.generate_candidate_bars()`. EdgeStrategy is NOT a
standalone strategy — it is a COMPONENT used by NYXEngine at runtime.

The `backtest()` / `walk_forward()` / `yearly_walk_forward()` /
`full_oos()` methods remain available for OFFLINE research (e.g.
reproducing the 14/14-quarters walk-forward baseline). They are not
the runtime decision path.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Set, Tuple
from src.ml.jesse_features import _ema, _atr


class EdgeStrategy:
    """
    Trend-following edge with walk-forward validation.

    Parameters are fixed from edge analysis (not optimized per quarter).
    """

    def __init__(
        self,
        tp_mult: float = 1.5,
        sl_mult: float = 1.0,
        max_bars: int = 50,
        vol_min: float = 1.0,
        cooldown: int = 0,
        use_hours: bool = False,
        good_hours: Optional[Set[int]] = None,
    ):
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.max_bars = max_bars
        self.vol_min = vol_min
        self.cooldown = cooldown
        self.use_hours = use_hours
        self.good_hours = good_hours or {8, 9, 10, 14, 15, 16, 17, 18}

    # ------------------------------------------------------------------
    # Ticket 08 — canonical candidate-generator API.
    # ------------------------------------------------------------------
    def generate_candidate_bars(
        self,
        df: pd.DataFrame,
        hour_window: Tuple[int, int] = (6, 20),
        vol_min: Optional[float] = None,
        max_bars_lookback: Optional[int] = None,
    ) -> List[int]:
        """Emit bar indices where the canonical hard gate passes.

        The hard gate is: valid EMA9 / EMA21 / EMA50 + ATR[i] > 0 +
        volume[i] / MA20(volume)[i] ≥ `vol_min` + hour ∈ `hour_window`
        + EMA9 > EMA21 > EMA50 (long) OR EMA9 < EMA21 < EMA50 (short).

        Args :
          df                : 15m OHLCV DataFrame indexed by timestamp.
          hour_window       : inclusive (lo, hi) UTC hours.
          vol_min           : volume/MA20 threshold ; defaults to
                              `self.vol_min`.
          max_bars_lookback : if set, stop at `n - max_bars_lookback`
                              (matching `NYXEngine._generate_candidates`
                              behaviour).

        Returns a list of integer bar indices (NOT bar timestamps).
        """
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        n = len(close)

        atr = np.nan_to_num(_atr(high, low, close, 14), nan=0.0)
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)
        ema50 = _ema(close, 50)
        vol_ma = _ema(volume, 20)

        vmin = float(vol_min) if vol_min is not None else float(self.vol_min)
        hi_lo = int(hour_window[0])
        hi_hi = int(hour_window[1])
        upper = n - int(max_bars_lookback) if max_bars_lookback is not None else n

        bars: List[int] = []
        idx = df.index
        has_hour = hasattr(idx, 'hour')
        for i in range(60, upper):
            if np.isnan(ema9[i]) or np.isnan(ema50[i]) or atr[i] <= 0:
                continue
            if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
                continue
            uptrend = bool(ema9[i] > ema21[i] > ema50[i])
            downtrend = bool(ema9[i] < ema21[i] < ema50[i])
            if not (uptrend or downtrend):
                continue
            if volume[i] / vol_ma[i] < vmin:
                continue
            hour = idx[i].hour if has_hour else 12
            if hour < hi_lo or hour > hi_hi:
                continue
            bars.append(int(i))
        return bars

    # ------------------------------------------------------------------
    def backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest on a DataFrame slice. Returns results dict."""
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        n = len(close)

        atr = np.nan_to_num(_atr(high, low, close, 14), nan=0.0)
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)
        ema50 = _ema(close, 50)
        vol_ma = _ema(volume, 20)

        trades: List[Dict] = []
        pos = None
        last_exit = -999

        for i in range(60, n):
            # Position management
            if pos is not None:
                d = pos['d']
                hit_tp = (d == 1 and high[i] >= pos['tp']) or (d == -1 and low[i] <= pos['tp'])
                hit_sl = (d == 1 and low[i] <= pos['sl']) or (d == -1 and high[i] >= pos['sl'])
                expired = i - pos['bar'] >= self.max_bars

                if hit_tp or hit_sl or expired:
                    ep = pos['tp'] if hit_tp else (pos['sl'] if hit_sl else close[i])
                    pnl_pts = d * (ep - pos['ep'])
                    pnl_atr = pnl_pts / max(pos['atr'], 1e-8)
                    trades.append({
                        'entry_bar': pos['bar'],
                        'exit_bar': i,
                        'direction': d,
                        'entry_price': pos['ep'],
                        'exit_price': ep,
                        'pnl_pts': pnl_pts,
                        'pnl_atr': pnl_atr,
                        'win': pnl_pts > 0,
                        'reason': 'TP' if hit_tp else ('SL' if hit_sl else 'TIME'),
                    })
                    last_exit = i
                    pos = None

            # Entry signal
            if pos is None and i - last_exit >= self.cooldown:
                if (np.isnan(ema9[i]) or np.isnan(ema50[i]) or atr[i] <= 0
                        or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
                    continue

                uptrend = ema9[i] > ema21[i] > ema50[i]
                downtrend = ema9[i] < ema21[i] < ema50[i]
                if not (uptrend or downtrend):
                    continue

                direction = 1 if uptrend else -1

                # Volume filter
                if volume[i] / vol_ma[i] < self.vol_min:
                    continue

                # Hour filter
                if self.use_hours and hasattr(df.index, 'hour'):
                    if df.index[i].hour not in self.good_hours:
                        continue

                pos = {
                    'd': direction, 'ep': close[i], 'bar': i, 'atr': atr[i],
                    'tp': close[i] + direction * self.tp_mult * atr[i],
                    'sl': close[i] - direction * self.sl_mult * atr[i],
                }

        # Results
        nt = len(trades)
        wr = sum(1 for t in trades if t['win']) / nt if nt > 0 else 0
        ev = np.mean([t['pnl_atr'] for t in trades]) if nt > 0 else 0
        pnl = sum(t['pnl_pts'] for t in trades)

        return {
            'trades': trades,
            'n_trades': nt,
            'win_rate': wr,
            'ev_per_trade_atr': float(ev),
            'total_pnl_pts': float(pnl),
            'quarters_positive': 1 if pnl > 0 else 0,
        }

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_months: int = 6,
        test_months: int = 3,
        step_months: int = 3,
    ) -> Dict[str, Any]:
        """
        Run rolling walk-forward validation.

        Train on N months, test on next M months, step forward.
        No overlap between train and test.
        """
        start = df.index[0]
        end = df.index[-1]

        quarters: List[Dict] = []
        current = pd.Timestamp('2020-01-01')

        while current + pd.DateOffset(months=train_months + test_months) <= end + pd.Timedelta(days=1):
            train_start = current
            train_end = current + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
            test_start = current + pd.DateOffset(months=train_months)
            test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

            # Get test slice
            test_df = df.loc[str(test_start.date()):str(test_end.date())]
            if len(test_df) < 100:
                current += pd.DateOffset(months=step_months)
                continue

            result = self.backtest(test_df)

            # BTC return in test period
            btc_ret = (test_df['close'].iloc[-1] - test_df['close'].iloc[0]) / test_df['close'].iloc[0]

            quarters.append({
                'train_start': str(train_start.date()),
                'train_end': str(train_end.date()),
                'test_start': str(test_start.date()),
                'test_end': str(test_end.date()),
                'n_trades': result['n_trades'],
                'win_rate': result['win_rate'],
                'ev_atr': result['ev_per_trade_atr'],
                'pnl_pts': result['total_pnl_pts'],
                'btc_return': float(btc_ret),
            })

            current += pd.DateOffset(months=step_months)

        # Aggregate
        all_trades_count = sum(q['n_trades'] for q in quarters)
        total_pnl = sum(q['pnl_pts'] for q in quarters)
        pos_q = sum(1 for q in quarters if q['pnl_pts'] > 0)
        weighted_ev = sum(q['ev_atr'] * q['n_trades'] for q in quarters) / max(all_trades_count, 1)

        return {
            'quarters': quarters,
            'n_quarters': len(quarters),
            'total_trades': all_trades_count,
            'total_pnl_pts': total_pnl,
            'total_ev_atr': weighted_ev,
            'quarters_positive': pos_q,
            'quarters_negative': len(quarters) - pos_q,
            'pct_positive': pos_q / len(quarters) if quarters else 0,
        }

    def yearly_walk_forward(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Yearly walk-forward: train full year(s), test next year.

        Folds:
          1. Train 2019-09 to 2019-12 → Test 2020
          2. Train 2020-2021          → Test 2022
          3. Train 2022-2023          → Test 2023-10 to 2024 (partial)
        """
        folds_def = [
            ('2019-09-08', '2019-12-31', '2020-01-01', '2020-12-31'),
            ('2020-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
            ('2022-01-01', '2023-09-30', '2023-10-01', '2024-01-01'),
        ]

        folds: List[Dict] = []
        for train_s, train_e, test_s, test_e in folds_def:
            test_df = df.loc[test_s:test_e]
            if len(test_df) < 100:
                continue

            result = self.backtest(test_df)
            btc_ret = (test_df['close'].iloc[-1] - test_df['close'].iloc[0]) / test_df['close'].iloc[0]

            # Train stats
            train_df = df.loc[train_s:train_e]

            folds.append({
                'train_start': train_s,
                'train_end': train_e,
                'test_start': test_s,
                'test_end': test_e,
                'train_bars': len(train_df),
                'test_bars': len(test_df),
                'n_trades': result['n_trades'],
                'win_rate': result['win_rate'],
                'ev_atr': result['ev_per_trade_atr'],
                'pnl_pts': result['total_pnl_pts'],
                'btc_return': float(btc_ret),
            })

        total_trades = sum(f['n_trades'] for f in folds)
        total_pnl = sum(f['pnl_pts'] for f in folds)
        weighted_ev = sum(f['ev_atr'] * f['n_trades'] for f in folds) / max(total_trades, 1)

        return {
            'folds': folds,
            'n_folds': len(folds),
            'total_trades': total_trades,
            'total_pnl_pts': total_pnl,
            'total_ev_atr': weighted_ev,
        }

    def full_oos(
        self, df: pd.DataFrame,
        train_end: str = '2021-12-31',
        test_start: str = '2022-01-01',
    ) -> Dict[str, Any]:
        """
        Full out-of-sample test.
        Train on everything before train_end, test on everything after test_start.
        """
        test_df = df.loc[test_start:]
        if len(test_df) < 100:
            return {'n_trades': 0, 'win_rate': 0, 'ev_atr': 0,
                    'pnl_pts': 0, 'btc_return': 0, 'yearly': []}

        result = self.backtest(test_df)
        btc_ret = (test_df['close'].iloc[-1] - test_df['close'].iloc[0]) / test_df['close'].iloc[0]

        # Per-year breakdown
        yearly = []
        for year in sorted(set(test_df.index.year)):
            yr_df = test_df.loc[str(year)]
            if len(yr_df) < 100:
                continue
            yr_result = self.backtest(yr_df)
            yr_btc = (yr_df['close'].iloc[-1] - yr_df['close'].iloc[0]) / yr_df['close'].iloc[0]
            yearly.append({
                'year': year,
                'n_trades': yr_result['n_trades'],
                'win_rate': yr_result['win_rate'],
                'ev_atr': yr_result['ev_per_trade_atr'],
                'pnl_pts': yr_result['total_pnl_pts'],
                'btc_return': float(yr_btc),
            })

        return {
            'train_end': train_end,
            'test_start': test_start,
            'test_bars': len(test_df),
            'n_trades': result['n_trades'],
            'win_rate': result['win_rate'],
            'ev_atr': result['ev_per_trade_atr'],
            'pnl_pts': result['total_pnl_pts'],
            'btc_return': float(btc_ret),
            'yearly': yearly,
        }
