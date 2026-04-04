#!/usr/bin/env python3
"""
NYX v0.9 — MTF Backtest Runner
Decisions on 15-minute bars, aligned MTF context (1D / 4H / 1H / 15M).
No look-ahead: each bar only sees closed candles.

Usage:
    python scripts/backtest_mtf.py --start 2023-02-01 --end 2023-02-15
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    'fractal': {
        'context_tf':   '1d',
        'structure_tf': '4h',
        'regime_tf':    '1h',
        'setup_tf':     '15m',
        'use_entry_agent': False,
    },
    'mtf': {
        'timeframes': {
            'context': '1d',
            'regime':  '1h',
            'setup':   '15m',
        }
    },
    'strategy': {
        'mtf_conditions': {
            'sdc_min':           5.0,
            'stability_4h_min':  0.60,
            'alignment_min':     0.40,
        },
        'intent_daily_projection_steps': 2,
    },
    'fractal_readiness': {
        'context_min_bars': 50,
        'regime_min_bars':  100,
        'setup_min_bars':   50,
    },
    'risk': {
        'max_position_pct':    0.10,
        'stop_loss_pct':       0.02,
        'take_profit_ratio':   2.0,
        'leverage':            10,
    },
    'macro': {'enabled': False},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mtf(pair: str, data_dir: str = 'data/raw/mtf') -> dict:
    """Load and index 1D/4H/1H/15M DataFrames for a pair."""
    tfs = {'1d': '1d', '4h': '4h', '1h': '1h', '15m': '15m'}
    out = {}
    for key, suffix in tfs.items():
        path = Path(data_dir) / f'{pair}_{suffix}.csv'
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df['datetime'])
        df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
        out[key] = df
    return out


def slice_mtf(mtf_all: dict, ts: pd.Timestamp, warmup: int = 200) -> dict:
    """
    Return MTF slices strictly before `ts` (no look-ahead), capped at
    `warmup` bars.  This keeps HSMM initialization fast regardless of
    how far back the raw data goes.
    """
    slices = {}
    for tf, df in mtf_all.items():
        sliced = df[df.index < ts].tail(warmup)
        slices[tf] = sliced
    return slices


def pct_str(v: float) -> str:
    return f'{v:+.2f}%'


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class MTFBacktest:
    def __init__(self, config: dict, initial_capital: float = 10_000.0):
        self.config = config
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.orchestrator = Orchestrator(config)

        self.position = None          # None | 'LONG'
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_size = 0.0      # fraction of capital

        self.trades: list = []
        self.equity_curve: list = []
        self.decision_log: list = []

    # -----------------------------------------------------------------------
    # Core loop
    # -----------------------------------------------------------------------

    def run(self, mtf_all: dict, start: str, end: str) -> dict:
        bars_15m = mtf_all['15m']
        bars_in_period = bars_15m[start:end]

        risk_cfg = self.config['risk']
        sl_pct   = risk_cfg['stop_loss_pct']
        tp_ratio = risk_cfg['take_profit_ratio']
        leverage = risk_cfg['leverage']
        size_pct = risk_cfg['max_position_pct']

        print(f"\n{'═'*70}")
        print(f"  NYX v0.9 — MTF BACKTEST")
        print(f"  Pair : BTCUSDT  |  TF décision : 15 min")
        print(f"  Période : {start}  →  {end}")
        print(f"  Barres  : {len(bars_in_period)}  |  Capital  : ${self.initial_capital:,.0f}")
        print(f"  Levier  : ×{leverage}  |  SL : {sl_pct*100:.1f}%  |  TP : {sl_pct*tp_ratio*100:.1f}%")
        print(f"{'═'*70}\n")

        total = len(bars_in_period)
        for bar_i, (ts, row) in enumerate(bars_in_period.iterrows()):
            if bar_i % 144 == 0:  # ~1 day of 15m bars
                pct = bar_i / total * 100
                print(f"  [{pct:5.1f}%]  {ts.strftime('%Y-%m-%d %H:%M')}  "
                      f"price=${float(row['close']):,.0f}  trades={len(self.trades)}")
            current_price = float(row['close'])

            # --- 1. Manage open position (SL/TP check before new signal) ---
            if self.position == 'LONG':
                if current_price <= self.stop_loss:
                    self._close('Stop-Loss', ts, current_price)
                elif current_price >= self.take_profit:
                    self._close('Take-Profit', ts, current_price)

            # --- 2. Build look-ahead-free MTF slices ---
            slices = slice_mtf(mtf_all, ts)
            if any(len(v) == 0 for v in slices.values()):
                continue  # Not enough history yet

            # --- 3. Orchestrator decision ---
            try:
                decision = self.orchestrator.decide(slices, current_price=current_price)
            except Exception as exc:
                decision = None
                action = 'ERROR'
                reason = str(exc)
            else:
                action = decision.action
                reason = decision.reason

            # --- 4. Entry ---
            if self.position is None and action == 'BUY':
                self.entry_price   = current_price
                self.stop_loss     = current_price * (1 - sl_pct)
                self.take_profit   = current_price * (1 + sl_pct * tp_ratio)
                self.position_size = size_pct
                self.position      = 'LONG'
                self._log_entry(ts, current_price, decision)

            # --- 5. Equity snapshot ---
            equity = self._equity(current_price)
            self.equity_curve.append({'ts': ts, 'price': current_price, 'equity': equity})

            # --- 6. Decision log (every bar) ---
            self.decision_log.append({
                'ts':     ts,
                'price':  current_price,
                'action': action,
                'reason': reason[:80] if reason else '',
                'equity': equity,
            })

        # Close any remaining position at end
        if self.position:
            last_row = bars_in_period.iloc[-1]
            self._close('End-of-period', bars_in_period.index[-1], float(last_row['close']))

        return self._compute_metrics(bars_in_period)

    # -----------------------------------------------------------------------
    # Position helpers
    # -----------------------------------------------------------------------

    def _equity(self, price: float) -> float:
        if self.position == 'LONG':
            pnl_pct = (price - self.entry_price) / self.entry_price
            return self.capital * (1 + pnl_pct * self.position_size *
                                   self.config['risk']['leverage'])
        return self.capital

    def _close(self, reason: str, ts, price: float):
        pnl_pct = (price - self.entry_price) / self.entry_price
        pnl = self.capital * pnl_pct * self.position_size * self.config['risk']['leverage']
        self.capital += pnl
        self.trades.append({
            'reason':      reason,
            'ts':          ts,
            'entry_price': self.entry_price,
            'exit_price':  price,
            'pnl':         pnl,
            'pnl_pct':     pnl_pct * 100,
            'won':         pnl > 0,
        })
        marker = '✅' if pnl > 0 else '❌'
        print(f"  {marker}  CLOSE LONG  {ts.strftime('%m/%d %H:%M')} @ ${price:,.0f}"
              f"  PnL: ${pnl:+,.2f} ({pnl_pct*100:+.2f}%)  [{reason}]")
        self.position = None
        self.entry_price = self.stop_loss = self.take_profit = self.position_size = 0.0

    def _log_entry(self, ts, price: float, decision):
        score = decision.score if decision else 0
        print(f"  🟢 ENTRY LONG  {ts.strftime('%m/%d %H:%M')} @ ${price:,.0f}"
              f"  score={score:.2f}  SL=${self.stop_loss:,.0f}  TP=${self.take_profit:,.0f}")

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def _compute_metrics(self, bars: pd.DataFrame) -> dict:
        n = len(self.trades)
        if n == 0:
            return {
                'trades': 0,
                'error': 'Aucun trade généré — pipeline en WAIT sur toute la période',
                'initial_capital': self.initial_capital,
                'final_capital':   self.capital,
                'log': self.decision_log,
            }

        winners = sum(1 for t in self.trades if t['won'])
        pnls    = [t['pnl'] for t in self.trades]
        win_pnls = [p for p in pnls if p > 0]
        los_pnls = [abs(p) for p in pnls if p < 0]

        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        win_rate     = winners / n * 100
        avg_win      = np.mean(win_pnls) if win_pnls else 0
        avg_loss     = np.mean(los_pnls) if los_pnls else 0
        profit_factor = (sum(win_pnls) / sum(los_pnls)) if los_pnls else float('inf')

        # Max drawdown from equity curve
        eq = [e['equity'] for e in self.equity_curve]
        peak = self.initial_capital
        max_dd = 0.0
        for e in eq:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # BTC buy-and-hold reference
        bh_return = (float(bars['close'].iloc[-1]) - float(bars['close'].iloc[0])) / float(bars['close'].iloc[0]) * 100

        return {
            'initial_capital': self.initial_capital,
            'final_capital':   round(self.capital, 2),
            'total_return':    round(total_return, 2),
            'bh_return':       round(bh_return, 2),
            'trades':          n,
            'winners':         winners,
            'losers':          n - winners,
            'win_rate':        round(win_rate, 1),
            'avg_win':         round(avg_win, 2),
            'avg_loss':        round(avg_loss, 2),
            'profit_factor':   round(profit_factor, 2),
            'max_drawdown':    round(max_dd, 2),
            'trade_list':      self.trades,
            'log':             self.decision_log,
        }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(res: dict, start: str, end: str):
    print(f"\n{'═'*70}")
    print(f"  RÉSULTATS  |  {start}  →  {end}")
    print(f"{'═'*70}")

    if 'error' in res:
        print(f"\n  ⚠  {res['error']}")
        print(f"\n  Capital initial : ${res['initial_capital']:,.2f}")
        print(f"  Capital final   : ${res['final_capital']:,.2f}")

        # Show decision breakdown
        log = res.get('log', [])
        if log:
            actions = {}
            for d in log:
                actions[d['action']] = actions.get(d['action'], 0) + 1
            print(f"\n  Distribution des décisions sur {len(log)} barres :")
            for act, cnt in sorted(actions.items(), key=lambda x: -x[1]):
                pct = cnt / len(log) * 100
                print(f"    {act:<8}  {cnt:>4}  ({pct:.1f}%)")

            # Sample of block reasons
            wait_bars = [d for d in log if d['action'] == 'WAIT']
            if wait_bars:
                from collections import Counter
                top_reasons = Counter(d['reason'][:60] for d in wait_bars).most_common(5)
                print(f"\n  Top raisons de WAIT :")
                for reason, cnt in top_reasons:
                    print(f"    [{cnt:>3}]  {reason}")
        return

    print(f"\n  Performance")
    print(f"  {'Capital initial':<22}  ${res['initial_capital']:>10,.2f}")
    print(f"  {'Capital final':<22}  ${res['final_capital']:>10,.2f}")
    print(f"  {'Rendement total':<22}  {pct_str(res['total_return']):>11}")
    print(f"  {'BTC Buy-and-Hold':<22}  {pct_str(res['bh_return']):>11}")
    print(f"  {'Max Drawdown':<22}  {res['max_drawdown']:>10.2f}%")

    print(f"\n  Trades")
    print(f"  {'Total':<22}  {res['trades']:>11}")
    print(f"  {'Gagnants':<22}  {res['winners']:>11}")
    print(f"  {'Perdants':<22}  {res['losers']:>11}")
    print(f"  {'Win Rate':<22}  {res['win_rate']:>10.1f}%")
    print(f"  {'Gain moyen':<22}  ${res['avg_win']:>10,.2f}")
    print(f"  {'Perte moyenne':<22}  ${res['avg_loss']:>10,.2f}")
    print(f"  {'Profit Factor':<22}  {res['profit_factor']:>11.2f}")

    print(f"\n  Détail des trades")
    for i, t in enumerate(res['trade_list'], 1):
        marker = '✅' if t['won'] else '❌'
        ts_str = t['ts'].strftime('%m/%d %H:%M') if hasattr(t['ts'], 'strftime') else str(t['ts'])
        print(f"    {i:>2}. {marker}  {ts_str}  "
              f"entrée ${t['entry_price']:,.0f} → sortie ${t['exit_price']:,.0f}  "
              f"PnL: ${t['pnl']:+,.2f} ({t['pnl_pct']:+.2f}%)  [{t['reason']}]")

    print(f"\n{'═'*70}\n")


def pct_str(v: float) -> str:
    return f'{v:+.2f}%'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='NYX v0.9 MTF Backtest')
    parser.add_argument('--pair',    default='BTCUSDT')
    parser.add_argument('--start',   default='2023-02-01')
    parser.add_argument('--end',     default='2023-02-15')
    parser.add_argument('--capital', type=float, default=10_000.0)
    parser.add_argument('--data-dir', default='data/raw/mtf')
    args = parser.parse_args()

    mtf_all = load_mtf(args.pair, args.data_dir)

    bt = MTFBacktest(BASE_CONFIG, args.capital)
    results = bt.run(mtf_all, args.start, args.end)
    print_report(results, args.start, args.end)


if __name__ == '__main__':
    main()
