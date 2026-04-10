#!/usr/bin/env python3
"""
NYX v1.0 — MTF Backtest Runner  [Institutional Risk Engine]
Decisions on 15-minute bars, aligned MTF context (1D / 4H / 1H / 15M).
No look-ahead: each bar only sees closed candles.

Usage:
    python scripts/backtest_mtf.py --start 2023-02-01 --end 2023-02-15
    python scripts/backtest_mtf.py --start 2023-01-01 --end 2023-12-31 --pretrain-all
    python scripts/backtest_mtf.py --start 2023-01-01 --end 2023-12-31 --pretrain-all --use-cache
"""

import sys
import pickle
import hashlib
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator import Orchestrator
from src.core.precomputed_runner import PrecomputedStates


# ---------------------------------------------------------------------------
# Pre-training cache
# ---------------------------------------------------------------------------
# EM training on 3+ years of 15m data takes ~15 min.  We cache the learned
# parameters to disk keyed by (pair, tf, pretrain_end, months, em_iters).
# Re-runs with identical settings load in <1s instead of re-training.
#
# Cache location: data/pretrain_cache/<hash>.pkl
# ---------------------------------------------------------------------------

CACHE_DIR = Path('data/pretrain_cache')


def _pretrain_cache_key(pair: str, pretrain_end: str,
                         pretrain_months: int, em_iters: int) -> str:
    """Deterministic hex key for this exact training config."""
    blob = f"{pair}|{pretrain_end}|{pretrain_months}|{em_iters}|v6_calibrate"
    return hashlib.md5(blob.encode()).hexdigest()[:16]


def _save_pretrain_cache(orchestrator, cache_path: Path) -> None:
    """Pickle HSMM learned params for both agents."""
    def _extract(agent) -> dict:
        h = agent.hsmm
        return {
            'transition_matrix':   h.transition_matrix.copy(),
            'initial_probs':       h.initial_probs.copy(),
            'emission_params':     {k: dict(v) for k, v in h.emission_params.items()},
            '_locked_transition':  h._locked_transition.copy() if h._locked_transition is not None else None,
            '_locked_initial':     h._locked_initial.copy()    if h._locked_initial    is not None else None,
            '_locked_structure':   h._locked_structure,
            '_locked':             h._locked,
        }

    payload = {
        'regime_agent': _extract(orchestrator.regime_agent),
        'setup_agent':  _extract(orchestrator.setup_agent),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"    [CACHE SAVED]  {cache_path}")


def _load_pretrain_cache(orchestrator, cache_path: Path) -> bool:
    """Restore HSMM params from cache.  Returns True on success."""
    if not cache_path.exists():
        return False

    def _restore(agent, params: dict) -> None:
        h = agent.hsmm
        h.transition_matrix  = params['transition_matrix']
        h.initial_probs      = params['initial_probs']
        h.emission_params    = params['emission_params']
        h._locked_transition = params['_locked_transition']
        h._locked_initial    = params['_locked_initial']
        h._locked_structure  = params['_locked_structure']
        h._locked            = params['_locked']
        # Invalidate inference caches
        h._init_fingerprint  = None
        h._fb_fingerprint    = None
        h._fb_cache          = None

    try:
        with open(cache_path, 'rb') as f:
            payload = pickle.load(f)
        _restore(orchestrator.regime_agent, payload['regime_agent'])
        _restore(orchestrator.setup_agent,  payload['setup_agent'])
        print(f"    [CACHE HIT]  {cache_path.name}  (pré-entraînement restauré en <1s)")
        return True
    except Exception as e:
        print(f"    [CACHE MISS / CORRUPT]  {e}")
        return False


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
            # 6-state HSMM calibration (P4b): equal prior = 1/6 ≈ 0.17.
            # Thresholds are ~2.6-3× baseline (vs ~2× for 3-state).
            # sdc_min 4.5 → dominant state prob > 0.45 (2.7× baseline)
            # stability 0.55 → self-transition > 0.55 (still meaningful persistence)
            'sdc_min':            4.5,
            'stability_4h_min':   0.55,
            # alignment_15m_min: SetupAgent 6-state HSMM threshold.
            # Bullish formula: P(Trend+) + 0.5×P(Squeeze) + 0.25×P(Range)
            # (Range = consolidation in bull trend, partial credit)
            # 0.40 requires clear directional signal over neutral noise.
            'alignment_15m_min':  0.40,
        },
        'intent_daily_projection_steps': 2,
        # Minimum aggregate score to enter a trade.
        # Recalibrated for 6-state HSMM (lower per-state probs → lower component scores).
        'min_entry_score': 0.78,
    },
    'fractal_readiness': {
        'context_min_bars': 200,   # SMA200 requires 200 bars minimum
        'regime_min_bars':  100,
        'setup_min_bars':   50,
    },
    'risk': {
        # --- Position sizing (volatility-adjusted) ---
        # We always risk exactly `risk_per_trade_pct` of capital if the ATR
        # stop is hit.  Position size = dollar_risk / (k_atr × ATR).
        # Capped at max_leverage × capital to prevent over-sizing in low-vol.
        'risk_per_trade_pct':      0.02,   # 2% of capital at risk per trade
        'max_leverage':            10,     # hard notional leverage cap

        # --- Trailing ATR stop ---
        # Uses ATR(50) — 50 × 15min = 12.5 hours of market context.
        # ATR(14) on 15m is ~$80 which is pure intrabar noise; ATR(50)
        # gives a more meaningful volatility estimate (~$250-400 in 2023).
        # Initial stop = entry ± k_atr × ATR(50). Trails in the direction
        # of the trade — never moves against it.
        'atr_period':              50,     # ATR period for stop/sizing (50×15m = 12.5h)
        'atr_sl_multiplier':       2.5,    # default k_atr (if no regime info)
        # k_atr = 2.5 across all regimes (simplified from per-regime table).
        # This places the fixed-TP at 2×2.5×ATR = 5×ATR above entry, which on
        # BTC at $25k with ATR(50)~$200 means a $1,000 (4%) target — reachable
        # in a single daily swing. With k=5.0 the TP was 10% away (rarely hit).
        'atr_k_by_regime': {
            'Trend+':       2.5,
            'Trend-':       2.5,
            'Range':        2.5,
            'Squeeze':      2.5,
            'Distribution': 2.5,
        },

        # --- Trailing take-profit (high-water-mark) ---
        # Trail TP kept as a fallback for very large moves that exceed the fixed TP.
        # min_profit_atr_mult=8.0 means it only activates after an 8×ATR gain —
        # effectively ensuring the fixed 2:1 TP fires first on normal winners.
        'min_profit_atr_mult':     8.0,    # large: trail TP is fallback only
        'trail_tp_retracement':    0.25,   # tight retrace to lock in big gains

        # --- Re-entry cooldown ---
        # Bars to wait after any close before new entry.
        # 96 bars × 15min = 24 hours. Prevents churning in choppy markets
        # where trail stops fire quickly then the system re-enters immediately.
        # Raised from 48 to 96 to further reduce over-trading.
        'cooldown_bars':           96,

        # --- Daily drawdown kill switch (prop-desk style) ---
        # If the account loses max_drawdown_pct of its value since the START
        # OF THE CURRENT DAY, all new entries are halted for the rest of
        # that day.  Resets at 00:00 UTC every day with fresh capital baseline.
        # This avoids the deadlock of an all-time-peak-based limit while
        # preserving daily capital protection.
        'max_drawdown_pct':        0.05,   # 5% daily drawdown limit

        # --- Transaction costs (applied on every entry AND exit) ---
        'fee_pct':                 0.0004, # 0.04%/side — Binance futures taker
        # Market-impact slippage: base + size-dependent component.
        # slippage = slippage_base + market_impact_factor × (notional / bar_dollar_vol)
        # At $10K notional vs $50M bar volume → ~0.02% + negligible impact
        # At $1M notional vs $50M bar volume → ~0.02% + 0.04% = 0.06%
        'slippage_base':           0.0002, # 0.02% base bid-ask spread
        'market_impact_factor':    2.0,    # Kyle-style: 2× notional/bar_vol
        # Funding rate (BTC perp): ~0.01%/8h avg in trending markets.
        # Long pays short. Charged every 32 bars (8h = 32 × 15m).
        # Use 0.0% for shorts (longs subsidise shorts historically).
        'funding_rate_long_8h':    0.0001, # 0.01%/8h for LONG
        'funding_rate_short_8h':   0.0000, # 0.00%/8h for SHORT
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


def build_aligned_index(mtf_all: dict, bar_index: pd.DatetimeIndex,
                         warmup: int = 200) -> list:
    """
    Pre-compute, for every 15m bar timestamp, the iloc positions in each
    TF that correspond to the last closed bar strictly before that timestamp.

    Returns a list of dicts: [{tf: (start_iloc, end_iloc), ...}, ...]
    Using iloc ranges avoids repeated boolean index scans at runtime.
    """
    tf_indices = {tf: df.index for tf, df in mtf_all.items()}
    aligned = []
    # Use searchsorted (vectorised) for each TF
    tf_sorted = {tf: np.array(idx.astype(np.int64)) for tf, idx in tf_indices.items()}
    bar_ns = bar_index.astype(np.int64)

    for ts_ns in bar_ns:
        pos = {}
        for tf, arr in tf_sorted.items():
            # last bar strictly before ts
            end = int(np.searchsorted(arr, ts_ns, side='left'))
            start = max(0, end - warmup)
            pos[tf] = (start, end)
        aligned.append(pos)
    return aligned


def slice_from_pos(mtf_all: dict, pos: dict) -> dict:
    """Fast iloc-based slice using pre-computed positions."""
    return {tf: mtf_all[tf].iloc[s:e] for tf, (s, e) in pos.items()}


def pct_str(v: float) -> str:
    return f'{v:+.2f}%'


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def pretrain_agents(orchestrator: Orchestrator, mtf_all: dict,
                    pretrain_end: str, pretrain_months: int,
                    em_iters: int = 30,
                    pair: str = 'BTCUSDT',
                    use_cache: bool = False) -> None:
    """
    Pre-train HSMM agents via Baum-Welch EM on historical data that
    precedes the backtest period.

    Lock mode: STRUCTURE LOCK (default after fit())
      - Transition matrix A and initial π → frozen from EM (structurally stable)
      - Emission params → re-estimated heuristically each bar (adapt to regime)
    This is the best of both worlds: learned structure, adaptive emissions.

    Args:
        orchestrator:    Orchestrator instance (exposes .regime_agent, .setup_agent)
        mtf_all:         Full MTF data dict
        pretrain_end:    ISO date string — last day of pre-training window
        pretrain_months: Number of months to look back for training data
        em_iters:        EM iterations per agent (default 30)
    """
    end_ts   = pd.Timestamp(pretrain_end)
    start_ts = end_ts - relativedelta(months=pretrain_months)
    # Cap at actual data start
    for df in mtf_all.values():
        if not df.empty:
            start_ts = max(start_ts, df.index[0])
            break

    print(f"\n  [EM PRE-TRAIN]  {start_ts.date()} → {end_ts.date()}  "
          f"({pretrain_months}M, {em_iters} iters, structure-lock)")

    # ---- Cache lookup ----
    if use_cache:
        cache_key  = _pretrain_cache_key(pair, pretrain_end, pretrain_months, em_iters)
        cache_path = CACHE_DIR / f"{cache_key}.pkl"
        if _load_pretrain_cache(orchestrator, cache_path):
            return   # parameters restored from disk — skip EM

    # ---- Regime agent — 1H data (matches inference TF), HTF = 4H ----
    regime_tf = '1h' if '1h' in mtf_all else list(mtf_all.keys())[0]
    htf_tf = '4h' if '4h' in mtf_all else None
    df_regime = mtf_all[regime_tf]
    regime_window = df_regime[(df_regime.index >= start_ts) & (df_regime.index < end_ts)]
    df_regime_htf = None
    if htf_tf:
        df_htf_full = mtf_all[htf_tf]
        df_regime_htf = df_htf_full[(df_htf_full.index >= start_ts) & (df_htf_full.index < end_ts)]
    if len(regime_window) >= 100:
        ll = orchestrator.regime_agent.pretrain(regime_window, n_iter=em_iters, df_htf=df_regime_htf)
        msg = f"{len(ll)} EM iters  LL={ll[-1]:.0f}" if ll else "EM skipped"
        print(f"    RegimeAgent  ({regime_tf}+htf={htf_tf})  {len(regime_window)} bars  {msg}")
    else:
        print(f"    RegimeAgent  SKIP ({len(regime_window)} bars < 100)")

    # ---- Setup agent — 15M data, HTF = 1H ----
    setup_tf = '15m' if '15m' in mtf_all else '1h'
    df_setup = mtf_all[setup_tf]
    setup_window = df_setup[(df_setup.index >= start_ts) & (df_setup.index < end_ts)]
    df_setup_htf = None
    if '1h' in mtf_all:
        df_1h_full = mtf_all['1h']
        df_setup_htf = df_1h_full[(df_1h_full.index >= start_ts) & (df_1h_full.index < end_ts)]
    if len(setup_window) >= 100:
        ll = orchestrator.setup_agent.pretrain(setup_window, n_iter=em_iters, df_htf=df_setup_htf)
        msg = f"{len(ll)} EM iters  LL={ll[-1]:.0f}" if ll else "EM skipped"
        print(f"    SetupAgent   ({setup_tf}+htf=1h)  {len(setup_window)} bars  {msg}")
    else:
        print(f"    SetupAgent   SKIP ({len(setup_window)} bars < 100)")

    # ---- Save to cache for future runs ----
    if use_cache:
        _save_pretrain_cache(orchestrator, cache_path)


class MTFBacktest:
    def __init__(self, config: dict, initial_capital: float = 10_000.0):
        self.config = config
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.orchestrator = Orchestrator(config)

        # ---- Position state ----
        self.position: str | None = None   # None | 'LONG' | 'SHORT'
        self.entry_price:  float = 0.0
        self.entry_atr:    float = 0.0     # ATR at entry (for trail calculations)
        self.entry_k:      float = 2.0     # regime-adaptive k_atr at entry
        self.notional:     float = 0.0     # position value in $ at entry
        self.trail_sl:     float = 0.0     # current trailing stop level
        self.high_water:   float = 0.0     # best price seen (LONG peak / SHORT trough)
        self.fixed_tp:     float = 0.0     # fixed 2:1 take-profit target (set at entry)

        # ---- Account risk ----
        # Daily DD kill switch: 5% loss vs that day's opening capital
        self._day_start_capital: float = initial_capital
        self._dd_halted:         bool  = False
        self._current_day:       object = None   # date object, reset each day

        # ---- Tracking ----
        self.trades:        list = []
        self.equity_curve:  list = []
        self.decision_log:  list = []
        self._last_close_bar: int = -1
        self._total_friction: float = 0.0  # cumulative fees + slippage in $

    # -----------------------------------------------------------------------
    # ATR (Wilder's) precomputation
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Wilder's ATR on a OHLCV DataFrame. Returns (T,) array."""
        high  = df['high'].values.astype(float)
        low   = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        prev  = np.roll(close, 1); prev[0] = close[0]
        tr = np.maximum(high - low,
                np.maximum(np.abs(high - prev), np.abs(low - prev)))
        atr = np.zeros(len(tr))
        if len(tr) < period:
            atr[:] = tr.mean() if len(tr) > 0 else 1.0
            return atr
        atr[period - 1] = tr[:period].mean()
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        # Fill warm-up bars with first valid value
        atr[:period - 1] = atr[period - 1]
        return atr

    # -----------------------------------------------------------------------
    # Position sizing (volatility-adjusted)
    # -----------------------------------------------------------------------

    def _compute_position_size(self, price: float, atr: float, k: float) -> tuple:
        """
        Return (notional_$, effective_leverage) for a new position.

        Logic:
          dollar_risk  = capital × risk_per_trade_pct
          stop_dist    = k × ATR
          qty_btc      = dollar_risk / stop_dist
          notional     = qty_btc × price
          leverage     = notional / capital  (capped at max_leverage)
        """
        risk_cfg    = self.config['risk']
        dollar_risk = self.capital * risk_cfg['risk_per_trade_pct']
        # Floor at 1.5% of price: prevents stops so tight that normal 15m noise
        # triggers them.  Without this, low-vol regimes produce hairline stops.
        stop_dist   = max(k * atr, price * 0.015)
        qty         = dollar_risk / stop_dist
        notional    = qty * price
        max_notional = self.capital * risk_cfg['max_leverage']
        notional    = min(notional, max_notional)
        leverage    = notional / self.capital
        return notional, leverage

    # -----------------------------------------------------------------------
    # Core loop
    # -----------------------------------------------------------------------

    def run(self, mtf_all: dict, start: str, end: str,
            precomputed_states: 'PrecomputedStates | None' = None) -> dict:
        import time as _time
        _precomputed = precomputed_states is not None and precomputed_states._ready

        risk_cfg = self.config['risk']
        fee      = risk_cfg['fee_pct']
        max_dd   = risk_cfg['max_drawdown_pct']
        k_regime = risk_cfg['atr_k_by_regime']
        min_mult = risk_cfg['min_profit_atr_mult']
        retr     = risk_cfg['trail_tp_retracement']

        bars_15m       = mtf_all['15m']
        bars_in_period = bars_15m[start:end]

        print(f"\n{'═'*70}")
        print(f"  NYX v1.0 — MTF BACKTEST  [Institutional Risk Engine]")
        print(f"  Pair : BTCUSDT  |  TF décision : 15 min")
        print(f"  Période : {start}  →  {end}")
        print(f"  Barres  : {len(bars_in_period)}  |  Capital  : ${self.initial_capital:,.0f}")
        atr_p = risk_cfg.get('atr_period', 50)
        print(f"  Risk/trade : {risk_cfg['risk_per_trade_pct']*100:.1f}%  |  "
              f"Max DD/day : {max_dd*100:.0f}%  |  "
              f"ATR({atr_p})×k  |  "
              f"Cooldown : {risk_cfg.get('cooldown_bars',16)} bars  |  "
              f"Fees : {risk_cfg['fee_pct']*100:.2f}%  |  "
              f"Slip base : {risk_cfg.get('slippage_base',0.0002)*100:.2f}%+impact  |  "
              f"Funding : {risk_cfg.get('funding_rate_long_8h',0.0001)*100:.3f}%/8h LONG  |  "
              f"Fill : open+1")
        print(f"{'═'*70}\n")

        # Pre-build aligned index
        print("  Pré-calcul de l'index aligné MTF...", end='', flush=True)
        t_idx = _time.time()
        aligned_idx = build_aligned_index(mtf_all, bars_in_period.index)
        print(f" {_time.time()-t_idx:.1f}s")

        # Precompute ATR on the full 15m dataset → index with iloc
        atr_period = risk_cfg.get('atr_period', 50)
        atr_full = self._compute_atr(bars_15m, period=atr_period)

        # Locate start iloc in full 15m df
        period_iloc_start = bars_15m.index.get_loc(bars_in_period.index[0]) \
            if bars_in_period.index[0] in bars_15m.index \
            else bars_15m.index.searchsorted(bars_in_period.index[0])

        bar_closes  = bars_in_period['close'].values.astype(float)
        bar_opens   = bars_in_period['open'].values.astype(float)
        bar_volumes = bars_in_period['volume'].values.astype(float)
        bar_index   = bars_in_period.index
        total       = len(bars_in_period)

        # Funding counters
        _funding_rate_long  = risk_cfg.get('funding_rate_long_8h',  0.0001)
        _funding_rate_short = risk_cfg.get('funding_rate_short_8h', 0.0000)
        _funding_interval   = 32   # every 32 bars = 8h
        _total_funding: float = 0.0

        for bar_i in range(total):
            ts            = bar_index[bar_i]
            current_price = float(bar_closes[bar_i])
            atr_now       = float(atr_full[period_iloc_start + bar_i])

            if bar_i % 144 == 0:
                pct = bar_i / total * 100
                dd_flag = ' [DD-HALT]' if self._dd_halted else ''
                print(f"  [{pct:5.1f}%]  {ts.strftime('%Y-%m-%d %H:%M')}  "
                      f"price=${current_price:,.0f}  trades={len(self.trades)}{dd_flag}")

            # ----------------------------------------------------------------
            # 1. Update trailing stop & trailing TP for open position
            # ----------------------------------------------------------------
            if self.position == 'LONG':
                # Fixed 2:1 TP target — check first (best price = limit exit)
                if self.fixed_tp > 0 and current_price >= self.fixed_tp:
                    self._close('Fixed-TP', ts, current_price)
                    self._last_close_bar = bar_i
                else:
                    # Trail stop upward
                    new_sl = current_price - self.entry_k * atr_now
                    self.trail_sl = max(self.trail_sl, new_sl)
                    # Track high-water
                    self.high_water = max(self.high_water, current_price)

                    # Check trail stop
                    if current_price <= self.trail_sl:
                        self._close('Trail-Stop', ts, current_price)
                        self._last_close_bar = bar_i
                    else:
                        # Check trail TP (only after min_profit_atr_mult × ATR gain)
                        gain = self.high_water - self.entry_price
                        if gain >= min_mult * self.entry_atr:
                            locked_floor = self.entry_price + gain * (1 - retr)
                            if current_price < locked_floor:
                                self._close('Trail-TP', ts, current_price)
                                self._last_close_bar = bar_i

            elif self.position == 'SHORT':
                # Fixed 2:1 TP target — check first
                if self.fixed_tp > 0 and current_price <= self.fixed_tp:
                    self._close('Fixed-TP', ts, current_price)
                    self._last_close_bar = bar_i
                else:
                    # Trail stop downward
                    new_sl = current_price + self.entry_k * atr_now
                    self.trail_sl = min(self.trail_sl, new_sl)
                    # Track low-water
                    self.high_water = min(self.high_water, current_price)

                    # Check trail stop
                    if current_price >= self.trail_sl:
                        self._close('Trail-Stop', ts, current_price)
                        self._last_close_bar = bar_i
                    else:
                        # Check trail TP
                        gain = self.entry_price - self.high_water
                        if gain >= min_mult * self.entry_atr:
                            locked_floor = self.entry_price - gain * (1 - retr)
                            if current_price > locked_floor:
                                self._close('Trail-TP', ts, current_price)
                                self._last_close_bar = bar_i

            # ----------------------------------------------------------------
            # 2a. Funding rate — charged every 32 bars (8h) on open positions
            # ----------------------------------------------------------------
            if self.position and bar_i % _funding_interval == 0 and bar_i > 0:
                rate = _funding_rate_long if self.position == 'LONG' else _funding_rate_short
                if rate > 0:
                    funding_cost = self.notional * rate
                    self.capital       -= funding_cost
                    _total_funding     += funding_cost
                    self._total_friction += funding_cost

            # ----------------------------------------------------------------
            # 2. Equity snapshot & daily drawdown check (prop-desk style)
            # ----------------------------------------------------------------
            equity = self._equity(current_price)

            # Daily reset: new trading day → fresh capital baseline, lift halt
            current_day = ts.date()
            if current_day != self._current_day:
                self._current_day       = current_day
                self._day_start_capital = self.capital   # reset to current capital
                self._dd_halted         = False          # fresh start each day

            # Activate daily kill switch
            daily_dd = (self._day_start_capital - equity) / self._day_start_capital
            if daily_dd >= max_dd and not self._dd_halted:
                self._dd_halted = True
                print(f"  ⛔ DD KILL SWITCH  {ts.strftime('%m/%d %H:%M')}  "
                      f"daily loss={daily_dd*100:.1f}%  "
                      f"day_start=${self._day_start_capital:,.0f}  equity=${equity:,.0f}")

            self.equity_curve.append({'ts': ts, 'price': current_price, 'equity': equity})

            # ----------------------------------------------------------------
            # 3. Skip orchestrator when in position or on cooldown
            # ----------------------------------------------------------------
            if self.position in ('LONG', 'SHORT'):
                self.decision_log.append({'ts': ts, 'price': current_price,
                                          'action': 'HOLD', 'reason': 'in position',
                                          'equity': equity})
                continue

            cooldown_bars = risk_cfg.get('cooldown_bars', 16)
            if self._last_close_bar >= 0 and bar_i - self._last_close_bar < cooldown_bars:
                self.decision_log.append({'ts': ts, 'price': current_price,
                                          'action': 'WAIT', 'reason': 'cooldown',
                                          'equity': equity})
                continue

            # ----------------------------------------------------------------
            # 4. Build look-ahead-free MTF slices (skipped in precomputed mode)
            # ----------------------------------------------------------------
            if _precomputed:
                # Need only the last 20 15m bars for entry filters
                _end_15m   = period_iloc_start + bar_i + 1
                _start_15m = max(0, _end_15m - 20)
                df_15m_filter = bars_15m.iloc[_start_15m:_end_15m]
                slices = None    # not used in precomputed mode
            else:
                slices = slice_from_pos(mtf_all, aligned_idx[bar_i])
                if any(len(v) == 0 for v in slices.values()):
                    continue
                df_15m_filter = slices.get('15m', pd.DataFrame())

            # ----------------------------------------------------------------
            # 5. Orchestrator decision
            # ----------------------------------------------------------------
            try:
                if _precomputed:
                    decision = precomputed_states.decide_fast(
                        aligned_idx[bar_i], current_price
                    )
                else:
                    decision = self.orchestrator.decide(slices, current_price=current_price)
            except Exception as exc:
                self.decision_log.append({'ts': ts, 'price': current_price,
                                          'action': 'ERROR', 'reason': str(exc)[:80],
                                          'equity': equity})
                continue

            action = decision.action
            reason = decision.reason

            # ----------------------------------------------------------------
            # 6. Entry — score filter + DD kill switch + vol-adjusted sizing
            # ----------------------------------------------------------------
            min_score = self.config.get('strategy', {}).get('min_entry_score', 0.0)

            if action in ('BUY', 'SELL'):
                if decision.score < min_score:
                    action = 'WAIT'
                    reason = f'Score {decision.score:.2f} < {min_score:.2f}'

                elif self._dd_halted:
                    action = 'WAIT'
                    reason = 'DD kill switch active'

                else:
                    df_15m = df_15m_filter

                    # --- Filter 1: 15m momentum confirmation ---
                    # Last bar must agree with trade direction to avoid entering
                    # just as price is reversing against the signal.
                    if not df_15m.empty and len(df_15m) >= 2:
                        last_bar = df_15m.iloc[-1]
                        prev_bar = df_15m.iloc[-2]
                        last_bullish = last_bar['close'] > last_bar['open']
                        last_bearish = last_bar['close'] < last_bar['open']
                        prev_bullish = prev_bar['close'] > prev_bar['open']
                        prev_bearish = prev_bar['close'] < prev_bar['open']
                        # Require at least 1 of the last 2 bars to confirm direction
                        if action == 'BUY' and not (last_bullish or prev_bullish):
                            action = 'WAIT'
                            reason = 'No 15m bullish momentum (last 2 bars both bearish)'
                        elif action == 'SELL' and not (last_bearish or prev_bearish):
                            action = 'WAIT'
                            reason = 'No 15m bearish momentum (last 2 bars both bullish)'

                    # --- Filter 2: Pullback entry filter ---
                    # Avoid entering when price is at or near a recent 3-hour high/low.
                    # The HSMM fires "bullish" after a sustained rally → we'd be
                    # buying the top.  Require at least 0.6% pullback from the 12-bar
                    # high (LONG) or 0.6% bounce from the 12-bar low (SHORT) before entry.
                    if action in ('BUY', 'SELL') and not df_15m.empty and len(df_15m) >= 12:
                        lookback = df_15m.tail(12)    # 12 × 15m = 3 hours
                        recent_high = float(lookback['high'].max())
                        recent_low  = float(lookback['low'].min())
                        pullback_from_high = (recent_high - current_price) / recent_high
                        bounce_from_low    = (current_price - recent_low)  / recent_low
                        if action == 'BUY' and pullback_from_high < 0.006:
                            action = 'WAIT'
                            reason = f'At 3h high (pullback={pullback_from_high:.2%} < 0.6%)'
                        elif action == 'SELL' and bounce_from_low < 0.006:
                            action = 'WAIT'
                            reason = f'At 3h low (bounce={bounce_from_low:.2%} < 0.6%)'
                    # --- Filter 3: Volume confirmation ---
                    # Only enter when 15m volume is above its 20-bar average.
                    # Low-volume moves are more likely to be noise; institutional
                    # participation requires above-average volume at entry.
                    if action in ('BUY', 'SELL') and not df_15m.empty \
                            and 'volume' in df_15m.columns and len(df_15m) >= 20:
                        vol_now  = float(df_15m['volume'].iloc[-1])
                        vol_avg  = float(df_15m['volume'].tail(20).mean())
                        if vol_avg > 0 and vol_now < 1.2 * vol_avg:
                            action = 'WAIT'
                            reason = f'Low volume ({vol_now/vol_avg:.2f}× avg, need 1.2×)'


                    # Guard: filters above may have changed action to 'WAIT'.
                    # Only enter if still BUY or SELL — never fall through to SHORT
                    # when a filtered-out BUY hits the else branch.
                    if action not in ('BUY', 'SELL'):
                        pass  # filtered out — skip entry silently
                    elif bar_i + 1 >= total:
                        pass  # no next bar to fill — skip (last bar of period)
                    else:
                        regime_comp = decision.components.get('regime')
                        dominant    = (regime_comp.metadata.get('dominant_state', 'Range')
                                       if regime_comp else 'Range')
                        k = k_regime.get(dominant, risk_cfg['atr_sl_multiplier'])

                        # ---- Fill at open of NEXT bar (realistic execution) ----
                        fill_price = float(bar_opens[bar_i + 1])

                        notional, lev = self._compute_position_size(fill_price, atr_now, k)

                        # Market-impact slippage: base + size/bar_vol component
                        bar_dol_vol  = max(fill_price * bar_volumes[bar_i + 1], 1.0)
                        impact_slip  = risk_cfg.get('market_impact_factor', 2.0) * (notional / bar_dol_vol)
                        total_slip   = risk_cfg.get('slippage_base', 0.0002) + impact_slip
                        total_slip   = min(total_slip, 0.005)   # cap at 0.5% per side
                        friction_now = risk_cfg['fee_pct'] + total_slip

                        # Apply entry friction (fee + slippage worsens fill)
                        eff_entry  = fill_price * (1 + friction_now) if action == 'BUY' \
                                     else fill_price * (1 - friction_now)
                        entry_cost = notional * friction_now
                        self.capital         -= entry_cost
                        self._total_friction += entry_cost

                        self.entry_price = eff_entry
                        self.entry_atr   = atr_now
                        self.entry_k     = k
                        self.notional    = notional

                        stop_dist = k * atr_now
                        if action == 'BUY':
                            self.trail_sl   = eff_entry - stop_dist
                            self.high_water = eff_entry
                            self.position   = 'LONG'
                            self.fixed_tp   = eff_entry + 2.5 * stop_dist
                        else:
                            self.trail_sl   = eff_entry + stop_dist
                            self.high_water = eff_entry
                            self.position   = 'SHORT'
                            self.fixed_tp   = eff_entry - 2.5 * stop_dist

                        self._log_entry(bar_index[bar_i + 1], eff_entry, decision,
                                        lev, k, atr_now,
                                        slip_bps=total_slip * 10_000)

            self.decision_log.append({
                'ts':     ts,
                'price':  current_price,
                'action': action,
                'reason': (reason or '')[:80],
                'equity': equity,
            })

        # Close any open position at end of period
        if self.position:
            last_price = float(bars_in_period['close'].iloc[-1])
            self._close('End-of-period', bar_index[-1], last_price)

        return self._compute_metrics(bars_in_period)

    # -----------------------------------------------------------------------
    # Position helpers
    # -----------------------------------------------------------------------

    def _equity(self, price: float) -> float:
        """Mark-to-market equity for open position."""
        if self.position is None:
            return self.capital
        if self.position == 'LONG':
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - price) / self.entry_price
        pnl = self.notional * pnl_pct
        return self.capital + pnl

    def _close(self, reason: str, ts, price: float):
        risk_cfg = self.config['risk']
        # Exit slippage: base only (passive fill — no market impact on stop exits)
        friction = risk_cfg['fee_pct'] + risk_cfg.get('slippage_base', 0.0002)

        side = self.position
        # Apply exit friction (worsens fill)
        eff_exit = price * (1 - friction) if side == 'LONG' \
                   else price * (1 + friction)

        if side == 'LONG':
            pnl_pct = (eff_exit - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - eff_exit) / self.entry_price

        pnl = self.notional * pnl_pct
        exit_cost = self.notional * friction
        self._total_friction += exit_cost

        self.capital += pnl
        self.trades.append({
            'side':        side,
            'reason':      reason,
            'ts':          ts,
            'entry_price': self.entry_price,
            'exit_price':  eff_exit,
            'pnl':         round(pnl, 2),
            'pnl_pct':     round(pnl_pct * 100, 3),
            'won':         pnl > 0,
            'notional':    round(self.notional, 2),
        })
        marker = '✅' if pnl > 0 else '❌'
        print(f"  {marker}  CLOSE {side:<5}  {ts.strftime('%m/%d %H:%M')} @ ${price:,.0f}"
              f"  PnL: ${pnl:+,.2f} ({pnl_pct*100:+.2f}%)  [{reason}]")
        self.position    = None
        self.entry_price = self.entry_atr = self.entry_k = 0.0
        self.notional    = self.trail_sl  = self.high_water = self.fixed_tp = 0.0

    def _log_entry(self, ts, price: float, decision, leverage: float,
                   k: float, atr: float, slip_bps: float = 0.0):
        score = decision.score if decision else 0
        side  = self.position
        icon  = '🟢' if side == 'LONG' else '🔴'
        print(f"  {icon} ENTRY {side:<5}  {ts.strftime('%m/%d %H:%M')} @ ${price:,.0f}"
              f"  score={score:.2f}  lev={leverage:.1f}×  "
              f"k={k:.1f}  ATR={atr:.0f}  slip={slip_bps:.1f}bps  "
              f"SL=${self.trail_sl:,.0f}  TP=${self.fixed_tp:,.0f}")

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

        winners  = sum(1 for t in self.trades if t['won'])
        pnls     = [t['pnl'] for t in self.trades]
        win_pnls = [p for p in pnls if p > 0]
        los_pnls = [abs(p) for p in pnls if p < 0]

        total_return  = (self.capital - self.initial_capital) / self.initial_capital * 100
        win_rate      = winners / n * 100
        avg_win       = float(np.mean(win_pnls)) if win_pnls else 0.0
        avg_loss      = float(np.mean(los_pnls)) if los_pnls else 0.0
        profit_factor = (sum(win_pnls) / sum(los_pnls)) if los_pnls else float('inf')

        # ---- Drawdown (depth + duration) ----
        eq_vals = np.array([e['equity'] for e in self.equity_curve])
        eq_ts   = [e['ts'] for e in self.equity_curve]
        peak_eq = eq_vals[0]
        max_dd  = 0.0
        dd_start_idx = 0
        max_dd_duration_bars = 0
        current_dd_start = 0
        for i, e in enumerate(eq_vals):
            if e > peak_eq:
                peak_eq = e
                current_dd_start = i
            dd = (peak_eq - e) / peak_eq * 100
            if dd > max_dd:
                max_dd = dd
                dd_start_idx = current_dd_start
            if dd > 0:
                max_dd_duration_bars = max(max_dd_duration_bars, i - current_dd_start)

        # ---- Sharpe & Sortino (annualised, 15-min bars) ----
        # 15m bars → 4×24×365 = 35040 bars/year
        bars_per_year = 35040.0
        eq_returns = np.diff(eq_vals) / eq_vals[:-1]  # bar-by-bar returns
        mean_r = float(np.mean(eq_returns))
        std_r  = float(np.std(eq_returns, ddof=1)) if len(eq_returns) > 1 else 1e-8
        sharpe = (mean_r / (std_r + 1e-10)) * np.sqrt(bars_per_year)

        downside = eq_returns[eq_returns < 0]
        sortino_denom = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-8
        sortino = (mean_r / (sortino_denom + 1e-10)) * np.sqrt(bars_per_year)

        # Annualised return for Calmar
        n_bars  = len(eq_vals)
        years   = n_bars / bars_per_year
        ann_ret = (self.capital / self.initial_capital) ** (1 / max(years, 1e-3)) - 1
        calmar  = (ann_ret * 100) / max(max_dd, 1e-4)

        # ---- Sharpe t-statistic (significance test) ----
        # t = Sharpe × √N_trades — rule: t > 2.0 = statistically significant
        sharpe_tstat = sharpe * np.sqrt(max(n, 1)) / np.sqrt(bars_per_year / (n_bars / max(n, 1)))

        # ---- Exposure & holding period ----
        hold_bars = [t.get('bars_held', 0) for t in self.trades]
        exposure_pct = sum(1 for e in self.decision_log
                          if e.get('action') == 'HOLD') / max(len(self.decision_log), 1) * 100

        # ---- BTC Buy-and-Hold ----
        bh_return = (float(bars['close'].iloc[-1]) - float(bars['close'].iloc[0])) \
                    / float(bars['close'].iloc[0]) * 100

        return {
            'initial_capital':   self.initial_capital,
            'final_capital':     round(self.capital, 2),
            'total_return':      round(total_return, 2),
            'ann_return':        round(ann_ret * 100, 2),
            'bh_return':         round(bh_return, 2),

            # Risk-adjusted
            'sharpe':            round(sharpe, 3),
            'sortino':           round(sortino, 3),
            'calmar':            round(calmar, 3),
            'sharpe_tstat':      round(sharpe_tstat, 2),

            # Drawdown
            'max_drawdown':      round(max_dd, 2),
            'max_dd_duration_bars': max_dd_duration_bars,

            # Trade stats
            'trades':            n,
            'winners':           winners,
            'losers':            n - winners,
            'win_rate':          round(win_rate, 1),
            'avg_win':           round(avg_win, 2),
            'avg_loss':          round(avg_loss, 2),
            'profit_factor':     round(profit_factor, 2),
            'exposure_pct':      round(exposure_pct, 1),

            # Costs
            'total_friction':    round(self._total_friction, 2),
            'friction_pct':      round(self._total_friction / self.initial_capital * 100, 2),

            'trade_list':        self.trades,
            'log':               self.decision_log,
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

    # ---- Performance ----
    print(f"\n  Performance")
    print(f"  {'Capital initial':<26}  ${res['initial_capital']:>10,.2f}")
    print(f"  {'Capital final':<26}  ${res['final_capital']:>10,.2f}")
    print(f"  {'Rendement total':<26}  {pct_str(res['total_return']):>11}")
    print(f"  {'Rendement annualisé':<26}  {pct_str(res.get('ann_return', 0)):>11}")
    print(f"  {'BTC Buy-and-Hold':<26}  {pct_str(res['bh_return']):>11}")

    # ---- Risk-adjusted (the institutional scorecard) ----
    print(f"\n  Risk-Adjusted  (cible: Sharpe>1.5 | DD<10% | PF>1.8)")
    sh   = res.get('sharpe', 0)
    so   = res.get('sortino', 0)
    cal  = res.get('calmar', 0)
    tstat = res.get('sharpe_tstat', 0)
    sh_flag  = '✅' if sh   >= 1.5  else ('⚠️ ' if sh  >= 0.8  else '❌')
    so_flag  = '✅' if so   >= 2.0  else ('⚠️ ' if so  >= 1.0  else '❌')
    cal_flag = '✅' if cal  >= 1.0  else ('⚠️ ' if cal >= 0.5  else '❌')
    t_flag   = '✅' if tstat >= 2.0  else ('⚠️ ' if tstat >= 1.0 else '❌')
    print(f"  {'Sharpe ratio':<26}  {sh:>10.3f}  {sh_flag}")
    print(f"  {'Sortino ratio':<26}  {so:>10.3f}  {so_flag}")
    print(f"  {'Calmar ratio':<26}  {cal:>10.3f}  {cal_flag}")
    print(f"  {'Sharpe t-stat (signif.)':<26}  {tstat:>10.2f}  {t_flag}  (>2.0 = significatif)")

    # ---- Drawdown ----
    dd    = res['max_drawdown']
    dd_flag = '✅' if dd <= 10 else ('⚠️ ' if dd <= 20 else '❌')
    dd_dur_h = res.get('max_dd_duration_bars', 0) / 4  # 15m → hours
    print(f"\n  Drawdown")
    print(f"  {'Max Drawdown':<26}  {dd:>10.2f}%  {dd_flag}")
    print(f"  {'Durée max DD (heures)':<26}  {dd_dur_h:>10.0f}h")

    # ---- Trades ----
    longs  = [t for t in res['trade_list'] if t.get('side') == 'LONG']
    shorts = [t for t in res['trade_list'] if t.get('side') == 'SHORT']
    exit_reasons = {}
    for t in res['trade_list']:
        r = t.get('reason', 'Unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    print(f"\n  Trades")
    print(f"  {'Total':<26}  {res['trades']:>11}")
    print(f"  {'  LONG':<26}  {len(longs):>11}")
    print(f"  {'  SHORT':<26}  {len(shorts):>11}")
    print(f"  {'Gagnants':<26}  {res['winners']:>11}")
    print(f"  {'Perdants':<26}  {res['losers']:>11}")
    print(f"  {'Win Rate':<26}  {res['win_rate']:>10.1f}%")
    print(f"  {'Gain moyen':<26}  ${res['avg_win']:>10,.2f}")
    print(f"  {'Perte moyenne':<26}  ${res['avg_loss']:>10,.2f}")
    print(f"  {'Profit Factor':<26}  {res['profit_factor']:>11.2f}")
    print(f"  {'Exposition marché':<26}  {res.get('exposure_pct', 0):>10.1f}%")
    print(f"\n  Sorties")
    for reason, cnt in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20}  {cnt:>3}  ({cnt/res['trades']*100:.0f}%)")

    # ---- Transaction costs ----
    print(f"\n  Friction (fees + slippage)")
    print(f"  {'Total coûts':<26}  ${res.get('total_friction', 0):>10,.2f}")
    print(f"  {'En % du capital':<26}  {res.get('friction_pct', 0):>10.2f}%")

    # ---- Trade detail ----
    print(f"\n  Détail des trades")
    for i, t in enumerate(res['trade_list'], 1):
        marker = '✅' if t['won'] else '❌'
        side   = t.get('side', 'LONG')
        ts_str = t['ts'].strftime('%m/%d %H:%M') if hasattr(t['ts'], 'strftime') else str(t['ts'])
        notional = t.get('notional', 0)
        print(f"    {i:>2}. {marker} [{side:<5}]  {ts_str}  "
              f"${t['entry_price']:,.0f}→${t['exit_price']:,.0f}  "
              f"notional=${notional:,.0f}  "
              f"PnL: ${t['pnl']:+,.2f} ({t['pnl_pct']:+.2f}%)  [{t['reason']}]")

    print(f"\n{'═'*70}\n")


def pct_str(v: float) -> str:
    return f'{v:+.2f}%'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='NYX v1.0 MTF Backtest — Institutional Risk Engine')
    parser.add_argument('--pair',     default='BTCUSDT')
    parser.add_argument('--start',    default='2023-02-01')
    parser.add_argument('--end',      default='2023-02-15')
    parser.add_argument('--capital',  type=float, default=10_000.0)
    parser.add_argument('--data-dir', default='data/raw/mtf')
    parser.add_argument('--pretrain-months', type=int, default=0,
                        help='Months before --start to pre-train via EM (0=disabled)')
    parser.add_argument('--pretrain-all', action='store_true',
                        help='Use ALL data before --start for EM pre-training')
    parser.add_argument('--em-iters', type=int, default=30,
                        help='EM iterations for pre-training (default: 30)')
    parser.add_argument('--use-cache', action='store_true',
                        help='Load/save EM pre-training from disk cache (~data/pretrain_cache/)')
    parser.add_argument('--precompute', action='store_true',
                        help='Precompute HSMM states + SMC for entire period (~70x faster hot loop)')
    parser.add_argument('--precompute-cache', action='store_true',
                        help='Load/save precomputed states from disk (~data/pretrain_cache/)')
    args = parser.parse_args()

    import time as _time
    mtf_all = load_mtf(args.pair, args.data_dir)

    bt = MTFBacktest(BASE_CONFIG, args.capital)

    # EM pre-training: all data before backtest start, or N months
    if args.pretrain_all:
        start_ts = pd.Timestamp(args.start)
        data_start = mtf_all['4h'].index[0]
        months_available = int((start_ts - data_start).days / 30)
        print(f"\n  [--pretrain-all]  {data_start.date()} → {start_ts.date()}"
              f"  ({months_available}M de données)")
        pretrain_agents(
            bt.orchestrator,
            mtf_all,
            pretrain_end=args.start,
            pretrain_months=months_available,
            em_iters=args.em_iters,
            pair=args.pair,
            use_cache=args.use_cache,
        )
    elif args.pretrain_months > 0:
        pretrain_agents(
            bt.orchestrator,
            mtf_all,
            pretrain_end=args.start,
            pretrain_months=args.pretrain_months,
            em_iters=args.em_iters,
            pair=args.pair,
            use_cache=args.use_cache,
        )

    # ---- Precomputed states (optional speedup) ----
    precomp = None
    if args.precompute:
        precomp = PrecomputedStates(bt.orchestrator)
        precomp_key = ''
        if args.precompute_cache:
            import hashlib as _hlib
            blob = f"{args.pair}|{args.start}|{args.end}|precomp_v1"
            precomp_key = _hlib.md5(blob.encode()).hexdigest()[:16]
        precomp.precompute(mtf_all, cache_key=precomp_key,
                           start_ts=args.start, end_ts=args.end)

    t0 = _time.time()
    results = bt.run(mtf_all, args.start, args.end, precomputed_states=precomp)
    elapsed = _time.time() - t0
    print(f"\n  Durée backtest : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print_report(results, args.start, args.end)


if __name__ == '__main__':
    main()
