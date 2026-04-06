"""
Smart Money Concepts (SMC) Detection - v2
Wraps the `smartmoneyconcepts` library for production-grade pattern detection
with a custom ChoCH/structure layer and a clean quality-score API.

Library (numba-accelerated):
  pip install smartmoneyconcepts

Patterns detected:
  Order Blocks (OB)         — swing-structure aware, volume-weighted
  Fair Value Gaps (FVG)     — with mitigation tracking
  Liquidity sweeps          — stop-hunt detection
  Break of Structure (BOS)  — trend continuation
  Change of Character (ChoCH) — trend reversal (first opposing structure break)
  Price-at-zone             — current price touching an unmitigated zone

Public API (backwards-compatible with v1):
  detect_all(df)  → dict with bool flags + smc_score_bullish/bearish
  smc_score(df, context) → float 0-1
  get_ob_zones(df, lookback) → list[dict]
  get_fvg_zones(df, lookback) → list[dict]
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

try:
    from smartmoneyconcepts import smc as _smc_lib
    _LIB_AVAILABLE = True
except ImportError:
    _LIB_AVAILABLE = False


class SMCDetector:
    """
    SMC v2 — Smart Money Concepts detector.

    Uses the `smartmoneyconcepts` library (numba-accelerated) when available
    for OB and FVG detection, with a vectorized fallback implementation.

    detect_all() returns:
      Boolean flags (backwards-compatible with v1):
        bullish_ob, bearish_ob, bullish_fvg, bearish_fvg,
        bos_bullish, bos_bearish, liquidity_sweep
        bullish_choch, bearish_choch       ← NEW
        bullish_at_zone, bearish_at_zone   ← NEW

      Aggregate scores:
        smc_score_bullish   float 0-1
        smc_score_bearish   float 0-1
    """

    def __init__(
        self,
        ob_range_threshold: float = 0.015,
        fvg_min_gap: float = 0.005,
        liquidity_lookback: int = 20,
        ob_lookback: int = 30,
        fvg_lookback: int = 30,
        swing_lookback: int = 5,
        displacement_atr_mult: float = 0.7,
        mitigation_tolerance: float = 0.003,
    ):
        if not isinstance(ob_range_threshold, (int, float)):
            raise TypeError(f"ob_range_threshold must be numeric, got {type(ob_range_threshold).__name__}")
        if not isinstance(fvg_min_gap, (int, float)):
            raise TypeError(f"fvg_min_gap must be numeric, got {type(fvg_min_gap).__name__}")
        if not isinstance(liquidity_lookback, int):
            raise TypeError(f"liquidity_lookback must be int, got {type(liquidity_lookback).__name__}")

        self.ob_range_threshold    = float(ob_range_threshold)
        self.fvg_min_gap           = float(fvg_min_gap)
        self.liquidity_lookback    = int(liquidity_lookback)
        self.ob_lookback           = int(ob_lookback)
        self.fvg_lookback          = int(fvg_lookback)
        self.swing_lookback        = int(swing_lookback)
        self.displacement_atr_mult = float(displacement_atr_mult)
        self.mitigation_tolerance  = float(mitigation_tolerance)

        self.diagnostics: Dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_all(self, data: pd.DataFrame) -> Dict:
        """
        Detect all SMC patterns. Returns dict with bool flags and quality scores.
        """
        if len(data) < 10:
            return self._empty_result()

        if _LIB_AVAILABLE:
            try:
                return self._detect_with_library(data)
            except Exception:
                pass   # silent fallback

        return self._detect_fallback(data)

    def smc_score(self, data: pd.DataFrame, context: str) -> float:
        """float 0-1 directional quality score. context: 'bullish'|'bearish'|'neutral'"""
        r = self.detect_all(data)
        if context == 'bullish':
            return float(r['smc_score_bullish'])
        elif context == 'bearish':
            return float(r['smc_score_bearish'])
        return float(max(r['smc_score_bullish'], r['smc_score_bearish']))

    # ------------------------------------------------------------------
    # Library-backed detection
    # ------------------------------------------------------------------

    def _detect_with_library(self, data: pd.DataFrame) -> Dict:
        """
        Use smartmoneyconcepts (numba) for OB and FVG, custom code for ChoCH.
        """
        # Library requires integer-indexed DataFrame
        df = data.reset_index(drop=True)
        T = len(df)
        cur_price = float(df['close'].iloc[-1])

        # ---- Swing detection (shared prerequisite) ----
        swings = _smc_lib.swing_highs_lows(df, swing_length=self.swing_lookback)

        # ---- Order Blocks ----
        ob_df = _smc_lib.ob(df, swings, close_mitigation=False)
        bull_obs, bear_obs = self._extract_active_zones(ob_df, T)

        # ---- Fair Value Gaps ----
        fvg_df = _smc_lib.fvg(df, join_consecutive=True)
        bull_fvgs, bear_fvgs = self._extract_active_zones(fvg_df, T)

        # ---- Price-at-zone ----
        bull_at_zone = self._price_in_any_zone(cur_price, bull_obs + bull_fvgs)
        bear_at_zone = self._price_in_any_zone(cur_price, bear_obs + bear_fvgs)

        # ---- BOS / ChoCH (custom vectorized) ----
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        c = df['close'].values.astype(float)
        sh_idx, sl_idx = self._find_swings(h, l, self.swing_lookback)

        bos_bull  = self._detect_bos(c, sh_idx, bullish=True)
        bos_bear  = self._detect_bos(c, sl_idx, bullish=False)
        choch_bull = self._detect_choch(c, sh_idx, sl_idx, bullish=True)
        choch_bear = self._detect_choch(c, sh_idx, sl_idx, bullish=False)

        # ---- Liquidity sweep ----
        liq_sweep = self._detect_liquidity_sweep_np(h, l, c)

        # ---- Volume spike ----
        vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
        vol_ok = self._volume_spike(vol)

        # ---- Aggregate scores ----
        bull_score = self._aggregate_score(
            has_ob=bool(bull_obs), has_fvg=bool(bull_fvgs),
            at_zone=bull_at_zone, choch=choch_bull, bos=bos_bull,
            sweep=liq_sweep, vol_ok=vol_ok,
        )
        bear_score = self._aggregate_score(
            has_ob=bool(bear_obs), has_fvg=bool(bear_fvgs),
            at_zone=bear_at_zone, choch=choch_bear, bos=bos_bear,
            sweep=liq_sweep, vol_ok=vol_ok,
        )

        self.diagnostics = {
            'backend': 'smartmoneyconcepts',
            'bull_obs': len(bull_obs), 'bear_obs': len(bear_obs),
            'bull_fvgs': len(bull_fvgs), 'bear_fvgs': len(bear_fvgs),
        }

        return {
            'bullish_ob':        bool(bull_obs),
            'bearish_ob':        bool(bear_obs),
            'bullish_fvg':       bool(bull_fvgs),
            'bearish_fvg':       bool(bear_fvgs),
            'liquidity_sweep':   liq_sweep,
            'bos_bullish':       bos_bull,
            'bos_bearish':       bos_bear,
            'bullish_choch':     choch_bull,
            'bearish_choch':     choch_bear,
            'bullish_at_zone':   bull_at_zone,
            'bearish_at_zone':   bear_at_zone,
            'smc_score_bullish': bull_score,
            'smc_score_bearish': bear_score,
        }

    def _extract_active_zones(self, df: pd.DataFrame, T: int):
        """
        From an OB or FVG DataFrame, extract unmitigated zones in the last
        `ob_lookback` / `fvg_lookback` bars.

        A zone is "active" (unmitigated) when MitigatedIndex == 0.0 in the
        smartmoneyconcepts convention (0 = not yet mitigated).

        Returns (bull_zones, bear_zones) as list of {'high', 'low'} dicts.
        """
        if df is None or df.empty:
            return [], []

        is_ob = 'OB' in df.columns
        signal_col = 'OB' if is_ob else 'FVG'
        top_col    = 'Top'
        bot_col    = 'Bottom'

        lookback = self.ob_lookback if is_ob else self.fvg_lookback
        start = max(0, T - lookback)

        recent = df.iloc[start:]

        bull_zones, bear_zones = [], []
        for idx, row in recent.iterrows():
            sig = row.get(signal_col)
            if pd.isna(sig) or sig == 0:
                continue
            top = row.get(top_col)
            bot = row.get(bot_col)
            if pd.isna(top) or pd.isna(bot):
                continue
            mit = row.get('MitigatedIndex', 0)
            is_unmitigated = (pd.isna(mit) or mit == 0.0)
            if not is_unmitigated:
                continue

            zone = {'high': float(top), 'low': float(bot)}
            if sig == 1:
                bull_zones.append(zone)
            elif sig == -1:
                bear_zones.append(zone)

        return bull_zones, bear_zones

    # ------------------------------------------------------------------
    # Fallback detection (pure numpy, no external library)
    # ------------------------------------------------------------------

    def _detect_fallback(self, data: pd.DataFrame) -> Dict:
        """Pure-numpy fallback when smartmoneyconcepts is unavailable."""
        o = data['open'].values.astype(float)
        h = data['high'].values.astype(float)
        l = data['low'].values.astype(float)
        c = data['close'].values.astype(float)
        vol = data['volume'].values.astype(float) if 'volume' in data.columns else None

        cur_price = c[-1]
        atr = self._atr(h, l, c, 14)
        sh_idx, sl_idx = self._find_swings(h, l, self.swing_lookback)

        bull_obs  = self._find_order_blocks(o, h, l, c, vol, atr, bullish=True)
        bear_obs  = self._find_order_blocks(o, h, l, c, vol, atr, bullish=False)
        bull_fvgs = self._find_fvgs(h, l, c, bullish=True)
        bear_fvgs = self._find_fvgs(h, l, c, bullish=False)

        bull_at_zone = self._price_in_any_zone(cur_price, bull_obs + bull_fvgs)
        bear_at_zone = self._price_in_any_zone(cur_price, bear_obs + bear_fvgs)

        bos_bull   = self._detect_bos(c, sh_idx, bullish=True)
        bos_bear   = self._detect_bos(c, sl_idx, bullish=False)
        choch_bull = self._detect_choch(c, sh_idx, sl_idx, bullish=True)
        choch_bear = self._detect_choch(c, sh_idx, sl_idx, bullish=False)
        liq_sweep  = self._detect_liquidity_sweep_np(h, l, c)
        vol_ok     = self._volume_spike(vol)

        bull_score = self._aggregate_score(
            has_ob=bool(bull_obs), has_fvg=bool(bull_fvgs),
            at_zone=bull_at_zone, choch=choch_bull, bos=bos_bull,
            sweep=liq_sweep, vol_ok=vol_ok,
        )
        bear_score = self._aggregate_score(
            has_ob=bool(bear_obs), has_fvg=bool(bear_fvgs),
            at_zone=bear_at_zone, choch=choch_bear, bos=bos_bear,
            sweep=liq_sweep, vol_ok=vol_ok,
        )

        self.diagnostics = {
            'backend': 'fallback',
            'bull_obs': len(bull_obs), 'bear_obs': len(bear_obs),
            'bull_fvgs': len(bull_fvgs), 'bear_fvgs': len(bear_fvgs),
        }

        return {
            'bullish_ob':        bool(bull_obs),
            'bearish_ob':        bool(bear_obs),
            'bullish_fvg':       bool(bull_fvgs),
            'bearish_fvg':       bool(bear_fvgs),
            'liquidity_sweep':   liq_sweep,
            'bos_bullish':       bos_bull,
            'bos_bearish':       bos_bear,
            'bullish_choch':     choch_bull,
            'bearish_choch':     choch_bear,
            'bullish_at_zone':   bull_at_zone,
            'bearish_at_zone':   bear_at_zone,
            'smc_score_bullish': bull_score,
            'smc_score_bearish': bear_score,
        }

    # ------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------

    def _find_swings(self, h: np.ndarray, l: np.ndarray, n: int):
        """Vectorized pivot detection: swing high/low indices."""
        T = len(h)
        if T < 2 * n + 1:
            return np.array([], dtype=int), np.array([], dtype=int)
        sh_idx, sl_idx = [], []
        for i in range(n, T - n):
            if h[i] == h[i - n: i + n + 1].max():
                sh_idx.append(i)
            if l[i] == l[i - n: i + n + 1].min():
                sl_idx.append(i)
        return np.array(sh_idx, dtype=int), np.array(sl_idx, dtype=int)

    # ------------------------------------------------------------------
    # Fallback OB / FVG
    # ------------------------------------------------------------------

    def _find_order_blocks(self, o, h, l, c, vol, atr, bullish: bool) -> List[Dict]:
        T = len(c)
        end, start = T, max(0, T - self.ob_lookback - 2)
        obs = []
        for i in range(start, end - 1):
            atr_here = float(atr[min(i+1, len(atr)-1)])
            imp_body = abs(c[i+1] - o[i+1])
            if bullish:
                if not (c[i] < o[i] and c[i+1] > o[i+1] and imp_body > atr_here * self.displacement_atr_mult):
                    continue
                zh, zl = o[i], c[i]
            else:
                if not (c[i] > o[i] and c[i+1] < o[i+1] and imp_body > atr_here * self.displacement_atr_mult):
                    continue
                zh, zl = c[i], o[i]
            if zh <= zl:
                continue
            post = slice(i+2, T)
            mitigated = (l[post].min() <= zh) if bullish else (h[post].max() >= zl)
            if mitigated:
                continue  # skip already-mitigated zones
            obs.append({'high': float(zh), 'low': float(zl)})
        return obs

    def _find_fvgs(self, h, l, c, bullish: bool) -> List[Dict]:
        T = len(c)
        end, start = T, max(1, T - self.fvg_lookback - 1)
        fvgs = []
        for i in range(start, end - 1):
            if bullish:
                gap_pct = (l[i+1] - h[i-1]) / max(h[i-1], 1e-6)
                if gap_pct <= self.fvg_min_gap:
                    continue
                fh, fl = l[i+1], h[i-1]
            else:
                gap_pct = (l[i-1] - h[i+1]) / max(l[i-1], 1e-6)
                if gap_pct <= self.fvg_min_gap:
                    continue
                fh, fl = l[i-1], h[i+1]
            if fh <= fl:
                continue
            post = slice(i+2, T)
            mitigated = (l[post].min() <= fh) if bullish else (h[post].max() >= fl)
            if mitigated:
                continue
            fvgs.append({'high': float(fh), 'low': float(fl)})
        return fvgs

    # ------------------------------------------------------------------
    # Price at zone
    # ------------------------------------------------------------------

    def _price_in_any_zone(self, cur_price: float, zones: List[Dict]) -> bool:
        """True if current price is within any zone (with small tolerance)."""
        for z in zones:
            tol = (z['high'] - z['low']) * 0.1  # 10% of zone width tolerance
            if z['low'] - tol <= cur_price <= z['high'] + tol:
                return True
        return False

    # ------------------------------------------------------------------
    # BOS / ChoCH
    # ------------------------------------------------------------------

    def _detect_bos(self, c: np.ndarray, swing_indices: np.ndarray, bullish: bool) -> bool:
        """Break of Structure: close exceeds the last swing high/low (continuation)."""
        relevant = swing_indices[swing_indices < len(c) - 1] if len(swing_indices) > 0 else np.array([])
        if len(relevant) == 0:
            return False
        level = float(c[relevant[-1]])
        return bool(c[-1] > level) if bullish else bool(c[-1] < level)

    def _detect_choch(self, c: np.ndarray, sh_idx: np.ndarray, sl_idx: np.ndarray,
                      bullish: bool) -> bool:
        """
        Change of Character: first structural break AGAINST the prevailing trend.

        Bullish ChoCH: descending swing highs (downtrend) + close breaks last swing HIGH
        Bearish ChoCH: ascending swing lows  (uptrend)   + close breaks last swing LOW
        """
        T = len(c)
        if T < 4:
            return False
        if bullish:
            rel = sh_idx[sh_idx < T - 1]
            if len(rel) < 2:
                return False
            if c[rel[-2]] <= c[rel[-1]]:   # not descending → no downtrend
                return False
            return bool(c[-1] > c[rel[-1]])
        else:
            rel = sl_idx[sl_idx < T - 1]
            if len(rel) < 2:
                return False
            if c[rel[-2]] >= c[rel[-1]]:   # not ascending → no uptrend
                return False
            return bool(c[-1] < c[rel[-1]])

    # ------------------------------------------------------------------
    # Liquidity sweep
    # ------------------------------------------------------------------

    def _detect_liquidity_sweep_np(self, h, l, c) -> bool:
        n = self.liquidity_lookback
        if len(c) < n + 2:
            return False
        hist_h = h[-(n+2):-1]
        hist_l = l[-(n+2):-1]
        swept_high = bool(h[-1] > hist_h.max() and c[-1] < c[-2])
        swept_low  = bool(l[-1] < hist_l.min() and c[-1] > c[-2])
        return swept_high or swept_low

    # ------------------------------------------------------------------
    # Volume & ATR helpers
    # ------------------------------------------------------------------

    def _volume_spike(self, vol: Optional[np.ndarray]) -> bool:
        if vol is None or len(vol) < 20:
            return False
        vol_ma = vol[-21:-1].mean()
        return bool(vol_ma > 0 and vol[-1] > 1.5 * vol_ma)

    def _atr(self, h, l, c, period: int = 14) -> np.ndarray:
        T = len(c)
        if T < 2:
            return np.full(T, float(h[0] - l[0]) if T > 0 else 1.0)
        tr = np.maximum(h[1:] - l[1:],
             np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr = np.empty(T)
        atr[0] = tr[0] if len(tr) > 0 else (h[0] - l[0])
        k = 1.0 / period
        for i in range(1, T):
            atr[i] = atr[i-1] * (1 - k) + (tr[i-1] * k if i <= len(tr) else 0)
        return atr

    # ------------------------------------------------------------------
    # Aggregate score
    # ------------------------------------------------------------------

    def _aggregate_score(self, has_ob, has_fvg, at_zone, choch, bos, sweep, vol_ok) -> float:
        """
        Weighted SMC quality score 0→1.
          at_zone  0.30  — price in unmitigated zone (strongest confluence)
          ob       0.20  — supply/demand zone exists
          choch    0.20  — structural reversal signal
          fvg      0.15  — imbalance zone exists
          bos      0.10  — trend continuation
          sweep    0.03  — stop hunt confirmation
          vol      0.02  — institutional volume
        """
        s = 0.0
        if at_zone:  s += 0.30
        if has_ob:   s += 0.20
        if choch:    s += 0.20
        if has_fvg:  s += 0.15
        if bos:      s += 0.10
        if sweep:    s += 0.03
        if vol_ok:   s += 0.02
        return float(min(s, 1.0))

    # ------------------------------------------------------------------
    # Backwards-compatible zone getters
    # ------------------------------------------------------------------

    def get_ob_zones(self, data: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        if _LIB_AVAILABLE:
            try:
                df = data.reset_index(drop=True)
                swings = _smc_lib.swing_highs_lows(df, swing_length=self.swing_lookback)
                ob_df = _smc_lib.ob(df, swings)
                bull, bear = self._extract_active_zones(ob_df, len(df))
                for z in bull: z['type'] = 'bullish'
                for z in bear: z['type'] = 'bearish'
                return bull + bear
            except Exception:
                pass
        o = data['open'].values.astype(float)
        h = data['high'].values.astype(float)
        l = data['low'].values.astype(float)
        c = data['close'].values.astype(float)
        vol = data['volume'].values.astype(float) if 'volume' in data.columns else None
        atr = self._atr(h, l, c)
        orig = self.ob_lookback; self.ob_lookback = lookback
        bull = self._find_order_blocks(o, h, l, c, vol, atr, bullish=True)
        bear = self._find_order_blocks(o, h, l, c, vol, atr, bullish=False)
        self.ob_lookback = orig
        for z in bull: z['type'] = 'bullish'
        for z in bear: z['type'] = 'bearish'
        return bull + bear

    def get_fvg_zones(self, data: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        if _LIB_AVAILABLE:
            try:
                df = data.reset_index(drop=True)
                fvg_df = _smc_lib.fvg(df)
                bull, bear = self._extract_active_zones(fvg_df, len(df))
                for z in bull: z['type'] = 'bullish'
                for z in bear: z['type'] = 'bearish'
                return bull + bear
            except Exception:
                pass
        h = data['high'].values.astype(float)
        l = data['low'].values.astype(float)
        c = data['close'].values.astype(float)
        orig = self.fvg_lookback; self.fvg_lookback = lookback
        bull = self._find_fvgs(h, l, c, bullish=True)
        bear = self._find_fvgs(h, l, c, bullish=False)
        self.fvg_lookback = orig
        for z in bull: z['type'] = 'bullish'
        for z in bear: z['type'] = 'bearish'
        return bull + bear

    # ------------------------------------------------------------------
    # Empty result
    # ------------------------------------------------------------------

    def _empty_result(self) -> Dict:
        return {
            'bullish_ob': False, 'bearish_ob': False,
            'bullish_fvg': False, 'bearish_fvg': False,
            'liquidity_sweep': False,
            'bos_bullish': False, 'bos_bearish': False,
            'bullish_choch': False, 'bearish_choch': False,
            'bullish_at_zone': False, 'bearish_at_zone': False,
            'smc_score_bullish': 0.0, 'smc_score_bearish': 0.0,
        }
