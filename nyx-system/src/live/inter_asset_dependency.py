"""
Inter-Asset Dependency Layer — Ticket 34.

Models dynamic lead/lag relationships, non-stationary correlation,
contagion, and regime-conditioned dependency between assets.

Operates AFTER per-asset GBM + Jesse + Fractal Quality,
BEFORE RiskEngine. Alpha untouched — only adjusts size_multiplier
and can suppress trades under high contagion.

Never changes trade direction. Never modifies GBM or Jesse.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# =========================================================================
# A1 — Lagged Feature Matrix
# =========================================================================

def compute_lagged_features(
    returns: Dict[str, pd.Series],
    asset_a: str,
    asset_b: str,
    lags: Sequence[int] = (1, 4, 8, 24),
    vol_window: int = 24,
) -> pd.DataFrame:
    """Build a lagged feature matrix for asset pair (A, B).

    For each lag L, computes ret_A(t-L) and ret_B(t-L).
    Also computes vol_ratio and momentum_spread.
    No future leakage — only uses past data at each row.
    """
    ra = returns[asset_a].copy()
    rb = returns[asset_b].copy()
    idx = ra.index.intersection(rb.index)
    ra = ra.reindex(idx)
    rb = rb.reindex(idx)

    df = pd.DataFrame(index=idx)
    df[f'ret_{asset_a}'] = ra
    df[f'ret_{asset_b}'] = rb

    for lag in lags:
        df[f'ret_{asset_a}_lag{lag}'] = ra.shift(lag)
        df[f'ret_{asset_b}_lag{lag}'] = rb.shift(lag)

    vol_a = ra.rolling(vol_window, min_periods=max(1, vol_window // 2)).std()
    vol_b = rb.rolling(vol_window, min_periods=max(1, vol_window // 2)).std()
    df['vol_ratio'] = vol_a / vol_b.clip(lower=1e-12)

    mom_a = ra.rolling(vol_window, min_periods=1).mean()
    mom_b = rb.rolling(vol_window, min_periods=1).mean()
    df['momentum_spread'] = mom_a - mom_b

    df = df.dropna()
    return df


# =========================================================================
# A2 — Lead/Lag Detection
# =========================================================================

def detect_lead_lag(
    returns: Dict[str, pd.Series],
    asset_a: str,
    asset_b: str,
    max_lag: int = 24,
) -> Dict[str, Any]:
    """Detect lead/lag relationship via cross-correlation.

    Returns leader_asset, follower_asset, lag_bars, confidence.
    """
    ra = returns[asset_a]
    rb = returns[asset_b]
    idx = ra.index.intersection(rb.index)
    ra = ra.reindex(idx).values
    rb = rb.reindex(idx).values

    ra = (ra - np.nanmean(ra)) / max(np.nanstd(ra), 1e-12)
    rb = (rb - np.nanmean(rb)) / max(np.nanstd(rb), 1e-12)
    n = len(ra)

    best_corr = 0.0
    best_lag = 0
    a_leads = True

    for lag in range(1, min(max_lag + 1, n // 4)):
        corr_a_leads = float(np.mean(ra[:-lag] * rb[lag:]))
        corr_b_leads = float(np.mean(rb[:-lag] * ra[lag:]))

        if abs(corr_a_leads) > abs(best_corr):
            best_corr = corr_a_leads
            best_lag = lag
            a_leads = True
        if abs(corr_b_leads) > abs(best_corr):
            best_corr = corr_b_leads
            best_lag = lag
            a_leads = False

    confidence = float(min(1.0, abs(best_corr)))

    if a_leads:
        leader, follower = asset_a, asset_b
    else:
        leader, follower = asset_b, asset_a

    return {
        'leader_asset': leader,
        'follower_asset': follower,
        'lag_bars': best_lag,
        'confidence': confidence,
        'raw_correlation': float(best_corr),
    }


# =========================================================================
# A3 — Regime-Conditioned Dependency Score
# =========================================================================

_REGIME_WEIGHT = {
    'trending': 1.3,
    'ranging': 0.6,
    'unknown': 1.0,
}


def compute_dependency_score(
    lead_lag_confidence: float,
    correlation: float,
    regime: str,
    contagion_risk: float,
) -> float:
    """Compute a regime-conditioned dependency score in [0, 1].

    Trending regimes amplify dependency (lead/lag more predictive).
    Ranging regimes dampen dependency (correlation breaks down).
    """
    regime_w = _REGIME_WEIGHT.get(regime, 1.0)
    raw = 0.4 * lead_lag_confidence + 0.3 * abs(correlation) + 0.3 * contagion_risk
    scaled = raw * regime_w
    return float(max(0.0, min(1.0, scaled)))


# =========================================================================
# A4 — Contagion Modeling
# =========================================================================

def detect_contagion(
    returns: Dict[str, pd.Series],
    leader: str,
    follower: str,
    threshold_std: float = 3.0,
    lookback: int = 10,
) -> Dict[str, Any]:
    """Detect shock propagation from leader to follower.

    A shock is a return > threshold_std standard deviations.
    Spillover probability = fraction of leader shocks followed by
    follower shocks within `lookback` bars.
    """
    rl = returns[leader]
    rf = returns[follower]
    idx = rl.index.intersection(rf.index)
    rl = rl.reindex(idx)
    rf = rf.reindex(idx)

    rl_std = float(rl.std())
    rf_std = float(rf.std())
    if rl_std < 1e-12 or rf_std < 1e-12:
        return {
            'contagion_risk': 0.0,
            'spillover_probability': 0.0,
            'shock_events_leader': 0,
        }

    leader_shocks = np.where(np.abs(rl.values) > threshold_std * rl_std)[0]
    follower_shocks_set = set(
        np.where(np.abs(rf.values) > threshold_std * rf_std)[0]
    )

    n_shocks = len(leader_shocks)
    if n_shocks == 0:
        return {
            'contagion_risk': 0.0,
            'spillover_probability': 0.0,
            'shock_events_leader': 0,
        }

    n_propagated = 0
    for si in leader_shocks:
        for offset in range(1, lookback + 1):
            if (si + offset) in follower_shocks_set:
                n_propagated += 1
                break

    spillover_prob = n_propagated / n_shocks
    contagion_risk = float(min(1.0, spillover_prob * (1.0 + np.log1p(n_shocks) * 0.1)))

    return {
        'contagion_risk': float(contagion_risk),
        'spillover_probability': float(spillover_prob),
        'shock_events_leader': int(n_shocks),
    }


# =========================================================================
# A6 — Aggregate Dependency
# =========================================================================

def aggregate_dependency(
    lead_lag: Dict[str, Any],
    contagion: Dict[str, Any],
    regime: str,
    rolling_correlation: float,
) -> Dict[str, Any]:
    """Aggregate all dependency signals into a single dict."""
    dep_strength = compute_dependency_score(
        lead_lag_confidence=lead_lag['confidence'],
        correlation=rolling_correlation,
        regime=regime,
        contagion_risk=contagion['contagion_risk'],
    )
    lag_alignment = float(min(1.0, lead_lag['confidence'] * (1.0 - lead_lag['lag_bars'] / 48.0)))
    lag_alignment = max(0.0, lag_alignment)
    divergence = float(max(0.0, min(1.0, 1.0 - abs(rolling_correlation))))

    return {
        'dependency_strength': dep_strength,
        'lag_alignment_score': lag_alignment,
        'divergence_score': divergence,
        'contagion_risk': contagion['contagion_risk'],
    }


# =========================================================================
# A7 — Decision Modulation
# =========================================================================

_CONTAGION_SUPPRESS_THRESHOLD = 0.8


def modulate_decision(
    symbol: str,
    direction: int,
    size_multiplier: float,
    dependency: Dict[str, float],
    leader_direction: int,
) -> Dict[str, Any]:
    """Modulate a per-asset decision based on inter-asset dependency.

    Rules:
    - NEVER changes direction
    - Leader (BTCUSDT) is not modulated by itself
    - Aligned with leader → boost size (up to 1.2×)
    - Misaligned with leader → reduce size (down to 0.5×)
    - High contagion risk + misalignment → suppress trade
    """
    if symbol == 'BTCUSDT':
        return {
            'adjusted_size_multiplier': size_multiplier,
            'confidence_adjustment': 0.0,
            'suppress_trade': False,
            'reason': 'leader_no_modulation',
        }

    dep_str = dependency.get('dependency_strength', 0.0)
    contagion = dependency.get('contagion_risk', 0.0)

    if contagion >= _CONTAGION_SUPPRESS_THRESHOLD and leader_direction != direction:
        return {
            'adjusted_size_multiplier': 0.0,
            'confidence_adjustment': -contagion,
            'suppress_trade': True,
            'reason': 'high_contagion_misaligned',
        }

    aligned = (direction == leader_direction) and leader_direction != 0
    if aligned:
        boost = 1.0 + dep_str * 0.25
        adj = min(boost, 1.25)
        conf_adj = dep_str * 0.2
    else:
        penalty = 1.0 - dep_str * 0.5
        adj = max(penalty, 0.5)
        conf_adj = -dep_str * 0.2

    return {
        'adjusted_size_multiplier': float(size_multiplier * adj),
        'confidence_adjustment': float(conf_adj),
        'suppress_trade': False,
        'reason': 'aligned' if aligned else 'misaligned',
    }


# =========================================================================
# A8 — InterAssetDependencyLayer (orchestrator)
# =========================================================================

class InterAssetDependencyLayer:
    """Orchestrator that accumulates returns and evaluates dependency."""

    def __init__(
        self,
        symbols: List[str],
        leader: str = 'BTCUSDT',
        min_bars: int = 100,
        max_buffer: int = 2000,
    ) -> None:
        self.symbols = symbols
        self.leader = leader
        self.min_bars = min_bars
        self.max_buffer = max_buffer
        self._return_buffer: Dict[str, List[float]] = defaultdict(list)
        self._ts_buffer: Dict[str, List[str]] = defaultdict(list)
        self._last_lead_lag: Dict[str, Dict[str, Any]] = {}
        self._last_contagion: Dict[str, Dict[str, Any]] = {}
        self._leader_direction: int = 0

    def update_return(self, symbol: str, timestamp: str, ret: float) -> None:
        """Append a return observation for a symbol."""
        self._return_buffer[symbol].append(ret)
        self._ts_buffer[symbol].append(timestamp)
        if len(self._return_buffer[symbol]) > self.max_buffer:
            self._return_buffer[symbol] = self._return_buffer[symbol][-self.max_buffer:]
            self._ts_buffer[symbol] = self._ts_buffer[symbol][-self.max_buffer:]

    def set_leader_direction(self, direction: int) -> None:
        self._leader_direction = direction

    def _has_enough_data(self, symbol: str) -> bool:
        return (
            len(self._return_buffer.get(self.leader, [])) >= self.min_bars
            and len(self._return_buffer.get(symbol, [])) >= self.min_bars
        )

    def _build_return_series(self, symbol: str) -> pd.Series:
        vals = self._return_buffer[symbol]
        ts = self._ts_buffer[symbol]
        idx = pd.to_datetime(ts)
        return pd.Series(vals, index=idx, name=symbol)

    def evaluate(
        self,
        symbol: str,
        direction: int,
        size_multiplier: float,
        regime: str = 'unknown',
    ) -> Dict[str, Any]:
        """Evaluate inter-asset dependency for a follower symbol.

        Returns modulation dict with adjusted_size_multiplier and suppress_trade.
        If insufficient data, returns pass-through (no modulation).
        """
        if not self._has_enough_data(symbol):
            return {
                'adjusted_size_multiplier': size_multiplier,
                'confidence_adjustment': 0.0,
                'suppress_trade': False,
                'reason': 'insufficient_data',
            }

        rets = {
            self.leader: self._build_return_series(self.leader),
            symbol: self._build_return_series(symbol),
        }

        lead_lag = detect_lead_lag(rets, self.leader, symbol, max_lag=24)
        contagion = detect_contagion(rets, self.leader, symbol)

        n = min(len(rets[self.leader]), len(rets[symbol]))
        rolling_corr = float(np.corrcoef(
            rets[self.leader].values[-n:],
            rets[symbol].values[-n:],
        )[0, 1])
        if np.isnan(rolling_corr):
            rolling_corr = 0.0

        dep = aggregate_dependency(
            lead_lag=lead_lag,
            contagion=contagion,
            regime=regime,
            rolling_correlation=rolling_corr,
        )

        self._last_lead_lag[symbol] = lead_lag
        self._last_contagion[symbol] = contagion

        result = modulate_decision(
            symbol=symbol,
            direction=direction,
            size_multiplier=size_multiplier,
            dependency=dep,
            leader_direction=self._leader_direction,
        )
        result['dependency'] = dep
        result['lead_lag'] = lead_lag
        return result

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for persistence / monitoring."""
        return {
            'leader': self.leader,
            'buffer_sizes': {s: len(v) for s, v in self._return_buffer.items()},
            'last_lead_lag': dict(self._last_lead_lag),
            'last_contagion': dict(self._last_contagion),
            'leader_direction': self._leader_direction,
        }
