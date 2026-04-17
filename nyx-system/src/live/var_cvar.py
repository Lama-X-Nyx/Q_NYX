"""
VaR / CVaR / Tail Risk — distribution-aware risk (Ticket 31).

Extends the existing RiskEngine (not replaces). Computed from
REAL trade PnL history. Deterministic and testable.

VaR  = maximum expected loss at confidence α over window N.
CVaR = average loss beyond VaR threshold (expected shortfall).
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np


def compute_var(pnl: np.ndarray, percentile: float = 5) -> float:
    """VaR at the given percentile (e.g. 5 = 95% confidence).

    Returns the PnL value below which `percentile` % of observations
    fall. Negative = loss. Returns 0 if empty.
    """
    if len(pnl) == 0:
        return 0.0
    return float(np.percentile(pnl, percentile))


def compute_cvar(pnl: np.ndarray, percentile: float = 5) -> float:
    """CVaR (Expected Shortfall) = mean of tail below VaR.

    Returns the average PnL of observations at or below the VaR
    threshold. More negative = worse tail. Returns 0 if empty.
    """
    if len(pnl) == 0:
        return 0.0
    var = compute_var(pnl, percentile)
    tail = pnl[pnl <= var]
    if len(tail) == 0:
        return float(var)
    return float(np.mean(tail))


class RollingRiskMetrics:
    """Rolling VaR / CVaR from realized trade PnL history."""

    def __init__(self, window: int = 100) -> None:
        self._window = int(window)
        self._pnl_buffer: Deque[float] = deque(maxlen=self._window)

    def add_trade_pnl(self, pnl: float) -> None:
        self._pnl_buffer.append(float(pnl))

    @property
    def _arr(self) -> np.ndarray:
        return np.array(self._pnl_buffer, dtype=float)

    @property
    def var_95(self) -> Optional[float]:
        if len(self._pnl_buffer) < 5:
            return None
        return compute_var(self._arr, 5)

    @property
    def var_99(self) -> Optional[float]:
        if len(self._pnl_buffer) < 5:
            return None
        return compute_var(self._arr, 1)

    @property
    def cvar_95(self) -> Optional[float]:
        if len(self._pnl_buffer) < 5:
            return None
        return compute_cvar(self._arr, 5)

    @property
    def cvar_99(self) -> Optional[float]:
        if len(self._pnl_buffer) < 5:
            return None
        return compute_cvar(self._arr, 1)

    def risk_scale_factor(self, max_cvar: float = -200.0) -> float:
        """Scale factor [0, 1] based on CVaR proximity to limit.

        1.0 = CVaR well within limit (full size).
        0.0 = CVaR at or beyond limit (block).
        Linear interpolation between.
        """
        cvar = self.cvar_95
        if cvar is None or max_cvar >= 0:
            return 1.0
        if cvar <= max_cvar:
            return 0.0
        ratio = cvar / max_cvar
        return float(max(0.0, min(1.0, 1.0 - ratio)))
