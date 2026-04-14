"""
Jesse Utilities Wrapper — Position Sizing, Crossovers, Kelly Criterion

Wraps jesse.utils when available, provides standalone implementations otherwise.
Key functions from Jesse used in Q_NYX:
  - risk_to_qty:     Position sizing based on risk %
  - crossed:         Detect crossovers between two series
  - kelly_criterion: Optimal bet sizing
  - anchor_timeframe: Map lower to higher TF
"""
import numpy as np
from typing import Any, Optional

jesse_utils: Any = None
try:
    from jesse import utils as jesse_utils  # type: ignore[import-untyped,no-redef]
    _JESSE_AVAILABLE = True
except ImportError:
    _JESSE_AVAILABLE = False


def risk_to_qty(
    capital: float,
    risk_per_capital: float,
    entry_price: float,
    stop_loss_price: float,
    precision: int = 8,
    fee_rate: float = 0.0,
) -> float:
    """
    Calculate position quantity based on risk percentage.

    Args:
        capital:           Total account capital.
        risk_per_capital:  Risk percentage (e.g., 0.02 for 2%).
        entry_price:       Planned entry price.
        stop_loss_price:   Planned stop-loss price.
        precision:         Decimal precision for quantity.
        fee_rate:          Trading fee rate (e.g., 0.001 for 0.1%).

    Returns:
        Position quantity (number of units to trade).
    """
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0.0

    if _JESSE_AVAILABLE:
        return jesse_utils.risk_to_qty(
            capital, risk_per_capital, entry_price, stop_loss_price,
            precision, fee_rate
        )

    risk_amount = capital * risk_per_capital
    fee_mult = 1 - fee_rate * 3
    qty = (risk_amount / risk_per_unit) * fee_mult
    return round(qty, precision)


def size_to_qty(
    position_size: float,
    entry_price: float,
    precision: int = 3,
    fee_rate: float = 0.0,
) -> float:
    """
    Convert dollar position size to quantity.

    Args:
        position_size: Dollar amount to allocate.
        entry_price:   Price per unit.
        precision:     Decimal precision.
        fee_rate:      Fee rate.

    Returns:
        Quantity of units.
    """
    if _JESSE_AVAILABLE:
        return jesse_utils.size_to_qty(position_size, entry_price, precision, fee_rate)

    if entry_price is None or entry_price == 0:
        return 0.0
    fee_mult = 1 - fee_rate * 3
    qty = (position_size * fee_mult) / entry_price
    return round(qty, precision)


def crossed(
    series1: np.ndarray,
    series2: np.ndarray,
    direction: Optional[str] = None,
    sequential: bool = False,
) -> np.ndarray | bool:
    """
    Detect crossover between two series.

    Args:
        series1:    First data series.
        series2:    Second data series (or scalar).
        direction:  'above' or 'below' (None = any cross).
        sequential: If True, check only last two values.

    Returns:
        Boolean or array of crossover points.
    """
    # Jesse's crossed() returns scalar, not array — we always use our own
    # implementation for consistent array-mode behavior.
    s1 = np.asarray(series1, dtype=float)
    s2 = np.asarray(series2, dtype=float) if not np.isscalar(series2) else np.full_like(s1, series2)

    if sequential:
        if len(s1) < 2:
            return False
        prev_above = s1[-2] > s2[-2]
        curr_above = s1[-1] > s2[-1]
        if direction == 'above':
            return bool(not prev_above and curr_above)
        elif direction == 'below':
            return bool(prev_above and not curr_above)
        return bool(prev_above != curr_above)

    # Full array mode
    above = s1 > s2
    crosses = np.zeros(len(s1), dtype=bool)
    for i in range(1, len(s1)):
        if direction == 'above':
            crosses[i] = not above[i - 1] and above[i]
        elif direction == 'below':
            crosses[i] = above[i - 1] and not above[i]
        else:
            crosses[i] = above[i - 1] != above[i]
    return crosses


def kelly_criterion(win_rate: float, ratio_avg_win_loss: float) -> float:
    """
    Kelly Criterion — optimal bet sizing.

    Args:
        win_rate:            Probability of winning (0-1).
        ratio_avg_win_loss:  Average win / average loss.

    Returns:
        Optimal fraction of capital to risk.
    """
    if _JESSE_AVAILABLE:
        return jesse_utils.kelly_criterion(win_rate, ratio_avg_win_loss)

    if ratio_avg_win_loss == 0:
        return 0.0
    return win_rate - ((1 - win_rate) / ratio_avg_win_loss)


def anchor_timeframe(timeframe: str) -> str:
    """
    Map lower timeframe to anchor (higher) timeframe.
    Uses Jesse's mapping when available, otherwise our own.
    """
    if _JESSE_AVAILABLE:
        return jesse_utils.anchor_timeframe(timeframe)

    mapping = {
        '1m': '5m', '3m': '15m', '5m': '30m',
        '15m': '1h', '30m': '2h', '45m': '3h',
        '1h': '4h', '2h': '6h', '3h': '1D',
        '4h': '1D', '6h': '1D', '1D': '1W',
    }
    return mapping.get(timeframe, '1D')


def estimate_risk(entry_price: float, stop_price: float) -> float:
    """Per-unit risk between entry and stop."""
    return abs(entry_price - stop_price)


def streaks(series: np.ndarray, use_diff: bool = True) -> np.ndarray:
    """Calculate consecutive positive/negative periods."""
    if use_diff:
        d = np.diff(series, prepend=series[0])
    else:
        d = series
    result = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        if d[i] > 0:
            result[i] = max(result[i - 1], 0) + 1
        elif d[i] < 0:
            result[i] = min(result[i - 1], 0) - 1
        else:
            result[i] = 0
    return result
