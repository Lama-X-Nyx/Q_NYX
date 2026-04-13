"""
Execution Policy — Only trade when alpha survives realistic execution

Rule: not "beautiful if perfect fill" but "still good in real conditions".

Functions:
  estimate_fill_probability: can we get maker fill?
  estimate_execution_cost: what will it actually cost?
  maker_viability_score: is post-only realistic?
  execution_check: full check → should_trade + order_type + cost
"""
import numpy as np
from typing import Any, Dict

# Binance Futures fee schedule
MAKER_FEE = 0.0002    # 0.02%
TAKER_FEE = 0.0004    # 0.04%
TAKER_SLIPPAGE = 0.0003  # 0.03% estimated


def estimate_fill_probability(
    volume_ratio: float,
    spread_pct: float,
    atr_pct: float,
    bar_range_pct: float,
) -> float:
    """
    Estimate probability of getting a maker fill within 1 bar.

    High volume + low spread + large bar range → price likely crosses our limit.
    Low volume + wide spread + tight range → order sits unfilled.
    """
    # Volume factor: more volume = more chance of fill
    vol_factor = min(1.0, volume_ratio / 3.0)

    # Spread factor: tight spread = easier to get crossed
    spread_factor = max(0, 1.0 - spread_pct * 10)

    # Range factor: large bar range = price moves through our level
    range_factor = min(1.0, bar_range_pct / 0.3)

    # Momentum penalty: fast moves may skip our price
    momentum_penalty = max(0, 1.0 - atr_pct * 2)

    prob = 0.3 * vol_factor + 0.25 * spread_factor + 0.25 * range_factor + 0.2 * momentum_penalty
    return float(np.clip(prob, 0.05, 0.95))


def estimate_execution_cost(
    fill_as_maker: bool,
    price: float,
    qty: float,
    spread_pct: float = 0.02,
) -> float:
    """
    Estimate total execution cost (fees + slippage) in dollars.

    Maker: fee only (limit order, no slippage).
    Taker: fee + spread crossing + slippage.
    """
    notional = price * qty

    if fill_as_maker:
        # Maker: 0.02% fee × 2 sides
        cost = notional * MAKER_FEE * 2
    else:
        # Taker: 0.04% fee × 2 sides + slippage
        fee_cost = notional * TAKER_FEE * 2
        slippage_cost = notional * TAKER_SLIPPAGE * 2
        spread_cost = notional * spread_pct / 100  # half-spread crossing
        cost = fee_cost + slippage_cost + spread_cost

    return float(cost)


def maker_viability_score(
    volume_ratio: float,
    atr_pct: float,
    momentum_abs: float,
    spread_pct: float,
) -> float:
    """
    Score [0,1] for whether a post-only limit order is viable.

    High viability: calm market, good volume, tight spread.
    Low viability: fast market, low volume, wide spread → need taker.
    """
    # Calm market = high viability
    calm = max(0, 1.0 - atr_pct * 3)

    # No strong momentum = limit won't get run over
    no_momentum = max(0, 1.0 - momentum_abs * 30)

    # Volume = liquidity at our price level
    vol = min(1.0, volume_ratio / 3.0)

    # Tight spread = easy to sit on bid/ask
    tight = max(0, 1.0 - spread_pct * 10)

    score = 0.3 * calm + 0.3 * no_momentum + 0.2 * vol + 0.2 * tight
    return float(np.clip(score, 0.0, 1.0))


def should_place_post_only(maker_viability: float, threshold: float = 0.50) -> bool:
    """Place post-only (maker) order if viability is high enough."""
    return maker_viability >= threshold


def should_cancel_unfilled(bars_waiting: int, max_wait: int = 3) -> bool:
    """Cancel unfilled limit order after max_wait bars."""
    return bars_waiting > max_wait


def execution_check(
    volume_ratio: float,
    spread_pct: float,
    atr_pct: float,
    momentum_abs: float,
    trade_alpha_pct: float,
    price: float = 30000.0,
    qty: float = 0.01,
) -> Dict[str, Any]:
    """
    Full execution check for a trade candidate.

    Returns:
        should_trade: bool — trade only if alpha > expected cost
        fill_probability: float
        maker_viability: float
        expected_cost_pct: float — as % of notional
        order_type: 'maker' or 'taker'
        alpha_after_cost: float — remaining alpha after execution
    """
    bar_range_pct = atr_pct * 0.7  # approximate

    fill_prob = estimate_fill_probability(volume_ratio, spread_pct, atr_pct, bar_range_pct)
    viability = maker_viability_score(volume_ratio, atr_pct, momentum_abs, spread_pct)
    use_maker = should_place_post_only(viability)

    if use_maker:
        cost = estimate_execution_cost(True, price, qty, spread_pct)
        order_type = 'maker'
        # Adjust for fill probability: if we miss the fill, we miss the trade
        # Expected cost = maker cost (when filled) + opportunity cost (when missed)
        effective_cost_pct = (MAKER_FEE * 2) * fill_prob + (trade_alpha_pct * (1 - fill_prob) * 0.3)
    else:
        cost = estimate_execution_cost(False, price, qty, spread_pct)
        order_type = 'taker'
        effective_cost_pct = TAKER_FEE * 2 + TAKER_SLIPPAGE * 2

    alpha_after = trade_alpha_pct - effective_cost_pct

    return {
        'should_trade': alpha_after > 0,
        'fill_probability': fill_prob,
        'maker_viability': viability,
        'expected_cost_pct': effective_cost_pct,
        'order_type': order_type,
        'alpha_after_cost': alpha_after,
    }
