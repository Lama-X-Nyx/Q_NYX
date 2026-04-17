"""
Execution Optimizer — fill probability + dynamic offset + maker/taker
(Ticket 32).

Operates AFTER RiskEngine, BEFORE OMS. Alpha untouched.
Deterministic, reproducible, lightweight.
"""
from __future__ import annotations

from typing import Any, Dict


def estimate_fill_probability(
    spread_bps: float,
    volatility_pct: float,
    price_move_pct: float,
    offset_bps: float,
) -> float:
    """Estimate maker fill probability ∈ [0, 1].

    Higher offset → higher fill prob (limit further from mid).
    Higher volatility → lower fill prob (price may run away).
    Higher recent price move → lower fill prob (momentum).
    """
    base = min(1.0, offset_bps / max(spread_bps + 1.0, 1.0))
    vol_penalty = min(0.5, volatility_pct * 0.2)
    move_penalty = min(0.3, abs(price_move_pct) * 0.5)
    p = max(0.0, min(1.0, base - vol_penalty - move_penalty))
    return float(p)


def compute_dynamic_offset_bps(
    volatility_pct: float,
    spread_bps: float,
    fill_probability: float,
) -> float:
    """Dynamic limit offset in bps.

    Higher vol → wider offset (more room for price to return).
    Lower fill_probability → wider offset (compensate).
    """
    base = max(spread_bps, 3.0)
    vol_adj = volatility_pct * 5.0
    fp_adj = max(0.0, (1.0 - fill_probability) * 10.0)
    return float(base + vol_adj + fp_adj)


def should_use_taker(
    expected_edge_bps: float,
    taker_cost_bps: float,
    fill_probability_maker: float,
) -> bool:
    """Maker vs taker decision.

    Use taker when :
    - expected edge > taker cost AND
    - maker fill probability is low (would likely miss)

    Use maker when :
    - fill probability is high (will likely fill as maker) OR
    - edge is too small to absorb taker cost
    """
    if expected_edge_bps <= taker_cost_bps:
        return False
    if fill_probability_maker >= 0.7:
        return False
    return True


def build_execution_plan(
    side: str,
    mark_price: float,
    volatility_pct: float,
    spread_bps: float,
    expected_edge_bps: float,
    taker_cost_bps: float = 8.0,
) -> Dict[str, Any]:
    """Build a complete execution plan for one trade.

    Returns :
      limit_price    : float
      order_type     : 'post_only_limit' | 'market'
      offset_bps     : float (dynamic)
      fill_probability : float
      use_taker      : bool
    """
    fp_est = estimate_fill_probability(
        spread_bps=spread_bps,
        volatility_pct=volatility_pct,
        price_move_pct=0.0,
        offset_bps=10.0,
    )

    offset = compute_dynamic_offset_bps(
        volatility_pct=volatility_pct,
        spread_bps=spread_bps,
        fill_probability=fp_est,
    )

    use_taker = should_use_taker(
        expected_edge_bps=expected_edge_bps,
        taker_cost_bps=taker_cost_bps,
        fill_probability_maker=fp_est,
    )

    if use_taker:
        limit_price = float(mark_price)
        order_type = 'market'
    else:
        offset_frac = offset / 10_000.0
        if side == 'buy':
            limit_price = mark_price * (1.0 - offset_frac)
        else:
            limit_price = mark_price * (1.0 + offset_frac)
        order_type = 'post_only_limit'

    return {
        'limit_price': float(limit_price),
        'order_type': order_type,
        'offset_bps': float(offset),
        'fill_probability': float(fp_est),
        'use_taker': use_taker,
    }
