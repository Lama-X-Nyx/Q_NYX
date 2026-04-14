"""
PostOnlyPaperBroker — honest maker-first simulator.

Design principles:
  1. Post-only strict: if the limit would cross the opposite side at placement
     time, the exchange rejects → order state = REJECTED. No stealth taker.
  2. Max-wait timeout: if the bar does not trade through the limit within
     max_wait_bars, the order is TIMED_OUT — this is a MISSED TRADE, logged
     as such. Explicitly NO silent taker fallback.
  3. Each order carries a typed state machine:
        NEW → POSTED → (FILLED | CANCELLED | TIMED_OUT | REJECTED)

This is the broker the paper-live pipeline should use so that the alpha
reported in backtest matches what the live strategy will actually capture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional


Side = Literal['buy', 'sell']


class OrderState(str, Enum):
    NEW       = "NEW"
    POSTED    = "POSTED"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    REJECTED  = "REJECTED"


@dataclass
class Order:
    oid: int
    pair: str
    side: Side
    qty: float
    limit_price: float
    mark_price_at_placement: float
    placed_at: str
    max_wait_bars: int
    state: OrderState = OrderState.NEW
    bars_waited: int = 0
    fills: List[dict] = field(default_factory=list)
    reject_reason: Optional[str] = None
    timeout_reason: Optional[str] = None
    timed_out_at: Optional[str] = None
    filled_at: Optional[str] = None


class PostOnlyPaperBroker:
    """Paper broker with strict post-only semantics."""

    def __init__(
        self,
        max_wait_bars: int = 3,
        maker_fee: float = 0.0002,
    ):
        self.default_max_wait = int(max_wait_bars)
        self.maker_fee = float(maker_fee)
        self._orders: Dict[int, Order] = {}
        self._next_oid = 1

    # ------------------------------------------------------------------
    # Placement (post-only check)
    # ------------------------------------------------------------------
    def place_post_only(
        self,
        pair: str,
        side: Side,
        qty: float,
        limit_price: float,
        mark_price: float,
        placed_at: str,
        max_wait_bars: Optional[int] = None,
    ) -> int:
        oid = self._next_oid
        self._next_oid += 1

        order = Order(
            oid=oid,
            pair=pair,
            side=side,
            qty=float(qty),
            limit_price=float(limit_price),
            mark_price_at_placement=float(mark_price),
            placed_at=placed_at,
            max_wait_bars=int(max_wait_bars if max_wait_bars is not None
                              else self.default_max_wait),
        )

        # Post-only check: would the order cross immediately?
        # Buy limit ≥ ask (we approximate ask ≈ mark) → would take liquidity.
        # Sell limit ≤ bid (bid ≈ mark) → would take liquidity.
        would_cross = (
            (side == 'buy' and limit_price >= mark_price) or
            (side == 'sell' and limit_price <= mark_price)
        )
        if would_cross:
            order.state = OrderState.REJECTED
            order.reject_reason = (
                f"post_only would cross: {side} @ {limit_price} "
                f"vs mark {mark_price}"
            )
        else:
            order.state = OrderState.POSTED

        self._orders[oid] = order
        return oid

    # ------------------------------------------------------------------
    # Bar advance
    # ------------------------------------------------------------------
    def on_bar(self, pair: str, ohlcv: dict, bar_ts: str) -> None:
        high = float(ohlcv['high'])
        low = float(ohlcv['low'])

        for order in self._orders.values():
            if order.state != OrderState.POSTED:
                continue
            if order.pair != pair:
                continue

            # Check maker fill: did the bar trade through our limit?
            filled = False
            if order.side == 'buy' and low <= order.limit_price:
                self._fill_maker(order, bar_ts=bar_ts)
                filled = True
            elif order.side == 'sell' and high >= order.limit_price:
                self._fill_maker(order, bar_ts=bar_ts)
                filled = True

            if filled:
                continue

            # No fill this bar → increment wait counter; timeout if exceeded.
            order.bars_waited += 1
            if order.bars_waited >= order.max_wait_bars:
                order.state = OrderState.TIMED_OUT
                order.timed_out_at = bar_ts
                order.timeout_reason = (
                    f"no fill after max_wait={order.max_wait_bars} bars"
                )

    # ------------------------------------------------------------------
    # Fill
    # ------------------------------------------------------------------
    def _fill_maker(self, order: Order, bar_ts: str) -> None:
        fee = order.limit_price * order.qty * self.maker_fee
        order.fills.append({
            'price': order.limit_price,
            'qty': order.qty,
            'side': order.side,
            'role': 'maker',
            'fee': fee,
            'ts': bar_ts,
        })
        order.state = OrderState.FILLED
        order.filled_at = bar_ts

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def cancel(self, oid: int) -> None:
        o = self._orders.get(oid)
        if o and o.state in (OrderState.POSTED, OrderState.NEW):
            o.state = OrderState.CANCELLED

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get(self, oid: int) -> Order:
        return self._orders[oid]

    def all_orders(self) -> List[Order]:
        return list(self._orders.values())

    def orders_in_state(self, state: OrderState) -> List[Order]:
        return [o for o in self._orders.values() if o.state == state]

    def missed_trades(self) -> List[Order]:
        return self.orders_in_state(OrderState.TIMED_OUT)

    def filled_orders(self) -> List[Order]:
        return self.orders_in_state(OrderState.FILLED)

    def rejected_orders(self) -> List[Order]:
        return self.orders_in_state(OrderState.REJECTED)

    def miss_rate(self) -> float:
        """TIMED_OUT / (TIMED_OUT + FILLED). Rejected are excluded from the ratio."""
        filled = len(self.filled_orders())
        missed = len(self.missed_trades())
        total = filled + missed
        if total == 0:
            return 0.0
        return missed / total
