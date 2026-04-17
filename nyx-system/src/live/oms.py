"""
Order Management System — single source of truth for execution state
(Ticket 23).

Wraps the existing PostOnlyPaperBroker (or a future exchange adapter)
as an execution CONTROLLER. NYXEngine expresses intent via
`oms.submit_order()`. OMS owns the canonical order lifecycle :

  SUBMITTED → PARTIALLY_FILLED → FILLED
  SUBMITTED → REJECTED
  SUBMITTED → CANCELLED
  PARTIALLY_FILLED → FILLED
  PARTIALLY_FILLED → CANCELLED

Terminal states (FILLED, REJECTED, CANCELLED) cannot be mutated.
Duplicate client_order_ids are rejected. Partial fills are tracked
with running filled_qty + avg_fill_price. Every state transition
is logged as a structured event.

No strategy code may bypass this layer.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

TERMINAL_STATES = frozenset({'FILLED', 'REJECTED', 'CANCELLED'})

VALID_TRANSITIONS = {
    'SUBMITTED':        {'PARTIALLY_FILLED', 'FILLED', 'REJECTED', 'CANCELLED'},
    'PARTIALLY_FILLED': {'FILLED', 'CANCELLED'},
}


@dataclass
class OMSOrder:
    """Canonical order object — the ONLY order schema in the system."""
    order_id: int
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    order_type: str = 'post_only_limit'
    post_only: bool = True
    status: str = 'SUBMITTED'
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    reject_reason: Optional[str] = None
    cancel_reason: Optional[str] = None
    exchange_order_id: Optional[str] = None


class OMS:
    """Order Management System — single source of truth."""

    def __init__(self) -> None:
        self._orders: Dict[int, OMSOrder] = {}
        self._by_client_id: Dict[str, int] = {}
        self._events: Dict[int, List[Dict[str, Any]]] = {}
        self._next_oid = 1

    def submit_order(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = 'post_only_limit',
    ) -> int:
        if client_order_id in self._by_client_id:
            raise ValueError(
                f'Duplicate client_order_id {client_order_id!r} — '
                'order already exists'
            )
        oid = self._next_oid
        self._next_oid += 1
        order = OMSOrder(
            order_id=oid,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            price=float(price),
            order_type=order_type,
        )
        self._orders[oid] = order
        self._by_client_id[client_order_id] = oid
        self._log_event(oid, 'SUBMITTED', {
            'symbol': symbol, 'side': side, 'qty': quantity,
            'price': price,
        })
        log.info('OMS submit %d (%s) %s %s %.4f @ %.2f',
                 oid, client_order_id, symbol, side, quantity, price)
        return oid

    def handle_fill(
        self,
        order_id: int,
        fill_qty: float,
        fill_price: float,
    ) -> None:
        order = self._require(order_id)
        self._require_non_terminal(order)

        new_filled = order.filled_qty + float(fill_qty)
        if new_filled > order.quantity * 1.001:
            raise ValueError(
                f'Overfill: filled {new_filled:.6f} > ordered '
                f'{order.quantity:.6f}'
            )

        total_cost = (
            order.avg_fill_price * order.filled_qty
            + float(fill_price) * float(fill_qty)
        )
        order.filled_qty = new_filled
        order.avg_fill_price = (
            total_cost / new_filled if new_filled > 0 else 0.0
        )
        order.updated_at = time.time()

        if abs(new_filled - order.quantity) < 1e-9:
            order.status = 'FILLED'
            self._log_event(order_id, 'FILLED', {
                'filled_qty': new_filled,
                'avg_fill_price': order.avg_fill_price,
            })
        else:
            order.status = 'PARTIALLY_FILLED'
            self._log_event(order_id, 'PARTIALLY_FILLED', {
                'fill_qty': fill_qty, 'fill_price': fill_price,
                'total_filled': new_filled,
            })

    def handle_reject(self, order_id: int, reason: str = '') -> None:
        order = self._require(order_id)
        self._require_non_terminal(order)
        order.status = 'REJECTED'
        order.reject_reason = reason
        order.updated_at = time.time()
        self._log_event(order_id, 'REJECTED', {'reason': reason})

    def cancel_order(self, order_id: int, reason: str = '') -> None:
        order = self._require(order_id)
        if order.status in TERMINAL_STATES:
            return
        order.status = 'CANCELLED'
        order.cancel_reason = reason
        order.updated_at = time.time()
        self._log_event(order_id, 'CANCELLED', {'reason': reason})

    def get_order(self, order_id: int) -> Optional[OMSOrder]:
        return self._orders.get(order_id)

    def get_events(self, order_id: int) -> List[Dict[str, Any]]:
        return list(self._events.get(order_id, []))

    def _require(self, oid: int) -> OMSOrder:
        order = self._orders.get(oid)
        if order is None:
            raise ValueError(f'Unknown order_id {oid}')
        return order

    def _require_non_terminal(self, order: OMSOrder) -> None:
        if order.status in TERMINAL_STATES:
            raise RuntimeError(
                f'Order {order.order_id} is in terminal state '
                f'{order.status} — cannot mutate'
            )

    def _log_event(
        self, oid: int, event: str, detail: Dict[str, Any],
    ) -> None:
        entry = {
            'event': event,
            'ts': time.time(),
            **detail,
        }
        self._events.setdefault(oid, []).append(entry)
