"""
MakerFirstBroker — paper execution that prefers maker fills.

Order life-cycle:
  1. Client calls place_limit_{buy,sell}() → order parked in _open.
  2. Each on_bar() checks if the bar traded through the limit price → fill as maker.
  3. If still unfilled after `max_wait_bars`, converts to taker at next close
     + slippage. Fee model:
       maker_fee   = 0.0002  (0.02%)
       taker_fee   = 0.0004  (0.04%)
       slippage    = 0.0003  (0.03%)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

Side = Literal['buy', 'sell']


@dataclass
class _Order:
    oid: int
    side: Side
    qty: float
    limit_price: float
    bars_waited: int = 0
    open: bool = True
    fills: List[dict] = field(default_factory=list)


class MakerFirstBroker:
    """Paper broker with maker-first fill logic."""

    def __init__(
        self,
        max_wait_bars: int = 3,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        taker_slippage: float = 0.0003,
    ):
        self.max_wait_bars = int(max_wait_bars)
        self.maker_fee = float(maker_fee)
        self.taker_fee = float(taker_fee)
        self.taker_slippage = float(taker_slippage)
        self._next_oid = 1
        self._orders: Dict[int, _Order] = {}

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------
    def place_limit_buy(self, qty: float, limit_price: float) -> int:
        return self._place('buy', qty, limit_price)

    def place_limit_sell(self, qty: float, limit_price: float) -> int:
        return self._place('sell', qty, limit_price)

    def _place(self, side: Side, qty: float, limit_price: float) -> int:
        oid = self._next_oid
        self._next_oid += 1
        self._orders[oid] = _Order(oid=oid, side=side, qty=qty, limit_price=limit_price)
        return oid

    # ------------------------------------------------------------------
    # Bar update
    # ------------------------------------------------------------------
    def on_bar(self, pair: str, ohlcv: dict) -> None:
        high = float(ohlcv['high'])
        low = float(ohlcv['low'])
        close = float(ohlcv['close'])

        for order in list(self._orders.values()):
            if not order.open:
                continue

            filled_maker = False
            if order.side == 'buy' and low <= order.limit_price:
                self._fill_maker(order, price=order.limit_price)
                filled_maker = True
            elif order.side == 'sell' and high >= order.limit_price:
                self._fill_maker(order, price=order.limit_price)
                filled_maker = True

            if filled_maker:
                continue

            # No maker fill this bar.
            order.bars_waited += 1
            if order.bars_waited > self.max_wait_bars:
                # Force taker at market close + slippage.
                slip = 1.0 + self.taker_slippage if order.side == 'buy' else 1.0 - self.taker_slippage
                self._fill_taker(order, price=close * slip)

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------
    def _fill_maker(self, order: _Order, price: float) -> None:
        fee = price * order.qty * self.maker_fee
        order.fills.append({
            'price': price, 'qty': order.qty, 'role': 'maker', 'fee': fee,
            'side': order.side,
        })
        order.open = False

    def _fill_taker(self, order: _Order, price: float) -> None:
        fee = price * order.qty * self.taker_fee
        order.fills.append({
            'price': price, 'qty': order.qty, 'role': 'taker', 'fee': fee,
            'side': order.side,
        })
        order.open = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def cancel(self, oid: int) -> None:
        if oid in self._orders:
            self._orders[oid].open = False

    def is_open(self, oid: int) -> bool:
        o = self._orders.get(oid)
        return bool(o and o.open)

    def fills(self, oid: int) -> List[dict]:
        o = self._orders.get(oid)
        return list(o.fills) if o else []
