"""
Position & Portfolio State — single source of truth (Ticket 24).

Derived ONLY from OMS fill events. No parallel state system.

Position lifecycle :
  fill(buy)  → open/add to long  OR  close/reduce short
  fill(sell) → open/add to short OR  close/reduce long
  partial close → reduce qty, realize PnL on closed portion
  full close → qty=0, side='flat', realize full PnL
  flip → close existing + open opposite in same fill

Portfolio :
  total_equity = initial_capital + realized_pnl_total - total_fees + Σ unrealized_pnl
  available_balance = initial_capital + realized_pnl_total - total_fees
  total_exposure = Σ |position.qty × mark_price| per symbol

NYXEngine READS this state. It never reconstructs it.
"""
from __future__ import annotations

from typing import Dict, Optional


class Position:
    """Single-symbol position tracker."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.side: str = 'flat'         # 'long' / 'short' / 'flat'
        self.quantity: float = 0.0
        self.avg_entry_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.open_timestamp: Optional[str] = None

    def on_fill(self, side: str, qty: float, price: float,
                timestamp: Optional[str] = None) -> float:
        """Process a fill. Returns realized PnL from this fill (0 if
        opening / adding, > 0 if closing profitably, < 0 if closing
        at a loss)."""
        qty = float(qty)
        price = float(price)
        rpnl = 0.0

        if self.side == 'flat':
            self.side = 'long' if side == 'buy' else 'short'
            self.quantity = qty
            self.avg_entry_price = price
            self.open_timestamp = timestamp

        elif (self.side == 'long' and side == 'buy') or \
             (self.side == 'short' and side == 'sell'):
            # Adding to position — VWAP avg entry.
            total_cost = self.avg_entry_price * self.quantity + price * qty
            self.quantity += qty
            self.avg_entry_price = total_cost / self.quantity

        else:
            # Closing (partially or fully) or flipping.
            close_qty = min(qty, self.quantity)
            if self.side == 'long':
                rpnl = close_qty * (price - self.avg_entry_price)
            else:
                rpnl = close_qty * (self.avg_entry_price - price)
            self.realized_pnl += rpnl
            self.quantity -= close_qty

            remainder = qty - close_qty
            if self.quantity < 1e-12:
                self.quantity = 0.0
                if remainder > 1e-12:
                    # Flip position.
                    self.side = 'long' if side == 'buy' else 'short'
                    self.quantity = remainder
                    self.avg_entry_price = price
                    self.open_timestamp = timestamp
                else:
                    self.side = 'flat'
                    self.avg_entry_price = 0.0

        return rpnl

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.quantity < 1e-12:
            return 0.0
        if self.side == 'long':
            return self.quantity * (mark_price - self.avg_entry_price)
        elif self.side == 'short':
            return self.quantity * (self.avg_entry_price - mark_price)
        return 0.0


class Portfolio:
    """Multi-symbol portfolio state — single source of truth."""

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self.initial_capital = float(initial_capital)
        self.positions: Dict[str, Position] = {}
        self.realized_pnl_total: float = 0.0
        self.total_fees: float = 0.0

    @property
    def available_balance(self) -> float:
        return self.initial_capital + self.realized_pnl_total - self.total_fees

    @property
    def total_equity(self) -> float:
        return self.available_balance

    @property
    def total_exposure(self) -> float:
        return sum(
            p.quantity * p.avg_entry_price
            for p in self.positions.values()
            if p.quantity > 0
        )

    def on_fill(self, symbol: str, side: str, qty: float,
                price: float, fee: float = 0.0,
                timestamp: Optional[str] = None) -> None:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        rpnl = self.positions[symbol].on_fill(side, qty, price, timestamp)
        self.realized_pnl_total += rpnl
        self.total_fees += float(fee)

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def exposure(self, symbol: str, mark_price: float) -> float:
        p = self.positions.get(symbol)
        if p is None or p.quantity < 1e-12:
            return 0.0
        return p.quantity * mark_price

    def total_exposure_at(self, marks: Dict[str, float]) -> float:
        total = 0.0
        for sym, p in self.positions.items():
            if p.quantity > 1e-12 and sym in marks:
                total += p.quantity * marks[sym]
        return total

    def equity_at(self, marks: Dict[str, float]) -> float:
        upnl = sum(
            p.unrealized_pnl(marks.get(sym, p.avg_entry_price))
            for sym, p in self.positions.items()
        )
        return self.available_balance + upnl
