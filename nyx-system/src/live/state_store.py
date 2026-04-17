"""
StateStore — atomic persistence + recovery (Ticket 26).

Saves and loads OMS orders, Portfolio positions, and event logs
to JSON files with atomic writes (tmp → fsync → rename). A crash
mid-write cannot corrupt the state file.

On restart :
  store.load_oms(oms)       → restores order states + client_id registry
  store.load_portfolio(pf)  → restores positions + realized PnL + fees
  Event log is append-only JSONL for replay / audit.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from src.live.oms import OMS, OMSOrder
from src.live.portfolio_state import Portfolio, Position


class StateStore:
    """Atomic JSON persistence for OMS + Portfolio + event log."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Atomic write helper
    # ------------------------------------------------------------------
    def _atomic_write(self, path: Path, data: Any) -> None:
        """Write JSON atomically : tmp file → fsync → rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # OMS persistence
    # ------------------------------------------------------------------
    def save_oms(self, oms: OMS) -> None:
        orders = {}
        for oid, order in oms._orders.items():
            orders[str(oid)] = {
                'order_id': order.order_id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'price': order.price,
                'order_type': order.order_type,
                'post_only': order.post_only,
                'status': order.status,
                'created_at': order.created_at,
                'updated_at': order.updated_at,
                'filled_qty': order.filled_qty,
                'avg_fill_price': order.avg_fill_price,
                'reject_reason': order.reject_reason,
                'cancel_reason': order.cancel_reason,
                'exchange_order_id': order.exchange_order_id,
            }
        payload = {
            'orders': orders,
            'next_oid': oms._next_oid,
            'events': {str(k): v for k, v in oms._events.items()},
        }
        self._atomic_write(self.state_dir / 'oms_state.json', payload)

    def load_oms(self, oms: OMS) -> None:
        path = self.state_dir / 'oms_state.json'
        if not path.exists():
            return
        data = json.loads(path.read_text())
        for oid_str, o in data.get('orders', {}).items():
            order = OMSOrder(
                order_id=int(o['order_id']),
                client_order_id=o['client_order_id'],
                symbol=o['symbol'],
                side=o['side'],
                quantity=float(o['quantity']),
                price=float(o['price']),
                order_type=o.get('order_type', 'post_only_limit'),
                post_only=o.get('post_only', True),
                status=o['status'],
                created_at=float(o.get('created_at', 0)),
                updated_at=float(o.get('updated_at', 0)),
                filled_qty=float(o.get('filled_qty', 0)),
                avg_fill_price=float(o.get('avg_fill_price', 0)),
                reject_reason=o.get('reject_reason'),
                cancel_reason=o.get('cancel_reason'),
                exchange_order_id=o.get('exchange_order_id'),
            )
            oms._orders[order.order_id] = order
            oms._by_client_id[order.client_order_id] = order.order_id
        oms._next_oid = int(data.get('next_oid', oms._next_oid))
        for oid_str, events in data.get('events', {}).items():
            oms._events[int(oid_str)] = list(events)

    # ------------------------------------------------------------------
    # Portfolio persistence
    # ------------------------------------------------------------------
    def save_portfolio(self, pf: Portfolio) -> None:
        positions = {}
        for sym, pos in pf.positions.items():
            positions[sym] = {
                'symbol': pos.symbol,
                'side': pos.side,
                'quantity': pos.quantity,
                'avg_entry_price': pos.avg_entry_price,
                'realized_pnl': pos.realized_pnl,
                'open_timestamp': pos.open_timestamp,
            }
        payload = {
            'initial_capital': pf.initial_capital,
            'realized_pnl_total': pf.realized_pnl_total,
            'total_fees': pf.total_fees,
            'positions': positions,
        }
        self._atomic_write(self.state_dir / 'portfolio_state.json', payload)

    def load_portfolio(self, pf: Portfolio) -> None:
        path = self.state_dir / 'portfolio_state.json'
        if not path.exists():
            return
        data = json.loads(path.read_text())
        pf.realized_pnl_total = float(data.get('realized_pnl_total', 0))
        pf.total_fees = float(data.get('total_fees', 0))
        for sym, p in data.get('positions', {}).items():
            pos = Position(sym)
            pos.side = p['side']
            pos.quantity = float(p['quantity'])
            pos.avg_entry_price = float(p['avg_entry_price'])
            pos.realized_pnl = float(p.get('realized_pnl', 0))
            pos.open_timestamp = p.get('open_timestamp')
            pf.positions[sym] = pos

    # ------------------------------------------------------------------
    # Event log (append-only JSONL)
    # ------------------------------------------------------------------
    def save_event_log(self, oms: OMS) -> None:
        path = self.state_dir / 'event_log.jsonl'
        with open(path, 'w') as f:
            for oid, events in oms._events.items():
                for ev in events:
                    line = json.dumps({'order_id': oid, **ev}, default=str)
                    f.write(line + '\n')
