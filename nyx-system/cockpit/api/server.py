"""
NYX Cockpit API — FastAPI backend for the operator dashboard.

Serves REST endpoints + WebSocket streams from the canonical
NYXRuntime, ExecutionMonitor, Portfolio, OMS, and OOS Engine.

All data comes from the backend. No signal logic here.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))

log = logging.getLogger('nyx.cockpit')

app = FastAPI(title='NYX Cockpit API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

_state: Dict[str, Any] = {
    'runtimes': {},
    'execution_monitor': None,
    'oos_engine': None,
    'last_bars': {},
    'last_results': {},
    'system_mode': 'paper',
    'system_state': 'stopped',
    'connected_assets': [],
}


def register_runtime(symbol: str, runtime: Any) -> None:
    _state['runtimes'][symbol] = runtime
    if symbol not in _state['connected_assets']:
        _state['connected_assets'].append(symbol)


def register_execution_monitor(monitor: Any) -> None:
    _state['execution_monitor'] = monitor


def register_oos_engine(engine: Any) -> None:
    _state['oos_engine'] = engine


def set_system_state(state: str) -> None:
    _state['system_state'] = state


def record_bar_result(symbol: str, bar: dict, result: dict) -> None:
    _state['last_bars'][symbol] = bar
    _state['last_results'][symbol] = result


# =========================================================================
# REST Endpoints
# =========================================================================

@app.get('/api/state')
def get_state() -> Dict[str, Any]:
    assets = _state['connected_assets']
    em = _state['execution_monitor']
    global_health = 'normal'
    if em:
        for sym in assets:
            h = em.get_health(sym)
            if h < 0.35:
                global_health = 'critical'
                break
            elif h < 0.7:
                global_health = 'degraded'
    return {
        'system_state': _state['system_state'],
        'system_mode': _state['system_mode'],
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'connected_assets': assets,
        'global_execution_state': global_health,
        'global_risk_multiplier': min(
            (em.get_multiplier(s) for s in assets), default=1.0
        ) if em and assets else 1.0,
    }


@app.get('/api/portfolio')
def get_portfolio() -> Dict[str, Any]:
    out: Dict[str, Any] = {'assets': {}}
    for sym, rt in _state['runtimes'].items():
        pf = rt.portfolio
        mark = _state['last_bars'].get(sym, {}).get('close', 0)
        pos = pf.get_position(sym)
        out['assets'][sym] = {
            'side': pos.side if pos else 'flat',
            'quantity': pos.quantity if pos else 0.0,
            'avg_entry': pos.avg_entry_price if pos else 0.0,
            'unrealized_pnl': pos.unrealized_pnl(mark) if pos and mark else 0.0,
            'realized_pnl': pos.realized_pnl if pos else 0.0,
        }
    first_rt = next(iter(_state['runtimes'].values()), None)
    if first_rt:
        marks = {s: _state['last_bars'].get(s, {}).get('close', 0)
                 for s in _state['runtimes']}
        out['total_equity'] = first_rt.portfolio.equity_at(marks)
        out['available_balance'] = first_rt.portfolio.available_balance
        out['total_exposure'] = first_rt.portfolio.total_exposure
        out['realized_pnl_total'] = first_rt.portfolio.realized_pnl_total
    return out


@app.get('/api/orders')
def get_orders(limit: int = 50) -> List[Dict[str, Any]]:
    orders = []
    for sym, rt in _state['runtimes'].items():
        for oid, order in list(rt.oms._orders.items())[-limit:]:
            orders.append({
                'order_id': order.order_id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'price': order.price,
                'status': order.status,
                'filled_qty': order.filled_qty,
                'avg_fill_price': order.avg_fill_price,
                'created_at': order.created_at,
                'updated_at': order.updated_at,
                'reject_reason': order.reject_reason,
                'cancel_reason': order.cancel_reason,
            })
    orders.sort(key=lambda x: x['created_at'], reverse=True)
    return orders[:limit]


@app.get('/api/health')
def get_health() -> Dict[str, Any]:
    em = _state['execution_monitor']
    if not em:
        return {'available': False}
    return em.snapshot()


@app.get('/api/metrics/{symbol}')
def get_metrics(symbol: str) -> Dict[str, Any]:
    rt = _state['runtimes'].get(symbol)
    if not rt:
        return {'error': f'unknown symbol: {symbol}'}
    return rt.metrics.snapshot()


@app.get('/api/heartbeat/{symbol}')
def get_heartbeat(symbol: str) -> Dict[str, Any]:
    rt = _state['runtimes'].get(symbol)
    if not rt:
        return {'error': f'unknown symbol: {symbol}'}
    return rt.heartbeat()


@app.get('/api/decision/{symbol}')
def get_last_decision(symbol: str) -> Dict[str, Any]:
    result = _state['last_results'].get(symbol)
    if not result:
        return {'symbol': symbol, 'action': 'NO_DATA', 'layers': {}}
    return result


@app.get('/api/runs')
def get_oos_runs() -> List[str]:
    engine = _state['oos_engine']
    if not engine:
        return []
    return engine.list_oos_runs()


@app.get('/api/runs/{run_id}')
def get_oos_run(run_id: str) -> Dict[str, Any]:
    engine = _state['oos_engine']
    if not engine:
        return {'error': 'no OOS engine'}
    result = engine.load_oos_report(run_id)
    return result.to_dict()


@app.get('/api/audit/{symbol}')
def get_audit(symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
    for sym, rt in _state['runtimes'].items():
        if sym == symbol and hasattr(rt, 'audit_store') and rt.audit_store:
            events = rt.audit_store.load_asset_events(symbol)
            return events[-limit:]
    return []


# =========================================================================
# WebSocket Streams
# =========================================================================

_ws_clients: List[WebSocket] = []


async def broadcast(channel: str, data: dict) -> None:
    msg = json.dumps({'channel': channel, 'data': data, 'ts': time.time()}, default=str)
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_clients.remove(ws)


@app.websocket('/ws/live')
async def ws_live(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.append(ws)
    log.info('WebSocket client connected (%d total)', len(_ws_clients))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
        log.info('WebSocket client disconnected (%d remain)', len(_ws_clients))


async def push_bar_result(symbol: str, bar: dict, result: dict) -> None:
    record_bar_result(symbol, bar, result)
    await broadcast('runtime', {'symbol': symbol, 'bar': bar, 'result': result})

    em = _state['execution_monitor']
    if em:
        await broadcast('health', em.snapshot())

    rt = _state['runtimes'].get(symbol)
    if rt:
        marks = {s: _state['last_bars'].get(s, {}).get('close', 0)
                 for s in _state['runtimes']}
        await broadcast('portfolio', {
            'equity': rt.portfolio.equity_at(marks),
            'balance': rt.portfolio.available_balance,
            'exposure': rt.portfolio.total_exposure,
            'pnl': rt.portfolio.realized_pnl_total,
        })
