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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))

from cockpit.api.security import (
    UserStore, SecurityAuditLog, RateLimiter, EnvironmentConfig,
    create_token, decode_token, check_permission,
)

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
    'paper_control': None,
    'last_bars': {},
    'last_results': {},
    'system_mode': 'paper',
    'system_state': 'stopped',
    'connected_assets': [],
    # Time-series history (ring buffers)
    'health_history': {},       # symbol → List[{ts, health, smoothed, multiplier}]
    'equity_history': [],       # List[{ts, equity, balance, exposure, pnl}]
    'decision_history': {},     # symbol → List[{ts, action, p_trade, size_mult}]
    'ohlcv': {},                # symbol → List[{ts, open, high, low, close, volume}]
}
_HISTORY_MAX = 500


def register_runtime(symbol: str, runtime: Any) -> None:
    _state['runtimes'][symbol] = runtime
    if symbol not in _state['connected_assets']:
        _state['connected_assets'].append(symbol)


def register_execution_monitor(monitor: Any) -> None:
    _state['execution_monitor'] = monitor


def register_oos_engine(engine: Any) -> None:
    _state['oos_engine'] = engine


def register_paper_control(svc: Any) -> None:
    _state['paper_control'] = svc


def set_system_state(state: str) -> None:
    _state['system_state'] = state


# =========================================================================
# Security (SEC-1)
# =========================================================================

_user_store = UserStore(create_default_admin=False)
_security_log = SecurityAuditLog()
_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)
_env_config = EnvironmentConfig(env=os.environ.get('NYX_ENV', 'paper'))


class LoginRequest(BaseModel):
    username: str
    password: str


def get_current_user(request: Request) -> Dict[str, str]:
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing token')
    token = auth[7:]
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail='Invalid or expired token')
    return {'user_id': payload['user_id'], 'role': payload['role']}


def require_role(min_role: str):
    def checker(user: Dict[str, str] = Depends(get_current_user)):
        if not check_permission(user['role'], min_role):
            raise HTTPException(status_code=403, detail=f'Insufficient role: need {min_role}')
        return user
    return checker


@app.post('/api/auth/login')
def login(body: LoginRequest) -> Dict[str, Any]:
    ip = 'unknown'
    if not _rate_limiter.allow(f'login:{body.username}'):
        _security_log.record('login_rate_limited', user_id=body.username, success=False)
        raise HTTPException(status_code=429, detail='Too many login attempts')
    user = _user_store.authenticate(body.username, body.password)
    if user is None:
        _security_log.record('login', user_id=body.username, success=False)
        raise HTTPException(status_code=401, detail='Invalid credentials')
    token = create_token(user_id=user['username'], role=user['role'])
    _security_log.record('login', user_id=user['username'], success=True)
    return {
        'token': token,
        'user_id': user['username'],
        'role': user['role'],
        'environment': _env_config.env,
    }


@app.post('/api/auth/bootstrap')
def bootstrap(body: LoginRequest) -> Dict[str, Any]:
    if not _rate_limiter.allow('bootstrap'):
        raise HTTPException(status_code=429, detail='Too many attempts')
    result = _user_store.bootstrap(body.username, body.password, audit_log=_security_log)
    if not result['success']:
        raise HTTPException(status_code=403, detail=result.get('reason', 'already initialized'))
    token = create_token(user_id=body.username, role='admin')
    return {'token': token, 'user_id': body.username, 'role': 'admin', 'environment': _env_config.env}


@app.get('/api/auth/needs-bootstrap')
def needs_bootstrap() -> Dict[str, bool]:
    return {'needs_bootstrap': _user_store.needs_bootstrap()}


@app.post('/api/auth/logout')
def logout(user: Dict[str, str] = Depends(get_current_user)) -> Dict[str, str]:
    _security_log.record('logout', user_id=user['user_id'], success=True)
    return {'status': 'logged_out'}


@app.get('/api/auth/me')
def get_me(user: Dict[str, str] = Depends(get_current_user)) -> Dict[str, Any]:
    return {
        'user_id': user['user_id'],
        'role': user['role'],
        'environment': _env_config.env,
    }


@app.get('/api/security/audit')
def get_security_audit(
    user: Dict[str, str] = Depends(require_role('admin')),
    limit: int = 100,
) -> List[Dict[str, Any]]:
    return _security_log.entries[-limit:]


def record_bar_result(symbol: str, bar: dict, result: dict) -> None:
    _state['last_bars'][symbol] = bar
    _state['last_results'][symbol] = result

    ts = bar.get('timestamp', time.time())

    ohlcv = _state['ohlcv'].setdefault(symbol, [])
    ohlcv.append({
        'ts': ts,
        'open': float(bar.get('open', 0)),
        'high': float(bar.get('high', 0)),
        'low': float(bar.get('low', 0)),
        'close': float(bar.get('close', 0)),
        'volume': float(bar.get('volume', 0)),
    })
    if len(ohlcv) > _HISTORY_MAX:
        _state['ohlcv'][symbol] = ohlcv[-_HISTORY_MAX:]

    em = _state['execution_monitor']
    if em and symbol in em.assets:
        hist = _state['health_history'].setdefault(symbol, [])
        hist.append({
            'ts': ts,
            'health': em.get_health(symbol),
            'smoothed': em.get_smoothed_health(symbol) if em.use_v2 else em.get_health(symbol),
            'multiplier': em.get_multiplier(symbol),
        })
        if len(hist) > _HISTORY_MAX:
            _state['health_history'][symbol] = hist[-_HISTORY_MAX:]

    layers = result.get('layers', {})
    signal = layers.get('signal', {})
    fq = layers.get('fractal_quality', {})
    em_layer = layers.get('execution_monitor', {})
    dhist = _state['decision_history'].setdefault(symbol, [])
    dhist.append({
        'ts': ts,
        'action': result.get('action', 'FLAT'),
        'p_trade': signal.get('p_trade', 0),
        'direction': signal.get('direction', 0),
        'size_mult': fq.get('size_multiplier', 0),
        'risk_mult': em_layer.get('risk_multiplier', 1.0),
    })
    if len(dhist) > _HISTORY_MAX:
        _state['decision_history'][symbol] = dhist[-_HISTORY_MAX:]

    rt = _state['runtimes'].get(symbol)
    if rt:
        marks = {s: _state['last_bars'].get(s, {}).get('close', 0)
                 for s in _state['runtimes']}
        eq_hist = _state['equity_history']
        eq_hist.append({
            'ts': ts,
            'equity': rt.portfolio.equity_at(marks),
            'balance': rt.portfolio.available_balance,
            'exposure': rt.portfolio.total_exposure,
            'pnl': rt.portfolio.realized_pnl_total,
        })
        if len(eq_hist) > _HISTORY_MAX:
            _state['equity_history'] = eq_hist[-_HISTORY_MAX:]


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
# Time-series endpoints (chart-ready)
# =========================================================================

@app.get('/api/series/health/{symbol}')
def get_health_series(symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
    hist = _state['health_history'].get(symbol, [])
    return hist[-limit:]


@app.get('/api/series/equity')
def get_equity_series(limit: int = 200) -> List[Dict[str, Any]]:
    return _state['equity_history'][-limit:]


@app.get('/api/series/decisions/{symbol}')
def get_decision_series(symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
    hist = _state['decision_history'].get(symbol, [])
    return hist[-limit:]


@app.get('/api/series/ohlcv/{symbol}')
def get_ohlcv_series(symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
    return _state['ohlcv'].get(symbol, [])[-limit:]


# =========================================================================
# Analytics / heatmap endpoints (from evaluators)
# =========================================================================

@app.get('/api/analytics/stability/{run_id}')
def get_stability_analytics(run_id: str) -> Dict[str, Any]:
    """Build chart-ready stability payloads for the frontend."""
    engine = _state['oos_engine']
    if not engine:
        return {'error': 'no engine'}
    try:
        from src.live.evaluators.stability import StabilityEvaluator
        result = engine.load_oos_report(run_id)
        ev = StabilityEvaluator()
        output = ev.evaluate(result)
    except Exception as e:
        return {'error': str(e)}

    per_asset = output.get('per_asset', {})

    heatmap_segments: List[Dict[str, Any]] = []
    stability_scores: List[Dict[str, Any]] = []
    regime_matrix: List[Dict[str, Any]] = []
    tail_distribution: List[Dict[str, Any]] = []

    for sym, d in per_asset.items():
        for seg in d.get('segments', []):
            heatmap_segments.append({
                'asset': sym,
                'segment': seg.get('label', '?'),
                'sharpe': seg.get('sharpe', 0),
                'pnl': seg.get('total_pnl', 0),
                'dd': seg.get('max_drawdown_pct', 0),
            })
        stability_scores.append({
            'asset': sym,
            'score_v1': d.get('metrics', {}).get('stability_score', 0),
            'score_v2': d.get('stability_score_v2', 0),
            'classification': d.get('classification', '?'),
            'flags': d.get('advanced_flags', []),
        })
        for regime, v in d.get('regime_matrix', {}).items():
            regime_matrix.append({
                'asset': sym, 'regime': regime,
                'sharpe': v.get('mean_sharpe', 0),
                'dd': v.get('mean_dd', 0),
                'wr': v.get('mean_wr', 0),
            })
        tr = d.get('tail_risk', {})
        tail_distribution.append({
            'asset': sym,
            'worst_trade': tr.get('worst_trade', 0),
            'tail_95': tr.get('tail_loss_95', 0),
            'tail_99': tr.get('tail_loss_99', 0),
            'skewness': tr.get('skewness', 0),
            'kurtosis': tr.get('kurtosis', 0),
        })

    return {
        'run_id': run_id,
        'heatmap_segments': heatmap_segments,
        'stability_scores': stability_scores,
        'regime_matrix': regime_matrix,
        'tail_distribution': tail_distribution,
    }


@app.get('/api/analytics/capacity/{run_id}')
def get_capacity_analytics(run_id: str) -> Dict[str, Any]:
    engine = _state['oos_engine']
    if not engine:
        return {'error': 'no engine'}
    try:
        from src.live.evaluators.capacity import CapacityEvaluator
        result = engine.load_oos_report(run_id)
        ev = CapacityEvaluator()
        output = ev.evaluate(result)
    except Exception as e:
        return {'error': str(e)}

    ladder_series: List[Dict[str, Any]] = []
    deployability: List[Dict[str, Any]] = []
    breakpoints: List[Dict[str, Any]] = []

    for sym, d in output.get('per_asset', {}).items():
        for r in d.get('scenario_results', []):
            ladder_series.append({
                'asset': sym,
                'capital': r.get('capital', 0),
                'sharpe': r.get('sharpe', 0),
                'pnl': r.get('total_pnl', 0),
                'retention': r.get('edge_retention', 1.0),
            })
        deployability.append({
            'asset': sym,
            'class': d.get('deployability', {}).get('class', '?'),
            'max_capital': d.get('deployability', {}).get('max_recommended_capital', 0),
            'flags': d.get('flags', []),
        })
        bp = d.get('breakpoint')
        if bp:
            breakpoints.append({'asset': sym, **bp})

    return {
        'run_id': run_id,
        'ladder_series': ladder_series,
        'deployability': deployability,
        'breakpoints': breakpoints,
    }


@app.get('/api/analytics/execution_stress/{run_id}')
def get_execution_stress_analytics(run_id: str) -> Dict[str, Any]:
    engine = _state['oos_engine']
    if not engine:
        return {'error': 'no engine'}
    try:
        from src.live.evaluators.capacity_execution import CapacityExecutionEvaluator
        result = engine.load_oos_report(run_id)
        ev = CapacityExecutionEvaluator()
        output = ev.evaluate(result)
    except Exception as e:
        return {'error': str(e)}

    # Build heatmap: capital × fee_multiplier × composite_retention
    heatmap_cells: List[Dict[str, Any]] = []
    sensitivities: List[Dict[str, Any]] = []
    deploy: List[Dict[str, Any]] = []

    for sym, d in output.get('per_asset', {}).items():
        for s in d.get('scenario_grid', []):
            heatmap_cells.append({
                'asset': sym,
                'capital': s.get('capital', 0),
                'offset_bps': s.get('offset_bps', 0),
                'fill_degradation': s.get('fill_degradation', 1.0),
                'fee_multiplier': s.get('fee_multiplier', 1.0),
                'composite_retention': s.get('composite_retention', 0),
                'miss_rate': s.get('miss_rate', 0),
            })
        sens = d.get('sensitivities', {})
        sensitivities.append({
            'asset': sym,
            'offset': sens.get('offset', 0),
            'fill': sens.get('fill', 0),
            'fee': sens.get('fee', 0),
        })
        deploy.append({
            'asset': sym,
            'class': d.get('deployability', {}).get('class', '?'),
            'flags': d.get('execution_flags', []),
        })

    return {
        'run_id': run_id,
        'heatmap_cells': heatmap_cells,
        'sensitivities': sensitivities,
        'deployability': deploy,
    }


@app.get('/api/analytics/fill_miss_bars')
def get_fill_miss_bars() -> List[Dict[str, Any]]:
    """Per-asset bar chart payload: placed / filled / timed_out."""
    em = _state['execution_monitor']
    if not em:
        return []
    out = []
    for sym in em.assets:
        snap = em._collector.snapshot(sym)
        out.append({
            'asset': sym,
            'placed': snap.get('placed', 0),
            'filled': snap.get('filled', 0),
            'timed_out': snap.get('missed', 0),
        })
    return out


# =========================================================================
# Paper Trading Control Plane (UI-2)
# =========================================================================

@app.get('/api/control/paper/status')
def get_paper_status() -> Dict[str, Any]:
    svc = _state['paper_control']
    if not svc:
        return {'current_state': 'no_control_service', 'restrictions': []}
    return svc.status()


@app.post('/api/control/paper/{action}')
async def paper_control_action(action: str, reason: str = '') -> Dict[str, Any]:
    svc = _state['paper_control']
    if not svc:
        return {'success': False, 'reason': 'no_control_service'}
    result = svc.execute(action, source='operator', reason=reason)
    await broadcast('paper_control', {
        'action': action,
        'result': result,
        'status': svc.status(),
    })
    return result


@app.get('/api/control/paper/audit')
def get_paper_audit(limit: int = 50) -> List[Dict[str, Any]]:
    svc = _state['paper_control']
    if not svc:
        return []
    return svc.audit_log[-limit:]


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
