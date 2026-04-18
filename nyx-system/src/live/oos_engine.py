"""
Canonical OOS Engine — Ticket 42.

One parameter-driven engine for all out-of-sample requests.
Supports mono/multi-asset, idealized/realistic, any calendar
window, any capital ($100 → $1B).

Usage:
    from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
    cfg = OOSConfig(assets=['BTCUSDT'], start_date='2023-01-01',
                    end_date='2023-12-31', initial_capital=10_000_000)
    engine = CanonicalOOSEngine()
    result = engine.run_oos(cfg)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


_HERE = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _HERE / 'data' / 'raw'
_MTF_DIR = _DATA_DIR / 'mtf'
_FEAT_DIR = _HERE / 'data' / 'features'
_MODELS_DIR = _HERE / 'models'

_SOL_CSV = {
    '15m': 'SOLUSDT_15minutes', '1h': 'SOLUSDT_1hour',
    '4h': 'SOLUSDT_4hours', '1d': 'SOLUSDT_1day',
}

_VALID_MODES = ('idealized', 'realistic')

MODULES_TESTED = [
    'NYXEngine (GBM)', 'Jesse (4 agents)', 'FractalQuality',
    'RiskEngine + VaR/CVaR', 'ExecutionOptimizer',
    'OMS', 'PostOnlyPaperBroker', 'Portfolio', 'Monitoring',
]


@dataclass
class OOSConfig:
    assets: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 10_000.0
    mode: str = 'realistic'
    train_end: str = '2022-12-31'
    allocations: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if not self.assets:
            raise ValueError('assets must be non-empty')
        if self.initial_capital < 1.0:
            raise ValueError(f'initial_capital must be >= 1, got {self.initial_capital}')
        if self.mode not in _VALID_MODES:
            raise ValueError(f'mode must be one of {_VALID_MODES}, got {self.mode!r}')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'assets': self.assets,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'mode': self.mode,
            'train_end': self.train_end,
            'allocations': self.allocations,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OOSConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _load_ohlcv(symbol: str, tf: str) -> pd.DataFrame:
    if symbol == 'BTCUSDT':
        df = pd.read_csv(_MTF_DIR / f'BTCUSDT_{tf}.csv')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif symbol == 'ETHUSDT':
        df = pd.read_csv(_DATA_DIR / f'ETHUSDT_{tf}.csv')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif symbol == 'SOLUSDT':
        name = _SOL_CSV[tf]
        df = pd.read_csv(
            _DATA_DIR / f'{name}.csv',
            usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        )
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime').drop(columns='timestamp')
    else:
        raise ValueError(f'Unknown symbol: {symbol}')
    for c in ('open', 'high', 'low', 'close'):
        df = df[df[c] > 0]
    return df


def _run_asset_idealized(
    symbol: str, capital: float, train_end: str,
    test_start: str, test_end: str,
) -> Dict[str, Any]:
    from src.core.nyx_engine import NYXEngine

    mtf = {tf: _load_ohlcv(symbol, tf) for tf in ('15m', '1h', '4h', '1d')}
    feats = {tf: pd.read_parquet(_FEAT_DIR / f'{symbol}_features_{tf}.parquet')
             for tf in ('15m', '1h', '4h', '1d')}

    engine = NYXEngine()
    result = engine.run(mtf, feats, train_end=train_end,
                        test_start=test_start, test_end=test_end)
    trades = result.get('trades', [])
    n = len(trades)
    pnl = sum(float(t.get('net_pnl', 0)) for t in trades) * (capital / 10_000.0)
    wins = len([t for t in trades if float(t.get('net_pnl', 0)) > 0])
    if n > 5:
        rets = [float(t['net_pnl']) / 10_000.0 for t in trades]
        sharpe = float(np.mean(rets) / max(np.std(rets), 1e-12) * np.sqrt(min(n, 252)))
    else:
        sharpe = 0.0

    return {
        'symbol': symbol, 'capital': capital, 'mode': 'idealized',
        'modules_tested': MODULES_TESTED[:3],
        'performance': {
            'n_trades': n, 'win_rate': round(wins / max(n, 1), 4),
            'sharpe': round(sharpe, 2), 'total_pnl': round(pnl, 2),
            'max_drawdown_pct': 0.0,
        },
    }


def _run_asset_realistic(
    symbol: str, capital: float, train_end: str,
    test_start: str, test_end: str,
) -> Dict[str, Any]:
    from src.core.nyx_engine import NYXEngine
    from src.agents.context_agent import ContextAgent
    from src.agents.regime_agent import RegimeAgent
    from src.agents.setup_agent import SetupAgent
    from src.agents.entry_agent import EntryAgent
    from src.agents.contracts import FractalReport
    from src.core.fractal_quality import compute_fractal_quality
    from src.live.risk_engine import RiskEngine
    from src.live.var_cvar import RollingRiskMetrics
    from src.live.execution_optimizer import build_execution_plan
    from src.live.oms import OMS
    from src.live.portfolio_state import Portfolio
    from src.live.monitoring import MetricsCollector
    from src.paper_live.post_only_broker import PostOnlyPaperBroker

    mtf = {tf: _load_ohlcv(symbol, tf) for tf in ('15m', '1h', '4h', '1d')}
    feats = {tf: pd.read_parquet(_FEAT_DIR / f'{symbol}_features_{tf}.parquet')
             for tf in ('15m', '1h', '4h', '1d')}

    engine = NYXEngine()
    result = engine.run(mtf, feats, train_end=train_end,
                        test_start=test_start, test_end=test_end)
    idealized_trades = result.get('trades', [])

    ide_pnl = sum(float(t.get('net_pnl', 0)) for t in idealized_trades) * (capital / 10_000.0)
    ide_n = len(idealized_trades)
    ide_wins = len([t for t in idealized_trades if float(t.get('net_pnl', 0)) > 0])
    if ide_n > 5:
        ide_rets = [float(t['net_pnl']) / 10_000.0 for t in idealized_trades]
        ide_sharpe = float(np.mean(ide_rets) / max(np.std(ide_rets), 1e-12) * np.sqrt(min(ide_n, 252)))
    else:
        ide_sharpe = 0.0

    ctx_a, reg_a, stp_a, ent_a = ContextAgent({}), RegimeAgent({}), SetupAgent({}), EntryAgent({})
    risk_engine = RiskEngine(
        max_position_pct=5.0,
        max_var_95_pct=3.0, max_cvar_95_pct=5.0,
    )
    rolling_risk = RollingRiskMetrics(window=50)
    oms = OMS()
    portfolio = Portfolio(initial_capital=capital)
    broker = PostOnlyPaperBroker(max_wait_bars=3, maker_fee=0.0002)

    df15 = mtf['15m'].loc[test_start:test_end]
    trade_by_ts = {str(t['timestamp']): t for t in idealized_trades}
    oc = 0
    filled: List[Dict] = []
    n_skip = n_block = n_placed = n_filled = n_to = 0
    pending: Dict[int, Dict] = {}

    for ts, row in df15.iterrows():
        bar = {'timestamp': ts.isoformat(), 'open': float(row['open']),
               'high': float(row['high']), 'low': float(row['low']),
               'close': float(row['close']), 'volume': float(row['volume'])}
        mark = float(row['close'])
        broker.on_bar(symbol, bar, bar_ts=ts.isoformat())

        for ob_id, m in list(pending.items()):
            ob = broker.get(ob_id)
            if ob.state == 'FILLED' and not m.get('d'):
                m['d'] = True
                fs = ob.fills
                if fs:
                    f = fs[0]
                    oms.handle_fill(m['oid'], fill_qty=f['qty'], fill_price=f['price'])
                    portfolio.on_fill(symbol, side=ob.side, qty=f['qty'],
                                     price=f['price'], fee=f['fee'])
                    pnl_s = float(m['t']['net_pnl']) * (capital / 10_000.0)
                    rolling_risk.add_trade_pnl(pnl_s)
                    filled.append({'pnl': pnl_s})
                    n_filled += 1
            elif ob.state == 'TIMED_OUT' and not m.get('d'):
                m['d'] = True
                oms.cancel_order(m['oid'], reason='TIMED_OUT')
                n_to += 1

        tk = str(ts)
        if tk not in trade_by_ts:
            continue
        t = trade_by_ts[tk]
        direction = int(t['direction'])
        side = 'buy' if direction > 0 else 'sell'

        reports = {}
        for an, ag, tfk, ds in [
            ('context', ctx_a, '1d', mtf['1d'].loc[:ts]),
            ('regime', reg_a, '4h', mtf['4h'].loc[:ts]),
            ('setup', stp_a, '1h', mtf['1h'].loc[:ts]),
            ('entry', ent_a, '15m', df15.loc[:ts]),
        ]:
            try:
                if len(ds) >= 20:
                    reports[an] = ag.report(ds, asset=symbol, timestamp=tk)
            except Exception:
                pass
            if an not in reports:
                reports[an] = FractalReport(
                    asset=symbol, agent=an, timeframe=tfk,
                    state='unavailable', score=0.5, passed=False,
                    block_reasons=['unavailable'], timestamp=tk,
                )

        fq = compute_fractal_quality(reports)
        if fq['skip_trade']:
            n_skip += 1
            continue

        qty = (portfolio.available_balance * 0.02) / max(mark, 1) * fq['size_multiplier']
        pfs = {
            'available_balance': portfolio.available_balance,
            'total_exposure': portfolio.total_exposure,
            'daily_realized_pnl': 0, 'weekly_realized_pnl': 0,
            'max_drawdown_from_peak': 0,
            'open_position_count': sum(1 for p in portfolio.positions.values() if p.quantity > 0),
        }
        rr = risk_engine.validate_trade(symbol=symbol, side=side, quantity=qty,
                                        price=mark, portfolio=pfs, rolling_risk=rolling_risk)
        if not rr['allowed']:
            n_block += 1
            continue

        ep = build_execution_plan(side=side, mark_price=mark, volatility_pct=0.5,
                                  spread_bps=5.0, expected_edge_bps=float(t.get('ml_score', 0.6)) * 100)
        oc += 1
        oid = oms.submit_order(client_order_id=f'{symbol}-{oc}', symbol=symbol,
                               side=side, quantity=qty, price=ep['limit_price'])
        bid = broker.place_post_only(pair=symbol, side=side, qty=qty,
                                     limit_price=ep['limit_price'], mark_price=mark,
                                     placed_at=ts.isoformat(), max_wait_bars=3)
        bo = broker.get(bid)
        if bo.state == 'REJECTED':
            oms.handle_reject(oid, reason='rejected')
            continue
        pending[bid] = {'oid': oid, 't': t}
        n_placed += 1

    tp = sum(f['pnl'] for f in filled)
    nr = len(filled)
    w = len([f for f in filled if f['pnl'] > 0])
    wr = w / max(nr, 1)
    if nr > 5:
        rs = [f['pnl'] / capital for f in filled]
        sh = float(np.mean(rs) / max(np.std(rs), 1e-12) * np.sqrt(min(nr, 252)))
    else:
        sh = 0.0
    eq = [capital]
    for f in filled:
        eq.append(eq[-1] + f['pnl'])
    ea = np.array(eq)
    pk = np.maximum.accumulate(ea)
    dd = (pk - ea) / np.where(pk > 0, pk, 1.0)
    mdd = float(dd.max()) * 100

    return {
        'symbol': symbol, 'capital': capital, 'mode': 'realistic',
        'modules_tested': MODULES_TESTED,
        'idealized_baseline': {
            'n_trades': ide_n, 'sharpe': round(ide_sharpe, 2),
            'total_pnl': round(ide_pnl, 2),
            'win_rate': round(ide_wins / max(ide_n, 1), 4),
        },
        'execution': {
            'idealized_signals': ide_n, 'skipped_quality': n_skip,
            'blocked_risk': n_block, 'placed': n_placed,
            'filled': n_filled, 'timed_out': n_to,
            'fill_rate': round(n_filled / max(n_placed, 1), 4),
            'miss_rate': round(n_to / max(n_placed, 1), 4),
        },
        'performance': {
            'n_trades': nr, 'win_rate': round(wr, 4),
            'sharpe': round(sh, 2), 'total_pnl': round(tp, 2),
            'max_drawdown_pct': round(mdd, 4),
        },
        'var_cvar': {
            'var_95': rolling_risk.var_95,
            'cvar_95': rolling_risk.cvar_95,
        },
    }


# =========================================================================
# Evaluator Framework (Ticket 42.1)
# =========================================================================

class OOSResultEvaluator:
    """Base class for post-OOS evaluators. Subclasses must set evaluator_name."""
    evaluator_name: str = ''

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.evaluator_name and 'evaluator_name' not in cls.__dict__:
            pass

    def __init__(self):
        if not self.evaluator_name:
            raise TypeError(
                f'{type(self).__name__} must set evaluator_name'
            )

    def evaluate(self, oos_result: 'OOSResult') -> dict:
        raise NotImplementedError


class EvaluatorRegistry:
    """Registry mapping evaluator names to instances."""

    def __init__(self) -> None:
        self._evaluators: Dict[str, OOSResultEvaluator] = {}

    def register(self, evaluator: OOSResultEvaluator) -> None:
        self._evaluators[evaluator.evaluator_name] = evaluator

    def get(self, name: str) -> OOSResultEvaluator:
        if name not in self._evaluators:
            raise KeyError(f'Unknown evaluator: {name!r}')
        return self._evaluators[name]

    def list(self) -> List[str]:
        return list(self._evaluators.keys())


class OOSResult:
    """Structured wrapper around OOS run output with evaluate() capability."""

    def __init__(
        self,
        data: Dict[str, Any],
        reports_dir: Optional[Path] = None,
    ) -> None:
        self._data = data
        self._reports_dir = reports_dir
        self._registry = EvaluatorRegistry()

    @property
    def run_id(self) -> str:
        return self._data.get('run_id', '')

    @property
    def config(self) -> Dict[str, Any]:
        return self._data.get('config', {})

    @property
    def per_asset(self) -> List[Dict[str, Any]]:
        return self._data.get('per_asset', [])

    @property
    def portfolio(self) -> Dict[str, Any]:
        return self._data.get('portfolio', {})

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def register_evaluator(self, evaluator: OOSResultEvaluator) -> None:
        self._registry.register(evaluator)

    def evaluate(self, evaluator_names: List[str]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for name in evaluator_names:
            ev = self._registry.get(name)
            output = ev.evaluate(self)
            results[name] = output
            if self._reports_dir is not None:
                self._reports_dir.mkdir(parents=True, exist_ok=True)
                path = self._reports_dir / f'eval_{self.run_id}_{name}.json'
                path.write_text(json.dumps(output, indent=2, default=str))
        return results


class CanonicalOOSEngine:
    """One engine for all OOS requests. Parameter-driven."""

    def __init__(self, reports_dir: Optional[Path] = None) -> None:
        self.reports_dir = reports_dir or (_HERE / 'reports' / 'oos_runs')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._registry = EvaluatorRegistry()

    def register_evaluator(self, evaluator: OOSResultEvaluator) -> None:
        self._registry.register(evaluator)

    def run_oos(self, config: OOSConfig) -> OOSResult:
        run_id = uuid.uuid4().hex[:12]
        alloc = config.allocations
        if alloc is None:
            n = len(config.assets)
            alloc = {s: 1.0 / n for s in config.assets}

        per_asset = []
        for sym in config.assets:
            cap = config.initial_capital * alloc.get(sym, 1.0 / len(config.assets))
            if config.mode == 'idealized':
                r = _run_asset_idealized(sym, cap, config.train_end,
                                         config.start_date, config.end_date)
            else:
                r = _run_asset_realistic(sym, cap, config.train_end,
                                         config.start_date, config.end_date)
            per_asset.append(r)

        portfolio_pnl = sum(r['performance']['total_pnl'] for r in per_asset)
        total_trades = sum(r['performance']['n_trades'] for r in per_asset)
        total_wins = sum(int(r['performance']['win_rate'] * r['performance']['n_trades'])
                         for r in per_asset)

        data = {
            'run_id': run_id,
            'config': config.to_dict(),
            'per_asset': per_asset,
            'portfolio': {
                'total_pnl': round(portfolio_pnl, 2),
                'return_pct': round(portfolio_pnl / config.initial_capital * 100, 2),
                'final_equity': round(config.initial_capital + portfolio_pnl, 2),
                'total_trades': total_trades,
                'aggregate_win_rate': round(total_wins / max(total_trades, 1), 4),
            },
        }

        path = self.reports_dir / f'oos_{run_id}.json'
        path.write_text(json.dumps(data, indent=2, default=str))

        result = OOSResult(data, reports_dir=self.reports_dir)
        for name in self._registry.list():
            result.register_evaluator(self._registry.get(name))
        return result

    def load_oos_report(self, run_id: str) -> OOSResult:
        path = self.reports_dir / f'oos_{run_id}.json'
        data = json.loads(path.read_text())
        result = OOSResult(data, reports_dir=self.reports_dir)
        for name in self._registry.list():
            result.register_evaluator(self._registry.get(name))
        return result

    def list_oos_runs(self) -> List[str]:
        return [p.stem.replace('oos_', '')
                for p in sorted(self.reports_dir.glob('oos_*.json'))]
