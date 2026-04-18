"""
NYXRuntime — THE single orchestrator (Ticket 28).

One file. One loop. Every layer visible. No hidden calls.

    bar = get_next_bar()
    ├── 1. HARD GATE (edge candidate detection)
    ├── 2. GBM SCORE (primary signal — model.predict_proba)
    ├── 3. JESSE REPORTS (4 fractal agents — context/regime/setup/entry)
    ├── 4. FRACTAL QUALITY (modulation from Jesse reports)
    ├── 5. RISK ENGINE (sovereign — can block any trade)
    ├── 6. OMS (submit order — single source of execution state)
    ├── 7. BROKER (PostOnlyPaperBroker or exchange adapter)
    ├── 8. PORTFOLIO (update positions + PnL from fills)
    ├── 9. PERSISTENCE (atomic state save)
    └── 10. MONITORING (metrics + alerts)

If you can't see it here, it doesn't exist in the system.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.live.bar_builder import BarBuilder
from src.live.feed_health import FeedHealth
from src.live.monitoring import AlertManager, MetricsCollector
from src.live.oms import OMS
from src.live.portfolio_state import Portfolio
from src.live.risk_engine import RiskEngine
from src.live.state_store import StateStore
from src.core.fractal_quality import compute_fractal_quality
from src.agents.contracts import FractalReport

log = logging.getLogger(__name__)


class NYXRuntime:
    """THE single orchestrator. Everything goes through here."""

    def __init__(
        self,
        symbol: str = 'BTCUSDT',
        models_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None,
        initial_capital: float = 10_000.0,
        dependency_layer: Optional[Any] = None,
        portfolio_allocator: Optional[Any] = None,
        candidate_store: Optional[Any] = None,
        audit_store: Optional[Any] = None,
        execution_monitor: Optional[Any] = None,
    ) -> None:
        self.symbol = symbol
        self.dependency_layer = dependency_layer
        self.portfolio_allocator = portfolio_allocator
        self.candidate_store = candidate_store
        self.execution_monitor = execution_monitor
        self.audit_store = audit_store

        # --- 0. MODELS (GBM + Jesse agents) ---
        from src.ml.nyx_live_decider import NYXLiveDecider
        models_path = models_dir or Path('models') / symbol
        self.decider = NYXLiveDecider(
            symbol=symbol,
            artifact_dir=models_path,
        )
        # GBM is inside: self.decider._meta (MetaGBM wrapping the
        # trained GradientBoostingClassifier). Exposed here for
        # traceability:
        self.gbm = self.decider._meta

        # Jesse agents (per-file, ML-native post-Ticket-18).
        from src.agents.context_agent import ContextAgent
        from src.agents.regime_agent import RegimeAgent
        from src.agents.setup_agent import SetupAgent
        from src.agents.entry_agent import EntryAgent
        self.context_agent = ContextAgent({})
        self.regime_agent = RegimeAgent({})
        self.setup_agent = SetupAgent({})
        self.entry_agent = EntryAgent({})

        # --- 1-10. ALL LAYERS ---
        self.bar_builder = BarBuilder()
        self.feed_health = FeedHealth(stale_seconds=120)
        self.risk_engine = RiskEngine()
        self.oms = OMS()
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.state_store = StateStore(
            state_dir or Path('state') / symbol,
        )
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()

        self._order_counter = 0
        self._last_persist_time = 0.0

        # Ticket 29 — operator control state.
        self._running = False
        self._paused = False

    # ------------------------------------------------------------------
    # TICKET 29 — Control Plane (operator layer)
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start(self) -> None:
        self._running = True
        self._paused = False
        log.info('SYSTEM STARTED')

    def stop(self) -> None:
        self._running = False
        self._paused = False
        self.state_store.save_oms(self.oms)
        self.state_store.save_portfolio(self.portfolio)
        log.info('SYSTEM STOPPED — state persisted')

    def pause(self) -> None:
        self._paused = True
        log.info('TRADING PAUSED')

    def resume(self) -> None:
        self._paused = False
        log.info('TRADING RESUMED')

    def emergency_stop(self, reason: str = 'operator') -> None:
        """Halt everything immediately : kill switch + stop."""
        self.risk_engine.activate_kill_switch(reason=f'EMERGENCY: {reason}')
        self.stop()
        log.critical('EMERGENCY STOP: %s', reason)

    def cancel_all_orders(self) -> int:
        """Cancel every non-terminal order in OMS. Returns count."""
        n = 0
        for oid, order in list(self.oms._orders.items()):
            if order.status in ('SUBMITTED', 'PARTIALLY_FILLED'):
                self.oms.cancel_order(oid, reason='cancel_all')
                n += 1
        self.state_store.save_oms(self.oms)
        log.info('CANCEL ALL — %d orders cancelled', n)
        return n

    def flatten_all(self) -> list:
        """Return close instructions for every open position.

        Does NOT place orders itself — the caller (or a follow-up
        command) submits them through the runtime. This keeps the
        control plane declarative, not imperative.
        """
        to_close = []
        for sym, pos in self.portfolio.positions.items():
            if pos.quantity > 1e-12 and pos.side != 'flat':
                close_side = 'sell' if pos.side == 'long' else 'buy'
                to_close.append({
                    'symbol': sym,
                    'side': close_side,
                    'quantity': pos.quantity,
                    'reason': 'flatten_all',
                })
        log.info('FLATTEN ALL — %d positions to close', len(to_close))
        return to_close

    # ------------------------------------------------------------------
    def on_bar(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        """Process ONE 15m bar through the ENTIRE pipeline.

        Returns a structured result dict documenting what happened
        at each layer — full traceability, no hidden calls.

        This is the ONLY method external code should call per bar.
        """
        result: Dict[str, Any] = {
            'timestamp': bar.get('timestamp', ''),
            'action': 'FLAT',
            'layers': {},
        }

        # Ticket 29 — control plane guards.
        if not self._running:
            result['action'] = 'SYSTEM_STOPPED'
            return result
        if self._paused:
            result['action'] = 'PAUSED'
            return result

        # ---- 1. HARD GATE + GBM SCORE (inside NYXLiveDecider) ----
        # NYXLiveDecider.on_15m_bar does:
        #   - stack.on_15m_bar (MTF buffer update)
        #   - hard gate (EMA alignment + vol + hour)
        #   - MetaGBM.decide → model.predict_proba (THE GBM call)
        #   - bear dial conditional threshold
        #   - cooldown + daily limit
        signal = self.decider.on_15m_bar(bar)
        result['layers']['signal'] = {
            'direction': signal.direction,
            'conviction': signal.conviction,
            'p_trade': signal.conviction,  # = GBM probability
        }

        if signal.direction == 0:
            result['action'] = 'FLAT'
            self._periodic_persist()
            self._update_monitoring(result)
            return result

        # ---- 2. JESSE FRACTAL REPORTS (4 agents, called HERE) ----
        # Context (1D), Regime (4H), Setup (1H), Entry (15M).
        # Each agent's .report() = ML-native (Ticket 18).
        ts_str = bar.get('timestamp', '')
        jesse_reports = self._call_jesse_agents(bar, ts_str)
        result['layers']['jesse'] = {
            agent: {
                'state': r.state,
                'score': r.score,
                'passed': r.passed,
            } for agent, r in jesse_reports.items()
        }

        # ---- 3. FRACTAL QUALITY MODULATION (from Jesse reports) ----
        fq = compute_fractal_quality(jesse_reports)
        result['layers']['fractal_quality'] = fq

        if fq['skip_trade']:
            result['action'] = 'SKIP_QUALITY'
            self._periodic_persist()
            self._update_monitoring(result)
            return result

        size_multiplier = fq['size_multiplier']

        # ---- CANDIDATE MODE (Ticket 36) ----
        # When candidate_store is set, runtime emits a CandidateDecision
        # and stops. The CentralOrchestrator handles dependency, allocation,
        # risk, and execution on the full cross-asset set.
        if self.candidate_store is not None:
            from src.live.central_orchestrator import CandidateDecision
            mark_price = float(bar.get('close', 0))
            candidate = CandidateDecision(
                symbol=self.symbol,
                timestamp=bar.get('timestamp', ''),
                direction=1 if signal.direction > 0 else -1,
                confidence=float(signal.conviction),
                expected_edge_bps=float(signal.conviction) * 40.0,
                size_hint=size_multiplier,
                quality_bucket=fq.get('quality_bucket', 'medium'),
                entry_price=mark_price,
                jesse_reports={
                    agent: {'state': r.state, 'score': r.score, 'passed': r.passed}
                    for agent, r in jesse_reports.items()
                },
            )
            self.candidate_store.add(candidate)
            result['action'] = 'CANDIDATE_EMITTED'
            result['layers']['candidate'] = candidate.to_dict()
            self._periodic_persist()
            self._update_monitoring(result)
            return result

        # ---- 3b. INTER-ASSET DEPENDENCY (Ticket 34) ----
        # Modulates size_multiplier based on cross-asset dynamics.
        # Only active when a shared dependency_layer is provided.
        if self.dependency_layer is not None:
            dep_direction = 1 if signal.direction > 0 else -1
            close_price = float(bar.get('close', 0))
            prev_close = float(bar.get('open', close_price))
            bar_ret = (close_price - prev_close) / max(prev_close, 1e-12)
            self.dependency_layer.update_return(
                self.symbol,
                bar.get('timestamp', ''),
                bar_ret,
            )
            dep_result = self.dependency_layer.evaluate(
                symbol=self.symbol,
                direction=dep_direction,
                size_multiplier=size_multiplier,
            )
            result['layers']['dependency'] = dep_result
            if dep_result.get('suppress_trade', False):
                result['action'] = 'SKIP_DEPENDENCY'
                self._periodic_persist()
                self._update_monitoring(result)
                return result
            size_multiplier = dep_result.get(
                'adjusted_size_multiplier', size_multiplier,
            )

        # ---- 3c. PORTFOLIO ALLOCATOR (Ticket 35) ----
        # Capital-constrained, dependency-aware trade arbiter.
        # Only active when a shared portfolio_allocator is provided.
        mark_price = float(bar.get('close', 0))
        side = 'buy' if signal.direction > 0 else 'sell'

        if self.portfolio_allocator is not None:
            dep = result['layers'].get('dependency', {}).get('dependency', {})
            candidate = {
                'symbol': self.symbol,
                'direction': 1 if signal.direction > 0 else -1,
                'confidence': float(signal.conviction),
                'expected_edge_bps': float(signal.conviction) * 40.0,
                'size_hint': size_multiplier,
                'quality_bucket': fq.get('quality_bucket', 'medium'),
                'entry_price': mark_price,
                'dependency': dep,
            }
            pf_ctx = {
                'total_equity': self.portfolio.available_balance + self.portfolio.total_exposure,
                'available_capital': self.portfolio.available_balance,
                'exposure_by_asset': {
                    sym: p.quantity * p.avg_entry_price
                    for sym, p in self.portfolio.positions.items()
                    if p.quantity > 0
                },
                'total_exposure': self.portfolio.total_exposure,
                'open_position_count': sum(
                    1 for p in self.portfolio.positions.values()
                    if p.quantity > 0
                ),
            }
            alloc_results = self.portfolio_allocator.allocate([candidate], pf_ctx)
            if alloc_results:
                alloc = alloc_results[0]
                result['layers']['allocator'] = alloc
                if alloc['decision'] in ('REJECT', 'DEFER'):
                    result['action'] = 'BLOCKED_ALLOCATOR'
                    self._periodic_persist()
                    self._update_monitoring(result)
                    return result
                size_multiplier = alloc['allocated_size'] * mark_price / max(
                    self._compute_quantity(mark_price, 1.0) * mark_price, 1e-12,
                )

        # ---- 3d. EXECUTION MONITOR (Ticket 46) ----
        # Applies adaptive risk_multiplier based on real-time execution
        # health (fill rate, fee level). 0.0 = critical (block trade),
        # 0.5 = degraded, 1.0 = normal. Does not alter signals.
        if self.execution_monitor is not None:
            exec_mult = self.execution_monitor.get_multiplier(self.symbol)
            result['layers']['execution_monitor'] = {
                'risk_multiplier': exec_mult,
                'health_score': self.execution_monitor.get_health(self.symbol),
                'flags': self.execution_monitor.get_flags(self.symbol),
            }
            if exec_mult <= 0.0:
                result['action'] = 'BLOCKED_EXECUTION_CRITICAL'
                self._periodic_persist()
                self._update_monitoring(result)
                return result
            size_multiplier *= exec_mult

        # ---- 4. RISK ENGINE (sovereign — can block) ----
        quantity = self._compute_quantity(mark_price, size_multiplier)

        pf_snapshot = {
            'available_balance': self.portfolio.available_balance,
            'total_exposure': self.portfolio.total_exposure,
            'daily_realized_pnl': 0.0,  # TODO: track per-day
            'weekly_realized_pnl': 0.0,
            'max_drawdown_from_peak': 0.0,
            'open_position_count': sum(
                1 for p in self.portfolio.positions.values()
                if p.quantity > 0
            ),
        }
        risk_result = self.risk_engine.validate_trade(
            symbol=self.symbol, side=side,
            quantity=quantity, price=mark_price,
            portfolio=pf_snapshot,
        )
        result['layers']['risk'] = risk_result

        if not risk_result['allowed']:
            result['action'] = 'BLOCKED_RISK'
            self._periodic_persist()
            self._update_monitoring(result)
            return result

        # ---- 5. OMS (submit order — single source of state) ----
        self._order_counter += 1
        client_id = f'{self.symbol}-{self._order_counter}'
        limit_price = mark_price * (0.999 if side == 'buy' else 1.001)

        oid = self.oms.submit_order(
            client_order_id=client_id,
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            price=limit_price,
        )
        result['layers']['oms'] = {
            'order_id': oid,
            'client_order_id': client_id,
            'side': side,
            'quantity': quantity,
            'limit_price': limit_price,
        }
        result['action'] = 'ORDER_SUBMITTED'
        if self.execution_monitor is not None:
            self.execution_monitor.record_order(self.symbol, placed=True)

        # ---- 6-10 happen on fill (broker callback) ----
        # In live: broker.on_bar() checks fills, then we call
        # self.on_fill(oid, fill_qty, fill_price) which updates
        # portfolio + persistence + monitoring.

        self.metrics.record_order_attempt(filled=False)  # pending
        self._periodic_persist()
        self._update_monitoring(result)
        return result

    # ------------------------------------------------------------------
    def on_fill(self, order_id: int, fill_qty: float,
                fill_price: float, fee: float = 0.0) -> None:
        """Called when broker confirms a fill. Updates OMS → Portfolio
        → Persistence → Monitoring in sequence."""
        # 7. OMS fill
        self.oms.handle_fill(order_id, fill_qty, fill_price)
        order = self.oms.get_order(order_id)

        if self.execution_monitor is not None:
            self.execution_monitor.record_fill(
                self.symbol, filled=True, fee=fee,
            )

        # 8. PORTFOLIO update
        if order is not None:
            self.portfolio.on_fill(
                symbol=order.symbol,
                side=order.side,
                qty=fill_qty,
                price=fill_price,
                fee=fee,
            )

        # 9. PERSISTENCE
        self.state_store.save_oms(self.oms)
        self.state_store.save_portfolio(self.portfolio)

        # 10. MONITORING
        self.metrics.record_order_attempt(filled=True)
        self.metrics.record_trade(
            pnl=0.0,  # PnL computed on close, not on fill
            side=order.side if order else '',
            filled=True,
        )
        equity = self.portfolio.equity_at({
            self.symbol: fill_price,
        })
        self.metrics.update_equity(equity)

        log.info(
            'FILL oid=%d %s %.6f @ %.2f | equity=%.2f',
            order_id, order.side if order else '?',
            fill_qty, fill_price, equity,
        )

    # ------------------------------------------------------------------
    def on_timeout(self, order_id: int) -> None:
        """Called when broker times out an order (missed trade)."""
        self.oms.cancel_order(order_id, reason='TIMED_OUT')
        self.metrics.record_order_attempt(filled=False)
        self.state_store.save_oms(self.oms)
        if self.execution_monitor is not None:
            self.execution_monitor.record_fill(self.symbol, filled=False)

    # ------------------------------------------------------------------
    def on_exchange_update(self, update: Dict[str, Any]) -> None:
        """Process an exchange execution report (fill / reject / cancel).

        This is the SINGLE entry point for exchange → system state.
        OMS → Portfolio → Persistence → Monitoring in sequence.
        """
        oid = int(update.get('order_id', 0))
        event_type = str(update.get('type', ''))

        if event_type == 'FILL':
            self.on_fill(
                order_id=oid,
                fill_qty=float(update.get('fill_qty', 0)),
                fill_price=float(update.get('fill_price', 0)),
                fee=float(update.get('fee', 0)),
            )
        elif event_type == 'REJECT':
            self.oms.handle_reject(oid, reason=str(update.get('reason', '')))
            self.state_store.save_oms(self.oms)
        elif event_type == 'CANCEL':
            self.oms.cancel_order(oid, reason=str(update.get('reason', '')))
            self.state_store.save_oms(self.oms)
        elif event_type == 'TIMEOUT':
            self.on_timeout(oid)
        else:
            log.warning('unknown exchange update type: %s', event_type)

    # ------------------------------------------------------------------
    def heartbeat(self) -> Dict[str, Any]:
        """Periodic system health check. Call every N seconds.

        Responsibilities :
        - check feed health
        - persist runtime state
        - trigger alerts if needed
        - return status snapshot for logging / dashboard

        This is NOT a bar event — it runs on a timer independent of
        market data. If the feed is stale, heartbeat still runs.
        """
        # Persist current state (regardless of trading activity).
        self.state_store.save_oms(self.oms)
        self.state_store.save_portfolio(self.portfolio)

        # Check alerts.
        alerts = self.alerts.check(
            ws_connected=self.feed_health.is_healthy(),
            equity=self.portfolio.available_balance,
        )

        # Activate kill switch on critical alerts.
        for a in alerts:
            if a.get('severity') == 'critical':
                self.risk_engine.activate_kill_switch(
                    reason=f'heartbeat alert: {a["message"]}'
                )
                log.critical('KILL SWITCH via heartbeat: %s', a['message'])

        status = {
            'feed_healthy': self.feed_health.is_healthy(),
            'feed_status': self.feed_health.status(),
            'risk_killed': self.risk_engine.is_killed,
            'metrics': self.metrics.snapshot(),
            'alerts': alerts,
            'open_orders': sum(
                1 for o in self.oms._orders.values()
                if o.status == 'SUBMITTED'
            ),
            'open_positions': sum(
                1 for p in self.portfolio.positions.values()
                if p.quantity > 0
            ),
        }
        log.info('heartbeat: %s', {
            k: v for k, v in status.items() if k != 'metrics'
        })
        return status

    # ------------------------------------------------------------------
    # Jesse agent calls — all 4, right here, visible.
    # ------------------------------------------------------------------
    def _call_jesse_agents(
        self, bar: Dict[str, Any], ts_str: str,
    ) -> Dict[str, FractalReport]:
        """Call ALL 4 Jesse agents and return their FractalReports.

        This is THE ONLY place where Jesse agents are called at runtime.
        If it's not here, it doesn't exist.
        """
        reports: Dict[str, FractalReport] = {}

        # Context (1D) — uses the decider's 1d buffer tail.
        try:
            df_1d = self.decider.stack.buf_1d._as_dataframe()
            if len(df_1d) >= 20:
                reports['context'] = self.context_agent.report(
                    df_1d, asset=self.symbol, timestamp=ts_str,
                )
        except Exception:
            pass

        # Regime (4H) — uses the decider's 4h buffer tail.
        try:
            df_4h = self.decider.stack.buf_4h._as_dataframe()
            if len(df_4h) >= 30:
                reports['regime'] = self.regime_agent.report(
                    df_4h, asset=self.symbol, timestamp=ts_str,
                )
        except Exception:
            pass

        # Setup (1H) — uses the decider's 1h buffer tail.
        try:
            df_1h = self.decider.stack.buf_1h._as_dataframe()
            if len(df_1h) >= 30:
                reports['setup'] = self.setup_agent.report(
                    df_1h, asset=self.symbol, timestamp=ts_str,
                )
        except Exception:
            pass

        # Entry (15M) — uses the decider's 15m buffer tail.
        try:
            df_15m = self.decider.stack.buf_15m._as_dataframe()
            if len(df_15m) >= 50:
                reports['entry'] = self.entry_agent.report(
                    df_15m, asset=self.symbol, timestamp=ts_str,
                )
        except Exception:
            pass

        # Fill missing agents with neutral defaults.
        for agent, tf in [('context', '1d'), ('regime', '4h'),
                          ('setup', '1h'), ('entry', '15m')]:
            if agent not in reports:
                reports[agent] = FractalReport(
                    asset=self.symbol, agent=agent, timeframe=tf,
                    state='unavailable', score=0.5, passed=False,
                    block_reasons=['agent unavailable'],
                    timestamp=ts_str,
                )

        return reports

    # ------------------------------------------------------------------
    def _compute_quantity(self, mark: float, size_mult: float) -> float:
        risk_pct = 0.02
        base_qty = (self.portfolio.available_balance * risk_pct) / max(mark, 1.0)
        return base_qty * size_mult

    def _periodic_persist(self) -> None:
        now = time.time()
        if now - self._last_persist_time > 60:
            self.state_store.save_oms(self.oms)
            self.state_store.save_portfolio(self.portfolio)
            self._last_persist_time = now

    def _update_monitoring(self, result: Dict[str, Any]) -> None:
        alerts = self.alerts.check(
            ws_connected=self.feed_health.is_healthy(),
            equity=self.portfolio.available_balance,
        )
        if alerts:
            for a in alerts:
                log.warning('ALERT: %s', a)
        if self.audit_store is not None:
            from src.live.audit_trail import AssetAuditEvent
            ev = AssetAuditEvent(
                symbol=self.symbol,
                timestamp=result.get('timestamp', ''),
                action=result.get('action', ''),
                layers=result.get('layers', {}),
            )
            self.audit_store.append_asset_event(ev)

    # ------------------------------------------------------------------
    def recover(self) -> None:
        """Reload state from disk after a crash."""
        self.state_store.load_oms(self.oms)
        self.state_store.load_portfolio(self.portfolio)
        log.info(
            'recovered: %d orders, %d positions, balance=%.2f',
            len(self.oms._orders),
            len(self.portfolio.positions),
            self.portfolio.available_balance,
        )

    def shutdown(self) -> None:
        """Clean shutdown — persist everything."""
        self.state_store.save_oms(self.oms)
        self.state_store.save_portfolio(self.portfolio)
        self.state_store.save_event_log(self.oms)
        log.info('shutdown complete — state persisted')
