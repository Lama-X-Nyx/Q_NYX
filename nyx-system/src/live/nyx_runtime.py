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
    ) -> None:
        self.symbol = symbol

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

        # ---- 4. RISK ENGINE (sovereign — can block) ----
        mark_price = float(bar.get('close', 0))
        quantity = self._compute_quantity(mark_price, size_multiplier)
        side = 'buy' if signal.direction > 0 else 'sell'

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
