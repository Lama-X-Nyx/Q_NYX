"""
PaperLiveRunner — wires every paper-live component into one object.

Flow per bar:
  1. DataValidator.validate(bar)          — reject corrupt / stale input
  2. broker.on_bar(pair, bar)              — advance open orders
  3. strategy.decide(pair, bar)            — produce action + agent results
  4. DecisionLogger.log(...)               — durable SQLite append
  5. StateManager.save(...)                — atomic state snapshot
  6. Heartbeat.tick(...)                   — liveness
  7. events.<semantic event>               — Telegram + Discord fan-out

Any unhandled exception in strategy.decide produces an `api_error` event
and is re-raised so upstream can restart the process.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from .broker import MakerFirstBroker
from .data_validator import DataValidator, DataValidationError
from .decision_logger import PersistentDecisionLogger
from .discord_alerter import DiscordAlerter
from .event_alerter import EventAlerter
from .heartbeat import Heartbeat
from .multi_alerter import MultiAlerter
from .state_manager import StateManager
from .telegram_alerter import TelegramAlerter


log = logging.getLogger(__name__)


class Strategy(Protocol):
    def decide(self, pair: str, bar: dict) -> dict: ...


class PaperLiveRunner:
    """End-to-end paper-live orchestrator."""

    def __init__(
        self,
        strategy: Strategy,
        db_path: Union[str, Path],
        state_path: Union[str, Path],
        heartbeat_path: Union[str, Path],
        telegram_token: Optional[str] = None,
        telegram_chat: Optional[str] = None,
        discord_webhook: Optional[str] = None,
        model_version: str = 'v0.3.2',
        max_wait_bars: int = 3,
    ):
        self.strategy = strategy
        self.model_version = model_version

        self.validators: Dict[str, DataValidator] = {}
        self.broker = MakerFirstBroker(max_wait_bars=max_wait_bars)
        self.decision_logger = PersistentDecisionLogger(db_path)
        self.state_manager = StateManager(state_path)
        self.heartbeat = Heartbeat(heartbeat_path)

        # Build alerter fan-out (Telegram + Discord, either may be disabled).
        telegram = TelegramAlerter(
            token=telegram_token, chat_id=telegram_chat, cooldown_s=60.0,
        )
        discord = DiscordAlerter(
            webhook_url=discord_webhook, cooldown_s=60.0,
        )
        self.alerter = MultiAlerter([telegram, discord])
        self.events = EventAlerter(self.alerter, cooldown_s=60.0)

        # Warm-restart state.
        self.last_bar_ts: Dict[str, str] = {}
        self.capital: float = 10_000.0
        self.equity: float = 10_000.0
        self.positions: Dict[str, dict] = {}
        self._load_warm_state()

        # Track orders already "alerted as unfilled" so we don't spam.
        self._alerted_taker_oids: set = set()

        self.bars_processed = 0

    # ------------------------------------------------------------------
    def _load_warm_state(self) -> None:
        try:
            st = self.state_manager.load()
        except Exception as e:
            log.warning("warm restart failed: %s", e)
            # events not built yet? At this point yes, constructor built it first.
            try:
                self.events.api_error(f"warm restart failed: {e}")
            except Exception:
                pass
            return
        if not st:
            return
        self.last_bar_ts = dict(st.get('last_bar_ts', {}))
        self.capital = float(st.get('capital', self.capital))
        self.equity = float(st.get('equity', self.equity))
        self.positions = dict(st.get('positions', {}))

    # ------------------------------------------------------------------
    def _validator_for(self, pair: str) -> DataValidator:
        if pair not in self.validators:
            v = DataValidator()
            last = self.last_bar_ts.get(pair)
            if last is not None:
                v._last_ts = last
            self.validators[pair] = v
        return self.validators[pair]

    # ------------------------------------------------------------------
    def _alert_unfilled_orders(self, pair: str) -> None:
        """Emit `order_unfilled` for any order that just became a taker fill
        (i.e. the limit aged out and the broker crossed the spread)."""
        for oid, order in self.broker._orders.items():
            if oid in self._alerted_taker_oids:
                continue
            if not order.fills:
                continue
            last = order.fills[-1]
            if last.get('role') == 'taker' and order.bars_waited >= self.broker.max_wait_bars:
                self.events.order_unfilled(
                    oid=oid,
                    pair=pair,
                    side=order.side,
                    price=order.limit_price,
                    waited_bars=order.bars_waited,
                )
                self._alerted_taker_oids.add(oid)

    # ------------------------------------------------------------------
    def on_bar(self, pair: str, bar: dict) -> None:
        """Single end-to-end bar step."""
        v = self._validator_for(pair)
        v.validate(bar)

        self.broker.on_bar(pair, bar)
        self._alert_unfilled_orders(pair)

        try:
            decision = self.strategy.decide(pair, bar)
        except Exception as e:
            log.exception("strategy.decide raised")
            self.events.api_error(f"strategy crash: {e}")
            raise

        ar = decision.get('agent_results', {})
        row = {
            'timestamp': bar['timestamp'],
            'pair': pair,
            'action': decision.get('action', 'WAIT'),
            'orch_score': decision.get('orch_score'),
            'size_factor': decision.get('size_factor'),
            'blocked_by': ','.join(decision.get('blocked_by', []) or []),
            'price': float(bar['close']),
            'atr': float(bar.get('atr', 0.0) or 0.0),
            'volume_ratio': float(bar.get('volume_ratio', 0.0) or 0.0),
            'model_version': self.model_version,
            'features_json': _to_json(decision.get('features', {})),
        }
        for name in ('context', 'regime', 'setup', 'entry'):
            a = ar.get(name, {})
            row[f'{name}_state']  = a.get('state', '')
            row[f'{name}_score']  = a.get('score', 0.0)
            row[f'{name}_passed'] = bool(a.get('passed', False))
        row['trade_entry_price'] = decision.get('trade_entry_price')
        row['trade_stop']        = decision.get('trade_stop')
        row['trade_tp']          = decision.get('trade_tp')
        row['trade_direction']   = int(decision.get('trade_direction', 0))
        row['trade_size']        = float(decision.get('trade_size', 0.0))
        self.decision_logger.log(row)

        # Semantic trade-decision event (BUY / SELL only).
        action = row['action']
        if action in ('BUY', 'SELL'):
            self.events.trade_decision(
                pair=pair,
                action=action,
                price=float(row['price']),
                score=float(row['orch_score'] or 0.0),
                size=float(row['trade_size'] or 0.0),
            )

        # State snapshot.
        self.last_bar_ts[pair] = bar['timestamp']
        self.state_manager.save({
            'schema_version': 1,
            'capital': self.capital,
            'equity': self.equity,
            'positions': self.positions,
            'last_bar_ts': self.last_bar_ts,
            'model_version': self.model_version,
        })

        # Heartbeat.
        self.bars_processed += 1
        self.heartbeat.tick(context={
            'bars_processed': self.bars_processed,
            'last_pair': pair,
            'last_ts': bar['timestamp'],
        })

    # ------------------------------------------------------------------
    # Manual event helpers (called from main.py / exchange adapter)
    # ------------------------------------------------------------------
    def notify_restart(self) -> None:
        """Call once at startup (main.py) so operators see we're back."""
        last = "never"
        if self.last_bar_ts:
            last = next(iter(sorted(self.last_bar_ts.values(), reverse=True)))
        self.events.restart(version=self.model_version, last_ts=last)

    def notify_service_down(self, reason: str) -> None:
        """Called by a watchdog when heartbeat goes stale."""
        self.events.service_down(reason)

    def notify_reconnect_exchange(self, reason: str) -> None:
        """Called by the exchange adapter after a successful reconnect."""
        self.events.reconnect_exchange(reason)

    def notify_api_error(self, reason: str) -> None:
        """Called by the exchange adapter on 4xx/5xx responses."""
        self.events.api_error(reason)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        try:
            self.decision_logger.close()
        except Exception:
            pass


def _to_json(x: Any) -> str:
    import json
    try:
        return json.dumps(x, default=str)
    except Exception:
        return '{}'
