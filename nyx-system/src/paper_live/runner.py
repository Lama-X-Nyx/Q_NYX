"""
PaperLiveRunner — wires every paper-live component into one object.

Execution semantics (honest):
  - Strategy emits `order_intent = {side, qty, limit_price, mark_price}` with
    every BUY / SELL decision.
  - Runner places the intent via `PostOnlyPaperBroker.place_post_only()`.
  - If the intent crosses the spread at placement → REJECTED → missed log.
  - If it posts but does not fill within `max_wait_bars` → TIMED_OUT → missed log.
  - Only FILLED orders count as actual trades.
  - Every REJECTED/TIMED_OUT emits an `order_unfilled` event (Telegram+Discord).

Every bar also:
  - validates the bar (DataValidator)
  - advances posted orders (broker.on_bar)
  - runs strategy.decide
  - persists the decision (PersistentDecisionLogger)
  - saves state (StateManager)
  - ticks heartbeat
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from .data_validator import DataValidator, DataValidationError
from .decision_logger import PersistentDecisionLogger
from .discord_alerter import DiscordAlerter
from .event_alerter import EventAlerter
from .heartbeat import Heartbeat
from .missed_trade_logger import MissedTradeLogger
from .multi_alerter import MultiAlerter
from .post_only_broker import OrderState, PostOnlyPaperBroker
from .state_manager import StateManager
from .telegram_alerter import TelegramAlerter


log = logging.getLogger(__name__)


class Strategy(Protocol):
    def decide(self, pair: str, bar: dict) -> dict: ...


class PaperLiveRunner:
    """End-to-end paper-live orchestrator with honest post-only execution."""

    def __init__(
        self,
        strategy: Strategy,
        db_path: Union[str, Path],
        state_path: Union[str, Path],
        heartbeat_path: Union[str, Path],
        missed_db_path: Optional[Union[str, Path]] = None,
        telegram_token: Optional[str] = None,
        telegram_chat: Optional[str] = None,
        discord_webhook: Optional[str] = None,
        model_version: str = 'v0.3.2',
        max_wait_bars: int = 3,
    ):
        self.strategy = strategy
        self.model_version = model_version
        self.max_wait_bars = max_wait_bars

        self.validators: Dict[str, DataValidator] = {}
        self.broker = PostOnlyPaperBroker(max_wait_bars=max_wait_bars)
        self.decision_logger = PersistentDecisionLogger(db_path)
        self.state_manager = StateManager(state_path)
        self.heartbeat = Heartbeat(heartbeat_path)

        # Missed-trade log defaults next to the decisions DB.
        if missed_db_path is None:
            missed_db_path = Path(db_path).with_name("missed_trades.db")
        self.missed_logger = MissedTradeLogger(missed_db_path)

        # Alerter fan-out (Telegram + Discord).
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

        # Track which oids we have already recorded as missed so we never
        # double-log or double-alert.
        self._logged_missed_oids: set = set()

        self.bars_processed = 0

    # ------------------------------------------------------------------
    def _load_warm_state(self) -> None:
        try:
            st = self.state_manager.load()
        except Exception as e:
            log.warning("warm restart failed: %s", e)
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
    def _handle_missed(self, order) -> None:
        """Persist + alert once for any REJECTED / TIMED_OUT order."""
        if order.oid in self._logged_missed_oids:
            return
        if order.state not in (OrderState.REJECTED, OrderState.TIMED_OUT):
            return

        self.missed_logger.log({
            'oid': order.oid,
            'pair': order.pair,
            'side': order.side,
            'qty': order.qty,
            'limit_price': order.limit_price,
            'mark_price': order.mark_price_at_placement,
            'placed_at': order.placed_at,
            'final_state': order.state.value,
            'reason': order.reject_reason or order.timeout_reason or '',
            'bars_waited': order.bars_waited,
            'timed_out_at': order.timed_out_at,
        })

        self.events.order_unfilled(
            oid=order.oid,
            pair=order.pair,
            side=order.side,
            price=order.limit_price,
            waited_bars=order.bars_waited,
        )
        self._logged_missed_oids.add(order.oid)

    def _sweep_missed(self) -> None:
        """Check every REJECTED/TIMED_OUT order and record if new."""
        for order in self.broker.all_orders():
            self._handle_missed(order)

    # ------------------------------------------------------------------
    def on_bar(self, pair: str, bar: dict) -> None:
        """Single end-to-end bar step."""
        v = self._validator_for(pair)
        v.validate(bar)

        ts = bar['timestamp']

        # 1. Advance posted orders first → may produce FILLED or TIMED_OUT.
        self.broker.on_bar(pair, bar, bar_ts=ts)
        self._sweep_missed()

        # 2. Strategy decides.
        try:
            decision = self.strategy.decide(pair, bar)
        except Exception as e:
            log.exception("strategy.decide raised")
            self.events.api_error(f"strategy crash: {e}")
            raise

        # 3. If the decision has an order intent, place it post-only.
        intent = decision.get('order_intent')
        if intent and decision.get('action') in ('BUY', 'SELL'):
            oid = self.broker.place_post_only(
                pair=pair,
                side=intent['side'],
                qty=float(intent['qty']),
                limit_price=float(intent['limit_price']),
                mark_price=float(intent['mark_price']),
                placed_at=ts,
                max_wait_bars=int(intent.get('max_wait_bars', self.max_wait_bars)),
            )
            # Immediate post-only rejection should be observed right away.
            self._handle_missed(self.broker.get(oid))

        # 4. Log the decision row.
        ar = decision.get('agent_results', {})
        row = {
            'timestamp': ts,
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

        # 5. Trade-decision event for BUY/SELL.
        action = row['action']
        if action in ('BUY', 'SELL'):
            self.events.trade_decision(
                pair=pair,
                action=action,
                price=float(row['price']),
                score=float(row['orch_score'] or 0.0),
                size=float(row['trade_size'] or 0.0),
            )

        # 6. State snapshot.
        self.last_bar_ts[pair] = ts
        self.state_manager.save({
            'schema_version': 1,
            'capital': self.capital,
            'equity': self.equity,
            'positions': self.positions,
            'last_bar_ts': self.last_bar_ts,
            'model_version': self.model_version,
        })

        # 7. Heartbeat.
        self.bars_processed += 1
        self.heartbeat.tick(context={
            'bars_processed': self.bars_processed,
            'last_pair': pair,
            'last_ts': ts,
            'broker_miss_rate': self.broker.miss_rate(),
        })

    # ------------------------------------------------------------------
    # Manual event helpers (called from main.py / exchange adapter)
    # ------------------------------------------------------------------
    def notify_restart(self) -> None:
        last = "never"
        if self.last_bar_ts:
            last = next(iter(sorted(self.last_bar_ts.values(), reverse=True)))
        self.events.restart(version=self.model_version, last_ts=last)

    def notify_service_down(self, reason: str) -> None:
        self.events.service_down(reason)

    def notify_reconnect_exchange(self, reason: str) -> None:
        self.events.reconnect_exchange(reason)

    def notify_api_error(self, reason: str) -> None:
        self.events.api_error(reason)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        try:
            self.decision_logger.close()
        except Exception:
            pass
        try:
            self.missed_logger.close()
        except Exception:
            pass


def _to_json(x: Any) -> str:
    import json
    try:
        return json.dumps(x, default=str)
    except Exception:
        return '{}'
