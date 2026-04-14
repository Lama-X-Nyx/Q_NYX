"""
MultiPairRunner — run N isolated PaperLiveRunner instances in parallel.

Each pair gets its own storage (decisions DB, missed DB, state, heartbeat)
so a crash or corrupt state in one pair can never poison the others.
Alerts can be shared (single stream tagged with the pair name in every
message) or per-pair if that matches the ops setup.

Failure isolation is part of the contract: `on_bars(batch,
raise_on_error=False)` collects per-pair exceptions and continues.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .event_alerter import EventAlerter
from .runner import PaperLiveRunner


log = logging.getLogger(__name__)


class UnknownPairError(KeyError):
    """Raised when a bar targets a pair not registered in the runner."""


class MultiPairRunner:
    """Container of per-pair PaperLiveRunner instances."""

    def __init__(
        self,
        config: Dict[str, Dict[str, Any]],
        shared_alerter: Optional[Any] = None,
    ):
        """
        config: {'ETHUSDT': {...}, 'XRPUSDT': {...}, ...}
            required:   'strategy', 'storage_dir'
            optional:   'max_wait_bars', 'model_version',
                        'telegram_token', 'telegram_chat', 'discord_webhook'

        shared_alerter: optional underlying alerter (TelegramAlerter,
            DiscordAlerter, MultiAlerter) used by every sub-runner's
            EventAlerter. If None, each sub-runner uses its own
            per-pair alerter (as configured in its cfg dict).
        """
        self._runners: Dict[str, PaperLiveRunner] = {}
        self._shared_alerter = shared_alerter

        for pair, pair_cfg in config.items():
            storage = Path(pair_cfg['storage_dir'])
            storage.mkdir(parents=True, exist_ok=True)

            runner = PaperLiveRunner(
                strategy=pair_cfg['strategy'],
                db_path=storage / 'decisions.db',
                state_path=storage / 'state.json',
                heartbeat_path=storage / 'heartbeat.json',
                missed_db_path=storage / 'missed_trades.db',
                telegram_token=pair_cfg.get('telegram_token'),
                telegram_chat=pair_cfg.get('telegram_chat'),
                discord_webhook=pair_cfg.get('discord_webhook'),
                model_version=pair_cfg.get('model_version', 'v0.3.2'),
                max_wait_bars=int(pair_cfg.get('max_wait_bars', 3)),
            )

            # If a shared alerter was provided, every sub-runner routes its
            # semantic events through it. Pair names are already baked into
            # the event messages (e.g. "BUY ETHUSDT @ ...").
            if shared_alerter is not None:
                runner.events = EventAlerter(shared_alerter, cooldown_s=0.0)

            self._runners[pair] = runner

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def pairs(self) -> List[str]:
        return list(self._runners.keys())

    def runner_for(self, pair: str) -> PaperLiveRunner:
        if pair not in self._runners:
            raise UnknownPairError(pair)
        return self._runners[pair]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def on_bar(self, pair: str, bar: dict) -> None:
        """Dispatch a single bar to its pair's runner."""
        self.runner_for(pair).on_bar(pair, bar)

    def on_bars(
        self,
        bars: Dict[str, dict],
        raise_on_error: bool = True,
    ) -> Dict[str, Exception]:
        """Dispatch one bar per pair. Returns {pair: exception} for failures.

        With `raise_on_error=True` (default), the first exception propagates.
        With `raise_on_error=False`, every pair is attempted; a crash in one
        pair does NOT prevent the other pairs' bars from being processed.
        """
        errors: Dict[str, Exception] = {}
        for pair, bar in bars.items():
            try:
                self.on_bar(pair, bar)
            except Exception as e:
                if raise_on_error:
                    raise
                log.exception("on_bar failed for pair=%s", pair)
                errors[pair] = e
        return errors

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    def miss_rate_all(self) -> Dict[str, float]:
        """{pair: post-only miss rate} — TIMED_OUT / (TIMED_OUT + FILLED)."""
        return {pair: r.broker.miss_rate() for pair, r in self._runners.items()}

    def heartbeat_status(self, max_age_s: float = 300.0) -> Dict[str, bool]:
        """{pair: True if its heartbeat is younger than max_age_s}."""
        return {
            pair: r.heartbeat.is_alive(max_age_s)
            for pair, r in self._runners.items()
        }

    def bars_processed_all(self) -> Dict[str, int]:
        return {pair: r.bars_processed for pair, r in self._runners.items()}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        for r in self._runners.values():
            try:
                r.shutdown()
            except Exception:
                log.exception("sub-runner shutdown failed")

    # ------------------------------------------------------------------
    # Operator helpers (broadcast)
    # ------------------------------------------------------------------
    def notify_restart_all(self) -> None:
        for r in self._runners.values():
            try:
                r.notify_restart()
            except Exception:
                pass

    def notify_service_down(self, pair: str, reason: str) -> None:
        self.runner_for(pair).notify_service_down(reason)

    def notify_reconnect_exchange(self, reason: str) -> None:
        """Fan-out a single reconnect event to every pair (one exchange,
        many streams)."""
        for r in self._runners.values():
            r.notify_reconnect_exchange(reason)
