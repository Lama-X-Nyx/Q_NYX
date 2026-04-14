"""
HubSpokeRunner — hub-and-spoke orchestrator.

Wires the architecture:

  bars (per symbol)
     │
     ▼
  SignalPod(symbol).on_bar()        ← per-asset signal producer
     │                               (returns a Signal each bar)
     ▼
  PortfolioAllocator.decide()       ← hub: ranks + risk caps + clusters
     │
     ▼
  PostOnlyPaperBroker               ← shared execution, per-asset broker
  + MissedTradeLogger               ← same stores per asset (via
  + PersistentDecisionLogger         sub-PaperLiveRunner instances)
  + EventAlerter / MultiAlerter

Pods only emit Signals. They do NOT place orders directly. This
enforces the "central portfolio allocator" layer the architecture
requires.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Union

from .portfolio_allocator import ApprovedTrade, PortfolioAllocator
from .signal import Signal
from ..paper_live.event_alerter import EventAlerter
from ..paper_live.multi_alerter import MultiAlerter
from ..paper_live.multi_pair_runner import UnknownPairError
from ..paper_live.runner import PaperLiveRunner


log = logging.getLogger(__name__)


class SignalPod(Protocol):
    symbol: str
    def on_bar(self, bar: dict) -> Signal: ...


class HubSpokeRunner:
    """Pods produce Signals; allocator decides; shared execution places."""

    def __init__(
        self,
        pods: Iterable[SignalPod],
        storage_root: Union[str, Path],
        shared_alerter=None,
        max_total_risk: float = 0.04,
        max_asset_risk: float = 0.015,
        max_cluster_risk: Optional[Dict[str, float]] = None,
        max_open_positions: int = 2,
    ):
        self._pods: Dict[str, SignalPod] = {p.symbol: p for p in pods}
        self._allocator = PortfolioAllocator(
            max_total_risk=max_total_risk,
            max_asset_risk=max_asset_risk,
            max_cluster_risk=max_cluster_risk,
            max_open_positions=max_open_positions,
        )

        # Build one PaperLiveRunner per symbol for the execution layer.
        root = Path(storage_root)
        self._runners: Dict[str, PaperLiveRunner] = {}
        for symbol in self._pods.keys():
            pair_dir = root / symbol
            pair_dir.mkdir(parents=True, exist_ok=True)
            runner = PaperLiveRunner(
                strategy=_NullStrategy(),   # pod is the brain; runner is the hands
                db_path=pair_dir / 'decisions.db',
                state_path=pair_dir / 'state.json',
                heartbeat_path=pair_dir / 'heartbeat.json',
                missed_db_path=pair_dir / 'missed_trades.db',
                telegram_token=None, telegram_chat=None,
            )
            if shared_alerter is not None:
                runner.events = EventAlerter(shared_alerter, cooldown_s=0.0)
            self._runners[symbol] = runner

        # Track open positions across all assets (for allocator awareness).
        # Each entry: { 'direction', 'risk', 'opened_bar', 'hold_bars' }.
        # `opened_bar` is a global bar counter (see `_global_bar_count`).
        # When `_global_bar_count - opened_bar > hold_bars`, the entry
        # is dropped in `_release_expired_positions()` so the same
        # symbol can take a new trade.
        self._open_positions: Dict[str, dict] = {}
        self._global_bar_count: int = 0

    # ------------------------------------------------------------------
    def pairs(self) -> List[str]:
        return list(self._pods.keys())

    # ------------------------------------------------------------------
    def _collect_signals(
        self,
        bars: Dict[str, dict],
    ) -> List[Signal]:
        signals: List[Signal] = []
        for symbol, bar in bars.items():
            if symbol not in self._pods:
                raise UnknownPairError(symbol)
            try:
                sig = self._pods[symbol].on_bar(bar)
            except Exception as e:
                # Pod crash isolation: alert + continue.
                log.exception("pod crash for %s", symbol)
                runner = self._runners.get(symbol)
                if runner is not None:
                    runner.events.api_error(f"pod crash {symbol}: {e}")
                continue
            if sig is not None:
                signals.append(sig)
            # Always advance the broker + do missed-trade sweep for this
            # symbol, even if signal is FLAT.
            runner = self._runners.get(symbol)
            if runner is not None:
                runner.broker.on_bar(symbol, bar, bar_ts=bar['timestamp'])
                runner._sweep_missed()
                runner.bars_processed += 1
                runner.heartbeat.tick(context={
                    'bars_processed': runner.bars_processed,
                    'last_pair': symbol,
                    'last_ts': bar['timestamp'],
                    'broker_miss_rate': runner.broker.miss_rate(),
                })
        return signals

    # ------------------------------------------------------------------
    def _release_expired_positions(self) -> None:
        """Drop any open position whose age exceeds its hold_bars.

        Called at the START of each on_bars() so the allocator sees an
        accurate view of currently-held positions.
        """
        now = self._global_bar_count
        expired = []
        for sym, info in self._open_positions.items():
            opened = int(info.get('opened_bar', now))
            hold = int(info.get('hold_bars', 50))
            if now - opened > hold:
                expired.append(sym)
        for sym in expired:
            self._open_positions.pop(sym, None)

    # ------------------------------------------------------------------
    def on_bars(self, bars: Dict[str, dict]) -> List[ApprovedTrade]:
        # 0. Age the global clock BEFORE processing this bar, then
        #    release any position whose hold window has expired.
        self._global_bar_count += 1
        self._release_expired_positions()

        # 1. Collect signals from every pod (crash-isolated).
        signals = self._collect_signals(bars)

        # 2. Hub decision: apply risk caps + cluster rules.
        trades = self._allocator.decide(signals, self._open_positions)

        # 3. Place post-only orders for approved trades.
        for t in trades:
            runner = self._runners[t.symbol]
            bar = bars[t.symbol]
            mark = float(bar['close'])
            # Maker-friendly limit: a hair below (buy) / above (sell) mark.
            if t.direction > 0:
                limit = mark * 0.999
                side = 'buy'
            else:
                limit = mark * 1.001
                side = 'sell'

            qty = float(t.final_risk)  # simplified: one unit per risk fraction
            oid = runner.broker.place_post_only(
                pair=t.symbol, side=side, qty=qty,
                limit_price=limit, mark_price=mark,
                placed_at=bar['timestamp'],
                max_wait_bars=runner.max_wait_bars,
            )
            runner._handle_missed(runner.broker.get(oid))

            # Event: always surface a trade_decision.
            runner.events.trade_decision(
                pair=t.symbol, action='BUY' if t.direction > 0 else 'SELL',
                price=mark, score=t.source_signal.score(),
                size=qty,
            )
            # Remember as open, tagged with the bar index at which it
            # opened and its expected hold time. _release_expired_positions
            # will drop it from self._open_positions once expired, freeing
            # the symbol for new signals.
            self._open_positions[t.symbol] = {
                'direction':  t.direction,
                'risk':       t.final_risk,
                'opened_bar': self._global_bar_count,
                'hold_bars':  int(t.source_signal.expected_hold_bars),
            }

        return trades

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        for r in self._runners.values():
            try:
                r.shutdown()
            except Exception:
                pass


# -----------------------------------------------------------------------------
class _NullStrategy:
    """Placeholder — pods are the brain, runner is just the hands."""
    def decide(self, pair, bar):
        return {
            'action': 'WAIT',
            'orch_score': 0.0,
            'size_factor': 0.0,
            'blocked_by': ['pod-driven'],
            'features': {},
            'trade_size': 0.0,
            'agent_results': {
                k: {'state': '', 'score': 0.0, 'passed': False}
                for k in ('context', 'regime', 'setup', 'entry')
            },
        }
