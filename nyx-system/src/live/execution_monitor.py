"""
Real-Time Execution Monitoring & Adaptive Risk Layer — Ticket 46.

Tracks execution quality in real-time and converts it into a
risk_multiplier that NYXRuntime applies to size_multiplier.

Does NOT modify alpha, signals, or model predictions.
Pure post-decision risk control.

Architecture:
  Market feedback → ExecutionMetricsCollector → RiskController
                  → risk_multiplier → NYXRuntime size scaling
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional


class ExecutionMetricsCollector:
    """Per-asset rolling window of fill/fee metrics."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._orders: Dict[str, Deque[Dict]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._fills: Dict[str, Deque[Dict]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def record_order(self, symbol: str, placed: bool = True) -> None:
        if placed:
            self._orders[symbol].append({'placed': True})

    def record_fill(
        self, symbol: str, filled: bool, fee: float = 0.0,
    ) -> None:
        self._fills[symbol].append({'filled': filled, 'fee': fee})

    def snapshot(self, symbol: str) -> Dict[str, float]:
        orders = list(self._orders.get(symbol, []))
        fills = list(self._fills.get(symbol, []))
        n_placed = len(orders)
        n_filled = sum(1 for f in fills if f['filled'])
        n_missed = sum(1 for f in fills if not f['filled'])
        n_fill_events = len(fills)

        fill_rate = n_filled / n_fill_events if n_fill_events else 1.0
        miss_rate = n_missed / n_fill_events if n_fill_events else 0.0
        filled_fees = [f['fee'] for f in fills if f['filled']]
        avg_fee = sum(filled_fees) / len(filled_fees) if filled_fees else 0.0

        return {
            'placed': n_placed,
            'filled': n_filled,
            'missed': n_missed,
            'fill_rate': round(fill_rate, 4),
            'miss_rate': round(miss_rate, 4),
            'avg_fee': round(avg_fee, 4),
        }


def compute_health_score(
    fill_rate: float,
    miss_rate: float,
    avg_fee: float,
    baseline_fee: float,
) -> float:
    """Normalized [0, 1] execution health.

    Formula:
      health = 0.5 × fill_rate + 0.3 × (1 - miss_rate) + 0.2 × fee_factor
      fee_factor = max(0, 1 - (fee_ratio - 1) × 2) where fee_ratio = avg/baseline

    Aligned with T44.1 sensitivities:
    - fill_sens ≈ 0.88 → heaviest weight on fill_rate
    - fee_sens ≈ 0.57 → moderate weight on fee_factor
    - offset_sens ≈ 0.09 → implicit, no direct signal
    """
    if baseline_fee > 0:
        fee_ratio = avg_fee / baseline_fee
        fee_factor = max(0.0, min(1.0, 1.0 - max(0.0, fee_ratio - 1.0) * 2.0))
    else:
        fee_factor = 1.0

    score = (
        0.5 * max(0.0, min(1.0, fill_rate))
        + 0.3 * max(0.0, min(1.0, 1.0 - miss_rate))
        + 0.2 * fee_factor
    )
    return round(max(0.0, min(1.0, score)), 4)


def detect_flags(
    fill_rate: float,
    miss_rate: float,
    avg_fee: float,
    baseline_fee: float,
    health_score: float,
) -> List[str]:
    flags = []
    if health_score >= 0.85:
        flags.append('execution_healthy')
    if 0.35 <= health_score < 0.7:
        flags.append('execution_degraded')
    if health_score < 0.35:
        flags.append('execution_critical')
    if fill_rate < 0.4:
        flags.append('fill_collapse_detected')
    if baseline_fee > 0 and avg_fee >= baseline_fee * 1.5:
        flags.append('fee_spike_detected')
    return flags


class RiskController:
    """Converts health score into risk_multiplier via 3-state machine."""

    def __init__(
        self,
        normal_threshold: float = 0.7,
        critical_threshold: float = 0.35,
        hysteresis_margin: float = 0.05,
        degraded_multiplier: float = 0.5,
    ) -> None:
        self.normal_threshold = normal_threshold
        self.critical_threshold = critical_threshold
        self.hysteresis_margin = hysteresis_margin
        self.degraded_multiplier = degraded_multiplier
        self.current_state = 'normal'

    def classify(self, health_score: float) -> str:
        if health_score >= self.normal_threshold:
            return 'normal'
        if health_score < self.critical_threshold:
            return 'critical'
        return 'degraded'

    def update_state(self, health_score: float) -> str:
        new_state = self.classify(health_score)
        if self.current_state == 'normal':
            if new_state == 'normal':
                pass
            elif health_score < self.normal_threshold - self.hysteresis_margin:
                self.current_state = new_state
        elif self.current_state == 'degraded':
            if new_state == 'critical':
                self.current_state = 'critical'
            elif health_score >= self.normal_threshold + self.hysteresis_margin:
                self.current_state = 'normal'
        elif self.current_state == 'critical':
            if health_score >= self.critical_threshold + self.hysteresis_margin:
                self.current_state = new_state
        return self.current_state

    def get_multiplier(self, health_score: float) -> float:
        state = self.classify(health_score)
        if state == 'normal':
            return 1.0
        if state == 'degraded':
            return self.degraded_multiplier
        return 0.0


class ExecutionMonitor:
    """Top-level orchestrator combining metrics + controller.

    Per-asset and global control. Used by NYXRuntime via
    `get_multiplier(symbol)`.
    """

    def __init__(
        self,
        assets: List[str],
        window_size: int = 100,
        baseline_fee: float = 1.0,
        min_samples: int = 10,
    ) -> None:
        self.assets = assets
        self.baseline_fee = baseline_fee
        self.min_samples = min_samples
        self._collector = ExecutionMetricsCollector(window_size=window_size)
        self._controllers: Dict[str, RiskController] = {
            sym: RiskController() for sym in assets
        }
        self._global_controller = RiskController()

    def record_order(self, symbol: str, placed: bool = True) -> None:
        self._collector.record_order(symbol, placed=placed)

    def record_fill(
        self, symbol: str, filled: bool, fee: float = 0.0,
    ) -> None:
        self._collector.record_fill(symbol, filled=filled, fee=fee)

    def get_health(self, symbol: str) -> float:
        snap = self._collector.snapshot(symbol)
        if snap['filled'] + snap['missed'] < self.min_samples:
            return 1.0
        return compute_health_score(
            fill_rate=snap['fill_rate'],
            miss_rate=snap['miss_rate'],
            avg_fee=snap['avg_fee'],
            baseline_fee=self.baseline_fee,
        )

    def get_multiplier(self, symbol: str) -> float:
        health = self.get_health(symbol)
        ctrl = self._controllers.get(symbol)
        if ctrl is None:
            ctrl = RiskController()
            self._controllers[symbol] = ctrl
        return ctrl.get_multiplier(health)

    def is_systemic_degradation(self, threshold: float = 0.5) -> bool:
        healths = [self.get_health(s) for s in self.assets]
        n_valid = sum(
            1 for s in self.assets
            if self._collector.snapshot(s)['filled'] + self._collector.snapshot(s)['missed']
            >= self.min_samples
        )
        if n_valid < len(self.assets):
            return False
        n_degraded = sum(1 for h in healths if h < threshold)
        return n_degraded >= (len(self.assets) + 1) // 2

    def get_flags(self, symbol: str) -> List[str]:
        snap = self._collector.snapshot(symbol)
        health = self.get_health(symbol)
        return detect_flags(
            fill_rate=snap['fill_rate'],
            miss_rate=snap['miss_rate'],
            avg_fee=snap['avg_fee'],
            baseline_fee=self.baseline_fee,
            health_score=health,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            'per_asset': {
                sym: {
                    **self._collector.snapshot(sym),
                    'health_score': self.get_health(sym),
                    'multiplier': self.get_multiplier(sym),
                    'flags': self.get_flags(sym),
                }
                for sym in self.assets
            },
            'systemic_degradation': self.is_systemic_degradation(),
        }
