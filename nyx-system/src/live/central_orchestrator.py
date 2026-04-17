"""
Central Orchestrator — Ticket 36.

Converts the system from asynchronous per-runtime allocation to
synchronous portfolio-level allocation cycles.

Per-asset runtimes produce CandidateDecision only.
CentralOrchestrator collects candidates, applies dependency layer +
portfolio allocator on the FULL set, then returns approved trades
for dispatch to per-asset Risk → OMS.

Alpha (GBM/Jesse) untouched. Dependency layer untouched.
Portfolio allocator untouched.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CandidateDecision:
    """A per-asset trade candidate emitted by NYXRuntime."""
    symbol: str
    timestamp: str
    direction: int
    confidence: float
    expected_edge_bps: float
    size_hint: float
    quality_bucket: str
    entry_price: float
    jesse_reports: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_bucket(self) -> str:
        return self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'direction': self.direction,
            'confidence': self.confidence,
            'expected_edge_bps': self.expected_edge_bps,
            'size_hint': self.size_hint,
            'quality_bucket': self.quality_bucket,
            'entry_price': self.entry_price,
            'dependency': self.metadata.get('dependency', {}),
        }


class CandidateStore:
    """Thread-safe store for candidate decisions, indexed by time bucket."""

    def __init__(
        self,
        expected_symbols: List[str],
        tolerance_seconds: float = 60.0,
    ) -> None:
        self.expected_symbols = expected_symbols
        self.tolerance_seconds = tolerance_seconds
        self._lock = threading.Lock()
        self._buckets: Dict[str, Dict[str, CandidateDecision]] = defaultdict(dict)
        self._bucket_first_arrival: Dict[str, float] = {}

    def add(self, candidate: CandidateDecision) -> None:
        with self._lock:
            bucket_key = candidate.time_bucket
            self._buckets[bucket_key][candidate.symbol] = candidate
            if bucket_key not in self._bucket_first_arrival:
                self._bucket_first_arrival[bucket_key] = time.monotonic()

    def get_bucket(self, time_bucket: str) -> List[CandidateDecision]:
        with self._lock:
            return list(self._buckets.get(time_bucket, {}).values())

    def is_bucket_ready(self, time_bucket: str) -> bool:
        with self._lock:
            bucket = self._buckets.get(time_bucket, {})
            if len(bucket) >= len(self.expected_symbols):
                return True
            first = self._bucket_first_arrival.get(time_bucket)
            if first is not None and len(bucket) > 0:
                elapsed = time.monotonic() - first
                if elapsed >= self.tolerance_seconds:
                    return True
            return False

    def consume(self, time_bucket: str) -> List[CandidateDecision]:
        with self._lock:
            candidates = list(self._buckets.get(time_bucket, {}).values())
            self._buckets.pop(time_bucket, None)
            self._bucket_first_arrival.pop(time_bucket, None)
            return candidates


class CentralOrchestrator:
    """Runs synchronized allocation cycles across all assets."""

    def __init__(
        self,
        symbols: List[str],
        dependency_layer: Optional[Any] = None,
        portfolio_allocator: Optional[Any] = None,
        portfolio_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.symbols = symbols
        self.dependency_layer = dependency_layer

        if portfolio_allocator is None:
            from src.live.portfolio_allocator import PortfolioAllocator
            self.allocator = PortfolioAllocator()
        else:
            self.allocator = portfolio_allocator

        self._default_portfolio_ctx = portfolio_ctx or {
            'total_equity': 10_000.0,
            'available_capital': 10_000.0,
            'exposure_by_asset': {},
            'total_exposure': 0.0,
            'open_position_count': 0,
        }
        self._total_cycles = 0
        self._last_cycle_results: List[Dict[str, Any]] = []

    def run_cycle(
        self,
        candidates: List[CandidateDecision],
        portfolio_ctx: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run one synchronized allocation cycle on the full candidate set.

        1. Convert candidates to allocator format
        2. Apply dependency layer (if present)
        3. Pass full set to portfolio allocator
        4. Return allocation decisions
        """
        if not candidates:
            self._total_cycles += 1
            self._last_cycle_results = []
            return []

        ctx = portfolio_ctx or self._default_portfolio_ctx

        alloc_candidates = []
        for c in candidates:
            dep_data: Dict[str, Any] = {}
            if self.dependency_layer is not None:
                dep_result = self.dependency_layer.evaluate(
                    symbol=c.symbol,
                    direction=c.direction,
                    size_multiplier=c.size_hint,
                )
                dep_data = dep_result.get('dependency', {})

            alloc_candidates.append({
                'symbol': c.symbol,
                'direction': c.direction,
                'confidence': c.confidence,
                'expected_edge_bps': c.expected_edge_bps,
                'size_hint': c.size_hint,
                'quality_bucket': c.quality_bucket,
                'entry_price': c.entry_price,
                'dependency': dep_data,
            })

        results = self.allocator.allocate(alloc_candidates, ctx)

        self._total_cycles += 1
        self._last_cycle_results = results
        return results

    def snapshot(self) -> Dict[str, Any]:
        return {
            'total_cycles': self._total_cycles,
            'last_cycle_results': self._last_cycle_results,
            'symbols': self.symbols,
        }
