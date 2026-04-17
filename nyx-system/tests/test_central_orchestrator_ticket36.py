"""
TDD Tests — Ticket 36 — Central Synchronous Orchestrator.

Converts the system from asynchronous per-runtime allocation to
synchronous portfolio-level allocation cycles.

Per-asset runtimes produce CandidateDecision only.
CentralOrchestrator collects, applies dependency + allocator on
the FULL set, then dispatches approved trades to per-asset
Risk → OMS.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict

import pytest


# =========================================================================
# A1 — CandidateDecision
# =========================================================================
class TestCandidateDecision:

    def test_create_candidate(self):
        from src.live.central_orchestrator import CandidateDecision
        c = CandidateDecision(
            symbol='BTCUSDT',
            timestamp='2023-06-15T12:00:00',
            direction=1,
            confidence=0.85,
            expected_edge_bps=25.0,
            size_hint=1.0,
            quality_bucket='high',
            entry_price=30000.0,
        )
        assert c.symbol == 'BTCUSDT'
        assert c.direction == 1
        assert c.confidence == 0.85

    def test_candidate_to_dict(self):
        from src.live.central_orchestrator import CandidateDecision
        c = CandidateDecision(
            symbol='ETHUSDT',
            timestamp='2023-06-15T12:00:00',
            direction=-1,
            confidence=0.70,
            expected_edge_bps=15.0,
            size_hint=0.8,
            quality_bucket='medium',
            entry_price=2000.0,
        )
        d = c.to_dict()
        assert d['symbol'] == 'ETHUSDT'
        assert d['direction'] == -1
        assert d['confidence'] == 0.70
        assert d['entry_price'] == 2000.0

    def test_candidate_has_time_bucket(self):
        from src.live.central_orchestrator import CandidateDecision
        c = CandidateDecision(
            symbol='BTCUSDT',
            timestamp='2023-06-15T12:15:00',
            direction=1,
            confidence=0.8,
            expected_edge_bps=20.0,
            size_hint=1.0,
            quality_bucket='high',
            entry_price=30000.0,
        )
        assert c.time_bucket == '2023-06-15T12:15:00'


# =========================================================================
# A3 — CandidateStore
# =========================================================================
class TestCandidateStore:

    def test_store_add_candidate(self):
        from src.live.central_orchestrator import CandidateDecision, CandidateStore
        store = CandidateStore(expected_symbols=['BTCUSDT', 'ETHUSDT'])
        c = CandidateDecision(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            direction=1, confidence=0.8, expected_edge_bps=20.0,
            size_hint=1.0, quality_bucket='high', entry_price=30000.0,
        )
        store.add(c)
        assert len(store.get_bucket('2023-06-15T12:00:00')) == 1

    def test_store_multiple_assets_same_bucket(self):
        from src.live.central_orchestrator import CandidateDecision, CandidateStore
        store = CandidateStore(expected_symbols=['BTCUSDT', 'ETHUSDT'])
        store.add(CandidateDecision(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            direction=1, confidence=0.8, expected_edge_bps=20.0,
            size_hint=1.0, quality_bucket='high', entry_price=30000.0,
        ))
        store.add(CandidateDecision(
            symbol='ETHUSDT', timestamp='2023-06-15T12:00:00',
            direction=1, confidence=0.7, expected_edge_bps=15.0,
            size_hint=0.9, quality_bucket='medium', entry_price=2000.0,
        ))
        bucket = store.get_bucket('2023-06-15T12:00:00')
        assert len(bucket) == 2
        symbols = {c.symbol for c in bucket}
        assert symbols == {'BTCUSDT', 'ETHUSDT'}

    def test_store_no_duplicate_per_symbol_per_bucket(self):
        from src.live.central_orchestrator import CandidateDecision, CandidateStore
        store = CandidateStore(expected_symbols=['BTCUSDT'])
        c1 = CandidateDecision(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            direction=1, confidence=0.8, expected_edge_bps=20.0,
            size_hint=1.0, quality_bucket='high', entry_price=30000.0,
        )
        c2 = CandidateDecision(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            direction=-1, confidence=0.6, expected_edge_bps=10.0,
            size_hint=0.5, quality_bucket='low', entry_price=30100.0,
        )
        store.add(c1)
        store.add(c2)
        bucket = store.get_bucket('2023-06-15T12:00:00')
        assert len(bucket) == 1
        assert bucket[0].confidence == 0.6  # latest wins

    def test_store_is_bucket_ready(self):
        from src.live.central_orchestrator import CandidateDecision, CandidateStore
        store = CandidateStore(expected_symbols=['BTCUSDT', 'ETHUSDT'])
        assert not store.is_bucket_ready('2023-06-15T12:00:00')
        store.add(CandidateDecision(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            direction=1, confidence=0.8, expected_edge_bps=20.0,
            size_hint=1.0, quality_bucket='high', entry_price=30000.0,
        ))
        assert not store.is_bucket_ready('2023-06-15T12:00:00')
        store.add(CandidateDecision(
            symbol='ETHUSDT', timestamp='2023-06-15T12:00:00',
            direction=1, confidence=0.7, expected_edge_bps=15.0,
            size_hint=0.9, quality_bucket='medium', entry_price=2000.0,
        ))
        assert store.is_bucket_ready('2023-06-15T12:00:00')

    def test_store_bucket_ready_partial_ok_after_timeout(self):
        from src.live.central_orchestrator import CandidateDecision, CandidateStore
        store = CandidateStore(
            expected_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
            tolerance_seconds=0.1,
        )
        store.add(CandidateDecision(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            direction=1, confidence=0.8, expected_edge_bps=20.0,
            size_hint=1.0, quality_bucket='high', entry_price=30000.0,
        ))
        assert not store.is_bucket_ready('2023-06-15T12:00:00')
        time.sleep(0.15)
        assert store.is_bucket_ready('2023-06-15T12:00:00')

    def test_store_consume_clears_bucket(self):
        from src.live.central_orchestrator import CandidateDecision, CandidateStore
        store = CandidateStore(expected_symbols=['BTCUSDT'])
        store.add(CandidateDecision(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            direction=1, confidence=0.8, expected_edge_bps=20.0,
            size_hint=1.0, quality_bucket='high', entry_price=30000.0,
        ))
        consumed = store.consume('2023-06-15T12:00:00')
        assert len(consumed) == 1
        assert store.get_bucket('2023-06-15T12:00:00') == []

    def test_store_thread_safe(self):
        from src.live.central_orchestrator import CandidateDecision, CandidateStore
        store = CandidateStore(expected_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        ts = '2023-06-15T12:00:00'

        def add_sym(sym: str):
            store.add(CandidateDecision(
                symbol=sym, timestamp=ts,
                direction=1, confidence=0.8, expected_edge_bps=20.0,
                size_hint=1.0, quality_bucket='high', entry_price=1000.0,
            ))

        threads = [threading.Thread(target=add_sym, args=(s,))
                   for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(store.get_bucket(ts)) == 3


# =========================================================================
# A5 — CentralOrchestrator
# =========================================================================
class TestCentralOrchestrator:

    def _make_candidate(self, symbol: str, direction: int = 1,
                        confidence: float = 0.8, price: float = 30000.0,
                        ts: str = '2023-06-15T12:00:00'):
        from src.live.central_orchestrator import CandidateDecision
        return CandidateDecision(
            symbol=symbol, timestamp=ts,
            direction=direction, confidence=confidence,
            expected_edge_bps=confidence * 30.0,
            size_hint=1.0, quality_bucket='high',
            entry_price=price,
        )

    def test_orchestrator_instantiation(self):
        from src.live.central_orchestrator import CentralOrchestrator
        orch = CentralOrchestrator(symbols=['BTCUSDT', 'ETHUSDT'])
        assert orch.symbols == ['BTCUSDT', 'ETHUSDT']

    def test_run_cycle_with_full_set(self):
        from src.live.central_orchestrator import CentralOrchestrator
        orch = CentralOrchestrator(symbols=['BTCUSDT', 'ETHUSDT'])
        candidates = [
            self._make_candidate('BTCUSDT', confidence=0.9, price=30000.0),
            self._make_candidate('ETHUSDT', confidence=0.7, price=2000.0),
        ]
        results = orch.run_cycle(candidates)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert 'symbol' in r
            assert 'decision' in r

    def test_run_cycle_deterministic(self):
        from src.live.central_orchestrator import CentralOrchestrator
        orch = CentralOrchestrator(symbols=['BTCUSDT', 'ETHUSDT'])
        candidates = [
            self._make_candidate('BTCUSDT', confidence=0.9),
            self._make_candidate('ETHUSDT', confidence=0.7),
        ]
        r1 = orch.run_cycle(candidates)
        r2 = orch.run_cycle(candidates)
        for a, b in zip(r1, r2):
            assert a['symbol'] == b['symbol']
            assert a['decision'] == b['decision']

    def test_run_cycle_ranks_across_assets(self):
        from src.live.central_orchestrator import CentralOrchestrator
        orch = CentralOrchestrator(symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        candidates = [
            self._make_candidate('SOLUSDT', confidence=0.5, price=100.0),
            self._make_candidate('BTCUSDT', confidence=0.95, price=30000.0),
            self._make_candidate('ETHUSDT', confidence=0.6, price=2000.0),
        ]
        results = orch.run_cycle(candidates)
        approved = [r for r in results if r['decision'].startswith('APPROVE')]
        assert len(approved) >= 1
        assert approved[0]['symbol'] == 'BTCUSDT'

    def test_run_cycle_empty_candidates(self):
        from src.live.central_orchestrator import CentralOrchestrator
        orch = CentralOrchestrator(symbols=['BTCUSDT'])
        results = orch.run_cycle([])
        assert results == []

    def test_run_cycle_updates_dependency_layer(self):
        from src.live.central_orchestrator import CentralOrchestrator
        from src.live.inter_asset_dependency import InterAssetDependencyLayer
        dep = InterAssetDependencyLayer(symbols=['BTCUSDT', 'ETHUSDT'])
        orch = CentralOrchestrator(
            symbols=['BTCUSDT', 'ETHUSDT'],
            dependency_layer=dep,
        )
        candidates = [
            self._make_candidate('BTCUSDT', confidence=0.9, price=30000.0),
            self._make_candidate('ETHUSDT', confidence=0.7, price=2000.0),
        ]
        results = orch.run_cycle(candidates)
        assert len(results) == 2

    def test_run_cycle_persists_results(self):
        from src.live.central_orchestrator import CentralOrchestrator
        orch = CentralOrchestrator(symbols=['BTCUSDT', 'ETHUSDT'])
        candidates = [
            self._make_candidate('BTCUSDT', confidence=0.9),
        ]
        orch.run_cycle(candidates)
        snap = orch.snapshot()
        assert snap['total_cycles'] == 1
        assert len(snap['last_cycle_results']) == 1

    def test_cycle_count_increments(self):
        from src.live.central_orchestrator import CentralOrchestrator
        orch = CentralOrchestrator(symbols=['BTCUSDT'])
        for i in range(3):
            orch.run_cycle([self._make_candidate('BTCUSDT')])
        assert orch.snapshot()['total_cycles'] == 3


# =========================================================================
# Runtime refactor — candidate-only mode
# =========================================================================
class TestRuntimeCandidateMode:

    def _make_bar(self, close: float = 30000.0,
                  ts: str = '2023-06-15T12:00:00') -> dict:
        return {
            'timestamp': ts,
            'open': close * 0.999,
            'high': close * 1.001,
            'low': close * 0.998,
            'close': close,
            'volume': 100.0,
        }

    def test_runtime_with_candidate_store_emits_candidate(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        from src.live.central_orchestrator import CandidateStore
        HERE = Path(__file__).resolve().parent.parent
        store = CandidateStore(expected_symbols=['BTCUSDT'])
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
            candidate_store=store,
        )
        rt.start()
        result = rt.on_bar(self._make_bar())
        assert result['action'] in (
            'FLAT', 'CANDIDATE_EMITTED', 'SKIP_QUALITY',
            'SYSTEM_STOPPED', 'PAUSED',
        )
        if result['action'] == 'CANDIDATE_EMITTED':
            assert 'candidate' in result['layers']

    def test_runtime_without_store_still_executes(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        HERE = Path(__file__).resolve().parent.parent
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt.start()
        result = rt.on_bar(self._make_bar())
        assert result['action'] in (
            'FLAT', 'ORDER_SUBMITTED', 'SKIP_QUALITY',
            'BLOCKED_RISK', 'SKIP_DEPENDENCY', 'BLOCKED_ALLOCATOR',
        )

    def test_runtime_candidate_mode_does_not_submit_orders(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        from src.live.central_orchestrator import CandidateStore
        HERE = Path(__file__).resolve().parent.parent
        store = CandidateStore(expected_symbols=['BTCUSDT'])
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
            candidate_store=store,
        )
        rt.start()
        for _ in range(20):
            rt.on_bar(self._make_bar())
        assert len(rt.oms._orders) == 0


# =========================================================================
# Full integration — multi-asset synchronized cycle
# =========================================================================
class TestFullIntegration:

    def test_three_runtimes_orchestrator_cycle(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        from src.live.central_orchestrator import (
            CandidateStore, CentralOrchestrator, CandidateDecision,
        )
        HERE = Path(__file__).resolve().parent.parent

        store = CandidateStore(
            expected_symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        )
        orch = CentralOrchestrator(
            symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        )

        runtimes = {}
        for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
            runtimes[sym] = NYXRuntime(
                symbol=sym,
                models_dir=HERE / 'models' / sym,
                state_dir=Path(tempfile.mkdtemp()) / sym,
                candidate_store=store,
            )
            runtimes[sym].start()

        ts = '2023-06-15T12:00:00'
        prices = {'BTCUSDT': 30000.0, 'ETHUSDT': 2000.0, 'SOLUSDT': 25.0}
        for sym, price in prices.items():
            bar = {
                'timestamp': ts,
                'open': price * 0.999,
                'high': price * 1.001,
                'low': price * 0.998,
                'close': price,
                'volume': 100.0,
            }
            runtimes[sym].on_bar(bar)

        candidates = store.consume(ts)
        if candidates:
            results = orch.run_cycle(candidates)
            assert isinstance(results, list)
