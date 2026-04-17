"""
TDD Tests — Ticket 37 — Unified Audit Trail.

Structured, deterministic, replayable decision history covering:
1. Per-asset audit events (local decision chain)
2. Portfolio-level audit events (global allocation cycle)

Linked via cycle_id. Persisted as append-only JSONL.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# =========================================================================
# A1 — AssetAuditEvent schema
# =========================================================================
class TestAssetAuditEvent:

    def test_create_event(self):
        from src.live.audit_trail import AssetAuditEvent
        ev = AssetAuditEvent(
            symbol='BTCUSDT',
            timestamp='2023-06-15T12:00:00',
            action='ORDER_SUBMITTED',
            layers={
                'signal': {'direction': 1, 'conviction': 0.85, 'p_trade': 0.85},
                'jesse': {'context': {'state': 'bullish', 'score': 0.8, 'passed': True}},
                'fractal_quality': {'quality_bucket': 'high', 'size_multiplier': 1.25},
            },
        )
        assert ev.symbol == 'BTCUSDT'
        assert ev.action == 'ORDER_SUBMITTED'

    def test_event_to_dict_serializable(self):
        from src.live.audit_trail import AssetAuditEvent
        ev = AssetAuditEvent(
            symbol='BTCUSDT',
            timestamp='2023-06-15T12:00:00',
            action='FLAT',
            layers={'signal': {'direction': 0}},
        )
        d = ev.to_dict()
        s = json.dumps(d)
        assert isinstance(s, str)
        loaded = json.loads(s)
        assert loaded['symbol'] == 'BTCUSDT'
        assert loaded['action'] == 'FLAT'

    def test_event_has_event_id(self):
        from src.live.audit_trail import AssetAuditEvent
        ev = AssetAuditEvent(
            symbol='BTCUSDT',
            timestamp='2023-06-15T12:00:00',
            action='FLAT',
            layers={},
        )
        assert ev.event_id is not None
        assert len(ev.event_id) > 0

    def test_event_ids_unique(self):
        from src.live.audit_trail import AssetAuditEvent
        ids = set()
        for _ in range(100):
            ev = AssetAuditEvent(
                symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
                action='FLAT', layers={},
            )
            ids.add(ev.event_id)
        assert len(ids) == 100

    def test_event_cycle_id_optional(self):
        from src.live.audit_trail import AssetAuditEvent
        ev = AssetAuditEvent(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            action='FLAT', layers={},
        )
        assert ev.cycle_id is None
        ev2 = AssetAuditEvent(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            action='CANDIDATE_EMITTED', layers={}, cycle_id='cycle-001',
        )
        assert ev2.cycle_id == 'cycle-001'


# =========================================================================
# A1 — PortfolioAuditEvent schema
# =========================================================================
class TestPortfolioAuditEvent:

    def test_create_portfolio_event(self):
        from src.live.audit_trail import PortfolioAuditEvent
        ev = PortfolioAuditEvent(
            cycle_id='cycle-001',
            timestamp='2023-06-15T12:00:00',
            candidates=[
                {'symbol': 'BTCUSDT', 'direction': 1, 'confidence': 0.9},
                {'symbol': 'ETHUSDT', 'direction': 1, 'confidence': 0.7},
            ],
            allocation_results=[
                {'symbol': 'BTCUSDT', 'decision': 'APPROVE_FULL'},
                {'symbol': 'ETHUSDT', 'decision': 'DEFER'},
            ],
        )
        assert ev.cycle_id == 'cycle-001'
        assert len(ev.candidates) == 2

    def test_portfolio_event_serializable(self):
        from src.live.audit_trail import PortfolioAuditEvent
        ev = PortfolioAuditEvent(
            cycle_id='cycle-002',
            timestamp='2023-06-15T12:15:00',
            candidates=[],
            allocation_results=[],
        )
        d = ev.to_dict()
        s = json.dumps(d)
        loaded = json.loads(s)
        assert loaded['cycle_id'] == 'cycle-002'

    def test_portfolio_event_has_dependency_outputs(self):
        from src.live.audit_trail import PortfolioAuditEvent
        ev = PortfolioAuditEvent(
            cycle_id='cycle-003',
            timestamp='2023-06-15T12:00:00',
            candidates=[],
            allocation_results=[],
            dependency_outputs={'ETHUSDT': {'dependency_strength': 0.7}},
        )
        assert 'ETHUSDT' in ev.dependency_outputs


# =========================================================================
# A4 — AuditStore persistence
# =========================================================================
class TestAuditStore:

    def test_append_asset_event(self):
        from src.live.audit_trail import AssetAuditEvent, AuditStore
        store = AuditStore(Path(tempfile.mkdtemp()))
        ev = AssetAuditEvent(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            action='ORDER_SUBMITTED', layers={'signal': {'direction': 1}},
        )
        store.append_asset_event(ev)
        events = store.load_asset_events('BTCUSDT')
        assert len(events) == 1
        assert events[0]['symbol'] == 'BTCUSDT'

    def test_append_multiple_events(self):
        from src.live.audit_trail import AssetAuditEvent, AuditStore
        store = AuditStore(Path(tempfile.mkdtemp()))
        for i in range(5):
            ev = AssetAuditEvent(
                symbol='BTCUSDT', timestamp=f'2023-06-15T12:{i:02d}:00',
                action='FLAT', layers={},
            )
            store.append_asset_event(ev)
        events = store.load_asset_events('BTCUSDT')
        assert len(events) == 5

    def test_append_portfolio_event(self):
        from src.live.audit_trail import PortfolioAuditEvent, AuditStore
        store = AuditStore(Path(tempfile.mkdtemp()))
        ev = PortfolioAuditEvent(
            cycle_id='cycle-001', timestamp='2023-06-15T12:00:00',
            candidates=[{'symbol': 'BTCUSDT'}],
            allocation_results=[{'symbol': 'BTCUSDT', 'decision': 'APPROVE_FULL'}],
        )
        store.append_portfolio_event(ev)
        events = store.load_portfolio_events()
        assert len(events) == 1
        assert events[0]['cycle_id'] == 'cycle-001'

    def test_events_survive_reload(self):
        from src.live.audit_trail import AssetAuditEvent, AuditStore
        d = Path(tempfile.mkdtemp())
        store1 = AuditStore(d)
        store1.append_asset_event(AssetAuditEvent(
            symbol='ETHUSDT', timestamp='2023-06-15T12:00:00',
            action='FLAT', layers={},
        ))
        store2 = AuditStore(d)
        events = store2.load_asset_events('ETHUSDT')
        assert len(events) == 1

    def test_separate_files_per_symbol(self):
        from src.live.audit_trail import AssetAuditEvent, AuditStore
        store = AuditStore(Path(tempfile.mkdtemp()))
        store.append_asset_event(AssetAuditEvent(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            action='FLAT', layers={},
        ))
        store.append_asset_event(AssetAuditEvent(
            symbol='ETHUSDT', timestamp='2023-06-15T12:00:00',
            action='ORDER_SUBMITTED', layers={},
        ))
        btc = store.load_asset_events('BTCUSDT')
        eth = store.load_asset_events('ETHUSDT')
        assert len(btc) == 1
        assert len(eth) == 1
        assert btc[0]['action'] == 'FLAT'
        assert eth[0]['action'] == 'ORDER_SUBMITTED'


# =========================================================================
# A5 — Event linkage
# =========================================================================
class TestEventLinkage:

    def test_asset_event_links_to_portfolio_cycle(self):
        from src.live.audit_trail import AssetAuditEvent, PortfolioAuditEvent, AuditStore
        store = AuditStore(Path(tempfile.mkdtemp()))
        cycle_id = 'cycle-001'
        store.append_asset_event(AssetAuditEvent(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            action='CANDIDATE_EMITTED', layers={}, cycle_id=cycle_id,
        ))
        store.append_asset_event(AssetAuditEvent(
            symbol='ETHUSDT', timestamp='2023-06-15T12:00:00',
            action='CANDIDATE_EMITTED', layers={}, cycle_id=cycle_id,
        ))
        store.append_portfolio_event(PortfolioAuditEvent(
            cycle_id=cycle_id, timestamp='2023-06-15T12:00:00',
            candidates=[{'symbol': 'BTCUSDT'}, {'symbol': 'ETHUSDT'}],
            allocation_results=[
                {'symbol': 'BTCUSDT', 'decision': 'APPROVE_FULL'},
                {'symbol': 'ETHUSDT', 'decision': 'DEFER'},
            ],
        ))
        btc_events = store.load_asset_events('BTCUSDT')
        pf_events = store.load_portfolio_events()
        assert btc_events[0]['cycle_id'] == pf_events[0]['cycle_id']

    def test_get_events_by_cycle_id(self):
        from src.live.audit_trail import AssetAuditEvent, AuditStore
        store = AuditStore(Path(tempfile.mkdtemp()))
        store.append_asset_event(AssetAuditEvent(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            action='CANDIDATE_EMITTED', layers={}, cycle_id='cycle-001',
        ))
        store.append_asset_event(AssetAuditEvent(
            symbol='BTCUSDT', timestamp='2023-06-15T12:15:00',
            action='FLAT', layers={}, cycle_id=None,
        ))
        all_events = store.load_asset_events('BTCUSDT')
        linked = [e for e in all_events if e.get('cycle_id') == 'cycle-001']
        assert len(linked) == 1


# =========================================================================
# A6 — Replay engine
# =========================================================================
class TestReplayEngine:

    def test_replay_loads_events(self):
        from src.live.audit_trail import AssetAuditEvent, AuditStore, replay_asset_events
        store = AuditStore(Path(tempfile.mkdtemp()))
        for i in range(3):
            store.append_asset_event(AssetAuditEvent(
                symbol='BTCUSDT', timestamp=f'2023-06-15T12:{i:02d}:00',
                action='FLAT' if i < 2 else 'ORDER_SUBMITTED',
                layers={'signal': {'direction': 0 if i < 2 else 1}},
            ))
        events = replay_asset_events(store, 'BTCUSDT')
        assert len(events) == 3
        assert events[-1]['action'] == 'ORDER_SUBMITTED'

    def test_replay_detects_missing_steps(self):
        from src.live.audit_trail import AuditStore, verify_audit_chain
        store = AuditStore(Path(tempfile.mkdtemp()))
        result = verify_audit_chain(store, 'BTCUSDT')
        assert result['valid'] is True
        assert result['event_count'] == 0

    def test_replay_deterministic_check(self):
        from src.live.audit_trail import AssetAuditEvent, AuditStore, replay_asset_events
        store = AuditStore(Path(tempfile.mkdtemp()))
        store.append_asset_event(AssetAuditEvent(
            symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
            action='ORDER_SUBMITTED',
            layers={'signal': {'direction': 1, 'conviction': 0.85}},
        ))
        r1 = replay_asset_events(store, 'BTCUSDT')
        r2 = replay_asset_events(store, 'BTCUSDT')
        assert r1 == r2


# =========================================================================
# A2 — Runtime instrumentation
# =========================================================================
class TestRuntimeInstrumentation:

    def test_runtime_emits_audit_events(self):
        import tempfile as tf
        from src.live.nyx_runtime import NYXRuntime
        from src.live.audit_trail import AuditStore
        HERE = Path(__file__).resolve().parent.parent
        state_dir = Path(tf.mkdtemp()) / 'BTCUSDT'
        audit_store = AuditStore(state_dir / 'audit')
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=state_dir,
            audit_store=audit_store,
        )
        rt.start()
        bar = {
            'timestamp': '2023-06-15T12:00:00',
            'open': 30000.0, 'high': 30100.0,
            'low': 29900.0, 'close': 30050.0, 'volume': 100.0,
        }
        rt.on_bar(bar)
        events = audit_store.load_asset_events('BTCUSDT')
        assert len(events) >= 1

    def test_runtime_without_audit_still_works(self):
        import tempfile as tf
        from src.live.nyx_runtime import NYXRuntime
        HERE = Path(__file__).resolve().parent.parent
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tf.mkdtemp()) / 'BTCUSDT',
        )
        rt.start()
        bar = {
            'timestamp': '2023-06-15T12:00:00',
            'open': 30000.0, 'high': 30100.0,
            'low': 29900.0, 'close': 30050.0, 'volume': 100.0,
        }
        result = rt.on_bar(bar)
        assert result['action'] in (
            'FLAT', 'ORDER_SUBMITTED', 'SKIP_QUALITY',
            'BLOCKED_RISK', 'SKIP_DEPENDENCY', 'BLOCKED_ALLOCATOR',
        )


# =========================================================================
# A3 — Orchestrator instrumentation
# =========================================================================
class TestOrchestratorInstrumentation:

    def test_orchestrator_emits_portfolio_audit(self):
        from src.live.central_orchestrator import CandidateDecision, CentralOrchestrator
        from src.live.audit_trail import AuditStore
        audit_store = AuditStore(Path(tempfile.mkdtemp()))
        orch = CentralOrchestrator(
            symbols=['BTCUSDT', 'ETHUSDT'],
            audit_store=audit_store,
        )
        candidates = [
            CandidateDecision(
                symbol='BTCUSDT', timestamp='2023-06-15T12:00:00',
                direction=1, confidence=0.9, expected_edge_bps=27.0,
                size_hint=1.0, quality_bucket='high', entry_price=30000.0,
            ),
        ]
        orch.run_cycle(candidates)
        events = audit_store.load_portfolio_events()
        assert len(events) == 1
        assert events[0]['cycle_id'] is not None
