"""
TDD Tests — Ticket 38 — Repository Refactor (Clean Architecture).

Structural tests enforcing:
1. src/nyx/ re-export layer exists and all canonical classes importable
2. Canonical modules do NOT import from legacy
3. docs/ARCHITECTURE.md exists and is comprehensive
4. legacy/MANIFEST.md exists
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent.parent


# =========================================================================
# A1 — src/nyx/ re-export layer exists
# =========================================================================
class TestCanonicalNamespace:

    def test_nyx_package_exists(self):
        assert (HERE / 'src' / 'nyx' / '__init__.py').exists()

    def test_import_runtime(self):
        from src.nyx.runtime import NYXRuntime, CentralOrchestrator, CandidateDecision, CandidateStore
        assert NYXRuntime is not None
        assert CentralOrchestrator is not None

    def test_import_decision(self):
        from src.nyx.decision import MetaGBM, NYXLiveDecider, compute_fractal_quality
        assert MetaGBM is not None
        assert NYXLiveDecider is not None

    def test_import_portfolio(self):
        from src.nyx.portfolio import PortfolioAllocator, InterAssetDependencyLayer, Portfolio
        assert PortfolioAllocator is not None

    def test_import_execution(self):
        from src.nyx.execution import OMS, RiskEngine
        assert OMS is not None
        assert RiskEngine is not None

    def test_import_data(self):
        from src.nyx.data import BarBuilder, FeedHealth, compute_stationary_features
        assert BarBuilder is not None

    def test_import_state(self):
        from src.nyx.state import StateStore, AuditStore, MetricsCollector, AlertManager
        assert StateStore is not None
        assert AuditStore is not None

    def test_import_contracts(self):
        from src.nyx.contracts import FractalReport, MetaDecision, Signal, CandidateDecision
        assert FractalReport is not None
        assert Signal is not None

    def test_top_level_import(self):
        from src.nyx import NYXRuntime, CentralOrchestrator
        assert NYXRuntime is not None
        assert CentralOrchestrator is not None


# =========================================================================
# A2 — Canonical modules do NOT import from legacy
# =========================================================================

CANONICAL_FILES = [
    'src/live/nyx_runtime.py',
    'src/live/central_orchestrator.py',
    'src/live/oms.py',
    'src/live/risk_engine.py',
    'src/live/var_cvar.py',
    'src/live/portfolio_state.py',
    'src/live/portfolio_allocator.py',
    'src/live/inter_asset_dependency.py',
    'src/live/state_store.py',
    'src/live/monitoring.py',
    'src/live/audit_trail.py',
    'src/live/bar_builder.py',
    'src/live/feed_health.py',
    'src/live/execution_optimizer.py',
    'src/core/meta_gbm.py',
    'src/core/fractal_quality.py',
]

LEGACY_IMPORT_PREFIXES = [
    'from src.paper_live.',
    'from src.validation.',
    'from src.execution.',
    'from src.cache.',
    'from src.macro.',
    'from src.runner.',
    'from src.data.',
]


class TestCanonicalBoundary:

    def test_canonical_files_do_not_import_legacy_packages(self):
        violations = []
        for rel_path in CANONICAL_FILES:
            full = HERE / rel_path
            if not full.exists():
                continue
            source = full.read_text()
            for prefix in LEGACY_IMPORT_PREFIXES:
                if prefix in source:
                    violations.append(f'{rel_path} imports {prefix}')
        assert violations == [], f'Legacy imports in canonical files: {violations}'

    def test_nyx_reexport_layer_does_not_contain_logic(self):
        """src/nyx/ files must be thin re-exports, not logic."""
        nyx_dir = HERE / 'src' / 'nyx'
        if not nyx_dir.exists():
            pytest.skip('src/nyx/ not created yet')
        for py in nyx_dir.glob('*.py'):
            source = py.read_text()
            assert len(source) < 2000, (
                f'{py.name} is too large ({len(source)} chars) for a re-export shim'
            )


# =========================================================================
# A3 — Documentation exists
# =========================================================================
class TestDocumentation:

    def test_architecture_md_exists(self):
        path = HERE / 'docs' / 'ARCHITECTURE.md'
        assert path.exists(), 'docs/ARCHITECTURE.md missing'
        text = path.read_text()
        assert len(text) > 1000, 'ARCHITECTURE.md too short'

    def test_architecture_has_pipeline_flow(self):
        path = HERE / 'docs' / 'ARCHITECTURE.md'
        text = path.read_text()
        for term in ['NYXRuntime', 'GBM', 'Jesse', 'RiskEngine', 'OMS']:
            assert term in text, f'{term} missing from ARCHITECTURE.md'

    def test_architecture_has_module_map(self):
        path = HERE / 'docs' / 'ARCHITECTURE.md'
        text = path.read_text()
        assert 'canonical' in text.lower()
        assert 'legacy' in text.lower()

    def test_legacy_manifest_exists(self):
        path = HERE / 'legacy' / 'MANIFEST.md'
        assert path.exists(), 'legacy/MANIFEST.md missing'
        text = path.read_text()
        assert len(text) > 500, 'MANIFEST.md too short'


# =========================================================================
# A4 — No circular imports in canonical layer
# =========================================================================
class TestNoCircularImports:

    def test_import_nyx_runtime(self):
        mod = importlib.import_module('src.live.nyx_runtime')
        assert hasattr(mod, 'NYXRuntime')

    def test_import_central_orchestrator(self):
        mod = importlib.import_module('src.live.central_orchestrator')
        assert hasattr(mod, 'CentralOrchestrator')

    def test_import_portfolio_allocator(self):
        mod = importlib.import_module('src.live.portfolio_allocator')
        assert hasattr(mod, 'PortfolioAllocator')

    def test_import_audit_trail(self):
        mod = importlib.import_module('src.live.audit_trail')
        assert hasattr(mod, 'AuditStore')
