"""
TDD Tests — `NYXEngine` is the single canonical runtime entrypoint.

Ticket 04 — Make NYXEngine the single runtime entrypoint.

Per user decision (recorded in ticket 04): rename the canonical batch
engine from `src.ml.nyx_pipeline.NYXPipeline` to
`src.core.nyx_engine.NYXEngine`. The legacy v0.8 engine (HSMM + SMC +
macro) moves to `src.core.nyx_engine_v08.NYXEngine` for its 9 legacy
callers (src/runner/run_paper.py, src/validation/*, scripts/run_backtest.py).

A thin backward-compat shim at `src.ml.nyx_pipeline` re-exports
`NYXEngine as NYXPipeline` so the ~45 existing canonical callers keep
working without touching them in this ticket.
"""
from __future__ import annotations

import importlib

import pytest


# ===========================================================================
class TestCanonicalImportPath:
    """NYXEngine must be importable from src.core.nyx_engine and be
    a class with a `.run()` method (= the canonical batch entrypoint)."""

    def test_nyx_engine_importable(self):
        mod = importlib.import_module('src.core.nyx_engine')
        assert hasattr(mod, 'NYXEngine'), \
            "src.core.nyx_engine must export NYXEngine"

    def test_nyx_engine_has_run_method(self):
        from src.core.nyx_engine import NYXEngine
        assert hasattr(NYXEngine, 'run'), \
            "NYXEngine must have `.run()` (canonical batch entrypoint)"

    def test_nyx_engine_instantiable_with_defaults(self):
        """`NYXEngine()` must construct with no args (same as old
        NYXPipeline())."""
        from src.core.nyx_engine import NYXEngine
        e = NYXEngine()
        assert e is not None


# ===========================================================================
class TestLegacyV08Segregated:
    """The old v0.8 monolithic engine (HSMM+SMC+macro) must live at a
    SEPARATE path so it cannot collide with the new canonical engine."""

    def test_v08_importable_from_its_own_module(self):
        """Legacy NYXEngine v0.8 (HSMM+SMC+macro) stays callable from
        src.core.nyx_engine_v08 for its 9 legacy callers — nothing in
        the canonical path imports it."""
        mod = importlib.import_module('src.core.nyx_engine_v08')
        assert hasattr(mod, 'NYXEngine'), \
            "src.core.nyx_engine_v08 must export the legacy NYXEngine class"

    def test_v08_is_not_canonical(self):
        """The legacy v0.8 class and the new canonical class must be
        DIFFERENT classes. Same name, two separate modules, zero
        overlap."""
        from src.core.nyx_engine import NYXEngine as Canonical
        from src.core.nyx_engine_v08 import NYXEngine as Legacy
        assert Canonical is not Legacy, (
            "canonical NYXEngine and legacy v0.8 NYXEngine must be "
            "distinct classes — no re-export collision"
        )


# ===========================================================================
class TestBackwardCompatShim:
    """`src.ml.nyx_pipeline` becomes a thin shim re-exporting NYXEngine
    so existing callers that `from src.ml.nyx_pipeline import NYXPipeline`
    keep working through the migration."""

    def test_shim_reexports_nyx_engine(self):
        from src.ml.nyx_pipeline import NYXPipeline
        from src.core.nyx_engine import NYXEngine
        assert NYXPipeline is NYXEngine, (
            "src.ml.nyx_pipeline.NYXPipeline must be the SAME class as "
            "src.core.nyx_engine.NYXEngine (shim re-export)"
        )

    def test_shim_has_no_independent_logic(self):
        """Guard: the shim must stay small. If it grows past a few
        lines it risks drifting from the canonical engine."""
        import src.ml.nyx_pipeline as shim_mod
        shim_path = shim_mod.__file__
        assert shim_path is not None
        with open(shim_path) as f:
            text = f.read()
        assert len(text) < 1500, (
            f"src/ml/nyx_pipeline.py has grown to {len(text)} chars — "
            "it must stay a thin shim re-exporting NYXEngine"
        )
        # The shim must not re-declare the class. Use a regex that
        # matches a Python class declaration at line-start (so mentions
        # of the old name inside docstrings don't trigger).
        import re
        assert not re.search(r'^class\s+NYXPipeline\b', text, re.MULTILINE), (
            "src/ml/nyx_pipeline.py must not re-declare NYXPipeline — "
            "it is a pure re-export of NYXEngine"
        )


# ===========================================================================
class TestCanonicalDocRefersToNYXEngine:
    """The canonical architecture doc must name `NYXEngine` as the
    runtime entrypoint (not `NYXPipeline`) post-Ticket 04."""

    def test_arch_doc_mentions_nyx_engine(self):
        from pathlib import Path
        arch = Path(__file__).parent.parent / 'docs' / 'ARCHITECTURE_CANONIQUE.md'
        text = arch.read_text()
        assert 'NYXEngine' in text, (
            'ARCHITECTURE_CANONIQUE.md must name NYXEngine as the '
            'canonical entrypoint post-ticket-04'
        )
        assert 'NYXEngine.run' in text, (
            'ARCHITECTURE_CANONIQUE.md must reference NYXEngine.run'
        )


# ===========================================================================
class TestLegacyCallersUnchanged:
    """The 9 legacy callers of v0.8 NYXEngine (src/runner/run_paper.py,
    src/validation/*, scripts/run_backtest.py) must now import from
    src.core.nyx_engine_v08 — not from src.core.nyx_engine (which is
    the new canonical)."""

    LEGACY_CALLERS = [
        'src/runner/run_paper.py',
        'src/validation/walk_forward.py',
        'src/validation/oos_report.py',
        'src/validation/benchmarks.py',
        'src/validation/smc_diagnostics.py',
        'src/validation/signal_funnel.py',
        'src/validation/pattern_quality.py',
        'scripts/run_backtest.py',
    ]

    @pytest.mark.parametrize('rel_path', LEGACY_CALLERS)
    def test_legacy_caller_uses_v08_path(self, rel_path):
        from pathlib import Path
        p = Path(__file__).parent.parent / rel_path
        if not p.exists():
            pytest.skip(f"{rel_path} does not exist in repo")
        text = p.read_text()
        # It must not import the NEW canonical engine (that would mean
        # a legacy v0.8 caller suddenly calls the new engine with a
        # different API — runtime crash).
        assert 'from src.core.nyx_engine import' not in text, (
            f"{rel_path} imports the NEW canonical NYXEngine — if it "
            "was a v0.8 caller it must be updated to "
            "`from src.core.nyx_engine_v08 import NYXEngine`"
        )
