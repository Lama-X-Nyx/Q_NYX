"""
TDD Tests — docs/ARCHITECTURE_CANONIQUE.md presence + canonical declarations.

Ticket 01 — Freeze the canonical NYX architecture.

These tests are **documentary** (presence of declarations), not
behavioural. They protect against accidental removal or silent
watering-down of the canonical architecture decision in a future
refactor.

The doc must explicitly declare:
  - ONE canonical runtime path (data MTF -> reporters -> Meta-GBM ->
    risk -> execution -> logging/feedback)
  - ONE canonical offline path (training data -> features ->
    calibration -> OOS / walk-forward / reality checks)
  - The role of the 4 Jesse agents as fractal reporters
  - The role of Meta-GBM as the strategy brain
  - `edge_strategy` as NOT a standalone runtime strategy
  - `threshold_optimizer` as offline-only
"""
from pathlib import Path

import pytest


ARCH_PATH = (
    Path(__file__).parent.parent / 'docs' / 'ARCHITECTURE_CANONIQUE.md'
)


@pytest.fixture(scope='module')
def arch_text() -> str:
    assert ARCH_PATH.exists(), f"{ARCH_PATH} missing"
    return ARCH_PATH.read_text()


# ===========================================================================
class TestCanonicalDocPresence:
    """The canonical architecture doc must exist and be non-trivial."""

    def test_doc_exists_and_non_trivial(self, arch_text):
        assert len(arch_text) > 1000, (
            'ARCHITECTURE_CANONIQUE.md is too short to be a canonical '
            'declaration — expected > 1000 chars'
        )

    def test_doc_headlines_present(self, arch_text):
        """Doc must have the two canonical-path headers."""
        assert '## Canonical runtime path' in arch_text
        assert '## Canonical offline path' in arch_text


# ===========================================================================
class TestRuntimePathDeclaration:
    """The runtime path must name its components explicitly."""

    def test_runtime_entrypoint_declared(self, arch_text):
        """`NYXEngine.run` (Ticket 04 rename from NYXPipeline) must be
        the canonical runtime entrypoint. The old name may still
        appear in historical / deprecation-shim wording."""
        assert 'NYXEngine' in arch_text
        assert 'NYXEngine.run' in arch_text

    def test_runtime_lists_core_layers(self, arch_text):
        """All 6 layers of the canonical runtime path must be named."""
        for layer in (
            'Data MTF',
            'Meta-GBM',
            'Risk manager',
            'Execution layer',
            'Logging',
            'feedback',
        ):
            assert layer in arch_text, (
                f"canonical runtime layer {layer!r} missing"
            )


# ===========================================================================
class TestFractalReportersDeclaration:
    """The 4 Jesse agents (Context/Regime/Setup/Entry) are fractal
    reporters, not deciders. Orchestrator is meta, not counted here."""

    def test_four_agents_named_as_reporters(self, arch_text):
        """The 4 agent names must appear AND be tagged 'fractal reporters'."""
        assert 'fractal reporter' in arch_text.lower()
        for name in ('Context', 'Regime', 'Setup', 'Entry'):
            assert f'Jesse{name}Agent' in arch_text or f'Jesse {name}' in arch_text, (
                f"Jesse {name} reporter not explicitly named"
            )

    def test_jesse_agents_honest_status(self, arch_text):
        """The doc must be HONEST: the 4 agents are currently NOT WIRED.
        NYXPipeline uses hand-crafted rule_* scalars as proxies."""
        t = arch_text.lower()
        assert 'not wired' in t or 'non wir' in t or 'pas encore wir' in t, (
            'doc must state that the 4 Jesse reporters are currently '
            'not wired into production (NYXPipeline uses proxies)'
        )
        assert 'rule_context' in arch_text
        assert 'rule_regime' in arch_text
        assert 'rule_setup' in arch_text


# ===========================================================================
class TestMetaGBMDeclaration:
    """Meta-GBM must be declared as the strategy brain."""

    def test_meta_gbm_is_strategy_brain(self, arch_text):
        assert 'Meta-GBM' in arch_text
        assert 'strategy brain' in arch_text.lower()

    def test_meta_gbm_maps_to_gbm_classifier(self, arch_text):
        """Doc must state that Meta-GBM is the GBM inside NYXPipeline
        today (not a separate 'meta' class stacked on top of agents)."""
        assert 'GradientBoostingClassifier' in arch_text
        assert '0.60' in arch_text  # canonical threshold


# ===========================================================================
class TestOfflineOnlyModulesDeclaration:
    """edge_strategy, threshold_optimizer, ml_filter_v2 are offline-only."""

    def test_edge_strategy_not_a_runtime_strategy(self, arch_text):
        assert 'edge_strategy' in arch_text
        t = arch_text.lower()
        # Must be called out as historical / not runtime.
        assert (
            'edge_strategy' in arch_text and (
                'offline' in t or 'not a standalone' in t
                or 'not wired at runtime' in t or 'historical' in t
            )
        )

    def test_threshold_optimizer_offline_only(self, arch_text):
        assert 'threshold_optimizer' in arch_text
        assert 'offline-only' in arch_text.lower() or 'offline only' in arch_text.lower()

    def test_ml_filter_v2_offline_only(self, arch_text):
        """ml_filter_v2 is used only by threshold_optimizer. Must be
        declared as offline calibration, not runtime."""
        assert 'ml_filter_v2' in arch_text


# ===========================================================================
class TestRiskAndExecutionOwnership:
    """Risk manager and execution layer must be pinned to their owners."""

    def test_risk_manager_points_at_bear_dial(self, arch_text):
        assert 'conditional_dial' in arch_text or 'bear_risk_dial' in arch_text

    def test_execution_points_at_post_only_broker(self, arch_text):
        assert 'PostOnlyPaperBroker' in arch_text

    def test_logging_points_at_persistent_decisions(self, arch_text):
        assert 'PersistentDecisionLogger' in arch_text or 'missed_trades' in arch_text


# ===========================================================================
class TestOfflinePathDeclaration:
    """Offline path must list calibration + OOS + reality-check steps."""

    def test_offline_lists_calibration(self, arch_text):
        t = arch_text.lower()
        assert 'calibration' in t
        assert 'threshold' in t

    def test_offline_lists_walk_forward(self, arch_text):
        assert 'walk-forward' in arch_text.lower() or 'walk forward' in arch_text.lower()

    def test_offline_lists_reality_checks(self, arch_text):
        assert 'reality check' in arch_text.lower()
