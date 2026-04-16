"""
TDD Tests — docs/RUNNER_INVENTORY.md + docs/PROJECT_TRUTH_MAP.md
presence + every entrypoint tagged.

Ticket 02 — Build the runner inventory and truth map.

These tests are **documentary** (presence of required entries +
status tags), not behavioural. They protect against a future session
silently dropping an entrypoint from the inventory.

Every path listed in the ticket's minimum scope must appear in
RUNNER_INVENTORY.md with a status tag ∈ {canonical runtime,
offline calibration, legacy, research/experimental}.
"""
from pathlib import Path

import pytest


DOCS = Path(__file__).parent.parent / 'docs'
INVENTORY = DOCS / 'RUNNER_INVENTORY.md'
TRUTH_MAP = DOCS / 'PROJECT_TRUTH_MAP.md'


@pytest.fixture(scope='module')
def inventory_text() -> str:
    assert INVENTORY.exists(), f"{INVENTORY} missing"
    return INVENTORY.read_text()


@pytest.fixture(scope='module')
def truth_map_text() -> str:
    assert TRUTH_MAP.exists(), f"{TRUTH_MAP} missing"
    return TRUTH_MAP.read_text()


# ===========================================================================
class TestInventoryDocPresence:

    def test_inventory_non_trivial(self, inventory_text):
        assert len(inventory_text) > 2000, (
            'RUNNER_INVENTORY.md too short — expected > 2000 chars '
            '(ticket requires purpose, status, asset/TF, caller, '
            'runtime-vs-offline for each entry)'
        )

    def test_inventory_has_four_status_tags(self, inventory_text):
        """The 4 canonical status tags must ALL be present somewhere."""
        for tag in (
            'canonical runtime',
            'offline calibration',
            'legacy',
            'research',  # covers 'research/experimental'
        ):
            assert tag in inventory_text.lower(), (
                f'status tag {tag!r} not used — inventory must classify '
                f'every entry with one of the 4 canonical statuses'
            )


# ===========================================================================
class TestMinimumScopeCovered:
    """Every path in the ticket's minimum scope must appear by name."""

    REQUIRED_PATHS = (
        'scripts/run_backtest.py',
        'scripts/backtest_mtf.py',
        'src/core/nyx_engine.py',
        'src/ml/edge_strategy.py',
        'src/ml/ml_filter_v2.py',
        'src/ml/threshold_optimizer.py',
        'src/ml/feedback_loop.py',
        # jesse_* glob — at minimum jesse_agents, jesse_features, jesse_labeler
        'jesse_agents',
        'jesse_features',
        'jesse_labeler',
    )

    @pytest.mark.parametrize('path', REQUIRED_PATHS)
    def test_path_present(self, inventory_text, path):
        assert path in inventory_text, (
            f'inventory must list {path!r} (ticket minimum scope)'
        )


# ===========================================================================
class TestCanonicalEntrypointPresent:
    """The canonical runtime entrypoints (per Ticket 01) must be
    present AND tagged 'canonical runtime'."""

    def _mentioned_near(self, text, needle, tag):
        """True if any occurrence of `needle` has `tag` within a wider
        window (300 chars either side) — tolerant enough for prose
        paragraphs, strict enough to require explicit tagging somewhere
        near the path mention."""
        t = text.lower()
        n = needle.lower()
        target = tag.lower()
        idx = 0
        while True:
            idx = t.find(n, idx)
            if idx < 0:
                return False
            window = t[max(0, idx-300): idx+300]
            if target in window:
                return True
            idx += len(n)

    def test_nyx_pipeline_canonical(self, inventory_text):
        assert 'nyx_pipeline' in inventory_text
        assert self._mentioned_near(
            inventory_text, 'nyx_pipeline', 'canonical'
        ), 'nyx_pipeline.py must be tagged canonical runtime'

    def test_nyx_live_decider_canonical(self, inventory_text):
        assert 'nyx_live_decider' in inventory_text
        assert self._mentioned_near(
            inventory_text, 'nyx_live_decider', 'canonical'
        ), 'nyx_live_decider.py must be tagged canonical runtime'


# ===========================================================================
class TestPaperLiveEntrypointsListed:
    """Paper-live entrypoints (if present) must be inventoried."""

    def test_paper_live_runner_listed(self, inventory_text):
        """src/paper_live/runner.py exists per current repo — must
        appear in the inventory with a status tag."""
        # We assert its presence; the status tag is checked globally
        # by TestInventoryDocPresence.test_inventory_has_four_status_tags.
        assert 'paper_live' in inventory_text.lower()


# ===========================================================================
class TestLegacyAndOfflineExplicitlyMarked:
    """The historical modules must be called out as legacy or offline,
    not left ambiguous."""

    def test_edge_strategy_not_standalone_strategy(self, inventory_text):
        """edge_strategy.py must be explicitly tagged as NOT a
        standalone strategy — whether it's labelled offline, legacy,
        or (post-Ticket-08) a canonical CANDIDATE-GENERATION COMPONENT
        of the runtime. What the test forbids is ambiguous wording
        that could be read as 'standalone strategy'."""
        idx = inventory_text.lower().find('edge_strategy')
        assert idx >= 0
        ctx = inventory_text.lower()[max(0, idx-50): idx+500]
        # Post-Ticket-08 acceptable tags: offline / legacy (old
        # reading) OR "candidate generat" (new reading — the canonical
        # candidate generator is NOT a standalone strategy).
        assert (
            'offline' in ctx
            or 'legacy' in ctx
            or 'candidate gen' in ctx
            or 'not a standalone' in ctx
        ), (
            'edge_strategy.py must be tagged as non-standalone '
            '(offline / legacy / candidate-generator component)'
        )

    def test_threshold_optimizer_offline(self, inventory_text):
        idx = inventory_text.lower().find('threshold_optimizer')
        assert idx >= 0
        ctx = inventory_text.lower()[max(0, idx-50): idx+400]
        assert 'offline' in ctx, (
            'threshold_optimizer.py must be tagged offline calibration'
        )

    def test_ml_filter_v2_offline_or_legacy(self, inventory_text):
        idx = inventory_text.lower().find('ml_filter_v2')
        assert idx >= 0
        ctx = inventory_text.lower()[max(0, idx-50): idx+400]
        assert 'offline' in ctx or 'legacy' in ctx


# ===========================================================================
class TestTruthMapPresence:

    def test_truth_map_non_trivial(self, truth_map_text):
        assert len(truth_map_text) > 1000, (
            'PROJECT_TRUTH_MAP.md too short — expected > 1000 chars'
        )

    def test_truth_map_links_to_inventory(self, truth_map_text):
        """The truth map should reference the inventory doc so readers
        can drill down from the high-level map to the per-module
        detail."""
        assert 'RUNNER_INVENTORY' in truth_map_text

    def test_truth_map_four_layers(self, truth_map_text):
        """Truth map is the high-level layered view — must mention all
        4 status layers explicitly."""
        t = truth_map_text.lower()
        assert 'canonical runtime' in t
        assert 'offline' in t
        assert 'legacy' in t
        assert 'research' in t or 'experimental' in t


# ===========================================================================
class TestNoMysteryRunner:
    """Ticket acceptance: 'no unnamed mystery runner left'.

    We walk scripts/ and flag any .py file with `if __name__ ==
    '__main__':` that is NOT mentioned in the inventory by name.
    """

    def _invocable_scripts(self) -> list:
        scripts_dir = Path(__file__).parent.parent / 'scripts'
        if not scripts_dir.is_dir():
            return []
        out = []
        for p in scripts_dir.glob('*.py'):
            try:
                text = p.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if "__name__ == '__main__'" in text or '__name__ == "__main__"' in text:
                out.append(p.name)
        return sorted(out)

    def test_every_invocable_script_inventoried(self, inventory_text):
        missing = [
            name for name in self._invocable_scripts()
            if name not in inventory_text
        ]
        assert not missing, (
            f'{len(missing)} invocable scripts(s) under scripts/ are '
            f'not inventoried: {missing}. Either add them to '
            f'RUNNER_INVENTORY.md or delete the file.'
        )
