"""
TDD — `docs/BRANCH_STORY.md` honesty guard.

The previous version of BRANCH_STORY started at "Part 1 — Foundations
(pyright + paper-live + hub-spoke)" and implicitly presented
`NYXPipeline` as legacy / pre-branch code. That was factually wrong:
`NYXPipeline` + the 5 Jesse agents were BUILT on this branch in the
first ~40 commits. They are the core of the work, and every published
A/B/C / walk-forward number comes from `NYXPipeline.run()` direct.

These tests guarantee the doc honestly reflects that.
"""
from pathlib import Path

import pytest


STORY_PATH = Path(__file__).parent.parent / 'docs' / 'BRANCH_STORY.md'


@pytest.fixture(scope='module')
def story() -> str:
    assert STORY_PATH.exists(), f"{STORY_PATH} missing"
    return STORY_PATH.read_text()


# ===========================================================================
class TestPhaseZeroPresent:
    """Part 0 MUST exist and credit NYXPipeline + Jesse as THE core."""

    def test_part_0_header_exists(self, story):
        assert "## Part 0" in story, \
            "Part 0 must exist — NYXPipeline + Jesse were built in phase 0"

    def test_nyxpipeline_is_the_core(self, story):
        markers = (
            "NYXPipeline",
            "core engine",
        )
        for m in markers:
            assert m in story, f"missing: {m!r}"
        # One of the explicit honest phrases must appear:
        honest_phrases = (
            "NYXPipeline is the core engine built in phase 0 of this branch",
            "NYXPipeline is the actual core engine",
            "NYXPipeline produces every A/B/C",
        )
        assert any(p in story for p in honest_phrases), (
            "At least one phrase acknowledging NYXPipeline as the core "
            "must be present. Current honest_phrases: "
            f"{honest_phrases}"
        )

    def test_jesse_agents_mentioned(self, story):
        for m in ('5 Jesse', 'JesseContextAgent', 'JesseOrchestrator',
                   'jesse_agents.py'):
            assert m in story, f"Jesse reference missing: {m!r}"


class TestIntegrationGapAcknowledged:
    """Part 3 / 4 must state that numbers come from NYXPipeline direct,
    NOT from HubSpokeRunner / PortfolioAllocator."""

    def test_gap_statement(self, story):
        # Accept several equivalent wordings.
        gap_phrases = (
            "Paper-live and hub-and-spoke layers are NOT yet wired to NYXPipeline",
            "Paper-live and hub-and-spoke layers are not yet wired to NYXPipeline",
            "numbers come from `NYXPipeline.run()` direct",
            "All CAGR/MC/BS numbers come from `NYXPipeline.run()` direct",
            "not through HubSpokeRunner",
            "never touched a real trade",
        )
        assert any(p in story for p in gap_phrases), (
            "The doc must explicitly acknowledge the integration gap "
            "(numbers from NYXPipeline direct, not HubSpokeRunner). "
            f"Acceptable phrasings: {gap_phrases}"
        )

    def test_jesse_alt_status(self, story):
        alt_phrases = (
            "alternative modular architecture",
            "not integrated",
            "not currently wired",
            "alternative archi",
        )
        assert any(p in story for p in alt_phrases), (
            "5 Jesse agents status must be labeled as alternative / "
            "not integrated. Accepted phrasings: " + str(alt_phrases)
        )


class TestLiveGapAcknowledged:
    """The doc must acknowledge that NYXPipeline is batch-only and that a
    real live moteur requires a separate NYXLiveDecider."""

    def test_batch_vs_live_mentioned(self, story):
        phrases = (
            "batch",
            "not real-time",
            "NYXLiveDecider",
            "live decider",
        )
        hits = sum(1 for p in phrases if p.lower() in story.lower())
        assert hits >= 2, (
            f"Doc must discuss batch-vs-live gap; found {hits} markers. "
            f"Looked for: {phrases}"
        )


class TestPartOrdering:
    """Parts 0 → 4 must appear in order (0 = phase 0, 4 = honesty pass)."""

    def test_parts_in_order(self, story):
        last = -1
        for i in range(5):  # Part 0, 1, 2, 3, 4
            header = f"## Part {i}"
            idx = story.find(header)
            assert idx > last, (
                f"Part {i} missing or out of order "
                f"(idx={idx}, last={last})"
            )
            last = idx
