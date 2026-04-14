"""
TDD Tests — Jesse agents documented status.

Task (c). The 5 Jesse agents (`src/ml/jesse_agents.py`) are kept in
the tree as an alternative modular architecture. They are NOT wired
into the production NYXPipeline path. This test guarantees the
status is explicitly documented in two places :

  1. A `STATUS:` line in the file header of `src/ml/jesse_agents.py`.
  2. A dedicated `docs/JESSE_AGENTS_STATUS.md` that explains the
     relationship to `NYXPipeline` and lists the 45 existing tests.
"""
from pathlib import Path

import pytest


AGENTS_PY = Path(__file__).parent.parent / 'src' / 'ml' / 'jesse_agents.py'
STATUS_DOC = Path(__file__).parent.parent / 'docs' / 'JESSE_AGENTS_STATUS.md'


# ===========================================================================
class TestAgentsFileHeader:

    def test_file_exists(self):
        assert AGENTS_PY.exists(), f"{AGENTS_PY} missing"

    def test_header_has_status_line(self):
        text = AGENTS_PY.read_text()
        # Header status must be within the first 50 lines.
        head = '\n'.join(text.splitlines()[:50])
        assert 'STATUS:' in head, \
            "jesse_agents.py must have a 'STATUS:' line in its header"

    def test_header_mentions_not_wired(self):
        text = AGENTS_PY.read_text()
        head = '\n'.join(text.splitlines()[:50])
        phrases = (
            'not currently wired',
            'not wired',
            'alternative modular architecture',
            'NOT currently wired',
        )
        assert any(p in head for p in phrases), (
            "jesse_agents.py header must say the agents are not wired "
            "into production (so future readers know they're alternative)"
        )

    def test_header_points_to_status_doc(self):
        text = AGENTS_PY.read_text()
        head = '\n'.join(text.splitlines()[:50])
        assert 'JESSE_AGENTS_STATUS.md' in head, \
            "header must reference docs/JESSE_AGENTS_STATUS.md"


# ===========================================================================
class TestStatusDoc:

    def test_doc_exists(self):
        assert STATUS_DOC.exists(), f"{STATUS_DOC} missing"

    @pytest.fixture(scope='class')
    def doc(self):
        return STATUS_DOC.read_text()

    def test_doc_names_all_5_agents(self, doc):
        for agent in ('JesseContextAgent', 'JesseRegimeAgent',
                       'JesseSetupAgent', 'JesseEntryAgent',
                       'JesseOrchestrator'):
            assert agent in doc, f"agent {agent!r} must be named"

    def test_doc_mentions_nyxpipeline_relationship(self, doc):
        """Doc must explain how the agents relate to NYXPipeline."""
        for marker in ('NYXPipeline', 'monolithic', 'alternative'):
            assert marker in doc, f"doc must contain {marker!r}"

    def test_doc_counts_existing_tests(self, doc):
        """Doc must say how many tests currently pass."""
        assert '45' in doc, \
            "doc must mention the 45 existing Jesse agent tests"

    def test_doc_has_how_to_swap_section(self, doc):
        """Explains what would be needed to swap NYXPipeline → 5 agents."""
        phrases = (
            'how to swap',
            'how to wire',
            'to integrate',
            'to wire',
            'to replace',
        )
        assert any(p.lower() in doc.lower() for p in phrases), (
            "doc must include a section explaining how one would "
            "integrate the 5 agents (even if not currently planned)"
        )

    def test_doc_notes_shared_utilities(self, doc):
        """Both systems share `jesse_features.py` + `soft_gate.py`."""
        assert 'jesse_features' in doc
        assert 'soft_gate' in doc or 'compute_disagreement' in doc
