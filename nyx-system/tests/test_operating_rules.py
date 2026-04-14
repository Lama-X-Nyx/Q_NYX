"""
TDD Tests — docs/OPERATING_RULES.md presence guard.

These tests are **documentary** (presence of rule text), not
behavioural. They protect against accidental removal or silent
watering-down of a permanent rule in a future refactor.

Every rule enumerated here must remain present with its exact header
and a canonical marker phrase.
"""
from pathlib import Path

import pytest


RULES_PATH = (Path(__file__).parent.parent / 'docs' / 'OPERATING_RULES.md')


@pytest.fixture(scope='module')
def rules_text() -> str:
    assert RULES_PATH.exists(), f"{RULES_PATH} missing"
    return RULES_PATH.read_text()


# ===========================================================================
class TestPermanentRulesPresence:
    """Each rule header must be present with its canonical wording."""

    def test_rule_1_tdd_strict(self, rules_text):
        assert "## Rule 1 — TDD strict" in rules_text
        assert "RED tests first" in rules_text

    def test_rule_2_mtf_training(self, rules_text):
        assert "## Rule 2 — Training is multi-timeframe" in rules_text
        assert "4 timeframes" in rules_text
        # 4 TF prefixes must be named
        for tag in ('15m', '1h', '4h', '1d', 'h1_', 'h4_', 'd1_'):
            assert tag in rules_text, f"prefix {tag!r} missing"

    def test_rule_3_honest_execution(self, rules_text):
        assert "## Rule 3 — Honest execution" in rules_text
        assert "PostOnlyPaperBroker" in rules_text
        assert "No silent taker fallback" in rules_text or \
               "No taker fallback" in rules_text or \
               "no silent taker fallback" in rules_text.lower()

    def test_rule_4_no_silent_data_loss(self, rules_text):
        assert "## Rule 4 — No silent data loss" in rules_text
        assert "PersistentDecisionLogger" in rules_text
        assert "StateManager" in rules_text

    def test_rule_5_events_alerted(self, rules_text):
        assert "## Rule 5 — Every operator-relevant event is alerted" in rules_text
        for event in ('service_down', 'reconnect_exchange', 'trade_decision',
                       'order_unfilled', 'api_error', 'restart'):
            assert event in rules_text, f"event {event!r} missing"

    def test_rule_6_pyright_clean(self, rules_text):
        assert "## Rule 6 — Pyright clean" in rules_text
        assert "0 errors" in rules_text

    def test_rule_7_read_context_first(self, rules_text):
        """NEW — must be present to prevent the Phase-0-ignored mistake."""
        assert "## Rule 7 — Always read session context FIRST" in rules_text, (
            "Rule 7 MUST be present. Without it, a new session can repeat "
            "the mistake of building in parallel to existing work."
        )
        for marker in (
            "docs/BRANCH_STORY.md",
            "docs/OPERATING_RULES.md",
            "git log --oneline origin/main..HEAD",
            "plan file",
            "session summary",
            "Never build in parallel to existing work",
        ):
            assert marker in rules_text, \
                f"Rule 7 marker missing: {marker!r}"


class TestRulesOrdering:
    """Rules must appear in numerical order so readers can scan them."""

    def test_rules_in_order(self, rules_text):
        last_idx = -1
        for i in range(1, 8):
            header = f"## Rule {i} —"
            idx = rules_text.find(header)
            assert idx >= 0, f"{header} missing"
            assert idx > last_idx, f"{header} appears before Rule {i - 1}"
            last_idx = idx


class TestPermanentGuardrailList:
    """The 'Permanent guardrail tests' section must stay present."""

    def test_guardrail_section_exists(self, rules_text):
        assert "Permanent guardrail tests" in rules_text

    def test_cornerstones_listed(self, rules_text):
        for test_file in (
            "tests/test_oos_final.py",
            "tests/test_mtf_4tf_coverage.py",
            "tests/test_post_only_broker.py",
            "tests/test_paper_live_runner.py",
        ):
            assert test_file in rules_text, \
                f"guardrail {test_file!r} missing"
