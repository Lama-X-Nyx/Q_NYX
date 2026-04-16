"""
TDD Tests — `src/core/meta_gbm.py::MetaGBM` (Ticket 06).

The MetaGBM is the canonical **strategy brain** that replaces the
vote-based "ALL must pass / any block → WAIT" logic of
`JesseOrchestrator` and `Orchestrator`. It interprets :

- 4 FractalReports (context / regime / setup / entry)
- agreement / disagreement across reports
- context strength, setup quality, entry timing quality
- edge features, market state features

and emits a canonical `MetaDecision` (Ticket 03) with :

- trade / no-trade   (→ `passed` + `direction`)
- direction          (→ `direction`)
- confidence         (→ `probability`)
- quality bucket     (→ `quality_bucket` ∈ {'high', 'medium', 'low'})
- risk hint          (→ `risk_hint` ∈ [0, 1])

Core semantic shift (Ticket 06 acceptance) :

- **"all must pass" is no longer the main strategy rule.** A
  MetaDecision can still `passed=True` when 1 or 2 reports have
  `passed=False`, provided the aggregate score + direction clear
  the threshold.
- **Disagreement becomes a feature, not an automatic failure.**
  It appears in `features_snapshot['disagreement']` and moderates
  `probability` and `risk_hint` — but does not hard-block.
- **The final decision is model-driven and strategy-aware** —
  even without retraining the GBM (out of scope per ticket), the
  decision logic is a principled aggregation rule that respects
  the FractalReport schema.
"""
from __future__ import annotations

from typing import Dict

import pytest


def _fr(agent: str, tf: str, state: str = 'neutral',
        score: float = 0.5, passed: bool = True,
        block_reasons=None, asset: str = 'BTCUSDT',
        timestamp: str = '2023-06-15T10:15:00'):
    """Build a valid FractalReport quickly."""
    from src.agents.contracts import FractalReport
    return FractalReport(
        asset=asset, agent=agent, timeframe=tf, state=state,
        score=score, passed=passed,
        block_reasons=list(block_reasons or ([] if passed else ['stub-block'])),
        timestamp=timestamp,
    )


def _four_reports(**scores_and_passes):
    """Build the canonical 4-report dict.

    Usage :
        _four_reports(context=(0.8, True), regime=(0.7, True),
                      setup=(0.6, True), entry=(0.7, True))
    """
    defaults = {
        'context': (0.7, True, '1d'),
        'regime':  (0.7, True, '4h'),
        'setup':   (0.7, True, '1h'),
        'entry':   (0.7, True, '15m'),
    }
    out = {}
    for agent, (def_score, def_pass, canon_tf) in defaults.items():
        if agent in scores_and_passes:
            score, passed = scores_and_passes[agent]
        else:
            score, passed = def_score, def_pass
        out[agent] = _fr(agent, canon_tf, score=score, passed=passed)
    return out


# ===========================================================================
class TestMetaGBMConstruction:

    def test_construct_defaults(self):
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM()
        assert brain.threshold == pytest.approx(0.60)

    def test_construct_custom_threshold(self):
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM(threshold=0.70)
        assert brain.threshold == pytest.approx(0.70)


# ===========================================================================
class TestMetaGBMOutputShape:
    """MetaGBM.decide(...) must always return a canonical MetaDecision."""

    def test_decide_returns_meta_decision(self):
        from src.core.meta_gbm import MetaGBM
        from src.agents.contracts import MetaDecision
        brain = MetaGBM()
        dec = brain.decide(
            fractal_reports=_four_reports(),
            features={},
            asset='BTCUSDT',
            timestamp='2023-06-15T10:15:00',
            timeframe='15m',
            hint_direction=1,
        )
        assert isinstance(dec, MetaDecision)

    def test_emits_quality_bucket_and_risk_hint(self):
        """Ticket 06 expects MetaGBM to fill the new MetaDecision
        optional outputs `quality_bucket` + `risk_hint`."""
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM()
        dec = brain.decide(
            fractal_reports=_four_reports(),
            features={}, asset='BTCUSDT', timestamp='2023-06-15T10:15:00',
            timeframe='15m', hint_direction=1,
        )
        assert dec.quality_bucket in ('high', 'medium', 'low')
        assert dec.risk_hint is not None
        assert 0.0 <= dec.risk_hint <= 1.0


# ===========================================================================
class TestNotAllMustPass:
    """ACCEPTANCE CRITERION 1: ALL-MUST-PASS is no longer the main rule."""

    def test_passes_with_one_agent_blocked_if_score_holds(self):
        """3 of 4 reports passed + strong aggregate score + clear
        direction → MetaDecision.passed=True.

        (Under the old JesseOrchestrator semantic, this would be
        WAIT because `any block → wait`.)
        """
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM(threshold=0.60)
        reports = _four_reports(
            context=(0.85, True),
            regime=(0.80, True),
            setup=(0.75, True),
            entry=(0.70, False),   # one agent blocks
        )
        dec = brain.decide(
            fractal_reports=reports,
            features={}, asset='BTCUSDT',
            timestamp='2023-06-15T10:15:00',
            timeframe='15m', hint_direction=1,
        )
        assert dec.passed is True, (
            f'MetaGBM should still pass with 3/4 agents passed and '
            f'aggregate score ~0.77 > threshold 0.60, got '
            f'passed={dec.passed}, block_reasons={dec.block_reasons}'
        )
        assert dec.direction == 1

    def test_passes_with_two_agents_blocked_if_score_holds(self):
        """2 of 4 blocked + very high score on the other 2 → still
        trade. This is the sharper test for non-vote-based logic."""
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM(threshold=0.60)
        reports = _four_reports(
            context=(0.95, True),
            regime=(0.95, True),
            setup=(0.50, False),
            entry=(0.50, False),
        )
        dec = brain.decide(
            fractal_reports=reports,
            features={}, asset='BTCUSDT',
            timestamp='2023-06-15T10:15:00',
            timeframe='15m', hint_direction=1,
        )
        # Aggregate = (0.95+0.95+0.50+0.50)/4 = 0.725 > 0.60 threshold
        # → should still pass (vote-based would have said WAIT).
        assert dec.passed is True

    def test_blocks_when_direction_is_zero_regardless_of_scores(self):
        """Even with all 4 agents passing, if there's no directional
        hint (hint_direction=0 and no consensus direction),
        MetaDecision must not pass."""
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM(threshold=0.60)
        dec = brain.decide(
            fractal_reports=_four_reports(),
            features={}, asset='BTCUSDT',
            timestamp='2023-06-15T10:15:00',
            timeframe='15m', hint_direction=0,
        )
        assert dec.passed is False
        assert dec.direction == 0
        assert dec.block_reasons  # non-empty

    def test_blocks_when_probability_below_threshold(self):
        """Even with direction=1 and all passed=True, if aggregate
        score is below threshold, MetaDecision must block."""
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM(threshold=0.80)
        reports = _four_reports(
            context=(0.55, True),
            regime=(0.50, True),
            setup=(0.50, True),
            entry=(0.50, True),
        )
        dec = brain.decide(
            fractal_reports=reports,
            features={}, asset='BTCUSDT',
            timestamp='2023-06-15T10:15:00',
            timeframe='15m', hint_direction=1,
        )
        assert dec.passed is False
        assert dec.probability < 0.80
        assert any('threshold' in r.lower() for r in dec.block_reasons)


# ===========================================================================
class TestDisagreementIsFeature:
    """ACCEPTANCE CRITERION 2: disagreement is a feature, not an
    automatic failure."""

    def test_disagreement_in_features_snapshot(self):
        """The MetaDecision.features_snapshot must carry a
        `disagreement` numeric feature (0 = perfect agreement,
        1 = all block)."""
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM()
        reports = _four_reports(
            context=(0.7, True), regime=(0.7, True),
            setup=(0.6, False), entry=(0.7, True),
        )
        dec = brain.decide(
            fractal_reports=reports,
            features={}, asset='BTCUSDT',
            timestamp='2023-06-15T10:15:00',
            timeframe='15m', hint_direction=1,
        )
        assert 'disagreement' in dec.features_snapshot
        # 1 out of 4 blocked → disagreement = 0.25
        assert dec.features_snapshot['disagreement'] == pytest.approx(0.25)

    def test_disagreement_moderates_probability(self):
        """Two decisions with same aggregate score but different
        disagreement levels must produce different probabilities —
        lower when disagreement higher."""
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM(threshold=0.40)   # low threshold to let both pass

        # Perfect agreement: all 4 passed, all 0.70.
        reports_all_pass = _four_reports(
            context=(0.70, True), regime=(0.70, True),
            setup=(0.70, True), entry=(0.70, True),
        )
        # Same aggregate score but 2 blocked.
        reports_mixed = _four_reports(
            context=(0.70, True), regime=(0.70, True),
            setup=(0.70, False), entry=(0.70, False),
        )
        dec_agree = brain.decide(
            fractal_reports=reports_all_pass, features={},
            asset='X', timestamp='t', timeframe='15m', hint_direction=1,
        )
        dec_mixed = brain.decide(
            fractal_reports=reports_mixed, features={},
            asset='X', timestamp='t', timeframe='15m', hint_direction=1,
        )
        assert dec_agree.probability > dec_mixed.probability, (
            'disagreement must reduce probability (not hard-block)'
        )


# ===========================================================================
class TestQualityBucketMapping:
    """Quality bucket maps aggregate score → {high, medium, low}."""

    def test_high_quality(self):
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM()
        reports = _four_reports(
            context=(0.90, True), regime=(0.90, True),
            setup=(0.90, True), entry=(0.90, True),
        )
        dec = brain.decide(
            fractal_reports=reports, features={}, asset='X',
            timestamp='t', timeframe='15m', hint_direction=1,
        )
        assert dec.quality_bucket == 'high'

    def test_medium_quality(self):
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM()
        reports = _four_reports(
            context=(0.65, True), regime=(0.65, True),
            setup=(0.65, True), entry=(0.65, True),
        )
        dec = brain.decide(
            fractal_reports=reports, features={}, asset='X',
            timestamp='t', timeframe='15m', hint_direction=1,
        )
        assert dec.quality_bucket == 'medium'

    def test_low_quality(self):
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM(threshold=0.30)
        reports = _four_reports(
            context=(0.35, True), regime=(0.35, True),
            setup=(0.35, True), entry=(0.35, True),
        )
        dec = brain.decide(
            fractal_reports=reports, features={}, asset='X',
            timestamp='t', timeframe='15m', hint_direction=1,
        )
        assert dec.quality_bucket == 'low'


# ===========================================================================
class TestRiskHint:
    """Risk hint [0, 1] — high when reports agree, low when mixed."""

    def test_risk_hint_highest_on_full_agreement(self):
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM()
        reports = _four_reports()
        dec = brain.decide(
            fractal_reports=reports, features={}, asset='X',
            timestamp='t', timeframe='15m', hint_direction=1,
        )
        assert dec.risk_hint == pytest.approx(1.0)

    def test_risk_hint_decreases_with_disagreement(self):
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM()
        reports_mixed = _four_reports(
            context=(0.70, True), regime=(0.70, False),
            setup=(0.70, False), entry=(0.70, True),
        )
        dec = brain.decide(
            fractal_reports=reports_mixed, features={}, asset='X',
            timestamp='t', timeframe='15m', hint_direction=1,
        )
        # 2 blocked / 4 = 0.5 disagreement → risk_hint = 0.5
        assert dec.risk_hint == pytest.approx(0.5)


# ===========================================================================
class TestLegacyOrchestratorDeprecated:
    """Acceptance: the vote-based orchestrators are documented as
    deprecated (docstring marker). Code stays for backward compat
    but must carry the deprecation note."""

    def test_jesse_orchestrator_has_deprecation_note(self):
        import src.ml.jesse_agents as mod
        assert mod.JesseOrchestrator.__doc__ is not None
        doc = mod.JesseOrchestrator.__doc__
        assert 'deprecated' in doc.lower() or 'ticket 06' in doc.lower(), (
            'JesseOrchestrator docstring must mark the vote-based logic '
            'as deprecated (Ticket 06) — the canonical strategy brain '
            'is now src.core.meta_gbm.MetaGBM'
        )

    def test_orchestrator_has_deprecation_note(self):
        import src.agents.orchestrator as mod
        assert mod.Orchestrator.__doc__ is not None
        doc = mod.Orchestrator.__doc__
        assert 'deprecated' in doc.lower() or 'ticket 06' in doc.lower(), (
            'Orchestrator docstring must mark the vote-based logic '
            'as deprecated (Ticket 06)'
        )
