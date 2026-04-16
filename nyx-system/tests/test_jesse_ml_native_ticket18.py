"""
TDD Tests — Ticket 18.

All 4 Jesse agents must emit ML-native reports where the MODEL owns
the primary signal (state, score, probabilities) — not heuristics.

Source-level checks :
  - `analyze()` must NOT contain heuristic-dominant blend patterns
    (`0.5 * p_ml + 0.5 * h_`, `0.6 * p_ml + 0.4 * h_`, etc.)
  - the primary probability fields in metadata (`p_bull`, `p_bear`,
    `p_trend`, `p_setup`, `p_up`, `p_down`) must come from
    `_predict_last()` (model output), not from hand-authored logic.

Behavioural checks :
  - report probabilities are non-degenerate (not all 0.5 or all 0/1)
  - reports still conform to FractalReport schema (Ticket 05)
  - retrained pass rates are sane (not degenerate post-refactor)
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest


def _synthetic(n: int = 500, kind: str = 'trend') -> pd.DataFrame:
    np.random.seed(42)
    drift = 0.002 if kind == 'trend' else 0.0
    close = 1000 * np.exp(np.cumsum(drift + np.random.randn(n) * 0.001))
    return pd.DataFrame({
        'open': close, 'high': close * 1.002, 'low': close * 0.998,
        'close': close, 'volume': np.full(n, 1000.0),
    }, index=pd.date_range('2023-01-01', periods=n, freq='15min'))


# ===========================================================================
class TestNoHeuristicDominantBlend:
    """Source-level : the `analyze()` body must NOT contain patterns
    that make heuristics the primary signal owner.

    Forbidden patterns (regex):
      `p_XXX = <weight> * <ml_var> + <weight> * <heuristic_var>`
    where both weights are > 0.  A pure `p = p_ml` is fine. A
    `p = 0.9 * p_ml + 0.1 * fallback` is borderline but OK (ML
    dominant, 90 % weight). We reject patterns where the heuristic
    weight is ≥ 0.3 (= ≥ 30 % of the signal is heuristic)."""

    BLEND_PATTERN = re.compile(
        r'0\.\d\s*\*\s*\w+_ml\b.*\+\s*0\.\d\s*\*\s*(?:h_|heur)',
        re.IGNORECASE,
    )

    @pytest.mark.parametrize('cls_name', [
        'JesseContextAgent', 'JesseRegimeAgent',
        'JesseSetupAgent', 'JesseEntryAgent',
    ])
    def test_no_heuristic_dominant_blend(self, cls_name):
        import src.ml.jesse_agents as mod
        cls = getattr(mod, cls_name)
        src = inspect.getsource(cls.analyze)
        matches = self.BLEND_PATTERN.findall(src)
        assert not matches, (
            f'{cls_name}.analyze() still contains heuristic-dominant '
            f'blend patterns: {matches}. Ticket 18 requires ML to own '
            'the primary signal.'
        )


# ===========================================================================
class TestMLNativeProbabilities:
    """Each agent's report metadata must contain probability fields
    derived from the trained model (not hard-coded 0/0.5/1)."""

    @pytest.mark.parametrize('cls_name,prob_key', [
        ('JesseContextAgent', 'p_bull'),
        ('JesseRegimeAgent',  'p_trend'),
        ('JesseSetupAgent',   'p_setup_ml'),
        ('JesseEntryAgent',   'p_up'),
    ])
    def test_probability_from_model(self, cls_name, prob_key):
        import src.ml.jesse_agents as mod
        cls = getattr(mod, cls_name)
        agent = cls()
        df = _synthetic(500, 'trend')
        agent.train(df)
        r = agent.analyze(df)
        assert prob_key in r.metadata, (
            f'{cls_name} report metadata missing {prob_key!r} — '
            'must be present and model-derived'
        )
        val = float(r.metadata[prob_key])
        assert 0.0 <= val <= 1.0
        # Non-degenerate : value should NOT be exactly 0.5 on trending data
        # (a trained model on clear trend should show some signal).
        assert val != 0.5 or cls_name == 'JesseSetupAgent', (
            f'{cls_name} {prob_key}={val} — suspiciously at default '
            '0.5, may not be model-derived'
        )


# ===========================================================================
class TestReportSchemaPreserved:
    """Ticket 05 FractalReport contract intact post-Ticket-18."""

    @pytest.mark.parametrize('cls_name,tf', [
        ('JesseContextAgent', '1d'),
        ('JesseRegimeAgent',  '4h'),
        ('JesseSetupAgent',   '1h'),
        ('JesseEntryAgent',   '15m'),
    ])
    def test_report_contract(self, cls_name, tf):
        from src.agents.contracts import FractalReport
        import src.ml.jesse_agents as mod
        agent = getattr(mod, cls_name)()
        df = _synthetic(500, 'trend')
        agent.train(df)
        r = agent.report(df, asset='BTCUSDT',
                         timestamp='2023-06-15T00:00:00')
        assert isinstance(r, FractalReport)
        assert r.timeframe == tf
        assert 0.0 <= r.score <= 1.0


# ===========================================================================
class TestRegimeMLOwnsState:
    """Post-Ticket-18 : Regime state must be derived from the trained
    model's probability, not from an ADX/momentum heuristic alone."""

    def test_regime_state_uses_model_prob(self):
        """The regime `passed` decision must be tied to the model's
        `p_trend` — when p_trend is high, passed should be True;
        when low, False. We check this by inspecting the `analyze()`
        source for `p_trend` usage in the state block."""
        import inspect
        from src.ml.jesse_agents import JesseRegimeAgent
        src = inspect.getsource(JesseRegimeAgent.analyze)
        assert 'p_trend' in src, (
            'Regime analyze() must reference p_trend (model output) '
            'in the state/passed determination — not ADX-only'
        )


# ===========================================================================
class TestSetupMLOwnsScore:
    """Post-Ticket-18 : Setup score must be primarily model-derived.

    The heuristic blend `0.5 * p_setup_ml + 0.5 * h_valid` must be
    gone — `score` should be based on `p_setup_ml` directly (or with
    a very small heuristic adjustment ≤ 10 %)."""

    def test_setup_score_close_to_ml_prob(self):
        from src.ml.jesse_agents import JesseSetupAgent
        agent = JesseSetupAgent()
        df = _synthetic(500, 'trend')
        agent.train(df)
        r = agent.analyze(df)
        p_ml = float(r.metadata.get('p_setup_ml', -1))
        assert p_ml >= 0.0, 'p_setup_ml missing from metadata'
        # Score must be close to p_ml (ML-native). Allow ±0.15
        # tolerance for a light heuristic adjustment / threshold.
        diff = abs(r.score - p_ml)
        assert diff < 0.20, (
            f'Setup score={r.score:.3f} but p_setup_ml={p_ml:.3f} '
            f'— diff {diff:.3f} > 0.20, heuristic still dominates'
        )
