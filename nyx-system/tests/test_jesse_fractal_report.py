"""
TDD Tests — Jesse agents emit canonical `FractalReport`.

Ticket 05 — Reposition Jesse agents as fractal reporters.

Every Jesse agent must expose a `.report(df, asset, timestamp=None,
**analyze_kwargs) -> FractalReport` method that :

- returns a `FractalReport` (canonical contract from Ticket 03)
- carries the canonical `agent` name ∈ CANONICAL_AGENTS
  ({context, regime, setup, entry})
- carries the canonical `timeframe` per NYX fractal stack
  (Context=1d, Regime=4h, Setup=1h, Entry=15m)
- has score ∈ [0, 1]
- has `passed` / `block_reasons` pair consistent with FractalReport
  validation

The 4 reports together must be able to feed a `MetaDecision`'s
`fractal_reports={'context': ..., 'regime': ..., 'setup': ...,
'entry': ...}` dict — proof of contract compatibility with the
Meta-GBM input layer.

Both implementations are covered :
  - mono-file `src/ml/jesse_agents.py` (Jesse*Agent classes)
  - per-file  `src/agents/*.py` (ContextAgent / RegimeAgent /
    SetupAgent / EntryAgent)

Existing `.analyze()` methods stay intact (45 GREEN tests) ; the new
reporter is additive.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tests.test_jesse_ml_tdd import (
    make_bullish_candles,
    make_mixed_synthetic,
)
from tests.test_jesse_agent_context import make_daily_bull
from tests.test_jesse_agent_regime import (
    make_hourly_bull,
    make_mixed_hourly,
)


# ---------------------------------------------------------------------------
# Canonical (agent, timeframe, impl) combinations.
#
# For each agent we test BOTH implementations :
#   mono  = src/ml/jesse_agents.py::Jesse*Agent
#   peri  = src/agents/*.py       (per-file class)
# ---------------------------------------------------------------------------

def _make_mono_context():
    from src.ml.jesse_agents import JesseContextAgent
    a = JesseContextAgent()
    a.train(make_daily_bull(200))
    return a, make_daily_bull(200), {}


def _make_mono_regime():
    from src.ml.jesse_agents import JesseRegimeAgent
    a = JesseRegimeAgent()
    a.train(make_mixed_hourly())
    return a, make_hourly_bull(200), {}


def _make_mono_setup():
    from src.ml.jesse_agents import JesseSetupAgent
    a = JesseSetupAgent()
    a.train(make_mixed_synthetic(400))
    return a, make_bullish_candles(200), {
        'context_score': 0.8, 'regime_score': 0.7,
    }


def _make_mono_entry():
    from src.ml.jesse_agents import JesseEntryAgent
    a = JesseEntryAgent()
    a.train(make_mixed_synthetic(500))
    return a, make_bullish_candles(200), {}


def _make_peri_context():
    from src.agents.context_agent import ContextAgent
    a = ContextAgent({})
    return a, make_daily_bull(200), {}


def _make_peri_regime():
    from src.agents.regime_agent import RegimeAgent
    a = RegimeAgent({})
    return a, make_hourly_bull(200), {}


def _make_peri_setup():
    from src.agents.setup_agent import SetupAgent
    a = SetupAgent({})
    return a, make_bullish_candles(200), {}


def _make_peri_entry():
    from src.agents.entry_agent import EntryAgent
    a = EntryAgent({})
    return a, make_bullish_candles(200), {}


# (id, factory, canonical_agent, canonical_timeframe)
AGENT_CASES = [
    ('mono-context', _make_mono_context, 'context', '1d'),
    ('mono-regime',  _make_mono_regime,  'regime',  '4h'),
    ('mono-setup',   _make_mono_setup,   'setup',   '1h'),
    ('mono-entry',   _make_mono_entry,   'entry',   '15m'),
    ('peri-context', _make_peri_context, 'context', '1d'),
    ('peri-regime',  _make_peri_regime,  'regime',  '4h'),
    ('peri-setup',   _make_peri_setup,   'setup',   '1h'),
    ('peri-entry',   _make_peri_entry,   'entry',   '15m'),
]

CASE_IDS = [c[0] for c in AGENT_CASES]


# ===========================================================================
class TestReporterMethodPresent:
    """Every agent (both implementations) must expose `.report()`."""

    @pytest.mark.parametrize(
        'factory,canonical_agent,canonical_tf',
        [c[1:] for c in AGENT_CASES],
        ids=CASE_IDS,
    )
    def test_agent_has_report_method(
        self, factory, canonical_agent, canonical_tf,
    ):
        agent, df, kw = factory()
        assert hasattr(agent, 'report'), (
            f'{type(agent).__name__} must expose `.report()` '
            '(Ticket 05)'
        )

    @pytest.mark.parametrize(
        'factory,canonical_agent,canonical_tf',
        [c[1:] for c in AGENT_CASES],
        ids=CASE_IDS,
    )
    def test_agent_declares_canonical_report_constants(
        self, factory, canonical_agent, canonical_tf,
    ):
        """Class-level `REPORT_AGENT` and `REPORT_TIMEFRAME` declare
        the canonical identity — independent of any config-driven
        `self.timeframe` used for internal analyze() logic."""
        agent, df, kw = factory()
        cls = type(agent)
        assert getattr(cls, 'REPORT_AGENT', None) == canonical_agent
        assert getattr(cls, 'REPORT_TIMEFRAME', None) == canonical_tf


# ===========================================================================
class TestReportReturnsFractalReport:

    @pytest.mark.parametrize(
        'factory,canonical_agent,canonical_tf',
        [c[1:] for c in AGENT_CASES],
        ids=CASE_IDS,
    )
    def test_returns_fractal_report(
        self, factory, canonical_agent, canonical_tf,
    ):
        from src.agents.contracts import FractalReport
        agent, df, kw = factory()
        r = agent.report(df, asset='BTCUSDT', timestamp='2023-06-15T00:00:00',
                         **kw)
        assert isinstance(r, FractalReport)

    @pytest.mark.parametrize(
        'factory,canonical_agent,canonical_tf',
        [c[1:] for c in AGENT_CASES],
        ids=CASE_IDS,
    )
    def test_report_canonical_agent_name(
        self, factory, canonical_agent, canonical_tf,
    ):
        agent, df, kw = factory()
        r = agent.report(df, asset='BTCUSDT', timestamp='2023-06-15T00:00:00',
                         **kw)
        assert r.agent == canonical_agent

    @pytest.mark.parametrize(
        'factory,canonical_agent,canonical_tf',
        [c[1:] for c in AGENT_CASES],
        ids=CASE_IDS,
    )
    def test_report_canonical_timeframe(
        self, factory, canonical_agent, canonical_tf,
    ):
        agent, df, kw = factory()
        r = agent.report(df, asset='BTCUSDT', timestamp='2023-06-15T00:00:00',
                         **kw)
        assert r.timeframe == canonical_tf

    @pytest.mark.parametrize(
        'factory,canonical_agent,canonical_tf',
        [c[1:] for c in AGENT_CASES],
        ids=CASE_IDS,
    )
    def test_report_score_in_range(
        self, factory, canonical_agent, canonical_tf,
    ):
        agent, df, kw = factory()
        r = agent.report(df, asset='BTCUSDT', timestamp='2023-06-15T00:00:00',
                         **kw)
        assert 0.0 <= r.score <= 1.0

    @pytest.mark.parametrize(
        'factory,canonical_agent,canonical_tf',
        [c[1:] for c in AGENT_CASES],
        ids=CASE_IDS,
    )
    def test_report_asset_preserved(
        self, factory, canonical_agent, canonical_tf,
    ):
        agent, df, kw = factory()
        r = agent.report(df, asset='ETHUSDT', timestamp='2023-06-15T00:00:00',
                         **kw)
        assert r.asset == 'ETHUSDT'


# ===========================================================================
class TestReportFeedsMetaDecision:
    """The 4 reports together must satisfy
    `MetaDecision(fractal_reports={'context': ..., 'regime': ...,
    'setup': ..., 'entry': ...})`. This is the ticket 05 acceptance
    criterion for contract compatibility with Meta-GBM input layer.
    """

    def _collect_mono_reports(self):
        """Build the 4 mono-file reports for a single timestamp."""
        reports = {}
        for case_id, factory, canonical_agent, _ in AGENT_CASES:
            if not case_id.startswith('mono-'):
                continue
            agent, df, kw = factory()
            reports[canonical_agent] = agent.report(
                df, asset='BTCUSDT', timestamp='2023-06-15T00:00:00',
                **kw,
            )
        return reports

    def _collect_peri_reports(self):
        reports = {}
        for case_id, factory, canonical_agent, _ in AGENT_CASES:
            if not case_id.startswith('peri-'):
                continue
            agent, df, kw = factory()
            reports[canonical_agent] = agent.report(
                df, asset='BTCUSDT', timestamp='2023-06-15T00:00:00',
                **kw,
            )
        return reports

    def test_mono_four_reports_feed_meta_decision(self):
        from src.agents.contracts import MetaDecision
        reports = self._collect_mono_reports()
        assert set(reports.keys()) == {'context', 'regime', 'setup', 'entry'}
        # This must NOT raise — proves type compatibility of the
        # fractal_reports dict with MetaDecision's validation.
        dec = MetaDecision(
            asset='BTCUSDT',
            timestamp='2023-06-15T00:00:00',
            timeframe='15m',
            direction=1,
            probability=0.72,
            threshold_used=0.60,
            passed=True,
            block_reasons=[],
            expected_edge_net=45.0,
            candidate_quality=0.72,
            fractal_reports=reports,
            features_snapshot={},
        )
        assert dec.fractal_reports['context'].agent == 'context'
        assert dec.fractal_reports['regime'].timeframe == '4h'
        assert dec.fractal_reports['setup'].timeframe == '1h'
        assert dec.fractal_reports['entry'].timeframe == '15m'

    def test_peri_four_reports_feed_meta_decision(self):
        from src.agents.contracts import MetaDecision
        reports = self._collect_peri_reports()
        assert set(reports.keys()) == {'context', 'regime', 'setup', 'entry'}
        dec = MetaDecision(
            asset='BTCUSDT',
            timestamp='2023-06-15T00:00:00',
            timeframe='15m',
            direction=1,
            probability=0.65,
            threshold_used=0.60,
            passed=True,
            block_reasons=[],
            expected_edge_net=30.0,
            candidate_quality=0.65,
            fractal_reports=reports,
            features_snapshot={},
        )
        assert dec.fractal_reports['context'].timeframe == '1d'


# ===========================================================================
class TestAnalyzeMethodUnaffected:
    """Regression guard : the existing `.analyze()` still returns
    `AgentResult` (45 pre-ticket-05 tests must stay GREEN)."""

    def test_mono_context_analyze_still_agent_result(self):
        from src.ml.jesse_agents import JesseContextAgent
        from src.agents.contracts import AgentResult
        a = JesseContextAgent()
        a.train(make_daily_bull(200))
        result = a.analyze(make_daily_bull(200))
        assert isinstance(result, AgentResult)
        assert result.agent == 'context'

    def test_peri_context_analyze_still_agent_result(self):
        from src.agents.context_agent import ContextAgent
        from src.agents.contracts import AgentResult
        a = ContextAgent({})
        result = a.analyze(make_daily_bull(200))
        assert isinstance(result, AgentResult)
        assert result.agent == 'context'
