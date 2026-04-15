"""
TDD Tests — canonical cross-layer contracts.

Ticket 03 — Unify contracts and system language.

4 canonical types that every layer uses the same way :

  FractalReport      — what a fractal agent (per timeframe) emits
  MetaDecision       — what the strategy brain (Meta-GBM) emits per bar
  TradePlan          — risk-sized + stop/TP resolved (ready to execute)
  ExecutionInstruction — broker-ready order payload

Chain :

  FractalReport(×N agents/TF)
      ↓  aggregated by Meta-GBM
  MetaDecision
      ↓  gated by risk + sized
  TradePlan
      ↓  translated by execution layer
  ExecutionInstruction

These tests enforce the canonical field set, validation rules, and
that adapters from legacy `AgentResult` / `Signal` produce equivalent
canonical values.
"""
from __future__ import annotations

from datetime import datetime

import pytest


# ===========================================================================
class TestFractalReport:
    """Canonical per-timeframe, per-agent report."""

    def _valid_kwargs(self):
        return dict(
            asset='ETHUSDT',
            agent='context',
            timeframe='1d',
            state='bullish',
            score=0.72,
            passed=True,
            block_reasons=[],
            timestamp='2023-01-01T10:00:00',
            metadata={'ema_spread': 0.01},
        )

    def test_valid_construction(self):
        from src.agents.contracts import FractalReport
        r = FractalReport(**self._valid_kwargs())
        assert r.asset == 'ETHUSDT'
        assert r.agent == 'context'
        assert r.timeframe == '1d'
        assert r.score == 0.72
        assert r.passed is True
        assert r.block_reasons == []

    def test_rejects_unknown_agent(self):
        from src.agents.contracts import FractalReport
        kw = self._valid_kwargs()
        kw['agent'] = 'macro'
        with pytest.raises((ValueError, AssertionError)):
            FractalReport(**kw)

    def test_rejects_unknown_timeframe(self):
        from src.agents.contracts import FractalReport
        kw = self._valid_kwargs()
        kw['timeframe'] = '2h'
        with pytest.raises((ValueError, AssertionError)):
            FractalReport(**kw)

    def test_score_out_of_range_rejected(self):
        from src.agents.contracts import FractalReport
        kw = self._valid_kwargs()
        kw['score'] = 1.5
        with pytest.raises((ValueError, AssertionError)):
            FractalReport(**kw)

    def test_passed_false_requires_block_reasons(self):
        from src.agents.contracts import FractalReport
        kw = self._valid_kwargs()
        kw['passed'] = False
        kw['block_reasons'] = []
        with pytest.raises((ValueError, AssertionError)):
            FractalReport(**kw)

    def test_to_dict_serialisable(self):
        from src.agents.contracts import FractalReport
        r = FractalReport(**self._valid_kwargs())
        d = r.to_dict()
        for k in ('asset', 'agent', 'timeframe', 'state', 'score',
                  'passed', 'block_reasons', 'timestamp', 'metadata'):
            assert k in d

    def test_from_agent_result_adapter(self):
        """Legacy AgentResult must convert to FractalReport losslessly
        for the shared fields."""
        from src.agents.contracts import AgentResult, FractalReport
        a = AgentResult(
            agent='regime',
            state='trend_plus',
            score=0.66,
            passed=True,
            reason='ADX > 25',
        )
        r = a.to_fractal_report(asset='BTCUSDT', timeframe='1h',
                                timestamp='2023-06-15T14:00:00')
        assert isinstance(r, FractalReport)
        assert r.agent == 'regime'
        assert r.score == 0.66
        assert r.passed is True
        assert r.timeframe == '1h'
        assert r.asset == 'BTCUSDT'


# ===========================================================================
class TestMetaDecision:
    """Canonical strategy-brain output (what the Meta-GBM decides)."""

    def _valid_kwargs(self):
        return dict(
            asset='ETHUSDT',
            timestamp='2023-01-01T10:15:00',
            timeframe='15m',
            direction=1,
            probability=0.72,
            threshold_used=0.60,
            passed=True,
            block_reasons=[],
            expected_edge_net=45.0,           # bps proxy
            candidate_quality=0.72,
            fractal_reports={},
            features_snapshot={'rule_context': 0.55},
        )

    def test_valid_construction(self):
        from src.agents.contracts import MetaDecision
        d = MetaDecision(**self._valid_kwargs())
        assert d.direction == 1
        assert d.probability == 0.72
        assert d.passed is True

    def test_direction_values_validated(self):
        from src.agents.contracts import MetaDecision
        kw = self._valid_kwargs()
        kw['direction'] = 2
        with pytest.raises((ValueError, AssertionError)):
            MetaDecision(**kw)

    def test_probability_range(self):
        from src.agents.contracts import MetaDecision
        kw = self._valid_kwargs()
        kw['probability'] = 1.2
        with pytest.raises((ValueError, AssertionError)):
            MetaDecision(**kw)

    def test_passed_true_requires_nonzero_direction(self):
        from src.agents.contracts import MetaDecision
        kw = self._valid_kwargs()
        kw['direction'] = 0
        kw['passed'] = True
        with pytest.raises((ValueError, AssertionError)):
            MetaDecision(**kw)

    def test_passed_false_requires_block_reasons(self):
        from src.agents.contracts import MetaDecision
        kw = self._valid_kwargs()
        kw['passed'] = False
        kw['direction'] = 0
        kw['block_reasons'] = []
        with pytest.raises((ValueError, AssertionError)):
            MetaDecision(**kw)

    def test_fractal_reports_keyed_by_agent(self):
        from src.agents.contracts import MetaDecision, FractalReport
        kw = self._valid_kwargs()
        kw['fractal_reports'] = {
            'context': FractalReport(
                asset='ETHUSDT', agent='context', timeframe='1d',
                state='bullish', score=0.8, passed=True,
                block_reasons=[], timestamp='2023-01-01T10:00:00',
            ),
        }
        d = MetaDecision(**kw)
        assert 'context' in d.fractal_reports
        assert d.fractal_reports['context'].score == 0.8


# ===========================================================================
class TestTradePlan:
    """Canonical risk-sized + stop/TP resolved plan (ready for execution)."""

    def _valid_kwargs(self):
        from src.agents.contracts import MetaDecision
        source = MetaDecision(
            asset='ETHUSDT', timestamp='2023-01-01T10:15:00',
            timeframe='15m', direction=1, probability=0.72,
            threshold_used=0.60, passed=True, block_reasons=[],
            expected_edge_net=45.0, candidate_quality=0.72,
            fractal_reports={}, features_snapshot={},
        )
        return dict(
            asset='ETHUSDT',
            timestamp='2023-01-01T10:15:00',
            direction=1,
            size_fraction=0.02,
            entry_price=1500.0,
            stop_loss=1485.0,
            take_profit=1530.0,
            expected_hold_bars=50,
            source=source,
        )

    def test_valid_long_plan(self):
        from src.agents.contracts import TradePlan
        p = TradePlan(**self._valid_kwargs())
        assert p.direction == 1
        assert p.size_fraction == 0.02
        assert p.stop_loss < p.entry_price < p.take_profit

    def test_valid_short_plan(self):
        from src.agents.contracts import TradePlan, MetaDecision
        kw = self._valid_kwargs()
        kw['source'] = MetaDecision(
            asset='ETHUSDT', timestamp='2023-01-01T10:15:00',
            timeframe='15m', direction=-1, probability=0.72,
            threshold_used=0.60, passed=True, block_reasons=[],
            expected_edge_net=45.0, candidate_quality=0.72,
            fractal_reports={}, features_snapshot={},
        )
        kw['direction'] = -1
        kw['stop_loss'] = 1515.0
        kw['take_profit'] = 1470.0
        p = TradePlan(**kw)
        assert p.direction == -1
        assert p.stop_loss > p.entry_price > p.take_profit

    def test_long_plan_rejects_inverted_stop(self):
        """For a long plan, stop_loss must be BELOW entry_price."""
        from src.agents.contracts import TradePlan
        kw = self._valid_kwargs()
        kw['stop_loss'] = 1520.0          # above entry 1500 — invalid for long
        with pytest.raises((ValueError, AssertionError)):
            TradePlan(**kw)

    def test_direction_must_be_actionable(self):
        """TradePlan for direction=0 makes no sense — must reject."""
        from src.agents.contracts import TradePlan
        kw = self._valid_kwargs()
        kw['direction'] = 0
        with pytest.raises((ValueError, AssertionError)):
            TradePlan(**kw)

    def test_source_direction_must_match(self):
        """TradePlan.direction must match source.direction."""
        from src.agents.contracts import TradePlan
        kw = self._valid_kwargs()
        # source.direction is +1 in _valid_kwargs; flip plan direction.
        kw['direction'] = -1
        kw['stop_loss'] = 1515.0
        kw['take_profit'] = 1470.0
        with pytest.raises((ValueError, AssertionError)):
            TradePlan(**kw)

    def test_size_fraction_range(self):
        from src.agents.contracts import TradePlan
        kw = self._valid_kwargs()
        kw['size_fraction'] = 1.5      # > 1 not allowed
        with pytest.raises((ValueError, AssertionError)):
            TradePlan(**kw)


# ===========================================================================
class TestExecutionInstruction:
    """Canonical broker-ready order payload."""

    def _valid_kwargs(self):
        from src.agents.contracts import MetaDecision, TradePlan
        dec = MetaDecision(
            asset='ETHUSDT', timestamp='2023-01-01T10:15:00',
            timeframe='15m', direction=1, probability=0.72,
            threshold_used=0.60, passed=True, block_reasons=[],
            expected_edge_net=45.0, candidate_quality=0.72,
            fractal_reports={}, features_snapshot={},
        )
        plan = TradePlan(
            asset='ETHUSDT', timestamp='2023-01-01T10:15:00',
            direction=1, size_fraction=0.02,
            entry_price=1500.0, stop_loss=1485.0, take_profit=1530.0,
            expected_hold_bars=50, source=dec,
        )
        return dict(
            asset='ETHUSDT',
            timestamp='2023-01-01T10:15:00',
            side='buy',
            quantity=0.1,
            order_type='post_only_limit',
            limit_price=1498.5,       # slightly below mark 1500 for maker buy
            max_wait_bars=4,
            source=plan,
        )

    def test_valid_buy_instruction(self):
        from src.agents.contracts import ExecutionInstruction
        e = ExecutionInstruction(**self._valid_kwargs())
        assert e.side == 'buy'
        assert e.quantity == 0.1
        assert e.order_type == 'post_only_limit'

    def test_side_matches_source_direction(self):
        """side='sell' must have source.direction == -1, else invalid."""
        from src.agents.contracts import ExecutionInstruction
        kw = self._valid_kwargs()
        kw['side'] = 'sell'             # but source.direction is +1
        with pytest.raises((ValueError, AssertionError)):
            ExecutionInstruction(**kw)

    def test_order_type_whitelist(self):
        from src.agents.contracts import ExecutionInstruction
        kw = self._valid_kwargs()
        kw['order_type'] = 'stop_market'
        with pytest.raises((ValueError, AssertionError)):
            ExecutionInstruction(**kw)

    def test_quantity_positive(self):
        from src.agents.contracts import ExecutionInstruction
        kw = self._valid_kwargs()
        kw['quantity'] = -0.1
        with pytest.raises((ValueError, AssertionError)):
            ExecutionInstruction(**kw)


# ===========================================================================
class TestSignalToMetaDecisionAdapter:
    """Existing pod Signal must convert to canonical MetaDecision so
    the hub-spoke layer can keep using Signal but the rest of the
    system speaks MetaDecision."""

    def test_signal_to_meta_decision(self):
        from src.assets.signal import Signal
        from src.agents.contracts import MetaDecision
        sig = Signal(
            symbol='ETHUSDT',
            timestamp='2023-06-15T10:15:00',
            direction=1,
            conviction=0.72,
            expected_edge_net=45.0,
            maker_viability=0.7,
            regime_tag='bull',
            bull_bear_tag='bull',
            size_suggestion=1.0,
            cluster_group='majors',
            expected_hold_bars=50,
        )
        dec = sig.to_meta_decision(threshold_used=0.60, timeframe='15m')
        assert isinstance(dec, MetaDecision)
        assert dec.asset == 'ETHUSDT'
        assert dec.direction == 1
        assert dec.probability == pytest.approx(0.72)
        assert dec.threshold_used == 0.60
        assert dec.passed is True
        assert dec.expected_edge_net == 45.0

    def test_flat_signal_to_flat_decision(self):
        """FLAT Signal (direction=0) produces a passed=False MetaDecision."""
        from src.assets.signal import Signal
        sig = Signal(
            symbol='ETHUSDT', timestamp='2023-06-15T10:15:00',
            direction=0, conviction=0.0, expected_edge_net=0.0,
            maker_viability=0.0, regime_tag='range',
            bull_bear_tag='range', size_suggestion=0.0,
            cluster_group='majors', expected_hold_bars=0,
        )
        dec = sig.to_meta_decision(threshold_used=0.60)
        assert dec.direction == 0
        assert dec.passed is False
        assert dec.block_reasons  # non-empty


# ===========================================================================
class TestContractsImportFromAgentsPackage:
    """Single import point — all 4 canonical types must be importable
    from src.agents.contracts (the natural owner)."""

    def test_four_types_exported(self):
        import src.agents.contracts as c
        for name in (
            'FractalReport', 'MetaDecision', 'TradePlan',
            'ExecutionInstruction',
        ):
            assert hasattr(c, name), f'{name} must be exported'

    def test_legacy_types_still_present(self):
        """Backward-compat: AgentResult + OrchestratorDecision must
        still be importable (existing callers depend on them)."""
        import src.agents.contracts as c
        assert hasattr(c, 'AgentResult')
        assert hasattr(c, 'OrchestratorDecision')
