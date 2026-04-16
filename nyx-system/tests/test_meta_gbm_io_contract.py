"""
TDD Tests — Meta-GBM I/O contract is FROZEN (Ticket 10).

Per Ticket 10 (also referred to as "Ticket 09 — Define Meta-GBM
inputs and outputs"), the canonical strategy-brain MUST publish a
frozen schema for both inputs and outputs so training, inference,
risk, and execution all agree without ad-hoc translation.

Inputs (frozen) :
- 4 Jesse fractal reports         → `fractal_reports`
- candidate-generation features   → `features`
- market state features           → `features` (merged)
- agreement / disagreement        → derived inside MetaGBM, returned
                                    in `features_snapshot`
- asset / timeframe context       → `asset`, `timestamp`, `timeframe`
- direction hint (hard gate)      → `hint_direction`
- pre-scored proba (batch helper) → `precomputed_proba` (optional)
- raw feature vector (live path)  → `feature_vector` (optional) +
                                    `already_scaled` flag

Outputs (frozen) :
- `trade_decision`           : 'BUY' | 'SELL' | 'WAIT'
- `direction`                : -1 / 0 / +1
- `confidence`               : alias for `probability` ∈ [0, 1]
- `expected_edge`            : alias for `expected_edge_net`
- `trade_quality_bucket`     : alias for `quality_bucket`
                               ∈ {'high', 'medium', 'low'}
- `risk_hint`                : ∈ [0, 1]

These properties live on `MetaDecision` so risk/execution downstream
(`TradePlan`, `ExecutionInstruction`) can consume them with the
exact names from the contract.

Acceptance : the schemas are exposed via `MetaGBM.INPUT_SCHEMA` and
`MetaGBM.OUTPUT_SCHEMA` class-level constants, and `MetaDecision`
exposes the 6 contract output fields as properties (4 are aliases
for backward-compat with existing internal field names).
"""
from __future__ import annotations

import pytest


# ===========================================================================
class TestMetaGBMInputSchema:
    """`MetaGBM.INPUT_SCHEMA` declares every input the .decide() method
    accepts, with type hints. Frozen — additive changes only."""

    def test_class_has_input_schema(self):
        from src.core.meta_gbm import MetaGBM
        assert hasattr(MetaGBM, 'INPUT_SCHEMA'), (
            'MetaGBM must publish INPUT_SCHEMA class attribute '
            '(Ticket 10 frozen contract)'
        )

    def test_input_schema_is_mapping(self):
        from src.core.meta_gbm import MetaGBM
        assert isinstance(MetaGBM.INPUT_SCHEMA, dict)
        assert len(MetaGBM.INPUT_SCHEMA) >= 6

    @pytest.mark.parametrize('key', [
        'fractal_reports',
        'features',
        'asset',
        'timestamp',
        'timeframe',
        'hint_direction',
    ])
    def test_required_input_key_documented(self, key):
        from src.core.meta_gbm import MetaGBM
        assert key in MetaGBM.INPUT_SCHEMA, (
            f'INPUT_SCHEMA must document {key!r} '
            '(Ticket 10 required input)'
        )

    @pytest.mark.parametrize('key', [
        'precomputed_proba',
        'feature_vector',
        'already_scaled',
    ])
    def test_optional_helper_input_documented(self, key):
        from src.core.meta_gbm import MetaGBM
        assert key in MetaGBM.INPUT_SCHEMA, (
            f'INPUT_SCHEMA must also document optional helper {key!r} '
            '(Ticket 07 batch + live shortcuts)'
        )


# ===========================================================================
class TestMetaGBMOutputSchema:
    """`MetaGBM.OUTPUT_SCHEMA` declares the canonical output fields the
    risk / execution layer consumes."""

    def test_class_has_output_schema(self):
        from src.core.meta_gbm import MetaGBM
        assert hasattr(MetaGBM, 'OUTPUT_SCHEMA'), (
            'MetaGBM must publish OUTPUT_SCHEMA class attribute '
            '(Ticket 10 frozen contract)'
        )

    def test_output_schema_is_mapping(self):
        from src.core.meta_gbm import MetaGBM
        assert isinstance(MetaGBM.OUTPUT_SCHEMA, dict)

    @pytest.mark.parametrize('key', [
        'trade_decision',
        'direction',
        'confidence',
        'expected_edge',
        'trade_quality_bucket',
        'risk_hint',
    ])
    def test_required_output_key_documented(self, key):
        from src.core.meta_gbm import MetaGBM
        assert key in MetaGBM.OUTPUT_SCHEMA, (
            f'OUTPUT_SCHEMA must document {key!r} '
            '(Ticket 10 required output)'
        )


# ===========================================================================
def _build_decision(direction: int, passed: bool, probability: float = 0.7,
                     quality_bucket=None, risk_hint=None,
                     edge_net: float = 42.0):
    from src.agents.contracts import MetaDecision
    return MetaDecision(
        asset='BTCUSDT',
        timestamp='2023-06-15T10:15:00',
        timeframe='15m',
        direction=direction,
        probability=probability,
        threshold_used=0.60,
        passed=passed,
        block_reasons=[] if passed else ['stub'],
        expected_edge_net=edge_net,
        candidate_quality=probability,
        fractal_reports={},
        features_snapshot={},
        quality_bucket=quality_bucket,
        risk_hint=risk_hint,
    )


# ===========================================================================
class TestMetaDecisionTradeDecisionProperty:
    """`MetaDecision.trade_decision` ∈ {'BUY', 'SELL', 'WAIT'}.

    Mapping :
      passed=True, direction=+1 → 'BUY'
      passed=True, direction=-1 → 'SELL'
      passed=False OR direction=0 → 'WAIT'
    """

    def test_buy(self):
        d = _build_decision(direction=1, passed=True)
        assert d.trade_decision == 'BUY'

    def test_sell(self):
        d = _build_decision(direction=-1, passed=True)
        assert d.trade_decision == 'SELL'

    def test_wait_when_not_passed(self):
        d = _build_decision(direction=1, passed=False, probability=0.30)
        assert d.trade_decision == 'WAIT'

    def test_wait_when_direction_zero(self):
        # passed must be False if direction==0 (validation rule).
        d = _build_decision(direction=0, passed=False, probability=0.30)
        assert d.trade_decision == 'WAIT'


# ===========================================================================
class TestMetaDecisionAliasProperties:
    """Ticket 10 aliases for the canonical output names. They MUST
    match the underlying internal field exactly (no rounding, no
    transformation)."""

    def test_confidence_aliases_probability(self):
        d = _build_decision(direction=1, passed=True, probability=0.84)
        assert d.confidence == pytest.approx(0.84)
        assert d.confidence == d.probability

    def test_expected_edge_aliases_expected_edge_net(self):
        d = _build_decision(direction=1, passed=True, edge_net=123.5)
        assert d.expected_edge == pytest.approx(123.5)
        assert d.expected_edge == d.expected_edge_net

    def test_trade_quality_bucket_aliases_quality_bucket(self):
        d = _build_decision(
            direction=1, passed=True, quality_bucket='medium',
        )
        assert d.trade_quality_bucket == 'medium'
        assert d.trade_quality_bucket == d.quality_bucket

    def test_risk_hint_already_canonical(self):
        d = _build_decision(direction=1, passed=True, risk_hint=0.42)
        assert d.risk_hint == pytest.approx(0.42)


# ===========================================================================
class TestRiskExecutionConsumesWithoutTranslation:
    """The risk + execution layer (`TradePlan`, `ExecutionInstruction`)
    must be constructible from a `MetaDecision` using only its
    canonical Ticket-10 outputs — no ad-hoc field renaming."""

    def test_trade_plan_takes_meta_decision_direction(self):
        from src.agents.contracts import TradePlan
        dec = _build_decision(
            direction=1, passed=True, probability=0.72,
            quality_bucket='high', risk_hint=0.9,
        )
        # Risk layer fills entry/stop/TP from market data — but the
        # DIRECTION must come straight from MetaDecision, no translation.
        plan = TradePlan(
            asset=dec.asset, timestamp=dec.timestamp,
            direction=dec.direction,           # <-- canonical
            size_fraction=0.02,
            entry_price=1500.0, stop_loss=1485.0, take_profit=1530.0,
            expected_hold_bars=50,
            source=dec,
        )
        assert plan.direction == dec.direction
        assert plan.source.confidence == dec.confidence
        assert plan.source.trade_quality_bucket == dec.trade_quality_bucket

    def test_execution_instruction_side_from_trade_decision(self):
        """`ExecutionInstruction.side` must be derivable from
        `MetaDecision.trade_decision` without ad-hoc mapping logic."""
        from src.agents.contracts import ExecutionInstruction, TradePlan
        dec = _build_decision(direction=1, passed=True, probability=0.72)
        plan = TradePlan(
            asset=dec.asset, timestamp=dec.timestamp,
            direction=dec.direction, size_fraction=0.02,
            entry_price=1500.0, stop_loss=1485.0, take_profit=1530.0,
            expected_hold_bars=50, source=dec,
        )
        # Canonical mapping :
        #   trade_decision='BUY'  → side='buy'
        #   trade_decision='SELL' → side='sell'
        side = dec.trade_decision.lower()
        instr = ExecutionInstruction(
            asset=dec.asset, timestamp=dec.timestamp,
            side=side, quantity=0.1,
            order_type='post_only_limit', limit_price=1498.5,
            max_wait_bars=4, source=plan,
        )
        assert instr.side == 'buy'


# ===========================================================================
class TestMetaGBMDecideEmitsCanonicalOutputs:
    """When MetaGBM.decide() is called, the returned MetaDecision
    must expose all 6 Ticket 10 output properties."""

    def _four_reports(self):
        from src.agents.contracts import FractalReport
        defaults = [
            ('context', '1d'), ('regime', '4h'),
            ('setup', '1h'), ('entry', '15m'),
        ]
        return {
            agent: FractalReport(
                asset='BTCUSDT', agent=agent, timeframe=tf,
                state='neutral', score=0.7, passed=True,
                block_reasons=[], timestamp='2023-06-15T10:15:00',
            ) for agent, tf in defaults
        }

    def test_decide_output_has_all_six_canonical_fields(self):
        from src.core.meta_gbm import MetaGBM
        brain = MetaGBM(threshold=0.40)
        dec = brain.decide(
            fractal_reports=self._four_reports(),
            features={}, asset='BTCUSDT',
            timestamp='2023-06-15T10:15:00',
            timeframe='15m', hint_direction=1,
        )
        # All 6 Ticket-10 outputs must be reachable.
        assert dec.trade_decision in ('BUY', 'SELL', 'WAIT')
        assert dec.direction in (-1, 0, 1)
        assert 0.0 <= dec.confidence <= 1.0
        assert isinstance(dec.expected_edge, float)
        assert dec.trade_quality_bucket in ('high', 'medium', 'low')
        assert dec.risk_hint is not None
        assert 0.0 <= dec.risk_hint <= 1.0
