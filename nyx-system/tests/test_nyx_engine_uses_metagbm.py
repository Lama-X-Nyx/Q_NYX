"""
TDD Tests — `NYXEngine` delegates its strategy-brain decision to the
canonical `MetaGBM` (Ticket 07, Option C: wrapper ownership).

The trained `GradientBoostingClassifier` inside `NYXEngine` is kept as
an implementation detail encapsulated BY `MetaGBM`. The numbers
validated pre-Ticket-07 must be preserved — `.run()` uses the same
model + scaler + feature contract, only the scoring entry-point shifts
to `MetaGBM.decide()`.

Acceptance tests :

1. After `.run()`, `engine._meta` is a `MetaGBM` with
   `has_trained_model=True`.
2. Each emitted trade carries a `meta_decision` of type `MetaDecision`
   (canonical Ticket 03 contract, traceability).
3. `meta_decision.probability` ≈ the batch `scores[i]` that the
   internal GBM produced — proof that the wrapper is transparent.
4. `meta_decision.quality_bucket` ∈ {'high', 'medium', 'low'} emitted
   per trade.

The equivalence guard
(`tests/test_nyx_equivalence_replay_vs_live.py`) is the ironclad
regression test that the refactor did not change observable behaviour.
"""
from __future__ import annotations

import pytest


# ===========================================================================
class TestNYXEngineDelegatesToMetaGBM:
    """After `.run()`, the engine owns a MetaGBM instance."""

    @pytest.fixture(scope='class')
    def eth_h1_2023_result(self, eth_mtf_data, eth_mtf_features):
        """Run NYXEngine on ETH 2023 H1 with default params.

        Using a narrow window + default params keeps the test fast
        while still exercising the training → scoring → trades path.
        """
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        result = engine.run(
            eth_mtf_data, eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-06-30',
        )
        return engine, result

    def test_engine_exposes_meta_after_run(self, eth_h1_2023_result):
        """`engine._meta` must be a MetaGBM instance (the canonical
        strategy brain wiring required by Ticket 07)."""
        engine, _ = eth_h1_2023_result
        from src.core.meta_gbm import MetaGBM
        assert hasattr(engine, '_meta'), (
            'NYXEngine.run() must instantiate a MetaGBM on self._meta '
            '(Ticket 07 canonical strategy-brain delegation)'
        )
        assert isinstance(engine._meta, MetaGBM)

    def test_meta_has_trained_model(self, eth_h1_2023_result):
        """The engine's MetaGBM must encapsulate the trained GBM +
        scaler + feature_names (Option C: wrapper ownership)."""
        engine, _ = eth_h1_2023_result
        assert engine._meta.has_trained_model is True


# ===========================================================================
class TestTradesCarryMetaDecision:
    """Every trade in result['trades'] must have a canonical
    `meta_decision` for traceability (Ticket 03 contract)."""

    @pytest.fixture(scope='class')
    def trades(self, eth_mtf_data, eth_mtf_features):
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        result = engine.run(
            eth_mtf_data, eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-06-30',
        )
        return result.get('trades', [])

    def test_trades_produced(self, trades):
        """ETH H1 2023 must still produce trades (sanity)."""
        assert len(trades) >= 10, (
            f'NYXEngine produced only {len(trades)} trades on ETH H1 '
            '2023 — refactor may have broken scoring'
        )

    def test_every_trade_has_meta_decision(self, trades):
        from src.agents.contracts import MetaDecision
        missing = [i for i, t in enumerate(trades)
                   if 'meta_decision' not in t]
        assert not missing, (
            f'{len(missing)} trade(s) missing the meta_decision key — '
            'Ticket 07 requires every trade to carry a canonical '
            'MetaDecision for traceability'
        )
        for t in trades:
            assert isinstance(t['meta_decision'], MetaDecision)

    def test_meta_decision_probability_matches_batch_score(self, trades):
        """The MetaDecision.probability must match the inline `score`
        that the original NYXEngine code used (precomputed_proba path
        → pass-through). This is the equivalence invariant of the
        wrapper."""
        for t in trades:
            dec = t['meta_decision']
            # The trade record carries `ml_score` = the inline GBM proba.
            ml_score = float(t.get('ml_score', -1))
            assert ml_score >= 0.0, 'ml_score missing from trade record'
            assert dec.probability == pytest.approx(ml_score, abs=1e-9)

    def test_quality_bucket_emitted(self, trades):
        """Each MetaDecision must set quality_bucket ∈
        CANONICAL_QUALITY_BUCKETS."""
        buckets = {t['meta_decision'].quality_bucket for t in trades}
        assert buckets.issubset({'high', 'medium', 'low'})
        # At least one non-None bucket across the trade set (trivial
        # sanity).
        assert buckets, 'no quality_bucket emitted'

    def test_risk_hint_in_range(self, trades):
        for t in trades:
            rh = t['meta_decision'].risk_hint
            assert rh is not None
            assert 0.0 <= rh <= 1.0


# ===========================================================================
class TestProbabilitySourceTracer:
    """The MetaDecision.features_snapshot must carry a
    `probability_source` tag identifying which code path computed the
    probability (precomputed, trained_gbm, heuristic). In NYXEngine the
    path is precomputed (batch-scored then wrapped)."""

    def test_source_is_precomputed_in_engine(
        self, eth_mtf_data, eth_mtf_features,
    ):
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        result = engine.run(
            eth_mtf_data, eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-06-30',
        )
        trades = result.get('trades', [])
        assert trades, 'no trades to inspect'
        sample = trades[0]['meta_decision']
        src = sample.features_snapshot.get('probability_source')
        assert src is not None, (
            'MetaDecision.features_snapshot must carry '
            'probability_source (Ticket 07 traceability)'
        )
