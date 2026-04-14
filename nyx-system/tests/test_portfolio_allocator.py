"""
TDD Tests — PortfolioAllocator

Takes N Signals from per-asset pods and returns a list of ApprovedTrades
that respects the global risk rules from the architecture spec:

  max_total_risk         default 4%
  max_asset_risk         default 1.5%
  max_cluster_risk       default 2.0%  (applies to 'alts' cluster)
  max_open_positions     default 2 (configurable)

Ranking: by Signal.score() (= max(0, edge) × viability × conviction),
descending. Tie-break deterministic (symbol ascending).

Correlation: same-direction alts conflict — only one is taken at full
size by default. The other is either dropped or down-weighted.

Contract:
  - `decide(signals, open_positions)` returns list[ApprovedTrade].
  - ApprovedTrade has: symbol, direction, final_risk (fraction of
    portfolio), source_signal.
  - Sum of final_risk across approved ≤ max_total_risk.
  - Per-asset risk ≤ max_asset_risk.
  - Per-cluster risk ≤ max_cluster_risk.
  - FLAT signals (direction=0) are skipped.
  - Signals with score()==0 (edge<=0 or conviction=0) are skipped.
"""
import pytest


def _sig(symbol='ETHUSDT', direction=1, edge=10.0, viability=0.8,
         conviction=0.7, size=0.3, cluster='majors', **over):
    from src.assets.signal import Signal
    kwargs = dict(
        symbol=symbol,
        timestamp='2025-01-01T00:00:00+00:00',
        direction=direction,
        conviction=conviction,
        expected_edge_net=edge,
        maker_viability=viability,
        regime_tag='trend_plus',
        bull_bear_tag='bull',
        size_suggestion=size,
        cluster_group=cluster,
    )
    kwargs.update(over)
    return Signal(**kwargs)


# ===========================================================================
class TestBasicApproval:

    def test_one_good_signal_approved(self):
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator()
        trades = a.decide([_sig()])
        assert len(trades) == 1
        assert trades[0].symbol == 'ETHUSDT'
        assert trades[0].direction == 1
        assert trades[0].final_risk > 0

    def test_flat_signals_skipped(self):
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator()
        trades = a.decide([_sig(direction=0)])
        assert trades == []

    def test_negative_edge_skipped(self):
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator()
        trades = a.decide([_sig(edge=-5.0)])
        assert trades == []


# ===========================================================================
class TestRiskCaps:

    def test_max_asset_risk_enforced(self):
        """Even if size_suggestion is huge, final_risk ≤ max_asset_risk."""
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator(max_asset_risk=0.015)
        trades = a.decide([_sig(size=5.0)])   # pod asked for 500% of cap
        assert len(trades) == 1
        assert trades[0].final_risk <= 0.015 + 1e-9

    def test_total_risk_cap_enforced(self):
        """Sum across approved ≤ max_total_risk."""
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator(max_total_risk=0.04, max_asset_risk=0.015)
        signals = [
            _sig(symbol='ETHUSDT',  edge=12, cluster='majors'),
            _sig(symbol='XRPUSDT',  edge=10, cluster='alts'),
            _sig(symbol='SOLUSDT',  edge=8,  cluster='alts'),
        ]
        trades = a.decide(signals)
        total = sum(t.final_risk for t in trades)
        assert total <= 0.04 + 1e-9

    def test_cluster_risk_cap(self):
        """Both alts together ≤ max_cluster_risk."""
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator(
            max_total_risk=0.05, max_asset_risk=0.015,
            max_cluster_risk={'alts': 0.02},
        )
        signals = [
            _sig(symbol='XRPUSDT', edge=10, cluster='alts'),
            _sig(symbol='SOLUSDT', edge=9,  cluster='alts'),
        ]
        trades = a.decide(signals)
        alt_total = sum(t.final_risk for t in trades if t.cluster_group == 'alts')
        assert alt_total <= 0.02 + 1e-9


# ===========================================================================
class TestRanking:

    def test_higher_score_wins_when_budget_limited(self):
        """With tight budget, the higher-score signal is approved first."""
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator(
            max_total_risk=0.015, max_asset_risk=0.015, max_open_positions=1,
        )
        signals = [
            _sig(symbol='ETHUSDT', edge=5,  viability=0.5),   # score = 5 × 0.5 × 0.7
            _sig(symbol='SOLUSDT', edge=15, viability=0.9),   # score = 15 × 0.9 × 0.7
        ]
        trades = a.decide(signals)
        assert len(trades) == 1
        assert trades[0].symbol == 'SOLUSDT'

    def test_max_open_positions_limit(self):
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator(max_open_positions=2)
        signals = [
            _sig(symbol='ETHUSDT', edge=15, cluster='majors'),
            _sig(symbol='XRPUSDT', edge=10, cluster='alts'),
            _sig(symbol='SOLUSDT', edge=8,  cluster='alts'),
        ]
        trades = a.decide(signals)
        assert len(trades) == 2

    def test_deterministic_tiebreak(self):
        """Exact-same score → sort by symbol ascending (reproducible)."""
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator(max_open_positions=1,
                               max_total_risk=0.015, max_asset_risk=0.015)
        # Same score for both
        s1 = _sig(symbol='ZRXUSDT', edge=10, viability=0.5, conviction=0.5,
                  cluster='alts')
        s2 = _sig(symbol='AAAUSDT', edge=10, viability=0.5, conviction=0.5,
                  cluster='alts')
        trades = a.decide([s1, s2])
        assert len(trades) == 1
        assert trades[0].symbol == 'AAAUSDT'


# ===========================================================================
class TestCorrelationVeto:

    def test_same_cluster_same_direction_reduces_second(self):
        """If two alts want to go long together and cluster cap is hit,
        the lower-score one is either dropped or down-sized."""
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator(
            max_total_risk=0.05, max_asset_risk=0.015,
            max_cluster_risk={'alts': 0.015},   # only 1 full-size alt fits
        )
        signals = [
            _sig(symbol='XRPUSDT', edge=10, direction=1, cluster='alts'),
            _sig(symbol='SOLUSDT', edge=8,  direction=1, cluster='alts'),
        ]
        trades = a.decide(signals)
        alt_total = sum(t.final_risk for t in trades if t.cluster_group == 'alts')
        assert alt_total <= 0.015 + 1e-9
        # First to be approved should be the highest-score (XRP here).
        if trades:
            assert trades[0].symbol == 'XRPUSDT'

    def test_opposite_direction_in_same_cluster_both_allowed(self):
        """A long alt and a short alt do not correlate — both can pass."""
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator(
            max_total_risk=0.05, max_asset_risk=0.015,
            max_cluster_risk={'alts': 0.03},
        )
        signals = [
            _sig(symbol='XRPUSDT', edge=10, direction=1,  cluster='alts'),
            _sig(symbol='SOLUSDT', edge=8,  direction=-1, cluster='alts'),
        ]
        trades = a.decide(signals)
        assert len(trades) == 2


# ===========================================================================
class TestOpenPositionAwareness:

    def test_already_open_counts_against_cap(self):
        """If ETH is already open, a new ETH signal cannot be re-approved."""
        from src.assets.portfolio_allocator import PortfolioAllocator
        a = PortfolioAllocator(max_open_positions=2)
        trades = a.decide(
            [_sig(symbol='ETHUSDT')],
            open_positions={'ETHUSDT': {'direction': 1}},
        )
        # Already open → skipped (no pyramiding).
        assert trades == []
