"""
TDD Tests — Ticket 35 — Portfolio Allocator.

Consumes per-asset decisions + inter-asset dependency outputs,
produces a coherent, capital-constrained, dependency-aware execution
plan. Operates AFTER dependency layer, BEFORE RiskEngine.

Alpha (GBM/Jesse) untouched. Dependency layer untouched.
"""
from __future__ import annotations

import pytest


def _make_candidate(
    symbol: str = 'BTCUSDT',
    direction: int = 1,
    confidence: float = 0.75,
    expected_edge_bps: float = 20.0,
    size_hint: float = 1.0,
    quality_bucket: str = 'high',
    entry_price: float = 30000.0,
    dependency_strength: float = 0.0,
    contagion_risk: float = 0.0,
    lag_alignment_score: float = 0.5,
    divergence_score: float = 0.2,
) -> dict:
    return {
        'symbol': symbol,
        'direction': direction,
        'confidence': confidence,
        'expected_edge_bps': expected_edge_bps,
        'size_hint': size_hint,
        'quality_bucket': quality_bucket,
        'entry_price': entry_price,
        'dependency': {
            'dependency_strength': dependency_strength,
            'contagion_risk': contagion_risk,
            'lag_alignment_score': lag_alignment_score,
            'divergence_score': divergence_score,
        },
    }


def _make_portfolio_ctx(
    total_equity: float = 10_000.0,
    available_capital: float = 10_000.0,
    exposure_by_asset: dict | None = None,
    total_exposure: float = 0.0,
    open_position_count: int = 0,
) -> dict:
    return {
        'total_equity': total_equity,
        'available_capital': available_capital,
        'exposure_by_asset': exposure_by_asset or {},
        'total_exposure': total_exposure,
        'open_position_count': open_position_count,
    }


# =========================================================================
# A1 — Input/Output schema
# =========================================================================
class TestAllocatorSchema:

    def test_allocator_instantiation(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        assert alloc is not None

    def test_allocate_returns_list(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        candidates = [_make_candidate()]
        ctx = _make_portfolio_ctx()
        result = alloc.allocate(candidates, ctx)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_allocation_has_required_fields(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        candidates = [_make_candidate()]
        ctx = _make_portfolio_ctx()
        result = alloc.allocate(candidates, ctx)
        r = result[0]
        assert 'symbol' in r
        assert 'decision' in r
        assert 'allocated_notional' in r
        assert 'allocated_size' in r
        assert 'allocation_reason' in r
        assert r['decision'] in (
            'APPROVE_FULL', 'APPROVE_REDUCED', 'DEFER', 'REJECT',
        )


# =========================================================================
# A2 — Scoring function
# =========================================================================
class TestAllocationScoring:

    def test_higher_confidence_scores_higher(self):
        from src.live.portfolio_allocator import compute_allocation_score
        s_high = compute_allocation_score(
            confidence=0.9, expected_edge_bps=20.0,
            quality_bucket='high', dependency_strength=0.0,
            contagion_risk=0.0, current_exposure_pct=0.0,
        )
        s_low = compute_allocation_score(
            confidence=0.5, expected_edge_bps=20.0,
            quality_bucket='high', dependency_strength=0.0,
            contagion_risk=0.0, current_exposure_pct=0.0,
        )
        assert s_high > s_low

    def test_higher_edge_scores_higher(self):
        from src.live.portfolio_allocator import compute_allocation_score
        s_high = compute_allocation_score(
            confidence=0.7, expected_edge_bps=40.0,
            quality_bucket='medium', dependency_strength=0.0,
            contagion_risk=0.0, current_exposure_pct=0.0,
        )
        s_low = compute_allocation_score(
            confidence=0.7, expected_edge_bps=10.0,
            quality_bucket='medium', dependency_strength=0.0,
            contagion_risk=0.0, current_exposure_pct=0.0,
        )
        assert s_high > s_low

    def test_high_contagion_reduces_score(self):
        from src.live.portfolio_allocator import compute_allocation_score
        s_clean = compute_allocation_score(
            confidence=0.7, expected_edge_bps=20.0,
            quality_bucket='high', dependency_strength=0.5,
            contagion_risk=0.1, current_exposure_pct=0.0,
        )
        s_contagion = compute_allocation_score(
            confidence=0.7, expected_edge_bps=20.0,
            quality_bucket='high', dependency_strength=0.5,
            contagion_risk=0.8, current_exposure_pct=0.0,
        )
        assert s_clean > s_contagion

    def test_high_exposure_reduces_score(self):
        from src.live.portfolio_allocator import compute_allocation_score
        s_fresh = compute_allocation_score(
            confidence=0.7, expected_edge_bps=20.0,
            quality_bucket='high', dependency_strength=0.0,
            contagion_risk=0.0, current_exposure_pct=0.0,
        )
        s_loaded = compute_allocation_score(
            confidence=0.7, expected_edge_bps=20.0,
            quality_bucket='high', dependency_strength=0.0,
            contagion_risk=0.0, current_exposure_pct=0.5,
        )
        assert s_fresh > s_loaded

    def test_score_non_negative(self):
        from src.live.portfolio_allocator import compute_allocation_score
        s = compute_allocation_score(
            confidence=0.1, expected_edge_bps=1.0,
            quality_bucket='low', dependency_strength=0.9,
            contagion_risk=0.9, current_exposure_pct=0.9,
        )
        assert s >= 0.0


# =========================================================================
# A3 — Capital constraints
# =========================================================================
class TestCapitalConstraints:

    def test_single_trade_within_limits(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator(max_capital_per_trade_pct=0.10)
        cands = [_make_candidate(entry_price=30000.0, confidence=0.9)]
        ctx = _make_portfolio_ctx(total_equity=10_000.0)
        result = alloc.allocate(cands, ctx)
        assert result[0]['allocated_notional'] <= 10_000.0 * 0.10 + 0.01

    def test_total_capital_usage_capped(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator(
            max_total_capital_pct=0.30,
            max_capital_per_trade_pct=0.15,
        )
        cands = [
            _make_candidate('BTCUSDT', confidence=0.9, entry_price=30000.0),
            _make_candidate('ETHUSDT', confidence=0.85, entry_price=2000.0),
            _make_candidate('SOLUSDT', confidence=0.80, entry_price=100.0),
        ]
        ctx = _make_portfolio_ctx(total_equity=10_000.0)
        result = alloc.allocate(cands, ctx)
        total_alloc = sum(r['allocated_notional'] for r in result)
        assert total_alloc <= 10_000.0 * 0.30 + 0.01

    def test_max_per_asset_enforced(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator(max_capital_per_asset_pct=0.15)
        cands = [_make_candidate('BTCUSDT', confidence=0.95, entry_price=30000.0)]
        ctx = _make_portfolio_ctx(
            total_equity=10_000.0,
            exposure_by_asset={'BTCUSDT': 1000.0},
        )
        result = alloc.allocate(cands, ctx)
        existing_plus_new = 1000.0 + result[0]['allocated_notional']
        assert existing_plus_new <= 10_000.0 * 0.15 + 100.01

    def test_capital_exhaustion_rejects_lower_ranked(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator(
            max_total_capital_pct=0.02,
            max_capital_per_trade_pct=0.10,
        )
        cands = [
            _make_candidate('BTCUSDT', confidence=0.95, entry_price=30000.0),
            _make_candidate('ETHUSDT', confidence=0.60, entry_price=2000.0),
        ]
        ctx = _make_portfolio_ctx(total_equity=10_000.0)
        result = alloc.allocate(cands, ctx)
        approved = [r for r in result if r['decision'].startswith('APPROVE')]
        rejected = [r for r in result if r['decision'] in ('REJECT', 'DEFER')]
        assert len(approved) >= 1
        assert len(rejected) >= 1


# =========================================================================
# A4 — Dependency-aware adjustment
# =========================================================================
class TestDependencyAdjustment:

    def test_follower_reduced_when_leader_selected(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        cands = [
            _make_candidate('BTCUSDT', confidence=0.9, dependency_strength=0.0),
            _make_candidate('ETHUSDT', confidence=0.85, dependency_strength=0.8),
        ]
        ctx = _make_portfolio_ctx(total_equity=10_000.0)
        result = alloc.allocate(cands, ctx)
        btc = next(r for r in result if r['symbol'] == 'BTCUSDT')
        eth = next(r for r in result if r['symbol'] == 'ETHUSDT')
        if btc['decision'].startswith('APPROVE') and eth['decision'].startswith('APPROVE'):
            assert eth['allocated_notional'] <= btc['allocated_notional']

    def test_high_dependency_overlap_penalized(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        cands_independent = [
            _make_candidate('BTCUSDT', confidence=0.8, dependency_strength=0.0),
            _make_candidate('ETHUSDT', confidence=0.8, dependency_strength=0.1),
        ]
        cands_dependent = [
            _make_candidate('BTCUSDT', confidence=0.8, dependency_strength=0.0),
            _make_candidate('ETHUSDT', confidence=0.8, dependency_strength=0.9),
        ]
        ctx = _make_portfolio_ctx(total_equity=10_000.0)
        r_ind = alloc.allocate(cands_independent, ctx)
        r_dep = alloc.allocate(cands_dependent, ctx)
        eth_ind = next(r for r in r_ind if r['symbol'] == 'ETHUSDT')
        eth_dep = next(r for r in r_dep if r['symbol'] == 'ETHUSDT')
        assert eth_dep['allocated_notional'] <= eth_ind['allocated_notional']


# =========================================================================
# A5 — Concentration control
# =========================================================================
class TestConcentrationControl:

    def test_directional_concentration_limited(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator(max_directional_pct=0.25)
        cands = [
            _make_candidate('BTCUSDT', direction=1, confidence=0.9),
            _make_candidate('ETHUSDT', direction=1, confidence=0.85),
            _make_candidate('SOLUSDT', direction=1, confidence=0.80),
        ]
        ctx = _make_portfolio_ctx(total_equity=10_000.0)
        result = alloc.allocate(cands, ctx)
        total_long = sum(
            r['allocated_notional'] for r in result if r.get('direction', 1) == 1
        )
        assert total_long <= 10_000.0 * 0.25 + 0.01

    def test_no_single_asset_dominates(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator(max_capital_per_asset_pct=0.15)
        cands = [_make_candidate('BTCUSDT', confidence=0.99, entry_price=30000.0)]
        ctx = _make_portfolio_ctx(total_equity=10_000.0)
        result = alloc.allocate(cands, ctx)
        assert result[0]['allocated_notional'] <= 10_000.0 * 0.15 + 0.01


# =========================================================================
# A6 — Trade selection
# =========================================================================
class TestTradeSelection:

    def test_ranking_is_deterministic(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        cands = [
            _make_candidate('BTCUSDT', confidence=0.9),
            _make_candidate('ETHUSDT', confidence=0.7),
            _make_candidate('SOLUSDT', confidence=0.5),
        ]
        ctx = _make_portfolio_ctx()
        r1 = alloc.allocate(cands, ctx)
        r2 = alloc.allocate(cands, ctx)
        for a, b in zip(r1, r2):
            assert a['symbol'] == b['symbol']
            assert a['decision'] == b['decision']
            assert abs(a['allocated_notional'] - b['allocated_notional']) < 0.01

    def test_higher_confidence_selected_first(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator(max_total_capital_pct=0.10)
        cands = [
            _make_candidate('SOLUSDT', confidence=0.5, entry_price=100.0),
            _make_candidate('BTCUSDT', confidence=0.95, entry_price=30000.0),
            _make_candidate('ETHUSDT', confidence=0.6, entry_price=2000.0),
        ]
        ctx = _make_portfolio_ctx(total_equity=10_000.0)
        result = alloc.allocate(cands, ctx)
        approved = [r for r in result if r['decision'].startswith('APPROVE')]
        assert len(approved) >= 1
        assert approved[0]['symbol'] == 'BTCUSDT'

    def test_empty_candidates_returns_empty(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        result = alloc.allocate([], _make_portfolio_ctx())
        assert result == []

    def test_all_rejected_when_no_capital(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        cands = [_make_candidate('BTCUSDT', confidence=0.9)]
        ctx = _make_portfolio_ctx(total_equity=10_000.0, available_capital=0.0)
        result = alloc.allocate(cands, ctx)
        assert result[0]['decision'] in ('REJECT', 'DEFER')


# =========================================================================
# A7 — Runtime integration
# =========================================================================
class TestRuntimeIntegration:

    def test_runtime_accepts_portfolio_allocator(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        from src.live.portfolio_allocator import PortfolioAllocator
        HERE = Path(__file__).resolve().parent.parent
        alloc = PortfolioAllocator()
        rt = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=HERE / 'models' / 'ETHUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'ETHUSDT',
            portfolio_allocator=alloc,
        )
        assert rt.portfolio_allocator is alloc

    def test_runtime_without_allocator_still_works(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        HERE = Path(__file__).resolve().parent.parent
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        assert rt.portfolio_allocator is None
        rt.start()
        bar = {
            'timestamp': '2023-06-15T12:00:00',
            'open': 30000.0, 'high': 30100.0,
            'low': 29900.0, 'close': 30050.0, 'volume': 100.0,
        }
        result = rt.on_bar(bar)
        assert result['action'] in (
            'FLAT', 'ORDER_SUBMITTED', 'SKIP_QUALITY',
            'BLOCKED_RISK', 'SKIP_DEPENDENCY',
        )

    def test_allocator_snapshot_for_persistence(self):
        from src.live.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        cands = [_make_candidate('BTCUSDT', confidence=0.9)]
        ctx = _make_portfolio_ctx()
        alloc.allocate(cands, ctx)
        snap = alloc.snapshot()
        assert 'last_cycle_results' in snap
        assert 'total_cycles' in snap
