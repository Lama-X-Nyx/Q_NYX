"""
TDD Tests — Ticket 32 — Execution Optimization.

Fill probability + dynamic offset + maker/taker decision.
Operates AFTER RiskEngine, BEFORE OMS. Alpha untouched.
"""
from __future__ import annotations

import pytest


# ===========================================================================
class TestFillProbability:

    def test_output_in_range(self):
        from src.live.execution_optimizer import estimate_fill_probability
        p = estimate_fill_probability(
            spread_bps=5.0, volatility_pct=0.5,
            price_move_pct=0.1, offset_bps=10.0,
        )
        assert 0.0 <= p <= 1.0

    def test_wider_offset_increases_fill_prob(self):
        from src.live.execution_optimizer import estimate_fill_probability
        p_tight = estimate_fill_probability(
            spread_bps=5.0, volatility_pct=0.5,
            price_move_pct=0.1, offset_bps=5.0,
        )
        p_wide = estimate_fill_probability(
            spread_bps=5.0, volatility_pct=0.5,
            price_move_pct=0.1, offset_bps=20.0,
        )
        assert p_wide >= p_tight

    def test_high_volatility_reduces_fill_prob(self):
        from src.live.execution_optimizer import estimate_fill_probability
        p_low_vol = estimate_fill_probability(
            spread_bps=5.0, volatility_pct=0.2,
            price_move_pct=0.0, offset_bps=10.0,
        )
        p_high_vol = estimate_fill_probability(
            spread_bps=5.0, volatility_pct=2.0,
            price_move_pct=0.0, offset_bps=10.0,
        )
        assert p_low_vol >= p_high_vol


# ===========================================================================
class TestDynamicOffset:

    def test_offset_positive(self):
        from src.live.execution_optimizer import compute_dynamic_offset_bps
        offset = compute_dynamic_offset_bps(
            volatility_pct=0.5, spread_bps=5.0, fill_probability=0.8,
        )
        assert offset > 0

    def test_high_vol_widens_offset(self):
        from src.live.execution_optimizer import compute_dynamic_offset_bps
        o_low = compute_dynamic_offset_bps(
            volatility_pct=0.2, spread_bps=5.0, fill_probability=0.8,
        )
        o_high = compute_dynamic_offset_bps(
            volatility_pct=2.0, spread_bps=5.0, fill_probability=0.8,
        )
        assert o_high > o_low

    def test_low_fill_prob_widens_offset(self):
        from src.live.execution_optimizer import compute_dynamic_offset_bps
        o_high_fp = compute_dynamic_offset_bps(
            volatility_pct=0.5, spread_bps=5.0, fill_probability=0.9,
        )
        o_low_fp = compute_dynamic_offset_bps(
            volatility_pct=0.5, spread_bps=5.0, fill_probability=0.3,
        )
        assert o_low_fp > o_high_fp


# ===========================================================================
class TestMakerTakerDecision:

    def test_taker_when_edge_exceeds_cost(self):
        from src.live.execution_optimizer import should_use_taker
        assert should_use_taker(
            expected_edge_bps=30.0, taker_cost_bps=8.0,
            fill_probability_maker=0.3,
        ) is True

    def test_maker_when_fill_prob_high(self):
        from src.live.execution_optimizer import should_use_taker
        assert should_use_taker(
            expected_edge_bps=15.0, taker_cost_bps=8.0,
            fill_probability_maker=0.95,
        ) is False

    def test_maker_when_edge_small(self):
        from src.live.execution_optimizer import should_use_taker
        assert should_use_taker(
            expected_edge_bps=5.0, taker_cost_bps=8.0,
            fill_probability_maker=0.6,
        ) is False


# ===========================================================================
class TestExecutionPlan:

    def test_plan_has_required_fields(self):
        from src.live.execution_optimizer import build_execution_plan
        plan = build_execution_plan(
            side='buy', mark_price=16500.0,
            volatility_pct=0.5, spread_bps=5.0,
            expected_edge_bps=20.0,
        )
        assert 'limit_price' in plan
        assert 'order_type' in plan
        assert 'offset_bps' in plan
        assert 'fill_probability' in plan

    def test_plan_limit_price_below_mark_for_buy(self):
        from src.live.execution_optimizer import build_execution_plan
        plan = build_execution_plan(
            side='buy', mark_price=16500.0,
            volatility_pct=0.5, spread_bps=5.0,
            expected_edge_bps=20.0,
        )
        if plan['order_type'] == 'post_only_limit':
            assert plan['limit_price'] < 16500.0

    def test_plan_limit_price_above_mark_for_sell(self):
        from src.live.execution_optimizer import build_execution_plan
        plan = build_execution_plan(
            side='sell', mark_price=16500.0,
            volatility_pct=0.5, spread_bps=5.0,
            expected_edge_bps=20.0,
        )
        if plan['order_type'] == 'post_only_limit':
            assert plan['limit_price'] > 16500.0
