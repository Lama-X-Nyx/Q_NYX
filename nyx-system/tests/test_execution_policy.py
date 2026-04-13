"""
TDD Tests — Execution Policy

Rule: only take trades whose alpha survives realistic execution.
Not "beautiful if perfect fill" but "still good in real conditions".

Modules:
  1. estimate_fill_probability: can we get maker fill given vol/spread?
  2. estimate_execution_cost: what will this trade actually cost?
  3. maker_viability_score: is post-only realistic here?
  4. should_place_post_only / should_cancel_unfilled
  5. Pipeline integration: execution filter before trade placement
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'


# ===========================================================================
# TEST 1: Fill probability estimation
# ===========================================================================
class TestFillProbability:

    def test_high_vol_high_fill(self):
        """High volume + low spread → high fill probability."""
        from src.ml.execution_policy import estimate_fill_probability
        prob = estimate_fill_probability(
            volume_ratio=3.0, spread_pct=0.01, atr_pct=0.3, bar_range_pct=0.2)
        assert prob >= 0.70, f"Fill prob {prob:.0%} too low for high vol"

    def test_low_vol_low_fill(self):
        """Low volume + wide spread → low fill probability."""
        from src.ml.execution_policy import estimate_fill_probability
        prob = estimate_fill_probability(
            volume_ratio=0.5, spread_pct=0.10, atr_pct=0.5, bar_range_pct=0.5)
        assert prob <= 0.50, f"Fill prob {prob:.0%} too high for low vol"

    def test_fill_prob_bounded(self):
        """Must be in [0, 1]."""
        from src.ml.execution_policy import estimate_fill_probability
        for vr, sp, ar, br in [(5.0, 0.005, 0.1, 0.1), (0.1, 0.5, 1.0, 1.0)]:
            prob = estimate_fill_probability(vr, sp, ar, br)
            assert 0.0 <= prob <= 1.0


# ===========================================================================
# TEST 2: Execution cost estimation
# ===========================================================================
class TestExecutionCost:

    def test_maker_cheaper_than_taker(self):
        """Maker cost must be lower than taker."""
        from src.ml.execution_policy import estimate_execution_cost
        maker = estimate_execution_cost(fill_as_maker=True, price=30000, qty=0.01)
        taker = estimate_execution_cost(fill_as_maker=False, price=30000, qty=0.01)
        assert maker < taker

    def test_cost_scales_with_size(self):
        """Bigger position → higher absolute cost."""
        from src.ml.execution_policy import estimate_execution_cost
        small = estimate_execution_cost(fill_as_maker=True, price=30000, qty=0.001)
        big = estimate_execution_cost(fill_as_maker=True, price=30000, qty=0.01)
        assert big > small

    def test_cost_includes_slippage(self):
        """Taker cost must include slippage estimate."""
        from src.ml.execution_policy import estimate_execution_cost
        cost = estimate_execution_cost(fill_as_maker=False, price=30000, qty=0.01)
        pure_fee = 30000 * 0.01 * 0.0004 * 2
        assert cost > pure_fee, "Cost should include slippage beyond pure fee"


# ===========================================================================
# TEST 3: Maker viability score
# ===========================================================================
class TestMakerViability:

    def test_calm_market_high_viability(self):
        """Low vol + high volume → post-only viable."""
        from src.ml.execution_policy import maker_viability_score
        score = maker_viability_score(
            volume_ratio=3.0, atr_pct=0.1, momentum_abs=0.001, spread_pct=0.01)
        assert score >= 0.60

    def test_fast_market_low_viability(self):
        """High momentum + low volume → post-only risky."""
        from src.ml.execution_policy import maker_viability_score
        score = maker_viability_score(
            volume_ratio=0.5, atr_pct=0.8, momentum_abs=0.02, spread_pct=0.05)
        assert score <= 0.40

    def test_viability_bounded(self):
        from src.ml.execution_policy import maker_viability_score
        score = maker_viability_score(5.0, 0.01, 0.0001, 0.001)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# TEST 4: Order decisions
# ===========================================================================
class TestOrderDecisions:

    def test_post_only_when_viable(self):
        from src.ml.execution_policy import should_place_post_only
        assert should_place_post_only(maker_viability=0.8) is True

    def test_no_post_only_when_risky(self):
        from src.ml.execution_policy import should_place_post_only
        assert should_place_post_only(maker_viability=0.2) is False

    def test_cancel_unfilled_after_delay(self):
        from src.ml.execution_policy import should_cancel_unfilled
        assert should_cancel_unfilled(bars_waiting=5, max_wait=3) is True
        assert should_cancel_unfilled(bars_waiting=2, max_wait=3) is False


# ===========================================================================
# TEST 5: Pipeline integration
# ===========================================================================
class TestExecutionFilter:

    def test_execution_check_filters_bad_fills(self):
        """Trades with low maker viability should be filtered or cost-adjusted."""
        from src.ml.execution_policy import execution_check
        # Good fill conditions
        good = execution_check(
            volume_ratio=3.0, spread_pct=0.01, atr_pct=0.2,
            momentum_abs=0.002, trade_alpha_pct=0.5)
        assert good['should_trade'] is True
        assert good['expected_cost_pct'] < 0.10

    def test_execution_rejects_when_cost_exceeds_alpha(self):
        """If execution cost > expected alpha → reject trade."""
        from src.ml.execution_policy import execution_check
        # Alpha smaller than execution cost → should reject
        bad = execution_check(
            volume_ratio=0.1, spread_pct=0.50, atr_pct=2.0,
            momentum_abs=0.10, trade_alpha_pct=0.0005)  # 0.05% alpha, cost ~0.14%
        assert bad['should_trade'] is False, \
            f"Should reject: cost={bad['expected_cost_pct']:.4f} > alpha=0.01"

    def test_execution_check_returns_all_fields(self):
        from src.ml.execution_policy import execution_check
        result = execution_check(
            volume_ratio=2.0, spread_pct=0.02, atr_pct=0.3,
            momentum_abs=0.003, trade_alpha_pct=0.3)
        for key in ['should_trade', 'fill_probability', 'maker_viability',
                     'expected_cost_pct', 'order_type', 'alpha_after_cost']:
            assert key in result, f"Missing: {key}"


# ===========================================================================
# TEST 6: Pipeline with execution policy
# ===========================================================================
class TestPipelineWithExecution:

    @pytest.fixture(scope='class')
    def mtf_data(self):
        data = {}
        for tf in ['15m', '1h', '1d']:
            d = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
            d['datetime'] = pd.to_datetime(d['datetime']); d = d.set_index('datetime')
            data[tf] = d
        return data

    @pytest.fixture(scope='class')
    def mtf_features(self):
        feats = {}
        for tf in ['15m', '1h', '1d']:
            feats[tf] = pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
        return feats

    def test_pipeline_with_execution_filter(self, mtf_data, mtf_features):
        """Pipeline must accept execution_filter=True."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(use_execution_filter=True)
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert 'n_trades' in r
        assert 'execution_reject_rate' in r

    def test_execution_filter_rejects_some(self, mtf_data, mtf_features):
        """Execution filter should reject some trades."""
        from src.ml.nyx_pipeline import NYXPipeline
        no_filter = NYXPipeline(use_execution_filter=False)
        with_filter = NYXPipeline(use_execution_filter=True)
        r_no = no_filter.run(mtf_data, mtf_features,
                             train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        r_yes = with_filter.run(mtf_data, mtf_features,
                                train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r_yes['n_trades'] <= r_no['n_trades']
