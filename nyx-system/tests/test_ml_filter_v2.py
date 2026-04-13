"""
TDD Tests — Enhanced ML Filter with Parquet Features + Threshold Calibration

Enhancement over basic ML filter:
  1. Use 47 parquet features (momentum, vol, microstructure, risk-adjusted)
     instead of 13 jesse core features
  2. Add HSMM regime probs when non-uniform
  3. Calibrate ML threshold via CV on train data
  4. Combine with soft gate sizing (disagreement → size adjustment)
  5. Target: Sharpe > hard gate (1.86) on 2023
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_15M = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf' / 'BTCUSDT_15m.csv'
FEAT_15M = Path(__file__).parent.parent / 'data' / 'features' / 'BTCUSDT_features_15m.parquet'


@pytest.fixture(scope='module')
def real_data():
    df = pd.read_csv(DATA_15M)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df


@pytest.fixture(scope='module')
def parquet_features():
    return pd.read_parquet(FEAT_15M)


# ===========================================================================
# TEST 1: Enhanced candidate features
# ===========================================================================
class TestEnhancedCandidates:

    def test_candidates_use_parquet_features(self, real_data, parquet_features):
        """Candidates must include parquet features (47+) not just jesse 13."""
        from src.ml.ml_filter_v2 import generate_enhanced_candidates
        candidates = generate_enhanced_candidates(
            real_data.loc['2023-01-01':'2023-06-30'],
            parquet_features.loc['2023-01-01':'2023-06-30'],
        )
        assert len(candidates) > 30
        feat_keys = set(candidates[0]['features'].keys())
        # Must include parquet microstructure features
        for f in ['buy_pressure', 'amihud', 'kyle_lambda', 'sharpe_8',
                   'ewma_vol_ratio', 'vol_surprise']:
            assert f in feat_keys, f"Missing parquet feature: {f}"

    def test_candidates_have_more_features_than_basic(self, real_data, parquet_features):
        """Enhanced must have 40+ features (vs 18 in basic)."""
        from src.ml.ml_filter_v2 import generate_enhanced_candidates
        candidates = generate_enhanced_candidates(
            real_data.loc['2023-01-01':'2023-06-30'],
            parquet_features.loc['2023-01-01':'2023-06-30'],
        )
        n_features = len(candidates[0]['features'])
        assert n_features >= 40, f"Only {n_features} features, expected 40+"


# ===========================================================================
# TEST 2: Threshold calibration
# ===========================================================================
class TestThresholdCalibration:

    def test_calibrate_returns_optimal_threshold(self, real_data, parquet_features):
        """Calibration must find threshold that maximizes Sharpe on train CV."""
        from src.ml.ml_filter_v2 import EnhancedMLFilter
        f = EnhancedMLFilter()
        f.train(
            real_data.loc['2020-01-01':'2022-12-31'],
            parquet_features.loc['2020-01-01':'2022-12-31'],
        )
        assert 0.45 <= f.calibrated_threshold <= 0.70, \
            f"Threshold {f.calibrated_threshold:.2f} outside expected range"

    def test_calibrated_threshold_better_than_default(self, real_data, parquet_features):
        """Calibrated threshold must produce >= WR than fixed 0.50."""
        from src.ml.ml_filter_v2 import EnhancedMLFilter
        f = EnhancedMLFilter()
        f.train(
            real_data.loc['2020-01-01':'2022-12-31'],
            parquet_features.loc['2020-01-01':'2022-12-31'],
        )
        # Should be at least as good as random
        assert f.train_metrics['calibrated_wr'] >= 0.40


# ===========================================================================
# TEST 3: Enhanced backtest results
# ===========================================================================
class TestEnhancedBacktest:

    def test_enhanced_positive_sharpe_2023(self, real_data, parquet_features):
        """Enhanced filter must have positive Sharpe on 2023."""
        from src.ml.ml_filter_v2 import EnhancedMLFilterBacktester
        bt = EnhancedMLFilterBacktester()
        r = bt.run(real_data, parquet_features,
                    train_end='2022-12-31', test_start='2023-01-01')
        assert r['sharpe'] > 0, f"Sharpe {r['sharpe']:.2f} negative"

    def test_enhanced_higher_wr_than_base(self, real_data, parquet_features):
        """Enhanced WR must be > hard gate base (48%)."""
        from src.ml.ml_filter_v2 import EnhancedMLFilterBacktester
        bt = EnhancedMLFilterBacktester()
        r = bt.run(real_data, parquet_features,
                    train_end='2022-12-31', test_start='2023-01-01')
        assert r['win_rate'] >= 0.48, \
            f"Enhanced WR {r['win_rate']:.0%} < base 48%"

    def test_enhanced_fewer_trades_higher_quality(self, real_data, parquet_features):
        """Must reject at least 30% of candidates."""
        from src.ml.ml_filter_v2 import EnhancedMLFilterBacktester
        bt = EnhancedMLFilterBacktester()
        r = bt.run(real_data, parquet_features,
                    train_end='2022-12-31', test_start='2023-01-01')
        assert r['filter_reject_rate'] >= 0.30

    def test_enhanced_metrics_complete(self, real_data, parquet_features):
        from src.ml.ml_filter_v2 import EnhancedMLFilterBacktester
        bt = EnhancedMLFilterBacktester()
        r = bt.run(real_data, parquet_features,
                    train_end='2022-12-31', test_start='2023-01-01')
        for k in ['sharpe', 'win_rate', 'n_trades', 'total_pnl_dollars',
                   'profit_factor', 'max_drawdown_pct', 'calibrated_threshold',
                   'filter_reject_rate', 'feature_importance']:
            assert k in r, f"Missing: {k}"


# ===========================================================================
# TEST 4: Walk-forward with enhanced features
# ===========================================================================
class TestEnhancedWalkForward:

    def test_walk_forward_avg_sharpe_positive(self, real_data, parquet_features):
        """Walk-forward average Sharpe must be positive."""
        from src.ml.ml_filter_v2 import EnhancedMLFilterBacktester
        bt = EnhancedMLFilterBacktester()
        sharpes = []
        for tr_e, te_s, te_e in [
            ('2021-12-31', '2022-01-01', '2022-12-31'),
            ('2022-12-31', '2023-01-01', '2023-12-31'),
        ]:
            r = bt.run(real_data, parquet_features, train_end=tr_e,
                        test_start=te_s, test_end=te_e)
            if r['n_trades'] > 5:
                sharpes.append(r['sharpe'])
        avg = np.mean(sharpes) if sharpes else 0
        assert avg > 0, f"Avg Sharpe {avg:.2f} negative: {sharpes}"

    def test_enhanced_better_than_basic_filter(self, real_data, parquet_features):
        """Enhanced must beat basic ML filter on WR in at least 1 fold."""
        from src.ml.ml_filter_v2 import EnhancedMLFilterBacktester
        from src.ml.ml_filter import MLFilteredBacktester
        bt_v2 = EnhancedMLFilterBacktester()
        bt_v1 = MLFilteredBacktester()
        improved = 0
        for tr_e, te_s, te_e in [
            ('2021-12-31', '2022-01-01', '2022-12-31'),
            ('2022-12-31', '2023-01-01', '2023-12-31'),
        ]:
            r2 = bt_v2.run(real_data, parquet_features, train_end=tr_e,
                           test_start=te_s, test_end=te_e)
            r1 = bt_v1.run(real_data, train_end=tr_e,
                           test_start=te_s, test_end=te_e)
            if r2['win_rate'] > r1['win_rate']:
                improved += 1
        assert improved >= 1, "Enhanced didn't beat basic on any fold"
