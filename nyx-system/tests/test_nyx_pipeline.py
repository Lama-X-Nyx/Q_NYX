"""
TDD Tests — Unified NYX Pipeline v0.3

Single pipeline that merges ALL modules:
  1. MTF data (15m + 1h + 1d) → features from all timeframes
  2. Edge candidates (trend + volume on 15m)
  3. MTF context features: 1D trend direction, 1H regime, as ML features
  4. ML Filter (47 parquet + MTF context + rule scores)
  5. Soft gate sizing (disagreement → size adjustment)
  6. Realistic execution (fees, slippage, position sizing)

Threshold: 0.60 (calibrated)
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'


@pytest.fixture(scope='module')
def mtf_data():
    """Load all 4 timeframes."""
    data = {}
    for tf in ['15m', '1h', '1d']:
        df = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        data[tf] = df
    return data


@pytest.fixture(scope='module')
def mtf_features():
    """Load parquet features for all timeframes."""
    feats = {}
    for tf in ['15m', '1h', '1d']:
        feats[tf] = pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
    return feats


# ===========================================================================
# TEST 1: Pipeline accepts MTF data
# ===========================================================================
class TestPipelineInput:

    def test_accepts_mtf_dict(self, mtf_data, mtf_features):
        """Pipeline must accept dict of DataFrames keyed by timeframe."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        result = pipe.run(
            mtf_data, mtf_features,
            train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31',
        )
        assert 'n_trades' in result

    def test_uses_all_three_timeframes(self, mtf_data, mtf_features):
        """Candidate features must include 1D and 1H context."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        result = pipe.run(
            mtf_data, mtf_features,
            train_end='2022-12-31', test_start='2023-01-01', test_end='2023-06-30',
        )
        feat_keys = set(result['feature_names'])
        # Must have MTF features
        assert any('ctx_1d' in f for f in feat_keys), f"No 1D context features in {feat_keys}"
        assert any('reg_1h' in f for f in feat_keys), f"No 1H regime features in {feat_keys}"


# ===========================================================================
# TEST 2: Pipeline merges all components
# ===========================================================================
class TestPipelineMerge:

    def test_has_ml_filter(self, mtf_data, mtf_features):
        """Pipeline must use ML filter (reject rate > 0)."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r['filter_reject_rate'] > 0.20, "ML filter not rejecting enough"

    def test_has_soft_gate_sizing(self, mtf_data, mtf_features):
        """Trades must have varying size_factor from soft gate."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        if r['n_trades'] >= 3:
            sfs = [t['size_factor'] for t in r['trades']]
            assert len(set(round(s, 2) for s in sfs)) > 1, "All same size_factor"

    def test_has_realistic_fees(self, mtf_data, mtf_features):
        """Fees must be tracked and non-zero."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r['total_fees'] > 0

    def test_threshold_is_0_60(self, mtf_data, mtf_features):
        """Default threshold must be 0.60."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        assert pipe.ml_threshold == 0.60


# ===========================================================================
# TEST 3: MTF features improve over 15m-only
# ===========================================================================
class TestMTFValue:

    def test_mtf_features_count_higher(self, mtf_data, mtf_features):
        """MTF pipeline must have more features than 15m-only."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-06-30')
        n_feat = len(r['feature_names'])
        assert n_feat >= 55, f"Only {n_feat} features — expected 55+ with MTF"


# ===========================================================================
# TEST 4: OOS results
# ===========================================================================
class TestOOSResults:

    def test_positive_sharpe_2023(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r['sharpe'] > 0, f"Sharpe {r['sharpe']:.2f} negative"

    def test_positive_pnl_2023(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r['total_pnl_dollars'] > 0

    def test_walk_forward_2022_2023(self, mtf_data, mtf_features):
        """Walk-forward on 2022+2023 must have positive avg Sharpe."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        sharpes = []
        for tr_e, te_s, te_e in [
            ('2021-12-31', '2022-01-01', '2022-12-31'),
            ('2022-12-31', '2023-01-01', '2023-12-31'),
        ]:
            r = pipe.run(mtf_data, mtf_features, train_end=tr_e, test_start=te_s, test_end=te_e)
            if r['n_trades'] > 3:
                sharpes.append(r['sharpe'])
        avg = np.mean(sharpes) if sharpes else 0
        assert avg > 0, f"Walk-forward avg Sharpe {avg:.2f} negative: {sharpes}"

    def test_max_dd_under_15pct(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r['max_drawdown_pct'] < 0.15, f"DD {r['max_drawdown_pct']:.1%} >= 15%"

    def test_complete_metrics(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        for k in ['n_trades', 'win_rate', 'sharpe', 'total_pnl_dollars',
                   'max_drawdown_pct', 'profit_factor', 'total_fees',
                   'filter_reject_rate', 'feature_names', 'trades']:
            assert k in r, f"Missing: {k}"
