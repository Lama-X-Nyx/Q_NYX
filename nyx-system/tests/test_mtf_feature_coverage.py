"""
TDD Tests — MTF feature coverage inside training candidates

The current NYXPipeline._generate_candidates was advertised as MTF but
only injected:
  * 24 15m stationary features
  * 3 hand-crafted 1d scalars (trend/mom/bullish)
  * 4 hand-crafted 1h scalars + at most 4 selected 1h parquet columns

→ The 24 stationary features computed on 1h and 1d were never used.

This test file pins the new contract: every candidate must carry the
full stationary feature vector from EACH timeframe, prefixed by its TF:

    no prefix   : 15m features (current behavior)
    h1_<name>   : 1h  stationary features (full 24 columns)
    d1_<name>   : 1d  stationary features (full 24 columns)
"""
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'


def _load_eth_ohlcv(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f'ETHUSDT_{tf}.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


@pytest.fixture(scope='module')
def eth_data_and_features():
    mtf = {tf: _load_eth_ohlcv(tf) for tf in ('15m', '1h', '1d')}
    feats = {
        tf: pd.read_parquet(FEAT_DIR / f'ETHUSDT_features_{tf}.parquet')
        for tf in ('15m', '1h', '1d')
    }
    return mtf, feats


# ===========================================================================
class TestFullMTFFeatures:

    def test_candidates_have_h1_prefixed_features(self, eth_data_and_features):
        """Each candidate must carry every 1h stationary feature as h1_*."""
        from src.ml.nyx_pipeline import NYXPipeline
        mtf, feats = eth_data_and_features
        p = NYXPipeline()
        ctx_1d = p._build_1d_context(mtf['1d'], feats['1d'])
        ctx_1h = p._build_1h_context(mtf['1h'], feats['1h'])
        # Slice to a short range to keep it fast.
        df_15m = mtf['15m'].loc['2022-06-01':'2022-09-30']
        feat_15m = feats['15m'].loc['2022-06-01':'2022-09-30']
        cands = p._generate_candidates(df_15m, feat_15m, ctx_1d, ctx_1h,
                                        feat_1h=feats['1h'], feat_1d=feats['1d'])
        assert cands, "expected ≥ 1 candidate on this window"

        h1_keys = [k for k in cands[0]['features'] if k.startswith('h1_')]
        # All 24 stationary features must be present (some may be filtered
        # out as constant, but we expect the majority).
        assert len(h1_keys) >= 15, f"only {len(h1_keys)} h1_* features"

    def test_candidates_have_d1_prefixed_features(self, eth_data_and_features):
        from src.ml.nyx_pipeline import NYXPipeline
        mtf, feats = eth_data_and_features
        p = NYXPipeline()
        ctx_1d = p._build_1d_context(mtf['1d'], feats['1d'])
        ctx_1h = p._build_1h_context(mtf['1h'], feats['1h'])
        df_15m = mtf['15m'].loc['2022-06-01':'2022-09-30']
        feat_15m = feats['15m'].loc['2022-06-01':'2022-09-30']
        cands = p._generate_candidates(df_15m, feat_15m, ctx_1d, ctx_1h,
                                        feat_1h=feats['1h'], feat_1d=feats['1d'])
        assert cands

        d1_keys = [k for k in cands[0]['features'] if k.startswith('d1_')]
        assert len(d1_keys) >= 15, f"only {len(d1_keys)} d1_* features"

    def test_total_feature_count_above_50(self, eth_data_and_features):
        """Full MTF training should see well above 50 distinct features
        (24 15m + 20+ h1_ + 20+ d1_ + rule scores + extras)."""
        from src.ml.nyx_pipeline import NYXPipeline
        mtf, feats = eth_data_and_features
        p = NYXPipeline()
        ctx_1d = p._build_1d_context(mtf['1d'], feats['1d'])
        ctx_1h = p._build_1h_context(mtf['1h'], feats['1h'])
        df_15m = mtf['15m'].loc['2022-06-01':'2022-09-30']
        feat_15m = feats['15m'].loc['2022-06-01':'2022-09-30']
        cands = p._generate_candidates(df_15m, feat_15m, ctx_1d, ctx_1h,
                                        feat_1h=feats['1h'], feat_1d=feats['1d'])
        assert cands
        assert len(cands[0]['features']) >= 50, \
            f"only {len(cands[0]['features'])} features"


# ===========================================================================
class TestMTFAlignmentSafety:

    def test_h1_ffill_leakage_safety(self, eth_data_and_features):
        """For a 15m bar at time T, the h1_* features must come from the
        LAST COMPLETED 1h bar (i.e. floor(T, 1h)), not from the 1h bar
        that closes at T+30m. No look-ahead."""
        from src.ml.nyx_pipeline import NYXPipeline
        mtf, feats = eth_data_and_features
        p = NYXPipeline()
        ctx_1d = p._build_1d_context(mtf['1d'], feats['1d'])
        ctx_1h = p._build_1h_context(mtf['1h'], feats['1h'])
        df_15m = mtf['15m'].loc['2022-06-01':'2022-06-10']
        feat_15m = feats['15m'].loc['2022-06-01':'2022-06-10']
        cands = p._generate_candidates(df_15m, feat_15m, ctx_1d, ctx_1h,
                                        feat_1h=feats['1h'], feat_1d=feats['1d'])
        if not cands:
            pytest.skip("no candidates in this window")

        # Pick one, check its h1_<col> value matches feats['1h'] at
        # ≤ candidate timestamp (reindex + ffill).
        cand = cands[0]
        ts = cand['timestamp']
        expected_row_idx = feats['1h'].index.get_indexer([ts], method='ffill')[0]
        assert expected_row_idx >= 0, "ffill must find a prior 1h bar"
        expected_row = feats['1h'].iloc[expected_row_idx]

        # Compare a column we know survives the nunique>2 filter.
        col = 'close_vs_ema50'
        assert f'h1_{col}' in cand['features'], \
            f"missing h1_{col} in candidate features"
        assert abs(cand['features'][f'h1_{col}'] - float(expected_row[col])) < 1e-9


# ===========================================================================
class TestETHEdgeWithFullMTF:

    def test_eth_2023_still_positive_with_full_mtf(self, eth_data_and_features):
        """With the expanded MTF features, ETH 2023 OOS must still have
        positive PnL and Sharpe (edge must not degrade)."""
        from src.ml.nyx_pipeline import NYXPipeline
        mtf, feats = eth_data_and_features
        p = NYXPipeline()
        r = p.run(mtf, feats,
                  train_end='2022-12-31',
                  test_start='2023-01-01',
                  test_end='2023-12-31')
        assert r['n_trades'] > 5
        assert r['total_pnl_dollars'] > 0, \
            f"ETH 2023 regressed to {r['total_pnl_dollars']:.2f}"
        assert r['sharpe'] > 0


# ===========================================================================
class TestBTCRegressionWithFullMTF:

    @pytest.fixture(scope='class')
    def btc_data_and_features(self):
        btc_dir = DATA_DIR / 'mtf'
        mtf = {
            tf: pd.read_csv(btc_dir / f'BTCUSDT_{tf}.csv').assign(
                datetime=lambda d: pd.to_datetime(d['datetime'])
            ).set_index('datetime')
            for tf in ('15m', '1h', '1d')
        }
        feats = {
            tf: pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
            for tf in ('15m', '1h', '1d')
        }
        return mtf, feats

    def test_btc_2023_positive_with_full_mtf(self, btc_data_and_features):
        """Expanded MTF must not break BTC edge."""
        from src.ml.nyx_pipeline import NYXPipeline
        mtf, feats = btc_data_and_features
        p = NYXPipeline()
        r = p.run(mtf, feats,
                  train_end='2022-12-31',
                  test_start='2023-01-01',
                  test_end='2023-12-31')
        assert r['n_trades'] > 5
        assert r['total_pnl_dollars'] > 0
