"""
TDD Tests — ETH Training (real 15m data)

Now that we have genuine ETH 15m data (not resampled from 1h), retrain
the pipeline on ETH end-to-end and measure if the edge holds.

The BTC edge is the reference baseline. We track:
  - does the pipeline run on ETH and produce trades?
  - is OOS Sharpe ≥ 0 on at least one recent year?
  - are the features saved to data/features/ for reuse?
  - per-asset model artefact saved to models/ETHUSDT/

Separate file from the old test_eth_pipeline.py — that one used
resampled 1h-as-15m and is kept to document the old behavior.
"""
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'
MODEL_DIR = Path(__file__).parent.parent / 'models'


# ---------------------------------------------------------------------------
# Module-scoped fixtures — compute features once, reuse across tests
# ---------------------------------------------------------------------------

def _load_eth_ohlcv(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f'ETHUSDT_{tf}.csv')
    # The ETH CSVs have an unnamed index column.
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    # Drop rows with any non-positive prices.
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


@pytest.fixture(scope='module')
def eth_mtf_data():
    """Native ETH 15m / 1h / 1d (no resampling)."""
    return {
        '15m': _load_eth_ohlcv('15m'),
        '1h':  _load_eth_ohlcv('1h'),
        '1d':  _load_eth_ohlcv('1d'),
    }


@pytest.fixture(scope='module')
def eth_mtf_features(eth_mtf_data):
    """Compute stationary features per TF. Cache to parquet."""
    from src.ml.jesse_features import compute_stationary_features
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    feats = {}
    for tf in ('15m', '1h', '1d'):
        cache = FEAT_DIR / f'ETHUSDT_features_{tf}.parquet'
        if cache.exists():
            feats[tf] = pd.read_parquet(cache)
        else:
            feat = compute_stationary_features(eth_mtf_data[tf], feature_set='full')
            feat.to_parquet(cache)
            feats[tf] = feat
    return feats


# ===========================================================================
# TEST 1: data quality on real 15m
# ===========================================================================
class TestETHDataQuality:

    def test_eth_15m_has_expected_coverage(self, eth_mtf_data):
        # 15m bars over ~4 years (Dec 2019 → Jan 2024): ~140k bars.
        assert len(eth_mtf_data['15m']) > 100_000

    def test_eth_prices_all_positive(self, eth_mtf_data):
        for tf, df in eth_mtf_data.items():
            for col in ('open', 'high', 'low', 'close'):
                assert (df[col] > 0).all(), f"{tf}.{col} has invalid prices"

    def test_eth_ohlc_geometry(self, eth_mtf_data):
        for tf, df in eth_mtf_data.items():
            assert (df['high'] >= df['low']).all(), f"{tf}: high<low"
            assert (df['close'] <= df['high']).all(), f"{tf}: close>high"
            assert (df['close'] >= df['low']).all(), f"{tf}: close<low"


# ===========================================================================
# TEST 2: features
# ===========================================================================
class TestETHFeatures:

    def test_features_computed_for_every_tf(self, eth_mtf_features):
        assert set(eth_mtf_features.keys()) == {'15m', '1h', '1d'}

    def test_features_cached_to_parquet(self, eth_mtf_features):
        for tf in ('15m', '1h', '1d'):
            assert (FEAT_DIR / f'ETHUSDT_features_{tf}.parquet').exists()

    def test_features_same_length_as_ohlcv(self, eth_mtf_data, eth_mtf_features):
        for tf in ('15m', '1h', '1d'):
            assert len(eth_mtf_features[tf]) == len(eth_mtf_data[tf])


# ===========================================================================
# TEST 3: pipeline runs on ETH with real 15m + produces candidates
# ===========================================================================
class TestETHPipelineRun:

    def test_pipeline_trains_and_tests_on_eth(self, eth_mtf_data, eth_mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(
            eth_mtf_data, eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-12-31',
        )
        assert 'n_trades' in r
        assert 'n_candidates' in r
        assert r['n_candidates'] > 0, "ETH must produce some hard-gate candidates"

    def test_eth_2023_positive_sharpe_if_enough_trades(self, eth_mtf_data, eth_mtf_features):
        """2023 was bull on ETH (+90%). Strategy must not lose on it."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(
            eth_mtf_data, eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-12-31',
        )
        if r['n_trades'] > 5:
            assert r['sharpe'] > 0, f"ETH 2023 Sharpe {r['sharpe']:.2f} must be positive"


# ===========================================================================
# TEST 4: model artefact persisted per asset
# ===========================================================================
class TestETHModelArtifact:

    def test_eth_model_saved_under_per_asset_dir(self, eth_mtf_data, eth_mtf_features,
                                                  tmp_path_factory):
        """Train an ETH model and save it under models/ETHUSDT/."""
        from src.ml.train_asset_model import train_and_save
        out_dir = tmp_path_factory.mktemp("eth_models")
        result = train_and_save(
            symbol='ETHUSDT',
            mtf_data=eth_mtf_data,
            mtf_features=eth_mtf_features,
            train_end='2022-12-31',
            out_dir=out_dir,
        )
        assert (out_dir / 'ml_filter_v1.pkl').exists()
        assert (out_dir / 'scaler.pkl').exists()
        assert (out_dir / 'feature_names.json').exists()
        assert 'n_train_candidates' in result
        assert result['n_train_candidates'] > 100


# ===========================================================================
# TEST 5: BTC regression — edge still present after refactor
# ===========================================================================
class TestBTCEdgeRegression:
    """Quick regression: BTC 2023 run with the same pipeline must still
    produce a positive expectancy. This catches any accidental change in
    behavior of NYXPipeline during the ETH work."""

    @pytest.fixture(scope='class')
    def btc_mtf_data(self):
        import pandas as pd
        btc_dir = DATA_DIR / 'mtf'
        return {
            tf: pd.read_csv(btc_dir / f'BTCUSDT_{tf}.csv').assign(
                datetime=lambda d: pd.to_datetime(d['datetime'])
            ).set_index('datetime')
            for tf in ('15m', '1h', '1d')
        }

    @pytest.fixture(scope='class')
    def btc_mtf_features(self):
        feat_dir = Path(__file__).parent.parent / 'data' / 'features'
        return {
            tf: pd.read_parquet(feat_dir / f'BTCUSDT_features_{tf}.parquet')
            for tf in ('15m', '1h', '1d')
        }

    def test_btc_2023_still_positive(self, btc_mtf_data, btc_mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(
            btc_mtf_data, btc_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-12-31',
        )
        # BTC was the reference edge: > 5 trades, total PnL > 0.
        assert r['n_trades'] > 5, "BTC must still generate enough trades"
        assert r['total_pnl_dollars'] > 0, \
            f"BTC 2023 PnL regressed: {r['total_pnl_dollars']:.2f}"
