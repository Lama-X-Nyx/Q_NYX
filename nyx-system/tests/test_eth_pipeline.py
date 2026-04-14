"""
TDD Tests — ETH Pipeline (1h execution timeframe)

Adapt pipeline to run on ETH with 1h as execution TF (no 15m available).
MTF: 1h (exec) + 4h (regime, resampled) + 1d (context, resampled).
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'


@pytest.fixture(scope='module')
def eth_mtf_data():
    """Load ETH 1h, resample to 4h + 1d. Clean invalid rows."""
    df_1h = pd.read_csv(DATA_DIR / 'ETHUSDT_1h.csv')
    df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])
    df_1h = df_1h.set_index('datetime')
    # Clean: remove rows with negative/zero prices
    for col in ['open', 'high', 'low', 'close']:
        df_1h = df_1h[df_1h[col] > 0]
    df_4h = df_1h.resample('4h').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()
    df_1d = df_1h.resample('1D').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()
    return {'15m': df_1h, '1h': df_4h, '1d': df_1d}  # map 1h→15m slot for pipeline


@pytest.fixture(scope='module')
def eth_mtf_features(eth_mtf_data):
    """Compute features from ETH OHLCV using existing feature engine."""
    from src.ml.jesse_features import compute_stationary_features
    feats = {}
    for tf in ['15m', '1h', '1d']:
        # Compute minimal features inline (feature engine works on any TF)
        df = eth_mtf_data[tf]
        feat = compute_stationary_features(df, feature_set='full')
        feats[tf] = feat
    return feats


class TestETHDataQuality:

    def test_eth_data_loaded(self, eth_mtf_data):
        assert len(eth_mtf_data['15m']) > 40000  # 5 years of 1h bars
        assert len(eth_mtf_data['1h']) > 10000   # 5 years of 4h
        assert len(eth_mtf_data['1d']) > 1500    # 5 years of 1d

    def test_eth_prices_valid(self, eth_mtf_data):
        """No negative or zero prices."""
        for tf, df in eth_mtf_data.items():
            for col in ['open', 'high', 'low', 'close']:
                assert (df[col] > 0).all(), f"{tf}.{col} has invalid prices"


class TestETHPipeline:

    def test_pipeline_runs_on_eth(self, eth_mtf_data, eth_mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(eth_mtf_data, eth_mtf_features,
                     train_end='2023-12-31', test_start='2024-01-01', test_end='2024-06-30')
        assert 'n_trades' in r

    def test_eth_produces_trades(self, eth_mtf_data, eth_mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(eth_mtf_data, eth_mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r['n_candidates'] > 0, "No candidates generated on ETH"


class TestETHOOS:

    def test_eth_positive_sharpe_2023(self, eth_mtf_data, eth_mtf_features):
        """ETH 2023 (bull: +90%) must produce positive Sharpe."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(eth_mtf_data, eth_mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        if r['n_trades'] > 5:
            assert r['sharpe'] > 0, f"ETH 2023 Sharpe {r['sharpe']:.2f}"

    def test_eth_3years_most_positive(self, eth_mtf_data, eth_mtf_features):
        """ETH OOS 2022-2024: at least 2/3 years positive."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        pos = 0; total = 0
        for year in [2022, 2023, 2024]:
            r = pipe.run(eth_mtf_data, eth_mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            if r['n_trades'] > 5:
                total += 1
                if r['total_pnl_dollars'] > 0:
                    pos += 1
        if total >= 2:
            assert pos >= total * 0.6, f"Only {pos}/{total} years positive on ETH"
