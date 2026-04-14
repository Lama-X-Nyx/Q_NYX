"""
Module-scoped fixtures shared by the OOS / Monte Carlo / Bootstrap suites
on BTC and ETH. Keeps feature computation + parquet loads out of every
test file.
"""
from pathlib import Path

import pandas as pd
import pytest


DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'
BTC_MTF_DIR = DATA_DIR / 'mtf'

# SOL CSV file names differ from BTC/ETH (timestamp in ms, richer columns).
_SOL_CSV_BY_TF = {
    '15m': 'SOLUSDT_15minutes',
    '1h':  'SOLUSDT_1hour',
    '4h':  'SOLUSDT_4hours',
    '1d':  'SOLUSDT_1day',
}


# ---------------------------------------------------------------------------
# ETH
# ---------------------------------------------------------------------------

def _load_eth_ohlcv(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f'ETHUSDT_{tf}.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


@pytest.fixture(scope='session')
def eth_mtf_data():
    return {tf: _load_eth_ohlcv(tf) for tf in ('15m', '1h', '4h', '1d')}


@pytest.fixture(scope='session')
def eth_mtf_features():
    return {
        tf: pd.read_parquet(FEAT_DIR / f'ETHUSDT_features_{tf}.parquet')
        for tf in ('15m', '1h', '4h', '1d')
    }


# ---------------------------------------------------------------------------
# BTC
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def btc_mtf_data():
    return {
        tf: pd.read_csv(BTC_MTF_DIR / f'BTCUSDT_{tf}.csv').assign(
            datetime=lambda d: pd.to_datetime(d['datetime'])
        ).set_index('datetime')
        for tf in ('15m', '1h', '4h', '1d')
    }


@pytest.fixture(scope='session')
def btc_mtf_features():
    return {
        tf: pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
        for tf in ('15m', '1h', '4h', '1d')
    }


# ---------------------------------------------------------------------------
# SOL
# ---------------------------------------------------------------------------

def _load_sol_ohlcv(tf: str) -> pd.DataFrame:
    name = _SOL_CSV_BY_TF[tf]
    df = pd.read_csv(DATA_DIR / f'{name}.csv',
                     usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime').drop(columns='timestamp')
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


@pytest.fixture(scope='session')
def sol_mtf_data():
    return {tf: _load_sol_ohlcv(tf) for tf in ('15m', '1h', '4h', '1d')}


@pytest.fixture(scope='session')
def sol_mtf_features():
    return {
        tf: pd.read_parquet(FEAT_DIR / f'SOLUSDT_features_{tf}.parquet')
        for tf in ('15m', '1h', '4h', '1d')
    }


# ---------------------------------------------------------------------------
# Helper: run both pipelines for a year and harvest trades.
# ---------------------------------------------------------------------------

def run_year(mtf_data, mtf_features, year: int, pipe=None):
    """Return pipeline result for the given year OOS."""
    from src.ml.nyx_pipeline import NYXPipeline
    pipe = pipe or NYXPipeline()
    return pipe.run(
        mtf_data, mtf_features,
        train_end=f'{year-1}-12-31',
        test_start=f'{year}-01-01',
        test_end=f'{year}-12-31',
    )
