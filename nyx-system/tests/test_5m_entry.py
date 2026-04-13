"""
TDD Tests — 5-Minute Entry Timeframe

Architecture originale NYX: entry sur 5m, setup sur 15m.
Le 5m permet un timing plus précis (3x plus de points d'entrée).

MTF complet: 1D context → 1H regime → 15M setup → 5M entry

Data: 5m synthétique dérivé du 15m (subdivide chaque bougie en 3).
Pour les vrais résultats il faudra des données 5m Binance.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'


@pytest.fixture(scope='module')
def mtf_data_5m():
    """Load MTF data including generated 5m."""
    from src.ml.data_5m import generate_5m_from_15m
    data = {}
    for tf in ['15m', '1h', '1d']:
        d = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
        d['datetime'] = pd.to_datetime(d['datetime']); d = d.set_index('datetime')
        data[tf] = d
    data['5m'] = generate_5m_from_15m(data['15m'])
    return data


@pytest.fixture(scope='module')
def mtf_features():
    feats = {}
    for tf in ['15m', '1h', '1d']:
        feats[tf] = pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
    return feats


# ===========================================================================
# TEST 1: 5m data generation
# ===========================================================================
class TestData5m:

    def test_5m_has_3x_more_bars(self, mtf_data_5m):
        """5m must have ~3x more bars than 15m."""
        n_15m = len(mtf_data_5m['15m'])
        n_5m = len(mtf_data_5m['5m'])
        ratio = n_5m / n_15m
        assert 2.5 <= ratio <= 3.5, f"Ratio {ratio:.1f}, expected ~3.0"

    def test_5m_has_ohlcv(self, mtf_data_5m):
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in mtf_data_5m['5m'].columns

    def test_5m_close_aligns_with_15m(self, mtf_data_5m):
        """Every 3rd 5m close should match the 15m close."""
        df_5m = mtf_data_5m['5m']
        df_15m = mtf_data_5m['15m']
        # Resample 5m back to 15m and compare closes
        resampled = df_5m['close'].resample('15min').last().dropna()
        common = resampled.index.intersection(df_15m.index)
        if len(common) > 100:
            diff = abs(resampled.loc[common] - df_15m.loc[common, 'close'])
            max_diff_pct = (diff / df_15m.loc[common, 'close']).max()
            assert max_diff_pct < 0.001, f"5m/15m close mismatch: {max_diff_pct:.4%}"

    def test_5m_high_low_consistent(self, mtf_data_5m):
        """high >= max(open, close), low <= min(open, close)."""
        df = mtf_data_5m['5m']
        assert (df['high'] >= df[['open', 'close']].max(axis=1) - 0.01).all()
        assert (df['low'] <= df[['open', 'close']].min(axis=1) + 0.01).all()


# ===========================================================================
# TEST 2: Pipeline with 5m entry
# ===========================================================================
class TestPipeline5mEntry:

    def test_pipeline_accepts_5m(self, mtf_data_5m, mtf_features):
        """Unified pipeline must accept 5m data."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(entry_tf='5m')
        r = pipe.run(mtf_data_5m, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-06-30')
        assert 'n_trades' in r

    def test_5m_more_candidates_than_15m(self, mtf_data_5m, mtf_features):
        """5m should generate more candidates (3x more entry points)."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe_15m = NYXPipeline(entry_tf='15m')
        pipe_5m = NYXPipeline(entry_tf='5m')
        r_15m = pipe_15m.run(mtf_data_5m, mtf_features,
                             train_end='2022-12-31', test_start='2023-01-01', test_end='2023-06-30')
        r_5m = pipe_5m.run(mtf_data_5m, mtf_features,
                           train_end='2022-12-31', test_start='2023-01-01', test_end='2023-06-30')
        assert r_5m['n_candidates'] >= r_15m['n_candidates'], \
            f"5m candidates {r_5m['n_candidates']} < 15m {r_15m['n_candidates']}"

    def test_5m_positive_sharpe_2023(self, mtf_data_5m, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(entry_tf='5m')
        r = pipe.run(mtf_data_5m, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r['sharpe'] > 0, f"Sharpe {r['sharpe']:.2f} negative on 5m"


# ===========================================================================
# TEST 3: 5m vs 15m comparison
# ===========================================================================
class TestCompare5mVs15m:

    def test_5m_better_or_comparable_to_15m(self, mtf_data_5m, mtf_features):
        """5m entry should be at least comparable to 15m on WR."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe_15m = NYXPipeline(entry_tf='15m')
        pipe_5m = NYXPipeline(entry_tf='5m')
        r_15m = pipe_15m.run(mtf_data_5m, mtf_features,
                             train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        r_5m = pipe_5m.run(mtf_data_5m, mtf_features,
                           train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        # 5m should not catastrophically underperform
        if r_5m['n_trades'] > 5 and r_15m['n_trades'] > 5:
            assert r_5m['win_rate'] >= r_15m['win_rate'] - 0.15, \
                f"5m WR {r_5m['win_rate']:.0%} much worse than 15m {r_15m['win_rate']:.0%}"
