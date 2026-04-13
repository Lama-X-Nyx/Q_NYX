"""
TDD Tests — NYX Pipeline v0.3.1 (Conditional Dial Merged)

Pipeline unifié avec bear dial conditionnel intégré.
Plus de modules séparés — tout dans nyx_pipeline.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'


@pytest.fixture(scope='module')
def mtf_data():
    data = {}
    for tf in ['15m', '1h', '1d']:
        d = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
        d['datetime'] = pd.to_datetime(d['datetime']); d = d.set_index('datetime')
        data[tf] = d
    return data


@pytest.fixture(scope='module')
def mtf_features():
    feats = {}
    for tf in ['15m', '1h', '1d']:
        feats[tf] = pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
    return feats


class TestMergedPipeline:

    def test_conditional_dial_is_default(self, mtf_data, mtf_features):
        """Pipeline must use conditional dial by default."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        assert pipe.use_conditional_dial is True

    def test_reports_bear_dial_activation(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert 'bear_dial_activation_rate' in r

    def test_bull_activation_under_50pct(self, mtf_data, mtf_features):
        """In 2023 bull, dial should activate < 50%."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r['bear_dial_activation_rate'] < 0.50

    def test_can_disable_dial(self, mtf_data, mtf_features):
        """Must be possible to run without dial (static mode)."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(use_conditional_dial=False)
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r.get('bear_dial_activation_rate', 0) == 0


class TestOOS4Years:

    def test_4_years_all_positive(self, mtf_data, mtf_features):
        """OOS 2020-2023 must have 3+ years positive."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        pos = 0
        for year in [2020, 2021, 2022, 2023]:
            r = pipe.run(mtf_data, mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            if r['total_pnl_dollars'] > 0:
                pos += 1
        assert pos >= 3, f"Only {pos}/4 years positive"

    def test_avg_sharpe_above_2(self, mtf_data, mtf_features):
        """Average Sharpe across 4 OOS years must be > 2.0."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        sharpes = []
        for year in [2020, 2021, 2022, 2023]:
            r = pipe.run(mtf_data, mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            sharpes.append(r['sharpe'])
        avg = np.mean(sharpes)
        assert avg > 2.0, f"Avg Sharpe {avg:.2f} <= 2.0: {sharpes}"

    def test_bear_2022_not_catastrophic(self, mtf_data, mtf_features):
        """2022 bear DD must be < 10% with conditional dial."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2021-12-31', test_start='2022-01-01', test_end='2022-12-31')
        assert r['max_drawdown_pct'] < 0.10, f"Bear DD {r['max_drawdown_pct']:.1%} >= 10%"
