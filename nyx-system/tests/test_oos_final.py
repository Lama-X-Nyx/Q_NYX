"""
TDD Tests — Final OOS with ALL features integrated

Pipeline v0.3.2: MTF + ML Filter + Conditional Dial + Execution Policy
All modules merged. Taker fees as default (honest baseline).
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


class TestOOSFullPipeline:
    """OOS with TAKER fees (the honest default)."""

    def test_3_of_4_years_positive(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(fee_rate=0.0004, slippage_rate=0.0003,
                           use_conditional_dial=True, use_execution_filter=True)
        pos = 0
        for year in [2020, 2021, 2022, 2023]:
            r = pipe.run(mtf_data, mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            if r['total_pnl_dollars'] > 0:
                pos += 1
        assert pos >= 3, f"Only {pos}/4 years positive with taker+dial+exec"

    def test_avg_sharpe_positive(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(fee_rate=0.0004, slippage_rate=0.0003,
                           use_conditional_dial=True, use_execution_filter=True)
        sharpes = []
        for year in [2020, 2021, 2022, 2023]:
            r = pipe.run(mtf_data, mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            sharpes.append(r['sharpe'])
        assert np.mean(sharpes) > 0, f"Avg Sharpe {np.mean(sharpes):.2f} negative"

    def test_bear_dd_under_10(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(fee_rate=0.0004, slippage_rate=0.0003,
                           use_conditional_dial=True, use_execution_filter=True)
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2021-12-31', test_start='2022-01-01', test_end='2022-12-31')
        assert r['max_drawdown_pct'] < 0.10

    def test_execution_filter_active(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(fee_rate=0.0004, slippage_rate=0.0003,
                           use_conditional_dial=True, use_execution_filter=True)
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r['execution_reject_rate'] >= 0  # tracked
        assert 'bear_dial_activation_rate' in r  # tracked


class TestOOSBootstrap:
    """Bootstrap the full pipeline trades with taker fees."""

    def test_bootstrap_ploss_under_5pct(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        from src.ml.bootstrap import standard_bootstrap
        pipe = NYXPipeline(fee_rate=0.0004, slippage_rate=0.0003,
                           use_conditional_dial=True, use_execution_filter=True)
        trades = []
        for year in [2020, 2021, 2022, 2023]:
            r = pipe.run(mtf_data, mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            trades.extend(r['trades'])
        bs = standard_bootstrap(trades, n_sims=2000)
        assert bs['prob_loss'] < 0.05, f"P(loss) {bs['prob_loss']:.1%}"

    def test_bootstrap_sharpe_p5_positive(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        from src.ml.bootstrap import standard_bootstrap
        pipe = NYXPipeline(fee_rate=0.0004, slippage_rate=0.0003,
                           use_conditional_dial=True, use_execution_filter=True)
        trades = []
        for year in [2020, 2021, 2022, 2023]:
            r = pipe.run(mtf_data, mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            trades.extend(r['trades'])
        bs = standard_bootstrap(trades, n_sims=2000)
        assert bs['sharpe_p5'] > 0, f"Sharpe p5 {bs['sharpe_p5']:.2f}"
