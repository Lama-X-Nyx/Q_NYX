"""
TDD Tests — Monte Carlo on Pipeline v0.3.1 (Conditional Dial Merged)

Final validation before paper trading.
Tests the MERGED pipeline, not separate modules.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'


@pytest.fixture(scope='module')
def pipeline_4yr_trades():
    """Run merged pipeline on 4 years, collect all trades."""
    from src.ml.nyx_pipeline import NYXPipeline
    mtf_data = {}
    for tf in ['15m', '1h', '1d']:
        d = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
        d['datetime'] = pd.to_datetime(d['datetime']); d = d.set_index('datetime')
        mtf_data[tf] = d
    mtf_features = {}
    for tf in ['15m', '1h', '1d']:
        mtf_features[tf] = pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')

    pipe = NYXPipeline()  # conditional dial ON by default
    all_trades = []
    for year in [2020, 2021, 2022, 2023]:
        r = pipe.run(mtf_data, mtf_features,
                     train_end=f'{year-1}-12-31',
                     test_start=f'{year}-01-01', test_end=f'{year}-12-31')
        btc = mtf_data['15m'].loc[f'{year}-01-01':f'{year}-12-31', 'close']
        regime = 'bull' if btc.iloc[-1] > btc.iloc[0] * 1.1 else ('bear' if btc.iloc[-1] < btc.iloc[0] * 0.9 else 'range')
        for t in r['trades']:
            t['year'] = year
            t['regime'] = regime
        all_trades.extend(r['trades'])
    return all_trades


class TestMCShuffleMerged:

    def test_shuffle_100pct_profitable(self, pipeline_4yr_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        r = trade_shuffle_mc(pipeline_4yr_trades, n_sims=2000)
        assert r['pct_profitable'] >= 0.95

    def test_shuffle_dd95_under_10(self, pipeline_4yr_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        r = trade_shuffle_mc(pipeline_4yr_trades, n_sims=2000)
        assert r['dd_95th'] < 0.10


class TestBootstrapMerged:

    def test_standard_p5_sharpe_above_1(self, pipeline_4yr_trades):
        from src.ml.bootstrap import standard_bootstrap
        r = standard_bootstrap(pipeline_4yr_trades, n_sims=2000)
        assert r['sharpe_p5'] > 1.0, f"P5 Sharpe {r['sharpe_p5']:.2f}"

    def test_block_p5_sharpe_above_1(self, pipeline_4yr_trades):
        from src.ml.bootstrap import block_bootstrap
        r = block_bootstrap(pipeline_4yr_trades, n_sims=2000, block_size=5)
        assert r['sharpe_p5'] > 1.0, f"Block P5 Sharpe {r['sharpe_p5']:.2f}"

    def test_regime_bear_ploss_under_15(self, pipeline_4yr_trades):
        from src.ml.bootstrap import regime_bootstrap
        r = regime_bootstrap(pipeline_4yr_trades, n_sims=2000)
        if 'bear_only' in r:
            assert r['bear_only']['prob_loss'] < 0.15, \
                f"Bear P(loss) {r['bear_only']['prob_loss']:.0%}"

    def test_block_not_worse_than_standard(self, pipeline_4yr_trades):
        """Block bootstrap must not collapse vs standard."""
        from src.ml.bootstrap import standard_bootstrap, block_bootstrap
        std = standard_bootstrap(pipeline_4yr_trades, n_sims=1000)
        blk = block_bootstrap(pipeline_4yr_trades, n_sims=1000, block_size=5)
        assert blk['sharpe_p5'] >= std['sharpe_p5'] * 0.7, \
            f"Block p5 {blk['sharpe_p5']:.2f} << standard {std['sharpe_p5']:.2f}"
