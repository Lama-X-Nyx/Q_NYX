"""
TDD Tests — Pipeline with TAKER fees (0.04% per side)

All previous results used maker fees (0.02%).
In reality: taker = default execution, maker = best case only with limit orders.
Must validate edge survives taker fees.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'

TAKER_FEE = 0.0004    # 0.04% per side (Binance Futures)
TAKER_SLIP = 0.0003   # 0.03% slippage (market order)


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


# ===========================================================================
# TEST 1: Pipeline must accept taker fees
# ===========================================================================
class TestTakerConfig:

    def test_pipeline_accepts_taker_fees(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(fee_rate=TAKER_FEE, slippage_rate=TAKER_SLIP)
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert 'n_trades' in r

    def test_taker_fees_higher_than_maker(self, mtf_data, mtf_features):
        from src.ml.nyx_pipeline import NYXPipeline
        maker = NYXPipeline(fee_rate=0.0002, slippage_rate=0.0001)
        taker = NYXPipeline(fee_rate=TAKER_FEE, slippage_rate=TAKER_SLIP)
        r_m = maker.run(mtf_data, mtf_features,
                        train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        r_t = taker.run(mtf_data, mtf_features,
                        train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r_t['total_fees'] >= r_m['total_fees'], "Taker fees should be higher"


# ===========================================================================
# TEST 2: OOS year-by-year with taker
# ===========================================================================
class TestTakerOOS:

    def test_3_of_4_years_positive_taker(self, mtf_data, mtf_features):
        """With taker fees, at least 3/4 years must be positive."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(fee_rate=TAKER_FEE, slippage_rate=TAKER_SLIP)
        pos = 0
        for year in [2020, 2021, 2022, 2023]:
            r = pipe.run(mtf_data, mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            if r['total_pnl_dollars'] > 0:
                pos += 1
        assert pos >= 3, f"Only {pos}/4 years positive with taker fees"

    def test_avg_sharpe_positive_taker(self, mtf_data, mtf_features):
        """Avg Sharpe must be > 0 with taker fees."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(fee_rate=TAKER_FEE, slippage_rate=TAKER_SLIP)
        sharpes = []
        for year in [2020, 2021, 2022, 2023]:
            r = pipe.run(mtf_data, mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            sharpes.append(r['sharpe'])
        assert np.mean(sharpes) > 0, f"Avg Sharpe {np.mean(sharpes):.2f} negative with taker"


# ===========================================================================
# TEST 3: Bootstrap survives taker
# ===========================================================================
class TestTakerBootstrap:

    def test_taker_bootstrap_ploss_under_10pct(self, mtf_data, mtf_features):
        """With taker fees, P(loss) must be < 10%."""
        from src.ml.nyx_pipeline import NYXPipeline
        from src.ml.bootstrap import standard_bootstrap
        pipe = NYXPipeline(fee_rate=TAKER_FEE, slippage_rate=TAKER_SLIP)
        trades = []
        for year in [2020, 2021, 2022, 2023]:
            r = pipe.run(mtf_data, mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            trades.extend(r['trades'])
        bs = standard_bootstrap(trades, n_sims=2000)
        assert bs['prob_loss'] < 0.10, f"Taker P(loss) {bs['prob_loss']:.1%} >= 10%"

    def test_taker_sharpe_p5_positive(self, mtf_data, mtf_features):
        """5th percentile Sharpe must be > 0 with taker."""
        from src.ml.nyx_pipeline import NYXPipeline
        from src.ml.bootstrap import standard_bootstrap
        pipe = NYXPipeline(fee_rate=TAKER_FEE, slippage_rate=TAKER_SLIP)
        trades = []
        for year in [2020, 2021, 2022, 2023]:
            r = pipe.run(mtf_data, mtf_features,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01', test_end=f'{year}-12-31')
            trades.extend(r['trades'])
        bs = standard_bootstrap(trades, n_sims=2000)
        assert bs['sharpe_p5'] > 0, f"Taker Sharpe p5 {bs['sharpe_p5']:.2f} negative"
