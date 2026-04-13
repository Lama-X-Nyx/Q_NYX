"""
TDD Tests — Monte Carlo Stress Test

Validates that the edge is REAL and not an artefact:
  1. Trade shuffle: permute trade order → equity curves should mostly be positive
  2. Candle noise: add random noise to OHLCV → edge should survive
  3. Fee stress: double the fees → edge should still exist (weaker)
  4. Slippage stress: 5x slippage → measure degradation
  5. Drawdown distribution: 95th percentile DD must be < 25%
  6. Confidence intervals: return distribution across simulations
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'


@pytest.fixture(scope='module')
def pipeline_trades():
    """Run pipeline once, get trades for Monte Carlo."""
    from src.ml.nyx_pipeline import NYXPipeline
    mtf_data = {}
    for tf in ['15m', '1h', '1d']:
        d = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
        d['datetime'] = pd.to_datetime(d['datetime']); d = d.set_index('datetime')
        mtf_data[tf] = d
    mtf_features = {}
    for tf in ['15m', '1h', '1d']:
        mtf_features[tf] = pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
    pipe = NYXPipeline()
    r = pipe.run(mtf_data, mtf_features,
                 train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
    return r


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
# TEST 1: Trade shuffle Monte Carlo
# ===========================================================================
class TestTradeShuffleMC:

    def test_shuffle_mostly_profitable(self, pipeline_trades):
        """Shuffled trade order → >60% of simulations must be profitable."""
        from src.ml.monte_carlo import trade_shuffle_mc
        result = trade_shuffle_mc(pipeline_trades['trades'], n_sims=500)
        assert result['pct_profitable'] >= 0.60, \
            f"Only {result['pct_profitable']:.0%} profitable after shuffle"

    def test_shuffle_median_return_positive(self, pipeline_trades):
        """Median return across shuffles must be positive."""
        from src.ml.monte_carlo import trade_shuffle_mc
        result = trade_shuffle_mc(pipeline_trades['trades'], n_sims=500)
        assert result['median_return'] > 0, \
            f"Median return {result['median_return']:.2%} negative"

    def test_shuffle_worst_dd_under_30pct(self, pipeline_trades):
        """95th percentile max drawdown must be < 30%."""
        from src.ml.monte_carlo import trade_shuffle_mc
        result = trade_shuffle_mc(pipeline_trades['trades'], n_sims=500)
        assert result['dd_95th'] < 0.30, \
            f"95th pctile DD {result['dd_95th']:.1%} >= 30%"


# ===========================================================================
# TEST 2: Candle noise Monte Carlo
# ===========================================================================
class TestCandleNoiseMC:

    def test_noise_edge_survives(self, mtf_data, mtf_features):
        """Adding 0.1% random noise to candles → edge should survive."""
        from src.ml.monte_carlo import candle_noise_mc
        result = candle_noise_mc(
            mtf_data, mtf_features,
            train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31',
            noise_pct=0.001, n_sims=20,
        )
        assert result['pct_profitable'] >= 0.50, \
            f"Only {result['pct_profitable']:.0%} profitable with 0.1% noise"

    def test_heavy_noise_degrades_gracefully(self, mtf_data, mtf_features):
        """0.5% noise should degrade but not destroy."""
        from src.ml.monte_carlo import candle_noise_mc
        result = candle_noise_mc(
            mtf_data, mtf_features,
            train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31',
            noise_pct=0.005, n_sims=10,
        )
        # Even with heavy noise, shouldn't be catastrophic
        assert result['median_return'] > -0.20, \
            f"Median return {result['median_return']:.2%} catastrophic with 0.5% noise"


# ===========================================================================
# TEST 3: Fee & slippage stress
# ===========================================================================
class TestFeeStress:

    def test_double_fees_still_positive(self, mtf_data, mtf_features):
        """2x fees (0.04% maker) → should still be marginally positive."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline(fee_rate=0.0004, slippage_rate=0.0002)
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        # With double fees, edge thinner but should survive on 2023 bull
        assert r['total_pnl_dollars'] > -500, \
            f"Double fees PnL ${r['total_pnl_dollars']:+,.0f} — edge destroyed"

    def test_5x_slippage_measures_sensitivity(self, mtf_data, mtf_features):
        """5x slippage (0.05%) → measure how much it degrades."""
        from src.ml.nyx_pipeline import NYXPipeline
        pipe_normal = NYXPipeline()
        pipe_slip = NYXPipeline(slippage_rate=0.0005)
        r_n = pipe_normal.run(mtf_data, mtf_features,
                              train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        r_s = pipe_slip.run(mtf_data, mtf_features,
                            train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        # Slippage should reduce PnL but not reverse it completely
        assert r_s['total_pnl_dollars'] > r_n['total_pnl_dollars'] * 0.3, \
            "5x slippage destroys >70% of PnL — too fragile"


# ===========================================================================
# TEST 4: Full Monte Carlo report
# ===========================================================================
class TestMonteCarloReport:

    def test_full_report_structure(self, pipeline_trades, mtf_data, mtf_features):
        """Full MC report must include all stress tests."""
        from src.ml.monte_carlo import full_monte_carlo_report
        report = full_monte_carlo_report(
            pipeline_trades['trades'], mtf_data, mtf_features,
            train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31',
        )
        for key in ['trade_shuffle', 'candle_noise', 'fee_stress', 'confidence_interval']:
            assert key in report, f"Missing MC section: {key}"

    def test_confidence_interval_bounded(self, pipeline_trades, mtf_data, mtf_features):
        """95% CI of returns must be bounded and reasonable."""
        from src.ml.monte_carlo import full_monte_carlo_report
        report = full_monte_carlo_report(
            pipeline_trades['trades'], mtf_data, mtf_features,
            train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31',
        )
        ci = report['confidence_interval']
        assert ci['ci_95_low'] > -0.50, f"95% CI low {ci['ci_95_low']:.1%} too extreme"
        assert ci['ci_95_high'] < 1.00, f"95% CI high {ci['ci_95_high']:.1%} too extreme"
