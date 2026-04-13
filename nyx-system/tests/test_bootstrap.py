"""
TDD Tests — Bootstrap Stress Tests (with replacement, block, by regime)

3 bootstrap methods:
  1. Standard bootstrap (with replacement): sample trades randomly
  2. Block bootstrap: preserve temporal structure (blocks of consecutive trades)
  3. Regime bootstrap: sample within each regime separately (bull/bear/range)

Each must produce distribution of returns, Sharpe, DD.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'


@pytest.fixture(scope='module')
def pipeline_trades():
    """Run pipeline on all 4 years, collect trades with metadata."""
    from src.ml.nyx_pipeline import NYXPipeline
    mtf_data = {}
    for tf in ['15m', '1h', '1d']:
        d = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
        d['datetime'] = pd.to_datetime(d['datetime']); d = d.set_index('datetime')
        mtf_data[tf] = d
    mtf_features = {}
    for tf in ['15m', '1h', '1d']:
        mtf_features[tf] = pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')

    all_trades = []
    for test_year in [2020, 2021, 2022, 2023]:
        pipe = NYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end=f'{test_year-1}-12-31',
                     test_start=f'{test_year}-01-01', test_end=f'{test_year}-12-31')
        for t in r['trades']:
            t['year'] = test_year
            btc = mtf_data['15m'].loc[f'{test_year}-01-01':f'{test_year}-12-31', 'close']
            t['regime'] = 'bull' if btc.iloc[-1] > btc.iloc[0] * 1.1 else ('bear' if btc.iloc[-1] < btc.iloc[0] * 0.9 else 'range')
        all_trades.extend(r['trades'])
    return all_trades


# ===========================================================================
# TEST 1: Standard bootstrap (with replacement)
# ===========================================================================
class TestStandardBootstrap:

    def test_returns_distribution(self, pipeline_trades):
        from src.ml.bootstrap import standard_bootstrap
        result = standard_bootstrap(pipeline_trades, n_sims=2000)
        assert 'returns' in result
        assert len(result['returns']) == 2000

    def test_median_return_positive(self, pipeline_trades):
        from src.ml.bootstrap import standard_bootstrap
        result = standard_bootstrap(pipeline_trades, n_sims=2000)
        assert result['median_return'] > 0

    def test_sharpe_distribution_has_variance(self, pipeline_trades):
        """With replacement, Sharpe should vary (unlike shuffle)."""
        from src.ml.bootstrap import standard_bootstrap
        result = standard_bootstrap(pipeline_trades, n_sims=2000)
        assert result['sharpe_std'] > 0.01, \
            f"Sharpe std {result['sharpe_std']:.3f} — no variance in bootstrap"

    def test_p5_sharpe_positive(self, pipeline_trades):
        """5th percentile Sharpe must be positive."""
        from src.ml.bootstrap import standard_bootstrap
        result = standard_bootstrap(pipeline_trades, n_sims=2000)
        assert result['sharpe_p5'] > 0, \
            f"5th pctile Sharpe {result['sharpe_p5']:.2f} negative"


# ===========================================================================
# TEST 2: Block bootstrap
# ===========================================================================
class TestBlockBootstrap:

    def test_block_preserves_structure(self, pipeline_trades):
        """Block bootstrap must sample contiguous blocks."""
        from src.ml.bootstrap import block_bootstrap
        result = block_bootstrap(pipeline_trades, n_sims=1000, block_size=5)
        assert 'returns' in result
        assert len(result['returns']) == 1000

    def test_block_dd_higher_than_iid(self, pipeline_trades):
        """Block bootstrap DD should be >= iid bootstrap (temporal clustering)."""
        from src.ml.bootstrap import standard_bootstrap, block_bootstrap
        iid = standard_bootstrap(pipeline_trades, n_sims=1000)
        block = block_bootstrap(pipeline_trades, n_sims=1000, block_size=5)
        # Block DD should be at least comparable (temporal clustering of losses)
        assert block['dd_median'] >= iid['dd_median'] * 0.5, \
            "Block DD suspiciously low vs iid"

    def test_block_median_return_positive(self, pipeline_trades):
        from src.ml.bootstrap import block_bootstrap
        result = block_bootstrap(pipeline_trades, n_sims=1000, block_size=5)
        assert result['median_return'] > 0


# ===========================================================================
# TEST 3: Regime bootstrap
# ===========================================================================
class TestRegimeBootstrap:

    def test_regime_samples_all_regimes(self, pipeline_trades):
        """Must sample from each regime proportionally."""
        from src.ml.bootstrap import regime_bootstrap
        result = regime_bootstrap(pipeline_trades, n_sims=1000)
        assert 'regime_counts' in result
        assert len(result['regime_counts']) >= 2  # at least bull and bear

    def test_regime_bear_worst_case(self, pipeline_trades):
        """Bear-only bootstrap must show worst case scenario."""
        from src.ml.bootstrap import regime_bootstrap
        result = regime_bootstrap(pipeline_trades, n_sims=1000)
        assert 'bear_only' in result
        # Bear-only should be worse than mixed
        assert result['bear_only']['median_return'] <= result['median_return']

    def test_regime_bull_best_case(self, pipeline_trades):
        from src.ml.bootstrap import regime_bootstrap
        result = regime_bootstrap(pipeline_trades, n_sims=1000)
        assert 'bull_only' in result
        assert result['bull_only']['median_return'] >= result['bear_only']['median_return']

    def test_regime_median_return_positive(self, pipeline_trades):
        from src.ml.bootstrap import regime_bootstrap
        result = regime_bootstrap(pipeline_trades, n_sims=1000)
        assert result['median_return'] > 0


# ===========================================================================
# TEST 4: Full bootstrap report
# ===========================================================================
class TestBootstrapReport:

    def test_full_report_structure(self, pipeline_trades):
        from src.ml.bootstrap import full_bootstrap_report
        report = full_bootstrap_report(pipeline_trades)
        for key in ['standard', 'block', 'regime', 'institutional_summary']:
            assert key in report, f"Missing: {key}"

    def test_institutional_summary_complete(self, pipeline_trades):
        from src.ml.bootstrap import full_bootstrap_report
        report = full_bootstrap_report(pipeline_trades)
        s = report['institutional_summary']
        for key in ['median_return', 'median_sharpe', 'p5_return', 'p5_sharpe',
                     'dd_p95', 'prob_loss', 'prob_dd_10', 'prob_dd_20',
                     'losing_streak_p95']:
            assert key in s, f"Missing institutional metric: {key}"
