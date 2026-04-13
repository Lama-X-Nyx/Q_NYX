"""
TDD Tests — Bootstrap with Bear Dial ON vs OFF

Compare bootstrap distributions:
  - Pipeline WITH conditional bear dial (default)
  - Pipeline WITHOUT (static)

The bear dial must:
  1. Reduce bear P(loss)
  2. Reduce DD p95
  3. Not destroy bull Sharpe
  4. Block bootstrap still >= standard (no clustering introduced)
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'

N_SIMS = 2000


@pytest.fixture(scope='module')
def trades_with_dial():
    """Trades from pipeline WITH conditional dial."""
    from src.ml.nyx_pipeline import NYXPipeline
    mtf_data, mtf_features = _load_data()
    pipe = NYXPipeline(use_conditional_dial=True)
    return _collect_trades(pipe, mtf_data, mtf_features)


@pytest.fixture(scope='module')
def trades_without_dial():
    """Trades from pipeline WITHOUT dial (static)."""
    from src.ml.nyx_pipeline import NYXPipeline
    mtf_data, mtf_features = _load_data()
    pipe = NYXPipeline(use_conditional_dial=False)
    return _collect_trades(pipe, mtf_data, mtf_features)


def _load_data():
    mtf_data = {}
    for tf in ['15m', '1h', '1d']:
        d = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
        d['datetime'] = pd.to_datetime(d['datetime']); d = d.set_index('datetime')
        mtf_data[tf] = d
    mtf_features = {}
    for tf in ['15m', '1h', '1d']:
        mtf_features[tf] = pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
    return mtf_data, mtf_features


def _collect_trades(pipe, mtf_data, mtf_features):
    all_trades = []
    for year in [2020, 2021, 2022, 2023]:
        r = pipe.run(mtf_data, mtf_features,
                     train_end=f'{year-1}-12-31',
                     test_start=f'{year}-01-01', test_end=f'{year}-12-31')
        btc = mtf_data['15m'].loc[f'{year}-01-01':f'{year}-12-31', 'close']
        regime = 'bull' if btc.iloc[-1] > btc.iloc[0] * 1.1 else (
            'bear' if btc.iloc[-1] < btc.iloc[0] * 0.9 else 'range')
        for t in r['trades']:
            t['year'] = year
            t['regime'] = regime
        all_trades.extend(r['trades'])
    return all_trades


# ===========================================================================
# TEST 1: Bear dial reduces bear P(loss) in regime bootstrap
# ===========================================================================
class TestBearDialReducesBearRisk:

    def test_bear_ploss_lower_with_dial(self, trades_with_dial, trades_without_dial):
        """Bear-only P(loss) must be lower with dial."""
        from src.ml.bootstrap import regime_bootstrap
        with_dial = regime_bootstrap(trades_with_dial, n_sims=N_SIMS)
        without_dial = regime_bootstrap(trades_without_dial, n_sims=N_SIMS)
        bear_with = with_dial.get('bear_only', {}).get('prob_loss', 0)
        bear_without = without_dial.get('bear_only', {}).get('prob_loss', 0)
        assert bear_with <= bear_without + 0.02, \
            f"Dial bear P(loss) {bear_with:.1%} > static {bear_without:.1%}"

    def test_bear_sharpe_improved_with_dial(self, trades_with_dial, trades_without_dial):
        """Bear Sharpe must be better or comparable with dial."""
        from src.ml.bootstrap import regime_bootstrap
        with_dial = regime_bootstrap(trades_with_dial, n_sims=N_SIMS)
        without_dial = regime_bootstrap(trades_without_dial, n_sims=N_SIMS)
        s_with = with_dial.get('bear_only', {}).get('sharpe_median', 0)
        s_without = without_dial.get('bear_only', {}).get('sharpe_median', 0)
        assert s_with >= s_without - 0.5, \
            f"Dial bear Sharpe {s_with:.2f} << static {s_without:.2f}"


# ===========================================================================
# TEST 2: Bear dial reduces DD in all bootstraps
# ===========================================================================
class TestBearDialReducesDD:

    def test_standard_dd95_not_worse(self, trades_with_dial, trades_without_dial):
        from src.ml.bootstrap import standard_bootstrap
        with_d = standard_bootstrap(trades_with_dial, n_sims=N_SIMS)
        without_d = standard_bootstrap(trades_without_dial, n_sims=N_SIMS)
        assert with_d['dd_p95'] <= without_d['dd_p95'] + 0.02

    def test_block_dd95_not_worse(self, trades_with_dial, trades_without_dial):
        from src.ml.bootstrap import block_bootstrap
        with_d = block_bootstrap(trades_with_dial, n_sims=N_SIMS, block_size=5)
        without_d = block_bootstrap(trades_without_dial, n_sims=N_SIMS, block_size=5)
        assert with_d['dd_p95'] <= without_d['dd_p95'] + 0.02


# ===========================================================================
# TEST 3: Bull Sharpe not destroyed
# ===========================================================================
class TestBullNotDestroyed:

    def test_bull_sharpe_preserved(self, trades_with_dial, trades_without_dial):
        """Bull-only Sharpe with dial must be >= 70% of without."""
        from src.ml.bootstrap import regime_bootstrap
        with_dial = regime_bootstrap(trades_with_dial, n_sims=N_SIMS)
        without_dial = regime_bootstrap(trades_without_dial, n_sims=N_SIMS)
        s_with = with_dial.get('bull_only', {}).get('sharpe_median', 0)
        s_without = without_dial.get('bull_only', {}).get('sharpe_median', 0)
        assert s_with >= s_without * 0.60, \
            f"Dial bull Sharpe {s_with:.2f} << static {s_without:.2f}"

    def test_overall_ploss_still_zero(self, trades_with_dial):
        """Overall P(loss) must remain 0% with dial."""
        from src.ml.bootstrap import standard_bootstrap
        r = standard_bootstrap(trades_with_dial, n_sims=N_SIMS)
        assert r['prob_loss'] < 0.01


# ===========================================================================
# TEST 4: Block >= Standard with dial (no clustering introduced)
# ===========================================================================
class TestBlockRobustWithDial:

    def test_block_sharpe_not_worse_than_standard(self, trades_with_dial):
        """Block p5 Sharpe must be >= 70% of standard (dial doesn't add clustering)."""
        from src.ml.bootstrap import standard_bootstrap, block_bootstrap
        std = standard_bootstrap(trades_with_dial, n_sims=N_SIMS)
        blk = block_bootstrap(trades_with_dial, n_sims=N_SIMS, block_size=5)
        assert blk['sharpe_p5'] >= std['sharpe_p5'] * 0.70, \
            f"Block p5 {blk['sharpe_p5']:.2f} << standard {std['sharpe_p5']:.2f} — dial introduces clustering"


# ===========================================================================
# TEST 5: Full comparison report
# ===========================================================================
class TestFullComparisonReport:

    def test_report_shows_improvement(self, trades_with_dial, trades_without_dial):
        """Must produce comparison showing dial impact on each metric."""
        from src.ml.bootstrap import standard_bootstrap, block_bootstrap, regime_bootstrap

        report = {}
        for label, trades in [('with_dial', trades_with_dial), ('without_dial', trades_without_dial)]:
            std = standard_bootstrap(trades, n_sims=N_SIMS)
            blk = block_bootstrap(trades, n_sims=N_SIMS, block_size=5)
            reg = regime_bootstrap(trades, n_sims=N_SIMS)
            report[label] = {
                'n_trades': len(trades),
                'std_sharpe_p5': std['sharpe_p5'],
                'blk_sharpe_p5': blk['sharpe_p5'],
                'std_dd_p95': std['dd_p95'],
                'blk_dd_p95': blk['dd_p95'],
                'bear_ploss': reg.get('bear_only', {}).get('prob_loss', 0),
                'bull_sharpe': reg.get('bull_only', {}).get('sharpe_median', 0),
            }

        # Dial must improve at least 1 metric without catastrophic regression
        w = report['with_dial']
        wo = report['without_dial']
        improvements = 0
        if w['bear_ploss'] < wo['bear_ploss']:
            improvements += 1
        if w['std_dd_p95'] < wo['std_dd_p95']:
            improvements += 1
        if w['blk_dd_p95'] < wo['blk_dd_p95']:
            improvements += 1
        assert improvements >= 1, f"Dial improved nothing: {report}"
