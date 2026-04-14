"""
TDD Tests — Walk-forward annualized trio.

Invariants:
  - Walk-forward covers 2020 → 2023 (4 full years of overlapping data).
  - 2020 composition is pair (BTC+ETH) because SOL has no training data.
  - 2021, 2022, 2023 compositions include the full trio.
  - Every year combined has ≥ 5 trades.
  - Every year combined is PnL-positive (the pipeline has been validated
    for each asset separately).
  - Annualized CAGR > 0.
  - Pooled trades bootstrap prob_loss < 5 %.
"""
import pytest
import numpy as np
import pandas as pd

from tests.conftest import run_year
from src.assets.combined_portfolio import combine_trades


_YEARS = (2020, 2021, 2022, 2023)
_INITIAL_CAPITAL = 10_000.0


@pytest.fixture(scope='module')
def wf_result(btc_mtf_data, btc_mtf_features,
              eth_mtf_data, eth_mtf_features,
              sol_mtf_data, sol_mtf_features):
    """Run the walk-forward once and share across tests."""
    per_year = {}
    all_trades = []
    for year in _YEARS:
        per_asset_trades = {}
        # BTC + ETH always participate.
        for name, mtf, feats in (
            ('BTCUSDT', btc_mtf_data, btc_mtf_features),
            ('ETHUSDT', eth_mtf_data, eth_mtf_features),
        ):
            r = run_year(mtf, feats, year)
            for t in r['trades']:
                t.setdefault('regime', 'bear' if year == 2022 else 'bull')
            per_asset_trades[name] = r['trades']

        # SOL participates only if training data exists before `year`.
        # SOL starts 2020-08 → for 2020 OOS there is no training yet.
        if year >= 2021:
            r = run_year(sol_mtf_data, sol_mtf_features, year)
            for t in r['trades']:
                t.setdefault('regime', 'bear' if year == 2022 else 'bull')
            per_asset_trades['SOLUSDT'] = r['trades']

        combined = combine_trades(per_asset_trades)
        pnls = np.array([t['net_pnl'] for t in combined], dtype=float) \
            if combined else np.array([], dtype=float)
        eq = _INITIAL_CAPITAL + np.cumsum(pnls)
        ret = (eq[-1] - _INITIAL_CAPITAL) / _INITIAL_CAPITAL if len(eq) else 0.0
        per_year[year] = {
            'composition':       sorted(per_asset_trades.keys()),
            'n_trades':          len(combined),
            'total_pnl':         float(pnls.sum()),
            'return_pct':        float(ret),
            'trades':            combined,
        }
        all_trades.extend(combined)
    return {'per_year': per_year, 'all_trades': all_trades}


# ===========================================================================
class TestComposition:

    def test_2020_is_pair(self, wf_result):
        assert wf_result['per_year'][2020]['composition'] == ['BTCUSDT', 'ETHUSDT']

    def test_2021_has_full_trio(self, wf_result):
        assert set(wf_result['per_year'][2021]['composition']) == {
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
        }

    def test_2022_and_2023_full_trio(self, wf_result):
        for year in (2022, 2023):
            assert set(wf_result['per_year'][year]['composition']) == {
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
            }


class TestPerYearMetrics:

    def test_each_year_has_enough_trades(self, wf_result):
        for year in _YEARS:
            n = wf_result['per_year'][year]['n_trades']
            assert n >= 5, f"year {year}: only {n} trades"

    def test_each_year_pnl_positive(self, wf_result):
        for year in _YEARS:
            pnl = wf_result['per_year'][year]['total_pnl']
            assert pnl > 0, f"year {year}: pnl {pnl:+.0f}"


class TestAnnualizedAggregate:

    def test_cagr_positive(self, wf_result):
        equity = _INITIAL_CAPITAL
        for year in _YEARS:
            equity *= (1 + wf_result['per_year'][year]['return_pct'])
        cagr = (equity / _INITIAL_CAPITAL) ** (1 / len(_YEARS)) - 1
        assert cagr > 0, f"CAGR {cagr:+.2%}"

    def test_years_positive_majority(self, wf_result):
        positives = sum(1 for y in _YEARS
                        if wf_result['per_year'][y]['return_pct'] > 0)
        assert positives >= 3, f"only {positives}/4 years positive"


class TestPooledStress:

    def test_pooled_bootstrap_prob_loss_low(self, wf_result):
        from src.ml.bootstrap import standard_bootstrap
        trades = wf_result['all_trades']
        if len(trades) < 30:
            pytest.skip()
        r = standard_bootstrap(trades, n_sims=1500)
        assert r['prob_loss'] < 0.05, \
            f"pooled prob_loss {r['prob_loss']:.1%}"

    def test_pooled_mc_profitable(self, wf_result):
        from src.ml.monte_carlo import trade_shuffle_mc
        trades = wf_result['all_trades']
        if len(trades) < 30:
            pytest.skip()
        r = trade_shuffle_mc(trades, n_sims=2000)
        assert r['pct_profitable'] >= 0.90
        assert r['median_return'] > 0
