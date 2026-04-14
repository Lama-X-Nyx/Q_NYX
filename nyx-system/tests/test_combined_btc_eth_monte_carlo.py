"""
TDD Tests — Combined BTC + ETH Monte Carlo.

Pools the trades from both assets across 2022 + 2023 and runs
trade_shuffle_mc. Combined should be AT LEAST as robust as the
weaker single asset.
"""
import pytest


@pytest.fixture(scope='module')
def combined_trades(btc_mtf_data, btc_mtf_features,
                    eth_mtf_data, eth_mtf_features):
    from src.assets.combined_portfolio import combine_trades
    from tests.conftest import run_year

    btc_all = []
    eth_all = []
    for year in (2022, 2023):
        tag = 'bear' if year == 2022 else 'bull'
        rb = run_year(btc_mtf_data, btc_mtf_features, year)
        re = run_year(eth_mtf_data, eth_mtf_features, year)
        for t in rb['trades']:
            t.setdefault('regime', tag)
        for t in re['trades']:
            t.setdefault('regime', tag)
        btc_all.extend(rb['trades'])
        eth_all.extend(re['trades'])

    return combine_trades({'BTCUSDT': btc_all, 'ETHUSDT': eth_all})


class TestCombinedShuffleMC:

    def test_pct_profitable_high(self, combined_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(combined_trades) < 20:
            pytest.skip()
        r = trade_shuffle_mc(combined_trades, n_sims=2000)
        assert r['pct_profitable'] >= 0.75, \
            f"combined pct_profitable={r['pct_profitable']:.0%}"

    def test_combined_dd_95_bounded(self, combined_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(combined_trades) < 20:
            pytest.skip()
        r = trade_shuffle_mc(combined_trades, n_sims=2000)
        assert r['dd_95th'] < 0.25, \
            f"combined 95th DD {r['dd_95th']:.1%}"

    def test_combined_median_return_positive(self, combined_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(combined_trades) < 20:
            pytest.skip()
        r = trade_shuffle_mc(combined_trades, n_sims=2000)
        assert r['median_return'] > 0
