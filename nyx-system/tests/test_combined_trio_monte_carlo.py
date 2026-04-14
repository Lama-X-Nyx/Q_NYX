"""TDD Tests — Trio BTC + ETH + SOL Monte Carlo."""
import pytest


@pytest.fixture(scope='module')
def trio_trades(
    btc_mtf_data, btc_mtf_features,
    eth_mtf_data, eth_mtf_features,
    sol_mtf_data, sol_mtf_features,
):
    from src.assets.combined_portfolio import combine_trades
    from tests.conftest import run_year
    btc_all, eth_all, sol_all = [], [], []
    for year in (2022, 2023):
        tag = 'bear' if year == 2022 else 'bull'
        rb = run_year(btc_mtf_data, btc_mtf_features, year)
        re = run_year(eth_mtf_data, eth_mtf_features, year)
        rs = run_year(sol_mtf_data, sol_mtf_features, year)
        for t in rb['trades']:
            t.setdefault('regime', tag)
        for t in re['trades']:
            t.setdefault('regime', tag)
        for t in rs['trades']:
            t.setdefault('regime', tag)
        btc_all.extend(rb['trades'])
        eth_all.extend(re['trades'])
        sol_all.extend(rs['trades'])
    return combine_trades({
        'BTCUSDT': btc_all, 'ETHUSDT': eth_all, 'SOLUSDT': sol_all,
    })


class TestTrioShuffleMC:

    def test_pct_profitable_very_high(self, trio_trades):
        """With 3 assets pooled, the MC prob_profit should be near-certainty."""
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(trio_trades) < 30:
            pytest.skip()
        r = trade_shuffle_mc(trio_trades, n_sims=2000)
        assert r['pct_profitable'] >= 0.85, \
            f"trio pct_profitable={r['pct_profitable']:.0%}"

    def test_median_return_exceeds_single_asset(
        self, trio_trades, btc_mtf_data, btc_mtf_features,
    ):
        """Pooled median return should exceed the median return of BTC alone
        (more edge sources = stronger aggregate return)."""
        from src.ml.monte_carlo import trade_shuffle_mc
        from tests.conftest import run_year
        if len(trio_trades) < 30:
            pytest.skip()
        r_trio = trade_shuffle_mc(trio_trades, n_sims=1500)

        btc_trades = []
        for y in (2022, 2023):
            btc_trades.extend(run_year(btc_mtf_data, btc_mtf_features, y)['trades'])
        r_btc = trade_shuffle_mc(btc_trades, n_sims=1500)
        assert r_trio['median_return'] > r_btc['median_return'], \
            f"trio {r_trio['median_return']:.1%} ≤ BTC alone {r_btc['median_return']:.1%}"

    def test_trio_dd_bounded(self, trio_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(trio_trades) < 30:
            pytest.skip()
        r = trade_shuffle_mc(trio_trades, n_sims=2000)
        assert r['dd_95th'] < 0.30
