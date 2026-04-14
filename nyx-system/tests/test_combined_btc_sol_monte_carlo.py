"""TDD Tests — Combined BTC + SOL Monte Carlo."""
import pytest


@pytest.fixture(scope='module')
def combined_trades(btc_mtf_data, btc_mtf_features,
                    sol_mtf_data, sol_mtf_features):
    from src.assets.combined_portfolio import combine_trades
    from tests.conftest import run_year
    btc_all, sol_all = [], []
    for year in (2022, 2023):
        tag = 'bear' if year == 2022 else 'bull'
        rb = run_year(btc_mtf_data, btc_mtf_features, year)
        rs = run_year(sol_mtf_data, sol_mtf_features, year)
        for t in rb['trades']:
            t.setdefault('regime', tag)
        for t in rs['trades']:
            t.setdefault('regime', tag)
        btc_all.extend(rb['trades'])
        sol_all.extend(rs['trades'])
    return combine_trades({'BTCUSDT': btc_all, 'SOLUSDT': sol_all})


class TestBTCSOLShuffleMC:

    def test_pct_profitable_high(self, combined_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(combined_trades) < 20:
            pytest.skip()
        r = trade_shuffle_mc(combined_trades, n_sims=2000)
        assert r['pct_profitable'] >= 0.70, \
            f"BTC+SOL pct_profitable={r['pct_profitable']:.0%}"

    def test_median_return_positive(self, combined_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(combined_trades) < 20:
            pytest.skip()
        r = trade_shuffle_mc(combined_trades, n_sims=2000)
        assert r['median_return'] > 0
