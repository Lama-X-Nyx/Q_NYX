"""
TDD Tests — SOL Monte Carlo.

Trade shuffle MC on SOL trades pooled across 2022 + 2023. SOL is the
most nervous asset, so we allow slightly looser bars than BTC.
"""
import pytest


@pytest.fixture(scope='module')
def sol_trades(sol_mtf_data, sol_mtf_features):
    from tests.conftest import run_year
    trades = []
    for year in (2022, 2023):
        r = run_year(sol_mtf_data, sol_mtf_features, year)
        for t in r['trades']:
            t.setdefault('regime', 'bear' if year == 2022 else 'bull')
        trades.extend(r['trades'])
    return trades


class TestSOLShuffleMC:

    def test_prob_profitable_high(self, sol_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(sol_trades) < 10:
            pytest.skip()
        r = trade_shuffle_mc(sol_trades, n_sims=2000)
        assert r['pct_profitable'] >= 0.55, \
            f"SOL pct_profitable={r['pct_profitable']:.0%}"

    def test_median_return_positive(self, sol_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(sol_trades) < 10:
            pytest.skip()
        r = trade_shuffle_mc(sol_trades, n_sims=2000)
        assert r['median_return'] > 0

    def test_dd_95_bounded(self, sol_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(sol_trades) < 10:
            pytest.skip()
        r = trade_shuffle_mc(sol_trades, n_sims=2000)
        assert r['dd_95th'] < 0.40, \
            f"SOL dd_95th {r['dd_95th']:.1%}"


class TestSOLNoiseMC:

    def test_noise_does_not_destroy_edge(self, sol_mtf_data, sol_mtf_features):
        from src.ml.monte_carlo import candle_noise_mc
        r = candle_noise_mc(
            sol_mtf_data, sol_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-12-31',
            n_sims=10, noise_pct=0.0005,
        )
        assert r['pct_profitable'] >= 0.55, \
            f"SOL candle-noise pct_profitable={r['pct_profitable']:.0%}"
