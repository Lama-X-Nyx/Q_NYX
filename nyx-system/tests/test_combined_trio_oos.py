"""TDD Tests — Trio BTC + ETH + SOL OOS."""
import pytest


def _year_trades(mtf_data, mtf_features, year: int):
    from tests.conftest import run_year
    r = run_year(mtf_data, mtf_features, year)
    return r, r['trades']


class TestTrioUnion:

    def test_combined_count_equals_sum(
        self, btc_mtf_data, btc_mtf_features,
        eth_mtf_data, eth_mtf_features,
        sol_mtf_data, sol_mtf_features,
    ):
        from src.assets.combined_portfolio import combine_trades
        _, btc = _year_trades(btc_mtf_data, btc_mtf_features, 2023)
        _, eth = _year_trades(eth_mtf_data, eth_mtf_features, 2023)
        _, sol = _year_trades(sol_mtf_data, sol_mtf_features, 2023)
        combined = combine_trades({
            'BTCUSDT': btc, 'ETHUSDT': eth, 'SOLUSDT': sol,
        })
        assert len(combined) == len(btc) + len(eth) + len(sol)

    def test_three_assets_tagged(
        self, btc_mtf_data, btc_mtf_features,
        eth_mtf_data, eth_mtf_features,
        sol_mtf_data, sol_mtf_features,
    ):
        from src.assets.combined_portfolio import combine_trades
        _, btc = _year_trades(btc_mtf_data, btc_mtf_features, 2023)
        _, eth = _year_trades(eth_mtf_data, eth_mtf_features, 2023)
        _, sol = _year_trades(sol_mtf_data, sol_mtf_features, 2023)
        combined = combine_trades({
            'BTCUSDT': btc, 'ETHUSDT': eth, 'SOLUSDT': sol,
        })
        assert {t['asset'] for t in combined} == {
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
        }


class TestTrioStrictlyBetterThanSubsets:

    def test_trio_trade_count_exceeds_any_pair(
        self, btc_mtf_data, btc_mtf_features,
        eth_mtf_data, eth_mtf_features,
        sol_mtf_data, sol_mtf_features,
    ):
        from src.assets.combined_portfolio import combine_trades
        years = (2022, 2023)
        def _harvest(mtf, feats):
            out = []
            for y in years:
                r = _year_trades(mtf, feats, y)[1]
                out.extend(r)
            return out
        btc = _harvest(btc_mtf_data, btc_mtf_features)
        eth = _harvest(eth_mtf_data, eth_mtf_features)
        sol = _harvest(sol_mtf_data, sol_mtf_features)

        trio = combine_trades({'BTCUSDT': btc, 'ETHUSDT': eth, 'SOLUSDT': sol})
        assert len(trio) > max(len(btc) + len(eth), len(btc) + len(sol),
                                len(eth) + len(sol))


class TestTrioNoLossYear:

    def test_trio_has_no_losing_year(
        self, btc_mtf_data, btc_mtf_features,
        eth_mtf_data, eth_mtf_features,
        sol_mtf_data, sol_mtf_features,
    ):
        for year in (2022, 2023):
            rb, _ = _year_trades(btc_mtf_data, btc_mtf_features, year)
            re, _ = _year_trades(eth_mtf_data, eth_mtf_features, year)
            rs, _ = _year_trades(sol_mtf_data, sol_mtf_features, year)
            total = (rb['total_pnl_dollars'] + re['total_pnl_dollars']
                     + rs['total_pnl_dollars'])
            assert total > 0, f"trio {year} combined PnL {total:+.0f}"
