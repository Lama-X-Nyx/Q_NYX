"""TDD Tests — Combined ETH + SOL OOS."""
import pytest


def _year_trades(mtf_data, mtf_features, year: int):
    from tests.conftest import run_year
    r = run_year(mtf_data, mtf_features, year)
    return r, r['trades']


class TestETHSOLUnion:

    def test_combined_count_equals_sum(self, eth_mtf_data, eth_mtf_features,
                                       sol_mtf_data, sol_mtf_features):
        from src.assets.combined_portfolio import combine_trades
        _, eth = _year_trades(eth_mtf_data, eth_mtf_features, 2023)
        _, sol = _year_trades(sol_mtf_data, sol_mtf_features, 2023)
        combined = combine_trades({'ETHUSDT': eth, 'SOLUSDT': sol})
        assert len(combined) == len(eth) + len(sol)

    def test_tagged(self, eth_mtf_data, eth_mtf_features,
                    sol_mtf_data, sol_mtf_features):
        from src.assets.combined_portfolio import combine_trades
        _, eth = _year_trades(eth_mtf_data, eth_mtf_features, 2023)
        _, sol = _year_trades(sol_mtf_data, sol_mtf_features, 2023)
        combined = combine_trades({'ETHUSDT': eth, 'SOLUSDT': sol})
        assert {t['asset'] for t in combined} == {'ETHUSDT', 'SOLUSDT'}


class TestETHSOLDiversification:

    def test_both_years_positive_combined(
        self, eth_mtf_data, eth_mtf_features, sol_mtf_data, sol_mtf_features,
    ):
        for year in (2022, 2023):
            re, _ = _year_trades(eth_mtf_data, eth_mtf_features, year)
            rs, _ = _year_trades(sol_mtf_data, sol_mtf_features, year)
            total = re['total_pnl_dollars'] + rs['total_pnl_dollars']
            assert total > 0, f"ETH+SOL {year}: combined PnL {total:+.0f}"
