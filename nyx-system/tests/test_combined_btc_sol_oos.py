"""TDD Tests — Combined BTC + SOL OOS (trade union)."""
import pytest


def _year_trades(mtf_data, mtf_features, year: int):
    from tests.conftest import run_year
    r = run_year(mtf_data, mtf_features, year)
    return r, r['trades']


class TestBTCSOLUnion:

    def test_combined_count_equals_sum(self, btc_mtf_data, btc_mtf_features,
                                       sol_mtf_data, sol_mtf_features):
        from src.assets.combined_portfolio import combine_trades
        _, btc = _year_trades(btc_mtf_data, btc_mtf_features, 2023)
        _, sol = _year_trades(sol_mtf_data, sol_mtf_features, 2023)
        combined = combine_trades({'BTCUSDT': btc, 'SOLUSDT': sol})
        assert len(combined) == len(btc) + len(sol)

    def test_every_trade_tagged(self, btc_mtf_data, btc_mtf_features,
                                 sol_mtf_data, sol_mtf_features):
        from src.assets.combined_portfolio import combine_trades
        _, btc = _year_trades(btc_mtf_data, btc_mtf_features, 2023)
        _, sol = _year_trades(sol_mtf_data, sol_mtf_features, 2023)
        combined = combine_trades({'BTCUSDT': btc, 'SOLUSDT': sol})
        assert {t['asset'] for t in combined} == {'BTCUSDT', 'SOLUSDT'}


class TestBTCSOLDiversification:

    def test_combined_both_years_positive_pnl(
        self, btc_mtf_data, btc_mtf_features, sol_mtf_data, sol_mtf_features,
    ):
        for year in (2022, 2023):
            rb, _ = _year_trades(btc_mtf_data, btc_mtf_features, year)
            rs, _ = _year_trades(sol_mtf_data, sol_mtf_features, year)
            total = rb['total_pnl_dollars'] + rs['total_pnl_dollars']
            assert total > 0, f"BTC+SOL {year}: combined PnL {total:+.0f}"
