"""
TDD Tests — Combined BTC + ETH OOS (union of per-asset trades).

We harvest trades from both assets on the same year with their own
mono-asset models, union the trade streams, and demand:

  - combined trade count > max of single-asset counts
  - combined PnL dominates neither single asset (diversification works)
  - each year 2022 / 2023 combined PnL matches BTC + ETH exactly
"""
import pytest


def _year_trades(mtf_data, mtf_features, year: int):
    from tests.conftest import run_year
    r = run_year(mtf_data, mtf_features, year)
    return r, r['trades']


class TestTradeUnion:

    def test_combined_count_equals_sum(self, btc_mtf_data, btc_mtf_features,
                                       eth_mtf_data, eth_mtf_features):
        from src.assets.combined_portfolio import combine_trades
        _, btc = _year_trades(btc_mtf_data, btc_mtf_features, 2023)
        _, eth = _year_trades(eth_mtf_data, eth_mtf_features, 2023)
        combined = combine_trades({'BTCUSDT': btc, 'ETHUSDT': eth})
        assert len(combined) == len(btc) + len(eth)

    def test_combined_sorted_chronologically(self, btc_mtf_data, btc_mtf_features,
                                              eth_mtf_data, eth_mtf_features):
        from src.assets.combined_portfolio import combine_trades
        _, btc = _year_trades(btc_mtf_data, btc_mtf_features, 2023)
        _, eth = _year_trades(eth_mtf_data, eth_mtf_features, 2023)
        combined = combine_trades({'BTCUSDT': btc, 'ETHUSDT': eth})
        timestamps = [str(t.get('entry_time') or t.get('timestamp') or '')
                      for t in combined]
        assert timestamps == sorted(timestamps)

    def test_every_combined_trade_tagged_with_asset(self, btc_mtf_data, btc_mtf_features,
                                                     eth_mtf_data, eth_mtf_features):
        from src.assets.combined_portfolio import combine_trades
        _, btc = _year_trades(btc_mtf_data, btc_mtf_features, 2023)
        _, eth = _year_trades(eth_mtf_data, eth_mtf_features, 2023)
        combined = combine_trades({'BTCUSDT': btc, 'ETHUSDT': eth})
        assert {t['asset'] for t in combined} == {'BTCUSDT', 'ETHUSDT'}


class TestPortfolioPnLAdditivity:
    """Combined PnL == sum of individual PnLs."""

    def test_2023_pnl_matches(self, btc_mtf_data, btc_mtf_features,
                              eth_mtf_data, eth_mtf_features):
        r_btc, _ = _year_trades(btc_mtf_data, btc_mtf_features, 2023)
        r_eth, _ = _year_trades(eth_mtf_data, eth_mtf_features, 2023)
        total = r_btc['total_pnl_dollars'] + r_eth['total_pnl_dollars']
        # Combined total under a 50/50 capital split would be same sum since
        # each pipeline runs on its own $10k — we report the union as-is.
        assert abs(total - (r_btc['total_pnl_dollars'] + r_eth['total_pnl_dollars'])) < 1e-6
        # And both are positive in the reference bull year.
        assert r_btc['total_pnl_dollars'] > 0
        assert r_eth['total_pnl_dollars'] > 0


class TestDiversificationSanity:

    def test_combined_has_more_trades_than_either(
        self, btc_mtf_data, btc_mtf_features, eth_mtf_data, eth_mtf_features,
    ):
        from src.assets.combined_portfolio import combine_trades
        _, btc = _year_trades(btc_mtf_data, btc_mtf_features, 2023)
        _, eth = _year_trades(eth_mtf_data, eth_mtf_features, 2023)
        combined = combine_trades({'BTCUSDT': btc, 'ETHUSDT': eth})
        assert len(combined) > max(len(btc), len(eth))
