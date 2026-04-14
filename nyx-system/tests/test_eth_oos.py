"""
TDD Tests — ETH OOS walk-forward validation.

Walk-forward train-end → test-year:
  train ≤ YYYY-1, test YYYY
  ... for years 2022, 2023.

Invariants we demand of the ETH edge:
  - ≥ 5 trades per year (otherwise metrics are not meaningful)
  - at least one year out of the window is clearly positive
  - 2022 (bear) max drawdown < 15%
  - in any positive-PnL year, Sharpe > 0
"""
import pytest


class TestETHWalkForward:

    @pytest.fixture(scope='class')
    def by_year(self, eth_mtf_data, eth_mtf_features):
        from tests.conftest import run_year
        results = {}
        for year in (2022, 2023):
            results[year] = run_year(eth_mtf_data, eth_mtf_features, year)
        return results

    def test_each_year_has_enough_trades(self, by_year):
        for year, r in by_year.items():
            assert r['n_trades'] >= 5, \
                f"ETH {year}: only {r['n_trades']} trades — too few for OOS"

    def test_at_least_one_year_positive(self, by_year):
        positives = [y for y, r in by_year.items() if r['total_pnl_dollars'] > 0]
        assert len(positives) >= 1, \
            f"ETH walk-forward: no positive year in {list(by_year)}"

    def test_sharpe_positive_on_pnl_positive_year(self, by_year):
        for year, r in by_year.items():
            if r['total_pnl_dollars'] > 0 and r['n_trades'] > 5:
                assert r['sharpe'] > 0, \
                    f"ETH {year}: PnL>0 but Sharpe={r['sharpe']:.2f}"

    def test_bear_year_drawdown_bounded(self, by_year):
        r = by_year[2022]
        assert r['max_drawdown_pct'] < 0.15, \
            f"ETH 2022 bear: DD {r['max_drawdown_pct']:.1%} > 15%"


class TestETHSpecificYears:
    """Individual year tests so each failure surfaces on its own."""

    def test_eth_2023_bull_sharpe_above_1(self, eth_mtf_data, eth_mtf_features):
        from tests.conftest import run_year
        r = run_year(eth_mtf_data, eth_mtf_features, 2023)
        if r['n_trades'] > 5:
            assert r['sharpe'] > 1.0, \
                f"ETH 2023 bull Sharpe {r['sharpe']:.2f} — expected > 1.0"

    def test_eth_2022_bear_pnl_not_catastrophic(self, eth_mtf_data, eth_mtf_features):
        """2022 was a bear year for ETH (-67%). Strategy must not bleed > 20% of capital."""
        from tests.conftest import run_year
        r = run_year(eth_mtf_data, eth_mtf_features, 2022)
        # Initial capital 10k → allow loss up to 20%.
        assert r['total_pnl_dollars'] > -2_000.0, \
            f"ETH 2022 PnL {r['total_pnl_dollars']:.0f} — catastrophic"
