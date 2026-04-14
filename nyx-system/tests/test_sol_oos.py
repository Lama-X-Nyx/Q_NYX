"""
TDD Tests — SOL OOS walk-forward.

SOL is the most nervous asset of the trio (momentum_fast session profile,
slippage high). Walk-forward on 2022 (bear) + 2023 (bull).

Invariants:
  - ≥ 5 trades per year
  - at least one year positive
  - 2022 bear DD bounded < 20%
  - bull year Sharpe > 0
"""
import pytest


class TestSOLWalkForward:

    @pytest.fixture(scope='class')
    def by_year(self, sol_mtf_data, sol_mtf_features):
        from tests.conftest import run_year
        return {year: run_year(sol_mtf_data, sol_mtf_features, year)
                for year in (2022, 2023)}

    def test_each_year_has_trades(self, by_year):
        for year, r in by_year.items():
            assert r['n_trades'] >= 5, \
                f"SOL {year}: only {r['n_trades']} trades"

    def test_at_least_one_year_positive(self, by_year):
        positives = [y for y, r in by_year.items() if r['total_pnl_dollars'] > 0]
        assert len(positives) >= 1, \
            f"SOL walk-forward: no positive year in {list(by_year)}"

    def test_bear_dd_bounded(self, by_year):
        r = by_year[2022]
        assert r['max_drawdown_pct'] < 0.20, \
            f"SOL 2022 bear DD {r['max_drawdown_pct']:.1%}"

    def test_bull_year_sharpe_positive(self, by_year):
        r = by_year[2023]
        if r['n_trades'] > 5:
            assert r['sharpe'] > 0, f"SOL 2023 Sharpe {r['sharpe']:.2f}"


class TestSOLNonCatastrophic:

    def test_2022_pnl_bounded_below(self, sol_mtf_data, sol_mtf_features):
        """SOL spot did -94% in 2022 — strategy must not bleed > 25% of capital."""
        from tests.conftest import run_year
        r = run_year(sol_mtf_data, sol_mtf_features, 2022)
        assert r['total_pnl_dollars'] > -2_500.0, \
            f"SOL 2022 PnL {r['total_pnl_dollars']:.0f}"
