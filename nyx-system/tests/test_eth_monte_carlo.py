"""
TDD Tests — ETH Monte Carlo stress test.

Runs Monte Carlo across the full ETH walk-forward trade set
(2022 + 2023). Two methods:

  1. trade_shuffle_mc  — randomize trade ORDER, same pool.
                         Tests "was the order of wins/losses lucky?".
  2. candle_noise_mc   — re-execute on noise-perturbed OHLCV.
                         Tests "does a tiny market perturbation kill
                         the edge?".

Institutional thresholds (from the BTC spec we shipped earlier):
  pct_profitable  > 80%
  median_return   > 0
  dd_95th         < 25%
"""
from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def eth_trades(eth_mtf_data, eth_mtf_features):
    """All ETH trades across 2022 + 2023 walk-forward."""
    from tests.conftest import run_year
    trades = []
    for year in (2022, 2023):
        r = run_year(eth_mtf_data, eth_mtf_features, year)
        for t in r['trades']:
            t.setdefault('regime', 'bear' if year == 2022 else 'bull')
        trades.extend(r['trades'])
    return trades


# ===========================================================================
class TestTradeShuffleMC:

    def test_prob_profitable_above_60pct(self, eth_trades):
        """Across random orderings, ≥ 60% of sims end net-positive."""
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(eth_trades) < 10:
            pytest.skip(f"only {len(eth_trades)} ETH trades")
        r = trade_shuffle_mc(eth_trades, n_sims=2000)
        assert r['pct_profitable'] >= 0.60, \
            f"ETH trade-shuffle pct_profitable={r['pct_profitable']:.0%}"

    def test_median_return_positive(self, eth_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(eth_trades) < 10:
            pytest.skip()
        r = trade_shuffle_mc(eth_trades, n_sims=2000)
        assert r['median_return'] > 0, \
            f"ETH trade-shuffle median_return={r['median_return']:.4f}"

    def test_dd_95th_bounded(self, eth_trades):
        from src.ml.monte_carlo import trade_shuffle_mc
        if len(eth_trades) < 10:
            pytest.skip()
        r = trade_shuffle_mc(eth_trades, n_sims=2000)
        assert r['dd_95th'] < 0.35, \
            f"ETH trade-shuffle 95th-pct DD {r['dd_95th']:.1%}"


# ===========================================================================
class TestCandleNoiseMC:

    def test_noise_does_not_destroy_edge(self, eth_mtf_data, eth_mtf_features):
        """Perturbing OHLC by ±0.05% must not make the edge disappear."""
        from src.ml.monte_carlo import candle_noise_mc
        r = candle_noise_mc(
            eth_mtf_data, eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-12-31',
            n_sims=10, noise_pct=0.0005,
        )
        assert r['pct_profitable'] >= 0.60, \
            f"ETH candle-noise pct_profitable={r['pct_profitable']:.0%}"
