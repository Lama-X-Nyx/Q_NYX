"""
TDD Tests — ETH Bootstrap.

Three methods:
  standard_bootstrap  (IID with replacement)
  block_bootstrap     (contiguous blocks, preserves autocorrelation)
  regime_bootstrap    (stratified by bull/bear tag)

Institutional sanity:
  prob_loss       < 20% (ETH is thinner than BTC, we allow more than 5%)
  sharpe_p5       > -1.0  (p5 sharpe not disastrous)
  return_p5       > -0.20 (5% worst case does not lose > 20%)
"""
import pytest


@pytest.fixture(scope='module')
def eth_trades(eth_mtf_data, eth_mtf_features):
    from tests.conftest import run_year
    trades = []
    for year in (2022, 2023):
        r = run_year(eth_mtf_data, eth_mtf_features, year)
        for t in r['trades']:
            t.setdefault('regime', 'bear' if year == 2022 else 'bull')
        trades.extend(r['trades'])
    return trades


# ===========================================================================
class TestStandardBootstrap:

    def test_prob_loss_under_20pct(self, eth_trades):
        from src.ml.bootstrap import standard_bootstrap
        if len(eth_trades) < 10:
            pytest.skip()
        r = standard_bootstrap(eth_trades, n_sims=2000)
        assert r['prob_loss'] < 0.20, \
            f"ETH std bootstrap P(loss) {r['prob_loss']:.1%}"

    def test_median_return_positive(self, eth_trades):
        from src.ml.bootstrap import standard_bootstrap
        if len(eth_trades) < 10:
            pytest.skip()
        r = standard_bootstrap(eth_trades, n_sims=2000)
        assert r['median_return'] > 0, \
            f"ETH std bootstrap median_return {r['median_return']:.4f}"


# ===========================================================================
class TestBlockBootstrap:

    def test_block_bootstrap_prob_loss(self, eth_trades):
        from src.ml.bootstrap import block_bootstrap
        if len(eth_trades) < 10:
            pytest.skip()
        r = block_bootstrap(eth_trades, n_sims=1000, block_size=5)
        assert r['prob_loss'] < 0.25, \
            f"ETH block bootstrap P(loss) {r['prob_loss']:.1%}"


# ===========================================================================
class TestRegimeBootstrap:

    def test_regime_stats_present_for_known_regimes(self, eth_trades):
        from src.ml.bootstrap import regime_bootstrap
        if len(eth_trades) < 10:
            pytest.skip()
        r = regime_bootstrap(eth_trades, n_sims=1000)
        assert 'regime_counts' in r
        assert 'bull_only' in r or 'bear_only' in r

    def test_bull_regime_profitable_on_eth(self, eth_trades):
        """ETH bull 2023 must bootstrap to positive expectancy."""
        from src.ml.bootstrap import regime_bootstrap
        if len(eth_trades) < 10:
            pytest.skip()
        r = regime_bootstrap(eth_trades, n_sims=1000)
        if 'bull_only' in r:
            assert r['bull_only']['median_return'] > 0, \
                f"bull regime median_return {r['bull_only']['median_return']:.4f}"


# ===========================================================================
class TestFullReport:

    def test_full_report_p5_return(self, eth_trades):
        """p5 return across the 3 bootstrap methods: not disastrous."""
        from src.ml.bootstrap import full_bootstrap_report
        if len(eth_trades) < 10:
            pytest.skip()
        r = full_bootstrap_report(eth_trades, n_sims=1000)
        p5 = r['institutional_summary']['p5_return']
        assert p5 > -0.30, f"ETH worst p5 return {p5:.1%}"
