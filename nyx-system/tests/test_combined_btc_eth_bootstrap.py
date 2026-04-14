"""
TDD Tests — Combined BTC + ETH Bootstrap.

The diversification hypothesis: pooling BTC + ETH trades yields a lower
prob_loss and higher p5 Sharpe than either asset alone.

If combined is WORSE than each single asset, the correlation structure
is not compensating the extra noise; we'd want to re-evaluate.
"""
import pytest


@pytest.fixture(scope='module')
def per_asset_trades(btc_mtf_data, btc_mtf_features,
                     eth_mtf_data, eth_mtf_features):
    from tests.conftest import run_year
    out = {'BTCUSDT': [], 'ETHUSDT': []}
    for year in (2022, 2023):
        tag = 'bear' if year == 2022 else 'bull'
        rb = run_year(btc_mtf_data, btc_mtf_features, year)
        re = run_year(eth_mtf_data, eth_mtf_features, year)
        for t in rb['trades']:
            t.setdefault('regime', tag)
        for t in re['trades']:
            t.setdefault('regime', tag)
        out['BTCUSDT'].extend(rb['trades'])
        out['ETHUSDT'].extend(re['trades'])
    return out


@pytest.fixture(scope='module')
def combined_trades(per_asset_trades):
    from src.assets.combined_portfolio import combine_trades
    return combine_trades(per_asset_trades)


class TestCombinedStandardBootstrap:

    def test_prob_loss_under_10pct(self, combined_trades):
        """Combined pool has many more samples → prob_loss should be tight."""
        from src.ml.bootstrap import standard_bootstrap
        if len(combined_trades) < 30:
            pytest.skip()
        r = standard_bootstrap(combined_trades, n_sims=2000)
        assert r['prob_loss'] < 0.10, \
            f"combined std bootstrap P(loss) {r['prob_loss']:.1%}"

    def test_combined_better_than_worst_single(self, combined_trades, per_asset_trades):
        """Combined prob_loss ≤ max of BTC alone and ETH alone."""
        from src.ml.bootstrap import standard_bootstrap
        if len(combined_trades) < 30:
            pytest.skip()
        r_combined = standard_bootstrap(combined_trades, n_sims=1500)
        r_btc = standard_bootstrap(per_asset_trades['BTCUSDT'], n_sims=1500)
        r_eth = standard_bootstrap(per_asset_trades['ETHUSDT'], n_sims=1500)
        worst_single = max(r_btc['prob_loss'], r_eth['prob_loss'])
        # Allow a small margin for MC noise
        assert r_combined['prob_loss'] <= worst_single + 0.02, (
            f"combined P(loss) {r_combined['prob_loss']:.1%} > worst single "
            f"{worst_single:.1%}"
        )


class TestCombinedBlockBootstrap:

    def test_block_prob_loss_under_15pct(self, combined_trades):
        from src.ml.bootstrap import block_bootstrap
        if len(combined_trades) < 30:
            pytest.skip()
        r = block_bootstrap(combined_trades, n_sims=1500, block_size=5)
        assert r['prob_loss'] < 0.15, \
            f"combined block bootstrap P(loss) {r['prob_loss']:.1%}"


class TestCombinedRegimeBootstrap:

    def test_bull_regime_positive(self, combined_trades):
        from src.ml.bootstrap import regime_bootstrap
        if len(combined_trades) < 30:
            pytest.skip()
        r = regime_bootstrap(combined_trades, n_sims=1500)
        if 'bull_only' in r:
            assert r['bull_only']['median_return'] > 0
