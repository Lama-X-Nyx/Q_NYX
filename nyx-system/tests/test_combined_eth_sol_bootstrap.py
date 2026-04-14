"""TDD Tests — Combined ETH + SOL Bootstrap."""
import pytest


@pytest.fixture(scope='module')
def per_asset_trades(eth_mtf_data, eth_mtf_features,
                     sol_mtf_data, sol_mtf_features):
    from tests.conftest import run_year
    out = {'ETHUSDT': [], 'SOLUSDT': []}
    for year in (2022, 2023):
        tag = 'bear' if year == 2022 else 'bull'
        re = run_year(eth_mtf_data, eth_mtf_features, year)
        rs = run_year(sol_mtf_data, sol_mtf_features, year)
        for t in re['trades']:
            t.setdefault('regime', tag)
        for t in rs['trades']:
            t.setdefault('regime', tag)
        out['ETHUSDT'].extend(re['trades'])
        out['SOLUSDT'].extend(rs['trades'])
    return out


@pytest.fixture(scope='module')
def combined_trades(per_asset_trades):
    from src.assets.combined_portfolio import combine_trades
    return combine_trades(per_asset_trades)


class TestETHSOLStandardBootstrap:

    def test_prob_loss_under_15pct(self, combined_trades):
        from src.ml.bootstrap import standard_bootstrap
        if len(combined_trades) < 30:
            pytest.skip()
        r = standard_bootstrap(combined_trades, n_sims=2000)
        assert r['prob_loss'] < 0.15


class TestETHSOLBlockBootstrap:

    def test_block_prob_loss(self, combined_trades):
        from src.ml.bootstrap import block_bootstrap
        if len(combined_trades) < 30:
            pytest.skip()
        r = block_bootstrap(combined_trades, n_sims=1500, block_size=5)
        assert r['prob_loss'] < 0.20
