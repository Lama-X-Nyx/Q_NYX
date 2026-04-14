"""TDD Tests — Trio BTC + ETH + SOL Bootstrap."""
import pytest


@pytest.fixture(scope='module')
def per_asset_trades(
    btc_mtf_data, btc_mtf_features,
    eth_mtf_data, eth_mtf_features,
    sol_mtf_data, sol_mtf_features,
):
    from tests.conftest import run_year
    out = {'BTCUSDT': [], 'ETHUSDT': [], 'SOLUSDT': []}
    for year in (2022, 2023):
        tag = 'bear' if year == 2022 else 'bull'
        rb = run_year(btc_mtf_data, btc_mtf_features, year)
        re = run_year(eth_mtf_data, eth_mtf_features, year)
        rs = run_year(sol_mtf_data, sol_mtf_features, year)
        for t in rb['trades']:
            t.setdefault('regime', tag)
        for t in re['trades']:
            t.setdefault('regime', tag)
        for t in rs['trades']:
            t.setdefault('regime', tag)
        out['BTCUSDT'].extend(rb['trades'])
        out['ETHUSDT'].extend(re['trades'])
        out['SOLUSDT'].extend(rs['trades'])
    return out


@pytest.fixture(scope='module')
def trio_trades(per_asset_trades):
    from src.assets.combined_portfolio import combine_trades
    return combine_trades(per_asset_trades)


class TestTrioStandardBootstrap:

    def test_prob_loss_very_low(self, trio_trades):
        from src.ml.bootstrap import standard_bootstrap
        if len(trio_trades) < 40:
            pytest.skip()
        r = standard_bootstrap(trio_trades, n_sims=2000)
        assert r['prob_loss'] < 0.05, \
            f"trio std prob_loss {r['prob_loss']:.1%}"

    def test_trio_no_worse_than_any_single(self, trio_trades, per_asset_trades):
        """prob_loss of trio ≤ prob_loss of worst single asset (diversification)."""
        from src.ml.bootstrap import standard_bootstrap
        if len(trio_trades) < 40:
            pytest.skip()
        rt = standard_bootstrap(trio_trades, n_sims=1500)
        worst_single = max(
            standard_bootstrap(per_asset_trades[s], n_sims=1500)['prob_loss']
            for s in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
        )
        assert rt['prob_loss'] <= worst_single + 0.02


class TestTrioBlockBootstrap:

    def test_block_prob_loss_low(self, trio_trades):
        from src.ml.bootstrap import block_bootstrap
        if len(trio_trades) < 40:
            pytest.skip()
        r = block_bootstrap(trio_trades, n_sims=1500, block_size=5)
        assert r['prob_loss'] < 0.10


class TestTrioRegimeBootstrap:

    def test_bull_positive(self, trio_trades):
        from src.ml.bootstrap import regime_bootstrap
        if len(trio_trades) < 40:
            pytest.skip()
        r = regime_bootstrap(trio_trades, n_sims=1500)
        if 'bull_only' in r:
            assert r['bull_only']['median_return'] > 0


class TestTrioFullReport:

    def test_p5_return_positive(self, trio_trades):
        from src.ml.bootstrap import full_bootstrap_report
        if len(trio_trades) < 40:
            pytest.skip()
        r = full_bootstrap_report(trio_trades, n_sims=1500)
        p5 = r['institutional_summary']['p5_return']
        assert p5 > 0, f"trio p5_return {p5:.1%}"
