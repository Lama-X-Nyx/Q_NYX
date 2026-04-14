"""TDD Tests — Combined BTC + SOL Bootstrap."""
import pytest


@pytest.fixture(scope='module')
def per_asset_trades(btc_mtf_data, btc_mtf_features,
                     sol_mtf_data, sol_mtf_features):
    from tests.conftest import run_year
    out = {'BTCUSDT': [], 'SOLUSDT': []}
    for year in (2022, 2023):
        tag = 'bear' if year == 2022 else 'bull'
        rb = run_year(btc_mtf_data, btc_mtf_features, year)
        rs = run_year(sol_mtf_data, sol_mtf_features, year)
        for t in rb['trades']:
            t.setdefault('regime', tag)
        for t in rs['trades']:
            t.setdefault('regime', tag)
        out['BTCUSDT'].extend(rb['trades'])
        out['SOLUSDT'].extend(rs['trades'])
    return out


@pytest.fixture(scope='module')
def combined_trades(per_asset_trades):
    from src.assets.combined_portfolio import combine_trades
    return combine_trades(per_asset_trades)


class TestBTCSOLStandardBootstrap:

    def test_prob_loss_under_15pct(self, combined_trades):
        from src.ml.bootstrap import standard_bootstrap
        if len(combined_trades) < 30:
            pytest.skip()
        r = standard_bootstrap(combined_trades, n_sims=2000)
        assert r['prob_loss'] < 0.15, \
            f"BTC+SOL std prob_loss {r['prob_loss']:.1%}"

    def test_combined_no_worse_than_worst_single(self, combined_trades, per_asset_trades):
        from src.ml.bootstrap import standard_bootstrap
        if len(combined_trades) < 30:
            pytest.skip()
        rc = standard_bootstrap(combined_trades, n_sims=1500)
        rb = standard_bootstrap(per_asset_trades['BTCUSDT'], n_sims=1500)
        rs = standard_bootstrap(per_asset_trades['SOLUSDT'], n_sims=1500)
        worst = max(rb['prob_loss'], rs['prob_loss'])
        assert rc['prob_loss'] <= worst + 0.03, (
            f"combined prob_loss {rc['prob_loss']:.1%} > worst single {worst:.1%}"
        )


class TestBTCSOLBlockBootstrap:

    def test_block_prob_loss(self, combined_trades):
        from src.ml.bootstrap import block_bootstrap
        if len(combined_trades) < 30:
            pytest.skip()
        r = block_bootstrap(combined_trades, n_sims=1500, block_size=5)
        assert r['prob_loss'] < 0.20


class TestBTCSOLRegimeBootstrap:

    def test_bull_regime_positive(self, combined_trades):
        from src.ml.bootstrap import regime_bootstrap
        if len(combined_trades) < 30:
            pytest.skip()
        r = regime_bootstrap(combined_trades, n_sims=1500)
        if 'bull_only' in r:
            assert r['bull_only']['median_return'] > 0
