"""
TDD Tests — SOL Bootstrap (standard + block + regime).
"""
import pytest


@pytest.fixture(scope='module')
def sol_trades(sol_mtf_data, sol_mtf_features):
    from tests.conftest import run_year
    trades = []
    for year in (2022, 2023):
        r = run_year(sol_mtf_data, sol_mtf_features, year)
        for t in r['trades']:
            t.setdefault('regime', 'bear' if year == 2022 else 'bull')
        trades.extend(r['trades'])
    return trades


class TestSOLStandardBootstrap:

    def test_prob_loss_under_25pct(self, sol_trades):
        from src.ml.bootstrap import standard_bootstrap
        if len(sol_trades) < 10:
            pytest.skip()
        r = standard_bootstrap(sol_trades, n_sims=2000)
        assert r['prob_loss'] < 0.25, \
            f"SOL std bootstrap P(loss) {r['prob_loss']:.1%}"

    def test_median_return_positive(self, sol_trades):
        from src.ml.bootstrap import standard_bootstrap
        if len(sol_trades) < 10:
            pytest.skip()
        r = standard_bootstrap(sol_trades, n_sims=2000)
        assert r['median_return'] > 0


class TestSOLBlockBootstrap:

    def test_block_prob_loss(self, sol_trades):
        from src.ml.bootstrap import block_bootstrap
        if len(sol_trades) < 10:
            pytest.skip()
        r = block_bootstrap(sol_trades, n_sims=1000, block_size=5)
        assert r['prob_loss'] < 0.30, \
            f"SOL block bootstrap P(loss) {r['prob_loss']:.1%}"


class TestSOLRegimeBootstrap:

    def test_regime_stats(self, sol_trades):
        from src.ml.bootstrap import regime_bootstrap
        if len(sol_trades) < 10:
            pytest.skip()
        r = regime_bootstrap(sol_trades, n_sims=1000)
        assert 'regime_counts' in r
        assert ('bull_only' in r or 'bear_only' in r)

    def test_bull_positive(self, sol_trades):
        from src.ml.bootstrap import regime_bootstrap
        if len(sol_trades) < 10:
            pytest.skip()
        r = regime_bootstrap(sol_trades, n_sims=1000)
        if 'bull_only' in r:
            assert r['bull_only']['median_return'] > 0


class TestSOLFullReport:

    def test_p5_return_bounded(self, sol_trades):
        from src.ml.bootstrap import full_bootstrap_report
        if len(sol_trades) < 10:
            pytest.skip()
        r = full_bootstrap_report(sol_trades, n_sims=1000)
        p5 = r['institutional_summary']['p5_return']
        assert p5 > -0.40, f"SOL worst p5 return {p5:.1%}"
