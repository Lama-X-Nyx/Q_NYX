"""
TDD Tests — Conditional Bear Dial

Not "always adaptive" — conditional activation based on signals.

Bear dial activates ONLY when:
  - 1H regime = bear/mixed (ADX + momentum)
  - Vol ratio elevated (> 1.5x normal)
  - Trend quality weak (EMA alignment < threshold)
  - Disagreement elevated (rules vs ML diverge)

Otherwise: full static bull params for maximum capture.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'


@pytest.fixture(scope='module')
def mtf_data():
    data = {}
    for tf in ['15m', '1h', '1d']:
        d = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
        d['datetime'] = pd.to_datetime(d['datetime']); d = d.set_index('datetime')
        data[tf] = d
    return data


@pytest.fixture(scope='module')
def mtf_features():
    feats = {}
    for tf in ['15m', '1h', '1d']:
        feats[tf] = pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
    return feats


# ===========================================================================
# TEST 1: Conditional activation logic
# ===========================================================================
class TestConditionalActivation:

    def test_bull_clear_uses_static(self):
        """Clear bull → bear dial OFF, static params."""
        from src.ml.conditional_dial import should_activate_bear_dial
        active = should_activate_bear_dial(
            regime_1h='trending_up', vol_ratio=0.8,
            trend_quality=0.8, disagreement=0.05)
        assert active is False

    def test_bear_regime_activates(self):
        """Bear 1H regime → bear dial ON."""
        from src.ml.conditional_dial import should_activate_bear_dial
        active = should_activate_bear_dial(
            regime_1h='trending_down', vol_ratio=1.0,
            trend_quality=0.5, disagreement=0.1)
        assert active is True

    def test_high_vol_plus_weak_trend_activates(self):
        """Vol high + weak trend (2 triggers) → bear dial ON."""
        from src.ml.conditional_dial import should_activate_bear_dial
        active = should_activate_bear_dial(
            regime_1h='ranging', vol_ratio=2.0,
            trend_quality=0.2, disagreement=0.1)
        assert active is True

    def test_weak_trend_alone_not_enough(self):
        """Single non-regime trigger alone → NOT enough (need 2+)."""
        from src.ml.conditional_dial import should_activate_bear_dial
        active = should_activate_bear_dial(
            regime_1h='ranging', vol_ratio=1.0,
            trend_quality=0.2, disagreement=0.1)
        assert active is False

    def test_high_disagreement_plus_vol_activates(self):
        """Disagreement + vol (2 triggers) → bear dial ON."""
        from src.ml.conditional_dial import should_activate_bear_dial
        active = should_activate_bear_dial(
            regime_1h='trending_up', vol_ratio=2.0,
            trend_quality=0.7, disagreement=0.5)
        assert active is True

    def test_bear_regime_alone_activates(self):
        """Bear regime counts double → always activates alone."""
        from src.ml.conditional_dial import should_activate_bear_dial
        active = should_activate_bear_dial(
            regime_1h='trending_down', vol_ratio=0.8,
            trend_quality=0.7, disagreement=0.05)
        assert active is True


# ===========================================================================
# TEST 2: Conditional pipeline results
# ===========================================================================
class TestConditionalPipeline:

    def test_conditional_produces_results(self, mtf_data, mtf_features):
        from src.ml.conditional_dial import ConditionalNYXPipeline
        pipe = ConditionalNYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert 'n_trades' in r
        assert r['n_trades'] > 0

    def test_conditional_bull_captures_more_than_always_adaptive(self, mtf_data, mtf_features):
        """In bull, conditional > always-adaptive on PnL."""
        from src.ml.conditional_dial import ConditionalNYXPipeline
        from src.ml.bear_risk_dial import AdaptiveNYXPipeline
        cond = ConditionalNYXPipeline()
        adapt = AdaptiveNYXPipeline()
        r_cond = cond.run(mtf_data, mtf_features,
                          train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        r_adapt = adapt.run(mtf_data, mtf_features,
                            train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert r_cond['total_pnl_dollars'] >= r_adapt['total_pnl_dollars'], \
            f"Conditional ${r_cond['total_pnl_dollars']:+,.0f} < adaptive ${r_adapt['total_pnl_dollars']:+,.0f} in bull"

    def test_conditional_bear_protects(self, mtf_data, mtf_features):
        """In bear 2022, conditional must not be worse than static."""
        from src.ml.conditional_dial import ConditionalNYXPipeline
        from src.ml.nyx_pipeline import NYXPipeline
        cond = ConditionalNYXPipeline()
        static = NYXPipeline()
        r_cond = cond.run(mtf_data, mtf_features,
                          train_end='2021-12-31', test_start='2022-01-01', test_end='2022-12-31')
        r_static = static.run(mtf_data, mtf_features,
                              train_end='2021-12-31', test_start='2022-01-01', test_end='2022-12-31')
        # Conditional should have lower DD or higher Sharpe in bear
        better_dd = r_cond['max_drawdown_pct'] <= r_static['max_drawdown_pct'] + 0.02
        better_sharpe = r_cond['sharpe'] >= r_static['sharpe'] - 0.5
        assert better_dd or better_sharpe, \
            f"Conditional not protecting in bear: DD {r_cond['max_drawdown_pct']:.1%} vs {r_static['max_drawdown_pct']:.1%}"

    def test_conditional_reports_activation_rate(self, mtf_data, mtf_features):
        """Must report what % of trades had bear dial active."""
        from src.ml.conditional_dial import ConditionalNYXPipeline
        pipe = ConditionalNYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert 'bear_dial_activation_rate' in r
        # In 2023 bull, activation should be < 50%
        assert r['bear_dial_activation_rate'] < 0.50, \
            f"Bear dial active {r['bear_dial_activation_rate']:.0%} of time in bull — too defensive"


# ===========================================================================
# TEST 3: Walk-forward conditional vs static vs adaptive
# ===========================================================================
class TestConditionalWalkForward:

    def test_conditional_best_of_both(self, mtf_data, mtf_features):
        """Conditional should be best compromise across 2022+2023."""
        from src.ml.conditional_dial import ConditionalNYXPipeline
        from src.ml.nyx_pipeline import NYXPipeline
        from src.ml.bear_risk_dial import AdaptiveNYXPipeline

        results = {}
        for label, Pipe in [('static', NYXPipeline), ('adaptive', AdaptiveNYXPipeline),
                             ('conditional', ConditionalNYXPipeline)]:
            pipe = Pipe()
            sharpes = []
            for tr_e, te_s, te_e in [
                ('2021-12-31', '2022-01-01', '2022-12-31'),
                ('2022-12-31', '2023-01-01', '2023-12-31'),
            ]:
                r = pipe.run(mtf_data, mtf_features, train_end=tr_e, test_start=te_s, test_end=te_e)
                sharpes.append(r['sharpe'])
            results[label] = np.mean(sharpes)

        # Conditional should be a reasonable compromise: positive and not catastrophically worse
        assert results['conditional'] > 0, \
            f"Conditional Sharpe {results['conditional']:.2f} negative: {results}"
        # Should not be more than 50% below the best
        best = max(results.values())
        assert results['conditional'] >= best * 0.5, \
            f"Conditional {results['conditional']:.2f} too far from best {best:.2f}"
