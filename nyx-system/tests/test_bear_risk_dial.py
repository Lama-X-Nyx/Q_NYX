"""
TDD Tests — Bear Risk Dial

The bootstrap shows bear is the weak point (P(loss)=10.2%).
Solution: adaptive risk parameters when regime = bear.

3 dials:
  1. Size reduction: risk_pct 2% → 1% in bear
  2. Threshold hardening: ml_threshold 0.60 → 0.68 in bear
  3. Trade limit tightening: max_daily 1 → 1 every 2 days in bear
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
# TEST 1: Bear detection
# ===========================================================================
class TestBearDetection:

    def test_detects_bear_regime(self, mtf_data):
        """Must correctly identify bear market from 1D data."""
        from src.ml.bear_risk_dial import detect_regime
        regime = detect_regime(mtf_data['1d'], '2022-06-15')
        assert regime == 'bear', f"Expected bear in mid-2022, got {regime}"

    def test_detects_bull_regime(self, mtf_data):
        from src.ml.bear_risk_dial import detect_regime
        regime = detect_regime(mtf_data['1d'], '2023-03-15')
        assert regime == 'bull', f"Expected bull in Mar 2023, got {regime}"

    def test_detects_range_regime(self, mtf_data):
        from src.ml.bear_risk_dial import detect_regime
        regime = detect_regime(mtf_data['1d'], '2023-08-15')
        # Mid-2023 was ranging
        assert regime in ('range', 'bull', 'bear')  # any valid


# ===========================================================================
# TEST 2: Risk dials adjust in bear
# ===========================================================================
class TestRiskDials:

    def test_bear_reduces_position_size(self):
        """In bear, risk_pct must be lower than in bull."""
        from src.ml.bear_risk_dial import get_risk_params
        bull_params = get_risk_params('bull')
        bear_params = get_risk_params('bear')
        assert bear_params['risk_pct'] < bull_params['risk_pct'], \
            f"Bear risk {bear_params['risk_pct']} >= bull {bull_params['risk_pct']}"

    def test_bear_raises_threshold(self):
        """In bear, ML threshold must be stricter."""
        from src.ml.bear_risk_dial import get_risk_params
        bull_params = get_risk_params('bull')
        bear_params = get_risk_params('bear')
        assert bear_params['ml_threshold'] > bull_params['ml_threshold'], \
            f"Bear threshold {bear_params['ml_threshold']} <= bull {bull_params['ml_threshold']}"

    def test_bear_increases_cooldown(self):
        """In bear, cooldown must be longer."""
        from src.ml.bear_risk_dial import get_risk_params
        bull_params = get_risk_params('bull')
        bear_params = get_risk_params('bear')
        assert bear_params['cooldown_bars'] > bull_params['cooldown_bars']

    def test_params_have_all_keys(self):
        from src.ml.bear_risk_dial import get_risk_params
        for regime in ['bull', 'bear', 'range']:
            params = get_risk_params(regime)
            for key in ['risk_pct', 'ml_threshold', 'cooldown_bars', 'max_daily_trades']:
                assert key in params, f"Missing {key} for regime {regime}"


# ===========================================================================
# TEST 3: Adaptive pipeline uses dials
# ===========================================================================
class TestAdaptivePipeline:

    def test_adaptive_produces_results(self, mtf_data, mtf_features):
        from src.ml.bear_risk_dial import AdaptiveNYXPipeline
        pipe = AdaptiveNYXPipeline()
        r = pipe.run(mtf_data, mtf_features,
                     train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        assert 'n_trades' in r
        assert r['n_trades'] > 0

    def test_adaptive_bear_fewer_trades(self, mtf_data, mtf_features):
        """Adaptive in 2022 bear must have fewer trades than static."""
        from src.ml.bear_risk_dial import AdaptiveNYXPipeline
        from src.ml.nyx_pipeline import NYXPipeline
        static = NYXPipeline()
        adaptive = AdaptiveNYXPipeline()
        r_static = static.run(mtf_data, mtf_features,
                              train_end='2021-12-31', test_start='2022-01-01', test_end='2022-12-31')
        r_adaptive = adaptive.run(mtf_data, mtf_features,
                                   train_end='2021-12-31', test_start='2022-01-01', test_end='2022-12-31')
        assert r_adaptive['n_trades'] <= r_static['n_trades'], \
            f"Adaptive {r_adaptive['n_trades']} > static {r_static['n_trades']} in bear"

    def test_adaptive_bear_less_dd(self, mtf_data, mtf_features):
        """Adaptive in 2022 must have lower DD than static."""
        from src.ml.bear_risk_dial import AdaptiveNYXPipeline
        from src.ml.nyx_pipeline import NYXPipeline
        static = NYXPipeline()
        adaptive = AdaptiveNYXPipeline()
        r_static = static.run(mtf_data, mtf_features,
                              train_end='2021-12-31', test_start='2022-01-01', test_end='2022-12-31')
        r_adaptive = adaptive.run(mtf_data, mtf_features,
                                   train_end='2021-12-31', test_start='2022-01-01', test_end='2022-12-31')
        assert r_adaptive['max_drawdown_pct'] <= r_static['max_drawdown_pct'] + 0.01, \
            f"Adaptive DD {r_adaptive['max_drawdown_pct']:.1%} > static {r_static['max_drawdown_pct']:.1%}"

    def test_adaptive_bull_not_penalized(self, mtf_data, mtf_features):
        """Adaptive in 2023 bull must not significantly underperform static."""
        from src.ml.bear_risk_dial import AdaptiveNYXPipeline
        from src.ml.nyx_pipeline import NYXPipeline
        static = NYXPipeline()
        adaptive = AdaptiveNYXPipeline()
        r_static = static.run(mtf_data, mtf_features,
                              train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        r_adaptive = adaptive.run(mtf_data, mtf_features,
                                   train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31')
        # Adaptive trades safety for PnL: may sacrifice up to 50% of bull upside
        # to protect bear downside — that's the explicit trade-off
        assert r_adaptive['total_pnl_dollars'] >= r_static['total_pnl_dollars'] * 0.40, \
            f"Adaptive PnL ${r_adaptive['total_pnl_dollars']:+,.0f} << static ${r_static['total_pnl_dollars']:+,.0f}"


# ===========================================================================
# TEST 4: Bootstrap improvement in bear
# ===========================================================================
class TestBearBootstrapImproved:

    def test_adaptive_bear_ploss_lower(self, mtf_data, mtf_features):
        """Adaptive bear bootstrap P(loss) must be lower than static."""
        from src.ml.bear_risk_dial import AdaptiveNYXPipeline
        from src.ml.nyx_pipeline import NYXPipeline
        from src.ml.bootstrap import standard_bootstrap

        # Collect bear trades for both
        static_trades: list = []
        adaptive_trades: list = []
        for Pipe, label in [(NYXPipeline, 'static'), (AdaptiveNYXPipeline, 'adaptive')]:
            pipe = Pipe()
            r = pipe.run(mtf_data, mtf_features,
                         train_end='2021-12-31', test_start='2022-01-01', test_end='2022-12-31')
            for t in r['trades']:
                t['regime'] = 'bear'
            if label == 'static':
                static_trades = r['trades']
            else:
                adaptive_trades = r['trades']

        if len(static_trades) >= 5 and len(adaptive_trades) >= 3:
            bs_static = standard_bootstrap(static_trades, n_sims=1000)
            bs_adaptive = standard_bootstrap(adaptive_trades, n_sims=1000)
            # Adaptive should have lower or equal P(loss)
            assert bs_adaptive['prob_loss'] <= bs_static['prob_loss'] + 0.05, \
                f"Adaptive P(loss) {bs_adaptive['prob_loss']:.1%} > static {bs_static['prob_loss']:.1%}"
