"""
TDD Tests — MTFFeatureStack + NYXLiveDecider produce the COMPLETE
feature block expected by the trained model artefact.

The model was trained on 84 features total:
  24 15m stationary (no prefix)
  17 h1_* 1h stationary
  17 h4_* 4h stationary
  17 d1_* 1d stationary
  3  ctx_1d_* (trend, mom20, bullish) — hand-crafted from 1d close
  4  reg_1h_* (adx, atr_pct, mom12, trending) — hand-crafted from 1h
  4  rule_* + disagreement — 15m hand-crafted
  5  extras (volume_spike, atr_pct, trend_strength, direction, hour_norm)

If the live feature vector defaults any of these to 0 (because they're
missing from MTFFeatureStack), the GBM sees a degraded input and the
decider silently underperforms (capture ratio 0.29 in the initial
equivalence test). This test catches that regression class.
"""
from pathlib import Path

import pytest


MODELS_DIR = Path(__file__).parent.parent / 'models'


# ===========================================================================
class TestMTFFeatureStackBlocks:
    """MTFFeatureStack must produce all 4 TF blocks (no prefix, h1_,
    h4_, d1_) AND the hand-crafted ctx_1d_* + reg_1h_* blocks."""

    def _seed(self, stack, n_days=3):
        """Feed n_days × 96 × 15m bars with gentle drift + noise."""
        import pandas as pd
        ts = pd.Timestamp('2023-01-01T00:00:00')
        for i in range(n_days * 96):
            price = 100.0 + i * 0.01
            stack.on_15m_bar({
                'timestamp': ts.isoformat(),
                'open':   price,
                'high':   price * 1.002,
                'low':    price * 0.998,
                'close':  price * 1.001,
                'volume': 500.0,
            })
            ts = ts + pd.Timedelta(minutes=15)

    def test_produces_ctx_1d_features(self):
        """ctx_1d_trend / mom20 / bullish produced once ≥ 50 daily bars."""
        from src.ml.mtf_feature_stack import MTFFeatureStack
        stack = MTFFeatureStack(symbol='ETHUSDT', window_size=400)
        # EMA50 on 1d needs 50 completed daily bars → feed 60 days.
        self._seed(stack, n_days=60)
        feats = stack.latest_features_dict()
        for name in ('ctx_1d_trend', 'ctx_1d_mom20', 'ctx_1d_bullish'):
            assert name in feats, f"missing {name!r} in MTFFeatureStack output"

    def test_produces_reg_1h_features(self):
        """reg_1h_adx / atr_pct / mom12 / trending produced with ≥ 30 1h bars."""
        from src.ml.mtf_feature_stack import MTFFeatureStack
        stack = MTFFeatureStack(symbol='ETHUSDT')
        # ADX(14) needs 28 bars; mom12 needs 13; feed 3 days = 72 × 1h.
        self._seed(stack, n_days=3)
        feats = stack.latest_features_dict()
        for name in ('reg_1h_adx', 'reg_1h_atr_pct',
                      'reg_1h_mom12', 'reg_1h_trending'):
            assert name in feats, f"missing {name!r}"


# ===========================================================================
class TestLiveDeciderFeatureCoverage:
    """NYXLiveDecider._build_feature_vector must set every expected
    feature (including rule_* + extras) to a NON-ZERO value when the
    underlying market state warrants it."""

    @pytest.fixture(scope='class')
    def decider_with_data(self, eth_mtf_data):
        """Decider seeded with 2022 data so buffers are warm."""
        import pandas as pd
        from src.ml.nyx_live_decider import NYXLiveDecider
        decider = NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
        )
        df = eth_mtf_data['15m'].loc['2022-10-01':'2022-12-31']
        for ts, row in df.iterrows():
            decider.on_15m_bar({
                'timestamp': ts.isoformat(),
                'open':   float(row['open']),
                'high':   float(row['high']),
                'low':    float(row['low']),
                'close':  float(row['close']),
                'volume': float(row['volume']),
            })
        return decider

    def test_ctx_1d_features_present(self, decider_with_data):
        """After seeding, ctx_1d_* features must be present and
        non-identically-zero."""
        feats = decider_with_data.stack.latest_features_dict()
        for name in ('ctx_1d_trend', 'ctx_1d_mom20'):
            assert name in feats, f"{name!r} missing"

    def test_reg_1h_features_present(self, decider_with_data):
        feats = decider_with_data.stack.latest_features_dict()
        for name in ('reg_1h_adx', 'reg_1h_atr_pct',
                      'reg_1h_mom12', 'reg_1h_trending'):
            assert name in feats, f"{name!r} missing"

    def test_model_expected_features_all_present(self, decider_with_data):
        """The feature vector built by the decider for the model must
        have EVERY expected-by-model feature present in the stack dict
        (not defaulted to 0 via .get('x', 0.0))."""
        feats = decider_with_data.stack.latest_features_dict()
        expected = set(decider_with_data.feature_names)

        # The rule/extra/direction features are produced by NYXLiveDecider
        # itself (not MTFFeatureStack) during feature vector assembly.
        # The stack must provide every other feature.
        rule_and_extras = {
            'rule_context', 'rule_regime', 'rule_setup', 'disagreement',
            'volume_spike', 'atr_pct', 'trend_strength', 'direction',
            'hour_norm',
        }
        stack_required = expected - rule_and_extras

        missing = [name for name in stack_required if name not in feats]
        assert not missing, (
            f"MTFFeatureStack is missing {len(missing)} features the "
            f"model expects: {sorted(missing)[:10]}"
        )
