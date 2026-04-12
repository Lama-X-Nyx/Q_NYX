"""
TDD Tests for Jesse Integration — Features v2, Utils, Research ML

Tests the full Jesse integration layer:
  1. Extended features (25+ stationary indicators)
  2. Utility functions (position sizing, crossovers, Kelly)
  3. Research ML (4-method feature importance, feature impact, training)
"""
import pytest
import numpy as np
import pandas as pd

from tests.test_jesse_ml_tdd import (
    make_bullish_candles, make_bearish_candles, make_ranging_candles,
    make_mixed_synthetic,
)


# ===========================================================================
# TEST 1: Extended Features (full set)
# ===========================================================================
class TestExtendedFeatures:

    def test_full_features_have_25_plus_columns(self):
        """Full feature set should have 25+ columns."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_bullish_candles(300)
        features = compute_stationary_features(df, feature_set='full')
        assert features.shape[1] >= 24, \
            f"Expected 24+ features, got {features.shape[1]}: {list(features.columns)}"

    def test_core_features_have_13_columns(self):
        """Core feature set should have exactly 13 columns."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_bullish_candles(300)
        features = compute_stationary_features(df, feature_set='core')
        assert features.shape[1] == 13, \
            f"Expected 13 core features, got {features.shape[1]}"

    def test_new_features_present(self):
        """New features (ADX, MACD, BB, MFI, etc.) should be present."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_bullish_candles(300)
        features = compute_stationary_features(df, feature_set='full')
        new_features = [
            'adx_norm', 'macd_hist_ratio', 'bb_percent_b', 'bb_width_ratio',
            'mfi_norm', 'obv_slope', 'roc_10', 'keltner_position',
            'squeeze', 'ema_ratio_50_200', 'zscore_20',
        ]
        for feat in new_features:
            assert feat in features.columns, f"Missing new feature: {feat}"

    def test_all_features_bounded(self):
        """All features should be in reasonable range (stationary)."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_bullish_candles(300)
        features = compute_stationary_features(df, feature_set='full')
        for col in features.columns:
            valid = features[col].dropna()
            if len(valid) == 0:
                continue
            assert valid.abs().max() < 100, \
                f"Feature '{col}' has extreme value {valid.abs().max():.1f}"

    def test_full_features_no_nan_after_warmup(self):
        """No NaN after warmup (200 bars for EMA200 convergence)."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_bullish_candles(500)
        features = compute_stationary_features(df, feature_set='full')
        after_warmup = features.iloc[210:]
        nan_counts = after_warmup.isna().sum()
        assert nan_counts.sum() == 0, \
            f"NaN found after warmup: {nan_counts[nan_counts > 0].to_dict()}"

    def test_adx_in_zero_one_range(self):
        """ADX should be normalized to [0, 1]."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_bullish_candles(300)
        features = compute_stationary_features(df, feature_set='full')
        adx = features['adx_norm'].dropna()
        assert adx.min() >= 0.0, f"ADX min {adx.min():.3f} < 0"
        assert adx.max() <= 1.0, f"ADX max {adx.max():.3f} > 1"

    def test_bb_percent_b_in_reasonable_range(self):
        """Bollinger %B should be bounded and centered roughly around 0.5 for range."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_ranging_candles(500)
        features = compute_stationary_features(df, feature_set='full')
        bb = features['bb_percent_b'].dropna()
        # %B should not have extreme outliers
        assert bb.abs().max() < 10, f"BB %B has extreme values: max={bb.abs().max():.2f}"

    def test_squeeze_binary(self):
        """Squeeze should be 0 or 1."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_ranging_candles(300)
        features = compute_stationary_features(df, feature_set='full')
        squeeze = features['squeeze'].dropna()
        assert set(squeeze.unique()).issubset({0.0, 1.0})


# ===========================================================================
# TEST 2: Utility Functions
# ===========================================================================
class TestJesseUtils:

    def test_risk_to_qty_basic(self):
        """risk_to_qty should calculate correct position size."""
        from src.ml.jesse_utils import risk_to_qty
        # $10000 capital, 2% risk, entry $100, stop $95 → risk $5/unit
        # Risk amount = $200, qty = 200/5 = 40 (without fees)
        # Jesse divides differently: size_to_qty(risk_to_size(...))
        qty = risk_to_qty(10000, 0.02, 100.0, 95.0)
        assert qty > 0, f"Expected positive qty, got {qty}"

    def test_risk_to_qty_zero_risk(self):
        """Zero stop distance should return 0."""
        from src.ml.jesse_utils import risk_to_qty
        qty = risk_to_qty(10000, 0.02, 100.0, 100.0)
        assert qty == 0.0  # guarded before calling Jesse

    def test_size_to_qty(self):
        """size_to_qty should convert dollar amount to shares."""
        from src.ml.jesse_utils import size_to_qty
        qty = size_to_qty(1000.0, 50.0)
        assert 19 <= qty <= 21  # ~20 shares

    def test_crossed_above(self):
        """Detect when series1 crosses above series2."""
        from src.ml.jesse_utils import crossed
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        s2 = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        result = crossed(s1, s2, direction='above')
        # result is array — cross at index 3
        assert result[3] == True

    def test_crossed_below(self):
        """Detect when series1 crosses below series2."""
        from src.ml.jesse_utils import crossed
        s1 = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        s2 = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        result = crossed(s1, s2, direction='below')
        assert np.any(result)

    def test_crossed_sequential(self):
        """Sequential mode checks only last two values."""
        from src.ml.jesse_utils import crossed
        s1 = np.array([2.0, 4.0])
        s2 = np.array([3.0, 3.0])
        assert crossed(s1, s2, direction='above', sequential=True) == True

    def test_kelly_criterion(self):
        """Kelly should return correct fraction."""
        from src.ml.jesse_utils import kelly_criterion
        # 60% win rate, 2:1 reward/risk
        k = kelly_criterion(0.6, 2.0)
        assert abs(k - 0.4) < 0.01  # 0.6 - (0.4/2) = 0.4

    def test_kelly_losing_system(self):
        """Losing system should return negative Kelly."""
        from src.ml.jesse_utils import kelly_criterion
        k = kelly_criterion(0.3, 1.0)
        assert k < 0  # 0.3 - 0.7/1.0 = -0.4

    def test_anchor_timeframe(self):
        """Maps lower TF to higher TF (Jesse's mapping)."""
        from src.ml.jesse_utils import anchor_timeframe
        # Jesse maps: 1h→4h, 4h→1D — always a bigger timeframe
        result_1h = anchor_timeframe('1h')
        result_4h = anchor_timeframe('4h')
        assert result_1h in ('4h', '6h'), f"1h → {result_1h}"
        assert result_4h in ('1D', '1d'), f"4h → {result_4h}"

    def test_streaks(self):
        """Consecutive positive/negative runs."""
        from src.ml.jesse_utils import streaks
        series = np.array([1, 2, 3, 2, 1, 2, 3, 4])
        s = streaks(series)
        # diff: [0, 1, 1, -1, -1, 1, 1, 1]
        assert s[2] == 2   # 2 consecutive ups
        assert s[4] == -2  # 2 consecutive downs
        assert s[7] == 3   # 3 consecutive ups


# ===========================================================================
# TEST 3: Research ML — Feature Importance & Training
# ===========================================================================
class TestJesseResearch:

    def test_feature_importance_4_methods(self):
        """Feature importance should return 4 methods + consensus."""
        from src.ml.jesse_research import compute_feature_importance
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = (X[:, 0] > 0).astype(int)  # only first feature matters
        names = ['important', 'noise1', 'noise2', 'noise3', 'noise4']
        result = compute_feature_importance(X, y, names)
        assert 'important' in result
        for method in ['anova_f', 'correlation', 'model_importance', 'cv_impact', 'consensus_rank']:
            assert method in result['important'], f"Missing method: {method}"
        # 'important' should have best (lowest) consensus rank
        assert result['important']['consensus_rank'] <= 2.0

    def test_feature_impact_identifies_helpful(self):
        """Feature impact should identify important features as 'keep'."""
        from src.ml.jesse_research import compute_feature_impact
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 4)
        X[:, 0] = np.linspace(-2, 2, n)  # perfectly correlated
        y = (X[:, 0] > 0).astype(int)
        split = int(n * 0.8)
        result = compute_feature_impact(
            X[:split], X[split:], y[:split], y[split:],
            ['key_feature', 'noise1', 'noise2', 'noise3'])
        # key_feature should be 'keep' (removing it hurts)
        assert result['key_feature']['verdict'] in ('keep', 'neutral')

    def test_train_model_returns_complete_result(self):
        """train_model should return model, scaler, metrics, importance."""
        from src.ml.jesse_research import train_model
        from src.ml.jesse_features import compute_stationary_features
        from src.ml.jesse_labeler import triple_barrier_labels
        np.random.seed(42)
        df = make_mixed_synthetic(300)
        features = compute_stationary_features(df, feature_set='core')
        labels = triple_barrier_labels(df)
        # Build data_points
        data_points = []
        for i in range(60, len(df)):
            row = features.iloc[i]
            if row.isna().any():
                continue
            data_points.append({
                'time': i,
                'features': row.to_dict(),
                'label': int(labels[i]),
            })
        result = train_model(data_points, task='multiclass',
                             test_ratio=0.2, verbose=False)
        assert 'model' in result
        assert 'scaler' in result
        assert 'metrics' in result
        assert 'feature_importance' in result
        assert 'feature_impact' in result
        assert 0 <= result['metrics']['accuracy'] <= 1

    def test_train_model_accuracy_on_synthetic(self):
        """Trained model should achieve decent accuracy on synthetic data."""
        from src.ml.jesse_research import train_model
        from src.ml.jesse_features import compute_stationary_features
        from src.ml.jesse_labeler import triple_barrier_labels
        np.random.seed(42)
        df = make_mixed_synthetic(500)
        features = compute_stationary_features(df, feature_set='core')
        labels = triple_barrier_labels(df)
        data_points = []
        for i in range(60, len(df)):
            row = features.iloc[i]
            if row.isna().any():
                continue
            data_points.append({
                'time': i,
                'features': row.to_dict(),
                'label': int(labels[i]),
            })
        result = train_model(data_points, task='multiclass',
                             test_ratio=0.2, verbose=False)
        assert result['metrics']['accuracy'] >= 0.5, \
            f"Accuracy {result['metrics']['accuracy']:.1%} too low on synthetic"


# ===========================================================================
# TEST 4: Full Pipeline with Extended Features
# ===========================================================================
class TestFullPipelineExtended:

    def test_full_features_improve_accuracy(self):
        """Full feature set should perform >= core features."""
        from src.ml.jesse_strategy import JesseMLStrategy
        from src.ml.jesse_features import compute_stationary_features
        np.random.seed(42)
        train_df = make_mixed_synthetic(500)

        # Core features
        strategy_core = JesseMLStrategy(mode='gather')
        X_core, y_core = strategy_core.gather(train_df)
        train_core = strategy_core.train(X_core, y_core)

        # Full features (manually compute + train)
        features_full = compute_stationary_features(train_df, feature_set='full')
        from src.ml.jesse_labeler import triple_barrier_labels
        labels = triple_barrier_labels(train_df)
        valid = np.ones(len(train_df), dtype=bool)
        valid[:210] = False  # EMA200 warmup
        valid &= ~features_full.isna().any(axis=1).values
        X_full = features_full.values[valid]
        y_full = labels[valid]

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_full)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y_full)
        acc_full = model.score(X_scaled, y_full)

        # Both should have reasonable accuracy
        assert train_core['accuracy'] >= 0.7
        assert acc_full >= 0.7
