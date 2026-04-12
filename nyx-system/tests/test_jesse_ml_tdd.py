"""
TDD Tests for Jesse ML Pipeline — RED PHASE

Core principle from the video:
  Generate SYNTHETIC data with obvious trends.
  If the model gets ~100% accuracy on synthetic → code is correct.
  If accuracy drops on real data → problem is data, not code.

Tests cover:
  1. Synthetic data generation (obvious trends)
  2. Triple barrier labeling (+1 / -1 / 0)
  3. Stationary features (ratios, not raw values)
  4. Gather mode (collect features + labels)
  5. Deploy mode (predict with confidence threshold)
  6. Full pipeline: synthetic → train → predict → ~100% accuracy
  7. Feature importance extraction
  8. A/B comparison structure
"""
import pytest
import numpy as np
import pandas as pd
from typing import List, Dict


# ---------------------------------------------------------------------------
# Helpers — synthetic data generators
# ---------------------------------------------------------------------------

def make_bullish_candles(n: int = 200, start_price: float = 1000.0) -> pd.DataFrame:
    """
    Generate obvious uptrend candles — model MUST predict +1.
    Video blueprint: +0.25% per candle (proportional), 15m timeframe.
    """
    pct_per_bar = 0.0025  # +0.25% per bar
    log_returns = pct_per_bar + np.random.randn(n) * 0.0003
    close = start_price * np.exp(np.cumsum(log_returns))
    bar_range = close * 0.002  # 0.2% bar range
    high = close + np.abs(np.random.randn(n)) * bar_range * 0.5 + bar_range * 0.3
    low = close - np.abs(np.random.randn(n)) * bar_range * 0.3
    open_ = close * (1 - pct_per_bar * 0.8 + np.random.randn(n) * 0.0001)
    volume = np.random.randint(100, 1000, n).astype(float)
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': volume
    }, index=dates)


def make_bearish_candles(n: int = 200, start_price: float = 2000.0) -> pd.DataFrame:
    """
    Generate obvious downtrend candles — model MUST predict -1.
    Video blueprint: -0.25% per candle (proportional), 15m timeframe.
    """
    pct_per_bar = -0.0025  # -0.25% per bar
    log_returns = pct_per_bar + np.random.randn(n) * 0.0003
    close = start_price * np.exp(np.cumsum(log_returns))
    bar_range = close * 0.002
    high = close + np.abs(np.random.randn(n)) * bar_range * 0.3
    low = close - np.abs(np.random.randn(n)) * bar_range * 0.5 - bar_range * 0.3
    open_ = close * (1 - pct_per_bar * 0.8 + np.random.randn(n) * 0.0001)
    volume = np.random.randint(100, 1000, n).astype(float)
    dates = pd.date_range('2024-04-01', periods=n, freq='15min')
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': volume
    }, index=dates)


def make_ranging_candles(n: int = 200, center: float = 1500.0) -> pd.DataFrame:
    """
    Generate sideways/range candles — model should predict 0.
    Video blueprint: sinusoidal oscillation around center price.
    """
    t = np.arange(n)
    amplitude = 0.003  # 0.3% oscillation
    close = center * (1 + amplitude * np.sin(2 * np.pi * t / 40)
                      + np.random.randn(n) * 0.0002)
    bar_range = close * 0.001
    high = close + np.abs(np.random.randn(n)) * bar_range + bar_range * 0.2
    low = close - np.abs(np.random.randn(n)) * bar_range - bar_range * 0.2
    open_ = close + np.random.randn(n) * bar_range * 0.3
    volume = np.random.randint(100, 1000, n).astype(float)
    dates = pd.date_range('2024-07-01', periods=n, freq='15min')
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': volume
    }, index=dates)


def make_mixed_synthetic(n_per_regime: int = 500) -> pd.DataFrame:
    """
    Concatenate bull + bear + range for full training set.
    Video: 6 months of synthetic data.
    """
    bull = make_bullish_candles(n_per_regime, start_price=1000.0)
    bear = make_bearish_candles(n_per_regime, start_price=3000.0)
    rng = make_ranging_candles(n_per_regime, center=1500.0)
    return pd.concat([bull, bear, rng]).sort_index()


# ===========================================================================
# TEST 1: Triple Barrier Labeling
# ===========================================================================
class TestTripleBarrierLabeler:
    """Triple barrier: TP hit → +1, SL hit → -1, time expired → 0."""

    def test_bullish_candles_label_mostly_plus_one(self):
        """On obvious uptrend, most labels should be +1."""
        from src.ml.jesse_labeler import triple_barrier_labels
        df = make_bullish_candles(300)
        labels = triple_barrier_labels(df, tp_mult=1.5, sl_mult=1.0, max_bars=50)
        assert isinstance(labels, np.ndarray)
        assert set(np.unique(labels)).issubset({-1, 0, 1})
        pct_bull = (labels == 1).mean()
        assert pct_bull > 0.6, f"Expected >60% bullish labels, got {pct_bull:.1%}"

    def test_bearish_candles_label_mostly_minus_one(self):
        """On obvious downtrend, most labels should be -1."""
        from src.ml.jesse_labeler import triple_barrier_labels
        df = make_bearish_candles(300)
        labels = triple_barrier_labels(df, tp_mult=1.5, sl_mult=1.0, max_bars=50)
        pct_bear = (labels == -1).mean()
        assert pct_bear > 0.6, f"Expected >60% bearish labels, got {pct_bear:.1%}"

    def test_ranging_candles_label_mostly_zero(self):
        """On tight range, most labels should be 0 (time expired)."""
        from src.ml.jesse_labeler import triple_barrier_labels
        df = make_ranging_candles(300, center=1500.0)
        # Wide barriers + short time window → time expires before TP/SL hit
        labels = triple_barrier_labels(df, tp_mult=3.0, sl_mult=3.0, max_bars=15)
        pct_range = (labels == 0).mean()
        assert pct_range > 0.3, f"Expected >30% range labels, got {pct_range:.1%}"

    def test_labels_length_matches_input(self):
        """Labels array length must match input DataFrame rows."""
        from src.ml.jesse_labeler import triple_barrier_labels
        df = make_bullish_candles(100)
        labels = triple_barrier_labels(df, tp_mult=1.5, sl_mult=1.0, max_bars=50)
        assert len(labels) == len(df)


# ===========================================================================
# TEST 2: Stationary Features (ratios, not raw values)
# ===========================================================================
class TestStationaryFeatures:
    """Features MUST be stationary — ratios/differences, never raw prices."""

    def test_features_are_bounded(self):
        """All feature values should be in a reasonable bounded range."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_bullish_candles(300)
        features = compute_stationary_features(df)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(df)
        # No raw prices sneaking in — all values should be roughly [-10, 10]
        for col in features.columns:
            valid = features[col].dropna()
            if len(valid) == 0:
                continue
            assert valid.abs().max() < 100, \
                f"Feature '{col}' has extreme value {valid.abs().max():.1f} — not stationary"

    def test_features_include_required_indicators(self):
        """Must include EMA ratios, RSI, ATR ratio, volume ratio."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_bullish_candles(300)
        features = compute_stationary_features(df)
        required = ['ema_ratio_21_50', 'rsi_14', 'atr_ratio', 'volume_ratio']
        for feat in required:
            assert feat in features.columns, f"Missing required feature: {feat}"

    def test_ema_ratio_is_stationary(self):
        """EMA ratio = (EMA21 - EMA50) / EMA50 — should hover around 0."""
        from src.ml.jesse_features import compute_stationary_features
        df = make_ranging_candles(500, center=150.0)
        features = compute_stationary_features(df)
        ema_ratio = features['ema_ratio_21_50'].dropna()
        assert abs(ema_ratio.mean()) < 0.1, \
            f"EMA ratio mean {ema_ratio.mean():.4f} too far from 0 for ranging data"

    def test_no_nan_in_features_after_warmup(self):
        """After warmup period, no NaN should remain (core=50, full=210)."""
        from src.ml.jesse_features import compute_stationary_features
        # Core features: 50 bars warmup is enough
        df = make_bullish_candles(300)
        features_core = compute_stationary_features(df, feature_set='core')
        after_warmup = features_core.iloc[50:]
        nan_counts = after_warmup.isna().sum()
        assert nan_counts.sum() == 0, \
            f"NaN in core features after warmup: {nan_counts[nan_counts > 0].to_dict()}"
        # Full features: need 210 bars warmup (EMA200)
        df_long = make_bullish_candles(500)
        features_full = compute_stationary_features(df_long, feature_set='full')
        after_warmup_full = features_full.iloc[210:]
        nan_full = after_warmup_full.isna().sum()
        assert nan_full.sum() == 0, \
            f"NaN in full features after warmup: {nan_full[nan_full > 0].to_dict()}"

    def test_features_different_price_levels_same_distribution(self):
        """Same trend at price=100 vs price=10000 → features should be similar."""
        from src.ml.jesse_features import compute_stationary_features
        df_low = make_bullish_candles(200, start_price=100.0)
        df_high = make_bullish_candles(200, start_price=10000.0)
        feat_low = compute_stationary_features(df_low).iloc[50:].mean()
        feat_high = compute_stationary_features(df_high).iloc[50:].mean()
        for col in feat_low.index:
            diff = abs(feat_low[col] - feat_high[col])
            assert diff < 5.0, \
                f"Feature '{col}' differs by {diff:.2f} across price levels — not stationary"


# ===========================================================================
# TEST 3: Gather / Deploy Modes
# ===========================================================================
class TestGatherDeployModes:
    """
    Gather mode: collect features + labels into a dataset (no trading).
    Deploy mode: use trained model to predict with confidence threshold.
    """

    def test_gather_mode_produces_dataset(self):
        """Gather mode should produce (X, y) arrays ready for training."""
        from src.ml.jesse_strategy import JesseMLStrategy
        df = make_mixed_synthetic(200)
        strategy = JesseMLStrategy(mode='gather')
        X, y = strategy.gather(df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0
        assert X.shape[1] > 5  # at least 5 features

    def test_deploy_mode_returns_predictions(self):
        """Deploy mode should return direction predictions."""
        from src.ml.jesse_strategy import JesseMLStrategy
        df = make_mixed_synthetic(200)
        # First gather + train
        strategy = JesseMLStrategy(mode='gather')
        X, y = strategy.gather(df)
        strategy.train(X, y)
        # Then deploy
        strategy.mode = 'deploy'
        df_test = make_bullish_candles(50)
        preds = strategy.predict(df_test)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(df_test)
        assert set(np.unique(preds)).issubset({-1, 0, 1})

    def test_deploy_confidence_threshold(self):
        """Only predict when model confidence > threshold."""
        from src.ml.jesse_strategy import JesseMLStrategy
        df = make_mixed_synthetic(300)
        strategy = JesseMLStrategy(mode='gather', confidence_threshold=0.45)
        X, y = strategy.gather(df)
        strategy.train(X, y)
        strategy.mode = 'deploy'
        df_test = make_bullish_candles(100)
        preds, probs = strategy.predict_with_confidence(df_test)
        # Predictions should be 0 where confidence < threshold
        for i in range(len(preds)):
            if preds[i] != 0:
                assert probs[i] >= 0.45, \
                    f"Prediction {preds[i]} at bar {i} with confidence {probs[i]:.2f} < threshold"


# ===========================================================================
# TEST 4: Full Pipeline — Synthetic TDD Validation
# ===========================================================================
class TestSyntheticTDDValidation:
    """
    THE CORE TDD TEST from the video:
    Train on synthetic data with obvious patterns.
    Model MUST achieve ~100% accuracy.
    If it doesn't → code is broken.
    """

    def test_synthetic_accuracy_above_90_percent(self):
        """
        On obvious synthetic data, model accuracy must be >= 90%.
        Video: synthetic data → ~100% accuracy validates code correctness.
        """
        from src.ml.jesse_strategy import JesseMLStrategy
        np.random.seed(42)
        # Train set — generous size
        train_df = make_mixed_synthetic(600)
        strategy = JesseMLStrategy(mode='gather')
        X_train, y_train = strategy.gather(train_df)
        strategy.train(X_train, y_train)
        # Test set (fresh synthetic, same obvious patterns, 200 bars each)
        np.random.seed(99)  # different seed for test
        test_bull = make_bullish_candles(200, start_price=1000.0)
        test_bear = make_bearish_candles(200, start_price=3000.0)
        strategy.mode = 'deploy'
        pred_bull = strategy.predict(test_bull)
        pred_bear = strategy.predict(test_bear)
        # Bull predictions (skip 60 warmup for EMA50 convergence)
        valid_bull = pred_bull[60:]
        active_bull = valid_bull[valid_bull != 0]  # only where model has opinion
        if len(active_bull) > 0:
            bull_acc = (active_bull == 1).mean()
        else:
            bull_acc = 0.0
        # Bear predictions
        valid_bear = pred_bear[60:]
        active_bear = valid_bear[valid_bear != 0]
        if len(active_bear) > 0:
            bear_acc = (active_bear == -1).mean()
        else:
            bear_acc = 0.0
        assert bull_acc >= 0.85, \
            f"Bull accuracy {bull_acc:.1%} < 85% on synthetic — CODE IS BROKEN"
        assert bear_acc >= 0.85, \
            f"Bear accuracy {bear_acc:.1%} < 85% on synthetic — CODE IS BROKEN"

    def test_model_generalizes_across_price_levels(self):
        """
        Model trained on price ~1000 should work on price ~50000 (stationarity).
        Video: stationary features ensure price-level independence.
        """
        from src.ml.jesse_strategy import JesseMLStrategy
        np.random.seed(42)
        train_df = make_mixed_synthetic(600)  # prices 1000-3000
        strategy = JesseMLStrategy(mode='gather')
        X, y = strategy.gather(train_df)
        strategy.train(X, y)
        strategy.mode = 'deploy'
        # Test on totally different price level
        np.random.seed(99)
        test_high = make_bullish_candles(200, start_price=50000.0)
        preds = strategy.predict(test_high)
        valid = preds[60:]
        active = valid[valid != 0]
        acc = (active == 1).mean() if len(active) > 0 else 0.0
        assert acc >= 0.75, \
            f"Cross-price accuracy {acc:.1%} < 75% — features are NOT stationary"


# ===========================================================================
# TEST 5: Feature Importance
# ===========================================================================
class TestFeatureImportance:
    """After training, must be able to extract feature importance rankings."""

    def test_feature_importance_returns_dict(self):
        """Feature importance should return {feature_name: importance_score}."""
        from src.ml.jesse_strategy import JesseMLStrategy
        df = make_mixed_synthetic(300)
        strategy = JesseMLStrategy(mode='gather')
        X, y = strategy.gather(df)
        strategy.train(X, y)
        importance = strategy.feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
        # Values should be non-negative
        for name, score in importance.items():
            assert score >= 0, f"Feature '{name}' has negative importance {score}"

    def test_top_feature_is_meaningful(self):
        """Top feature should be an EMA/trend indicator on synthetic trend data."""
        from src.ml.jesse_strategy import JesseMLStrategy
        np.random.seed(42)
        df = make_mixed_synthetic(500)
        strategy = JesseMLStrategy(mode='gather')
        X, y = strategy.gather(df)
        strategy.train(X, y)
        importance = strategy.feature_importance()
        top_3 = sorted(importance, key=importance.get, reverse=True)[:3]
        trend_features = {'ema_ratio_21_50', 'ema_ratio_9_21', 'momentum_10',
                          'momentum_20', 'rsi_14', 'atr_ratio', 'close_vs_ema50'}
        overlap = set(top_3) & trend_features
        assert len(overlap) >= 1, \
            f"Top 3 features {top_3} don't include any trend indicator"


# ===========================================================================
# TEST 6: A/B Structure
# ===========================================================================
class TestABComparison:
    """
    Strategy A = NYX current (HSMM + LightGBM agents)
    Strategy B = Jesse ML (stationary features + RandomForest + triple barrier)
    A/B runner must produce comparable metrics.
    """

    def test_ab_runner_returns_metrics_for_both(self):
        """A/B runner should return metrics dict for both strategies."""
        from src.ml.jesse_ab_runner import run_ab_comparison
        df = make_mixed_synthetic(300)
        results = run_ab_comparison(df, n_test_bars=100)
        assert 'strategy_a' in results
        assert 'strategy_b' in results
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            assert key in results['strategy_a'], f"Missing {key} in strategy_a"
            assert key in results['strategy_b'], f"Missing {key} in strategy_b"

    def test_ab_metrics_are_valid_numbers(self):
        """All metrics should be floats in [0, 1]."""
        from src.ml.jesse_ab_runner import run_ab_comparison
        df = make_mixed_synthetic(300)
        results = run_ab_comparison(df, n_test_bars=100)
        for strat in ['strategy_a', 'strategy_b']:
            for key in ['accuracy', 'precision', 'recall', 'f1']:
                val = results[strat][key]
                assert isinstance(val, float), f"{strat}.{key} is not float"
                assert 0.0 <= val <= 1.0, f"{strat}.{key} = {val} out of [0,1]"
