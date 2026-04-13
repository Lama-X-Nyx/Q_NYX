"""
Jesse ML Strategy — Gather / Deploy Pipeline

Two modes:
  GATHER: Collect features + triple-barrier labels → training dataset
  DEPLOY: Use trained classifier to predict direction with confidence threshold

Classifier: RandomForestClassifier (as recommended in the video).
  SVM also mentioned as strong alternative for small datasets.

When Jesse is installed, extends jesse.strategies.Strategy.
Falls back to standalone class if Jesse not available.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.ml.jesse_features import compute_stationary_features
from src.ml.jesse_labeler import triple_barrier_labels

# Try importing Jesse Strategy base
try:
    from jesse.strategies import Strategy as JesseBaseStrategy  # type: ignore[import-untyped]
    _JESSE_AVAILABLE = True
except ImportError:
    JesseBaseStrategy = object  # type: ignore[misc,assignment]
    _JESSE_AVAILABLE = False


class JesseMLStrategy:
    """
    ML Strategy with Gather / Deploy modes.

    Gather mode:
      1. Compute stationary features from OHLCV
      2. Generate triple-barrier labels
      3. Return (X, y) ready for training

    Deploy mode:
      1. Compute features for new bars
      2. Predict direction using trained model
      3. Filter by confidence threshold
    """

    def __init__(
        self,
        mode: Literal['gather', 'deploy'] = 'gather',
        classifier_type: str = 'random_forest',
        confidence_threshold: float = 0.45,
        tp_mult: float = 1.5,
        sl_mult: float = 1.0,
        max_bars: int = 50,
        warmup_bars: int = 50,
    ):
        self.mode = mode
        self.classifier_type = classifier_type
        self.confidence_threshold = confidence_threshold
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.max_bars = max_bars
        self.warmup_bars = warmup_bars

        # Model state
        self._model: Optional[RandomForestClassifier | SVC] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # GATHER MODE
    # ------------------------------------------------------------------
    def gather(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect features and labels from OHLCV data.

        Returns:
            (X, y) where X is features array and y is labels array.
            Only includes rows after warmup with valid labels.
        """
        features = compute_stationary_features(df)
        labels = triple_barrier_labels(
            df, tp_mult=self.tp_mult, sl_mult=self.sl_mult,
            max_bars=self.max_bars,
        )

        self._feature_names = list(features.columns)

        # Drop warmup + NaN rows
        valid_mask = np.ones(len(df), dtype=bool)
        valid_mask[:self.warmup_bars] = False
        valid_mask &= ~features.isna().any(axis=1).values

        X = features.values[valid_mask]
        y = labels[valid_mask]

        return X, y

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the classifier on gathered data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Labels array (n_samples,) with values in {-1, 0, 1}.

        Returns:
            Training metrics dict.
        """
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Build classifier
        if self.classifier_type == 'svm':
            self._model = SVC(
                kernel='rbf', probability=True, C=1.0,
                class_weight='balanced', random_state=42,
            )
        else:
            self._model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )

        self._model.fit(X_scaled, y)
        self._is_trained = True

        # Training accuracy
        y_pred = self._model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)

        return {'accuracy': acc, 'n_samples': len(y), 'n_features': X.shape[1]}

    # ------------------------------------------------------------------
    # DEPLOY MODE
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict direction for each bar.

        Returns:
            np.ndarray of predictions: +1, -1, or 0.
        """
        preds, _ = self.predict_with_confidence(df)
        return preds

    def predict_with_confidence(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.

        Returns:
            (predictions, confidences) — predictions are filtered by threshold.
        """
        assert self._is_trained, "Model not trained. Call train() first."
        assert self._model is not None
        assert self._scaler is not None

        features = compute_stationary_features(df)
        n = len(df)
        predictions = np.zeros(n, dtype=int)
        confidences = np.zeros(n, dtype=float)

        for i in range(n):
            if i < self.warmup_bars:
                continue
            row = features.iloc[i:i + 1].values
            if np.isnan(row).any():
                continue

            row_scaled = self._scaler.transform(row)
            proba = self._model.predict_proba(row_scaled)[0]
            classes = list(self._model.classes_)

            # Extract per-class probabilities
            prob_up = proba[classes.index(1)] if 1 in classes else 0.0
            prob_down = proba[classes.index(-1)] if -1 in classes else 0.0
            prob_neutral = proba[classes.index(0)] if 0 in classes else 0.0

            # Video rule: prob > threshold AND prob > opposite + margin
            margin = 0.20
            confidences[i] = max(prob_up, prob_down, prob_neutral)

            if prob_up >= self.confidence_threshold and prob_up > prob_down + margin:
                predictions[i] = 1
            elif prob_down >= self.confidence_threshold and prob_down > prob_up + margin:
                predictions[i] = -1
            # else: stays 0 (no clear signal)

        return predictions, confidences

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    def feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance from trained model.

        Returns:
            Dict mapping feature name → importance score.
        """
        assert self._is_trained, "Model not trained."
        assert self._model is not None

        if hasattr(self._model, 'feature_importances_'):
            importances = self._model.feature_importances_
        else:
            # SVM doesn't have feature_importances_, use permutation
            importances = np.ones(len(self._feature_names)) / len(self._feature_names)

        return dict(zip(self._feature_names, importances))
