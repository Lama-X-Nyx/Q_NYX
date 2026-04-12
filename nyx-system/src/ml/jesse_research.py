"""
Jesse Research ML Integration — Feature Importance & Model Training

Integrates Jesse's research.ml approach with Q_NYX:
  - 4-method feature importance (RFE, ANOVA, correlation, CV impact)
  - Feature impact analysis (retrain without each feature)
  - Calibration analysis
  - Model training with chronological split + StandardScaler

When Jesse is installed, delegates to jesse.research.ml.
Otherwise provides standalone implementation.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, r2_score, confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier

try:
    from jesse.research.ml import train_model as jesse_train_model  # type: ignore[import-untyped]
    _JESSE_AVAILABLE = True
except ImportError:
    _JESSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature importance — 4-method consensus (a la Jesse)
# ---------------------------------------------------------------------------

def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    estimator: Any = None,
    n_splits: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    4-method consensus feature importance ranking.

    Methods:
      1. ANOVA F-statistic (univariate)
      2. Absolute Pearson correlation with target
      3. Model-based importance (RandomForest feature_importances_)
      4. CV impact (baseline - score_without_feature)

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target labels.
        feature_names: List of feature names.
        estimator: Sklearn estimator (default: RandomForestClassifier).
        n_splits: Number of CV folds.

    Returns:
        Dict with per-method scores and consensus ranking.
    """
    from sklearn.feature_selection import f_classif
    n_features = X.shape[1]
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    # Method 1: ANOVA F-scores
    try:
        f_scores, _ = f_classif(X, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
    except Exception:
        f_scores = np.zeros(n_features)

    # Method 2: Absolute correlation
    correlations = np.array([
        abs(np.corrcoef(X[:, i], y)[0, 1]) if np.std(X[:, i]) > 0 else 0.0
        for i in range(n_features)
    ])

    # Method 3: Model-based importance
    model = clone(estimator)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    if hasattr(model, 'feature_importances_'):
        model_importance = model.feature_importances_
    else:
        model_importance = np.ones(n_features) / n_features

    # Method 4: CV impact (baseline - score without each feature)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    try:
        baseline_cv = np.mean(cross_val_score(
            clone(estimator), X_scaled, y, cv=tscv, scoring='accuracy'))
    except Exception:
        baseline_cv = 0.5

    cv_impacts = np.zeros(n_features)
    for i in range(n_features):
        X_without = np.delete(X_scaled, i, axis=1)
        try:
            score_without = np.mean(cross_val_score(
                clone(estimator), X_without, y, cv=tscv, scoring='accuracy'))
            cv_impacts[i] = baseline_cv - score_without  # positive = feature helps
        except Exception:
            cv_impacts[i] = 0.0

    # Consensus: convert all to ranks, average
    def _to_ranks(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores)  # descending
        ranks = np.zeros(len(scores))
        for rank, idx in enumerate(order):
            ranks[idx] = rank + 1
        return ranks

    rank_anova = _to_ranks(f_scores)
    rank_corr = _to_ranks(correlations)
    rank_model = _to_ranks(model_importance)
    rank_cv = _to_ranks(cv_impacts)
    consensus = (rank_anova + rank_corr + rank_model + rank_cv) / 4.0

    result = {}
    for i, name in enumerate(feature_names):
        result[name] = {
            'anova_f': float(f_scores[i]),
            'correlation': float(correlations[i]),
            'model_importance': float(model_importance[i]),
            'cv_impact': float(cv_impacts[i]),
            'consensus_rank': float(consensus[i]),
        }

    return result


# ---------------------------------------------------------------------------
# Feature impact analysis — retrain without each feature
# ---------------------------------------------------------------------------

def compute_feature_impact(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    estimator: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Measure impact of each feature by retraining without it.

    Returns:
        Dict per feature: {delta_accuracy, verdict: 'keep'|'drop'|'neutral'}.
    """
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    model = clone(estimator)
    model.fit(X_tr_scaled, y_train)
    baseline_acc = accuracy_score(y_test, model.predict(X_te_scaled))

    impacts = {}
    for i, name in enumerate(feature_names):
        X_tr_without = np.delete(X_tr_scaled, i, axis=1)
        X_te_without = np.delete(X_te_scaled, i, axis=1)
        m = clone(estimator)
        m.fit(X_tr_without, y_train)
        acc_without = accuracy_score(y_test, m.predict(X_te_without))
        delta = baseline_acc - acc_without

        if delta > 0.01:
            verdict = 'keep'
        elif delta < -0.01:
            verdict = 'drop'
        else:
            verdict = 'neutral'

        impacts[name] = {
            'baseline_accuracy': float(baseline_acc),
            'accuracy_without': float(acc_without),
            'delta': float(delta),
            'verdict': verdict,
        }

    return impacts


# ---------------------------------------------------------------------------
# Train model — Jesse-compatible pipeline
# ---------------------------------------------------------------------------

def train_model(
    data_points: List[Dict],
    estimator: Any = None,
    task: str = 'multiclass',
    test_ratio: float = 0.2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train model using Jesse-compatible data format.

    Args:
        data_points: List of dicts with 'time', 'features' (dict), 'label'.
        estimator:   Sklearn estimator (default: RandomForestClassifier).
        task:        'binary', 'multiclass', or 'regression'.
        test_ratio:  Fraction of data for test set.
        verbose:     Print training report.

    Returns:
        Dict with: model, scaler, metrics, feature_importance, feature_impact.
    """
    if _JESSE_AVAILABLE:
        try:
            return jesse_train_model(
                data_points, estimator, task=task,
                test_ratio=test_ratio, verbose=verbose)
        except Exception:
            pass  # Fall through to standalone

    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1)

    # Sort by time
    sorted_data = sorted(data_points, key=lambda x: x.get('time', 0))

    # Extract features and labels
    feature_names = sorted(sorted_data[0]['features'].keys())
    X = np.array([[dp['features'][f] for f in feature_names] for dp in sorted_data])

    if task == 'multiclass':
        y = np.array([int(dp['label']) for dp in sorted_data])
    elif task == 'binary':
        y = np.array([1 if dp['label'] > 0 else 0 for dp in sorted_data])
    else:
        y = np.array([float(dp['label']) for dp in sorted_data])

    # Chronological split
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    # Train
    model = clone(estimator)
    model.fit(X_tr_scaled, y_train)
    y_pred = model.predict(X_te_scaled)

    # Metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
    }

    # Feature importance (4-method)
    importance = compute_feature_importance(
        X_tr_scaled, y_train, feature_names, estimator)

    # Feature impact
    impact = compute_feature_impact(
        X_train, X_test, y_train, y_test, feature_names, estimator)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  MODEL TRAINING REPORT ({task})")
        print(f"{'='*60}")
        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")
        print(f"\n  Feature Impact:")
        for name in sorted(impact, key=lambda n: impact[n]['delta'], reverse=True):
            v = impact[name]
            print(f"    {name:25s}  delta={v['delta']:+.4f}  [{v['verdict']}]")
        print(f"{'='*60}\n")

    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'feature_importance': importance,
        'feature_impact': impact,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }
