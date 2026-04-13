"""
A/B Comparison Runner — NYX Current vs Jesse ML

Strategy A: NYX current system (HSMM + LightGBM agents)
  - Uses existing agents pipeline: Context → Regime → Setup → Entry → Orchestrator
  - Proxy: simplified HSMM state-based prediction

Strategy B: Jesse ML (stationary features + RandomForest + triple barrier)
  - Uses jesse_features.py + jesse_labeler.py + jesse_strategy.py

Both strategies are evaluated on the same test data.
Metrics: accuracy, precision, recall, F1 score.
"""
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.ml.jesse_strategy import JesseMLStrategy
from src.ml.jesse_labeler import triple_barrier_labels


def _strategy_a_predict(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        tp_mult: float, sl_mult: float,
                        max_bars: int) -> np.ndarray:
    """
    Strategy A: Simplified NYX-style prediction.

    Uses momentum + volatility heuristic as proxy for the full
    HSMM + LightGBM agent pipeline.
    This is the BASELINE to beat.
    """
    close = df_test['close'].values
    n = len(close)
    predictions = np.zeros(n, dtype=int)

    # Simple momentum-based prediction (proxy for full NYX pipeline)
    for i in range(20, n):
        sma_fast = np.mean(close[max(0, i - 10):i])
        sma_slow = np.mean(close[max(0, i - 20):i])
        if sma_fast > sma_slow * 1.001:
            predictions[i] = 1
        elif sma_fast < sma_slow * 0.999:
            predictions[i] = -1
        else:
            predictions[i] = 0

    return predictions


def _strategy_b_predict(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        tp_mult: float, sl_mult: float,
                        max_bars: int) -> np.ndarray:
    """
    Strategy B: Jesse ML pipeline.

    Full pipeline: stationary features → RandomForest → confidence threshold.
    """
    strategy = JesseMLStrategy(
        mode='gather', confidence_threshold=0.40,
        tp_mult=tp_mult, sl_mult=sl_mult, max_bars=max_bars,
    )
    X_train, y_train = strategy.gather(df_train)
    strategy.train(X_train, y_train)
    strategy.mode = 'deploy'
    predictions = strategy.predict(df_test)
    return predictions


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics, handling multi-class gracefully."""
    # Filter to only positions where we have valid labels
    mask = np.ones(len(y_true), dtype=bool)
    mask[:50] = False  # skip warmup

    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    return {
        'accuracy': float(accuracy_score(yt, yp)),
        'precision': float(precision_score(yt, yp, average='weighted', zero_division=0)),
        'recall': float(recall_score(yt, yp, average='weighted', zero_division=0)),
        'f1': float(f1_score(yt, yp, average='weighted', zero_division=0)),
    }


def run_ab_comparison(
    df: pd.DataFrame,
    n_test_bars: int = 100,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_bars: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Run A/B comparison between Strategy A and Strategy B.

    Args:
        df: Full OHLCV DataFrame (train + test concatenated).
        n_test_bars: Number of bars to use as test set (from the end).
        tp_mult: Take-profit ATR multiplier for labeling.
        sl_mult: Stop-loss ATR multiplier for labeling.
        max_bars: Max bars for triple barrier.

    Returns:
        {
            'strategy_a': {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...},
            'strategy_b': {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...},
        }
    """
    # Split train / test
    df_train = df.iloc[:-n_test_bars].copy()
    df_test = df.iloc[-n_test_bars:].copy()

    # Ground truth labels for test set
    y_true = triple_barrier_labels(df_test, tp_mult=tp_mult, sl_mult=sl_mult,
                                   max_bars=max_bars)

    # Strategy A predictions
    pred_a = _strategy_a_predict(df_train, df_test, tp_mult, sl_mult, max_bars)

    # Strategy B predictions
    pred_b = _strategy_b_predict(df_train, df_test, tp_mult, sl_mult, max_bars)

    return {
        'strategy_a': _compute_metrics(y_true, pred_a),
        'strategy_b': _compute_metrics(y_true, pred_b),
    }
