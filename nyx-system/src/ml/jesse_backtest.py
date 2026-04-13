"""
Jesse Fast Backtester — Vectorized precomputation + numpy hot loop

Speed target: 8,640 bars in < 5 seconds total (precompute + train + backtest).

Architecture:
  1. precompute(df) — compute ALL features + labels ONCE as numpy arrays
  2. train(X, y) — fit RandomForest on training portion
  3. run(df) — walk-forward: train on train portion, backtest on test portion

Zero DataFrame slicing in the hot loop. Pure numpy arrays.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.ml.jesse_features import compute_stationary_features
from src.ml.jesse_labeler import triple_barrier_labels
from src.ml.jesse_strategy import JesseMLStrategy


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    direction: int = 1  # 1=long, -1=short
    size: float = 1.0
    pnl: float = 0.0
    exit_reason: str = ''


class FastBacktester:
    """
    Vectorized backtester with numpy hot loop.

    Usage:
        bt = FastBacktester()
        result = bt.run(df, train_end='2023-02-28', test_start='2023-03-01')
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.02,
        tp_mult: float = 1.5,
        sl_mult: float = 1.0,
        max_bars_in_trade: int = 50,
        confidence_threshold: float = 0.45,
        warmup_bars: int = 60,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.max_bars_in_trade = max_bars_in_trade
        self.confidence_threshold = confidence_threshold
        self.warmup_bars = warmup_bars

    # ------------------------------------------------------------------
    # PRECOMPUTE — all features + labels as numpy arrays (ONCE)
    # ------------------------------------------------------------------
    def precompute(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute all features, labels, and price arrays ONCE.

        Returns dict of numpy arrays:
            features: (n, n_features) float64
            labels:   (n,) int — triple barrier labels
            close:    (n,) float64
            high:     (n,) float64
            low:      (n,) float64
            atr:      (n,) float64
        """
        # Features — vectorized
        feat_df = compute_stationary_features(df, feature_set='core')
        features = feat_df.values.astype(np.float64)
        # Replace NaN with 0 for warmup period
        features = np.nan_to_num(features, nan=0.0)

        # Labels — vectorized triple barrier
        labels = triple_barrier_labels(
            df, tp_mult=self.tp_mult, sl_mult=self.sl_mult,
            max_bars=self.max_bars_in_trade,
        )

        # Price arrays
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)

        # ATR — vectorized
        from src.ml.jesse_features import _atr
        atr = _atr(high, low, close, 14)
        atr = np.nan_to_num(atr, nan=0.0)

        return {
            'features': features,
            'feature_names': list(feat_df.columns),
            'labels': labels,
            'close': close,
            'high': high,
            'low': low,
            'atr': atr,
            'index': df.index,
        }

    # ------------------------------------------------------------------
    # RUN — full walk-forward pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        train_end: str = '2023-02-28',
        test_start: str = '2023-03-01',
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.

        1. Precompute features + labels
        2. Train model on train period
        3. Backtest on test period with numpy hot loop

        Returns:
            Dict with metrics, trades, equity_curve, etc.
        """
        # --- Step 1: Precompute ---
        precomp = self.precompute(df)
        features = precomp['features']
        labels = precomp['labels']
        close = precomp['close']
        high = precomp['high']
        low = precomp['low']
        atr = precomp['atr']
        index = precomp['index']

        # --- Step 2: Split train/test ---
        train_mask = index <= pd.Timestamp(train_end)
        test_mask = index >= pd.Timestamp(test_start)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        # Filter warmup from training
        valid_train = train_idx[train_idx >= self.warmup_bars]
        X_train = features[valid_train]
        y_train = labels[valid_train]

        # --- Step 3: Train model ---
        strategy = JesseMLStrategy(
            mode='gather',
            confidence_threshold=self.confidence_threshold,
            tp_mult=self.tp_mult,
            sl_mult=self.sl_mult,
            max_bars=self.max_bars_in_trade,
        )
        train_metrics = strategy.train(X_train, y_train)
        strategy.mode = 'deploy'
        strategy._feature_names = precomp['feature_names']

        # --- Step 4: Predict on test set ---
        X_test = features[test_idx]
        # Batch predict (vectorized, no per-bar loop)
        X_scaled = strategy._scaler.transform(X_test)  # type: ignore
        proba = strategy._model.predict_proba(X_scaled)  # type: ignore
        classes = list(strategy._model.classes_)  # type: ignore

        # Extract per-class probabilities
        prob_up = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X_test))
        prob_down = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(X_test))

        # Apply video rules: prob > threshold AND prob > opposite + margin
        margin = 0.20
        predictions = np.zeros(len(X_test), dtype=int)
        confidences = np.maximum(prob_up, prob_down)
        predictions[(prob_up >= self.confidence_threshold) & (prob_up > prob_down + margin)] = 1
        predictions[(prob_down >= self.confidence_threshold) & (prob_down > prob_up + margin)] = -1

        # --- Step 5: Backtest hot loop (pure numpy) ---
        test_close = close[test_idx]
        test_high = high[test_idx]
        test_low = low[test_idx]
        test_atr = atr[test_idx]
        test_index = index[test_idx]
        n_test = len(test_idx)

        capital = self.initial_capital
        equity_curve = np.zeros(n_test)
        trades: List[Trade] = []

        # Position state
        in_position = False
        entry_price = 0.0
        entry_bar = 0
        direction = 0
        tp_level = 0.0
        sl_level = 0.0
        position_size = 0.0

        for i in range(n_test):
            # --- Update existing position ---
            if in_position:
                bars_held = i - entry_bar

                # Check TP / SL / time expiry
                hit_tp = (direction == 1 and test_high[i] >= tp_level) or \
                         (direction == -1 and test_low[i] <= tp_level)
                hit_sl = (direction == 1 and test_low[i] <= sl_level) or \
                         (direction == -1 and test_high[i] >= sl_level)
                time_expired = bars_held >= self.max_bars_in_trade

                if hit_tp or hit_sl or time_expired:
                    if hit_tp:
                        exit_price = tp_level
                        reason = 'TP'
                    elif hit_sl:
                        exit_price = sl_level
                        reason = 'SL'
                    else:
                        exit_price = test_close[i]
                        reason = 'TIME'

                    pnl = direction * (exit_price - entry_price) * position_size
                    capital += pnl
                    trades.append(Trade(
                        entry_time=test_index[entry_bar],
                        entry_price=entry_price,
                        exit_time=test_index[i],
                        exit_price=exit_price,
                        direction=direction,
                        size=position_size,
                        pnl=pnl,
                        exit_reason=reason,
                    ))
                    in_position = False

            # --- Check for new entry ---
            if not in_position and predictions[i] != 0 and test_atr[i] > 0:
                direction = predictions[i]
                entry_price = test_close[i]
                entry_bar = i

                # Position sizing: risk_per_trade of capital
                risk_per_unit = self.sl_mult * test_atr[i]
                if risk_per_unit > 0:
                    position_size = (capital * self.risk_per_trade) / risk_per_unit
                else:
                    position_size = 0

                if position_size > 0:
                    # Set TP / SL
                    if direction == 1:  # long
                        tp_level = entry_price + self.tp_mult * test_atr[i]
                        sl_level = entry_price - self.sl_mult * test_atr[i]
                    else:  # short
                        tp_level = entry_price - self.tp_mult * test_atr[i]
                        sl_level = entry_price + self.sl_mult * test_atr[i]
                    in_position = True

            equity_curve[i] = capital

        # --- Close any open position at end ---
        if in_position:
            exit_price = test_close[-1]
            pnl = direction * (exit_price - entry_price) * position_size
            capital += pnl
            trades.append(Trade(
                entry_time=test_index[entry_bar],
                entry_price=entry_price,
                exit_time=test_index[-1],
                exit_price=exit_price,
                direction=direction,
                size=position_size,
                pnl=pnl,
                exit_reason='EOD',
            ))
            equity_curve[-1] = capital

        # --- Compute metrics ---
        trade_dicts = [
            {
                'entry_time': t.entry_time, 'entry_price': t.entry_price,
                'exit_time': t.exit_time, 'exit_price': t.exit_price,
                'direction': t.direction, 'size': t.size,
                'pnl': t.pnl, 'exit_reason': t.exit_reason,
            }
            for t in trades
        ]

        total_pnl = sum(t.pnl for t in trades)
        n_trades = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return {
            'metrics': {
                'total_pnl': total_pnl,
                'total_return': total_pnl / self.initial_capital,
                'n_trades': n_trades,
                'win_rate': win_rate,
                'max_drawdown': max_dd,
                'avg_pnl': total_pnl / n_trades if n_trades > 0 else 0.0,
            },
            'trades': trade_dicts,
            'equity_curve': equity_curve.tolist(),
            'train_bars': len(valid_train),
            'test_bars': n_test,
            'train_accuracy': train_metrics['accuracy'],
            'predictions_summary': {
                'total': n_test,
                'long': int((predictions == 1).sum()),
                'short': int((predictions == -1).sum()),
                'neutral': int((predictions == 0).sum()),
            },
        }
