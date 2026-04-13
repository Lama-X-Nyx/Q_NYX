"""
Enhanced ML Trade Filter v2 — Parquet Features + Threshold Calibration

Improvements over v1:
  1. 47 parquet features (microstructure, risk-adjusted, HSMM)
  2. Threshold calibration via time-series CV
  3. Soft gate sizing integration (disagreement → size)
  4. GBM with careful regularization
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from src.ml.jesse_features import _ema, _atr
from src.ml.soft_gate import compute_disagreement, compute_size_factor


def generate_enhanced_candidates(
    df_ohlcv: pd.DataFrame,
    df_features: pd.DataFrame,
    vol_min: float = 3.0,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_bars: int = 50,
    fee_rate: float = 0.0002,
    slippage_rate: float = 0.0001,
) -> List[Dict]:
    """Generate candidates with 47 parquet features + rule scores."""
    close = df_ohlcv['close'].values.astype(float)
    high = df_ohlcv['high'].values.astype(float)
    low = df_ohlcv['low'].values.astype(float)
    volume = df_ohlcv['volume'].values.astype(float)
    n = len(close)

    atr = np.nan_to_num(_atr(high, low, close, 14), nan=0.0)
    ema9 = _ema(close, 9); ema21 = _ema(close, 21); ema50 = _ema(close, 50)
    vol_ma = _ema(volume, 20)

    # Align parquet features with OHLCV index
    common_idx = df_ohlcv.index.intersection(df_features.index)
    feat_aligned = df_features.reindex(df_ohlcv.index)
    feat_cols = [c for c in df_features.columns if 'hsmm' not in c or
                 df_features[c].nunique() > 2]  # skip uniform HSMM
    feat_arr = np.nan_to_num(feat_aligned[feat_cols].values, nan=0.0)

    candidates = []
    for i in range(60, n - max_bars):
        if np.isnan(ema9[i]) or np.isnan(ema50[i]) or atr[i] <= 0:
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
            continue

        uptrend = ema9[i] > ema21[i] > ema50[i]
        downtrend = ema9[i] < ema21[i] < ema50[i]
        if not (uptrend or downtrend):
            continue
        if volume[i] / vol_ma[i] < vol_min:
            continue
        hour = df_ohlcv.index[i].hour if hasattr(df_ohlcv.index, 'hour') else 12
        if hour < 6 or hour > 20:
            continue

        direction = 1 if uptrend else -1

        # Simulate outcome
        entry_price = close[i] * (1 + direction * slippage_rate)
        tp = entry_price + direction * tp_mult * atr[i]
        sl = entry_price - direction * sl_mult * atr[i]

        outcome = 0.0; reason = 'TIME'
        for j in range(i + 1, min(i + max_bars + 1, n)):
            hit_tp = (direction == 1 and high[j] >= tp) or (direction == -1 and low[j] <= tp)
            hit_sl = (direction == 1 and low[j] <= sl) or (direction == -1 and high[j] >= sl)
            if hit_tp:
                exit_p = tp * (1 - direction * slippage_rate)
                outcome = direction * (exit_p - entry_price); reason = 'TP'; break
            if hit_sl:
                exit_p = sl * (1 - direction * slippage_rate)
                outcome = direction * (exit_p - entry_price); reason = 'SL'; break
        else:
            exit_p = close[min(i + max_bars, n - 1)] * (1 - direction * slippage_rate)
            outcome = direction * (exit_p - entry_price)

        fee = entry_price * fee_rate + abs(exit_p) * fee_rate
        outcome_net = outcome - fee

        # Build feature dict from parquet
        features = {}
        for k, col in enumerate(feat_cols):
            features[col] = float(feat_arr[i, k]) if i < len(feat_arr) else 0.0

        # Rule scores
        ctx_ratio = (ema9[i] - ema50[i]) / max(ema50[i], 1e-8)
        features['rule_context'] = float(np.clip(ctx_ratio * 20 + 0.5, 0, 1))
        features['rule_regime'] = float(np.clip(atr[i] / close[i] * 200, 0, 1))
        features['rule_setup'] = float(np.clip(abs(ema9[i] - ema21[i]) / max(ema21[i], 1e-8) * 100, 0, 1))
        features['disagreement'] = compute_disagreement(direction, {
            'context': features['rule_context'],
            'regime': features['rule_regime'],
            'setup': features['rule_setup'],
        })
        features['volume_spike'] = float(volume[i] / vol_ma[i])
        features['trend_strength'] = float(np.clip(abs(ema9[i] - ema50[i]) / max(ema50[i], 1e-8) * 50, 0, 10))
        features['direction'] = float(direction)
        features['hour_norm'] = float(hour / 24.0)

        candidates.append({
            'bar_idx': i, 'timestamp': df_ohlcv.index[i],
            'direction': direction, 'entry_price': entry_price,
            'outcome_net': float(outcome_net), 'reason': reason,
            'features': features,
        })

    return candidates


class EnhancedMLFilter:
    """ML filter with parquet features + calibrated threshold."""

    def __init__(self):
        self._model = None
        self._scaler = None
        self._feature_names: List[str] = []
        self.is_trained = False
        self.calibrated_threshold = 0.50
        self.train_metrics: Dict[str, float] = {}

    def train(self, df_ohlcv: pd.DataFrame, df_features: pd.DataFrame) -> None:
        candidates = generate_enhanced_candidates(df_ohlcv, df_features)
        if len(candidates) < 50:
            self.is_trained = True
            self.train_metrics = {'calibrated_wr': 0.5}
            return

        self._feature_names = sorted(candidates[0]['features'].keys())
        X = np.array([[c['features'].get(f, 0.0) for f in self._feature_names] for c in candidates])
        y = np.array([1 if c['outcome_net'] > 0 else 0 for c in candidates])
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        X = np.clip(X, -1e6, 1e6)

        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X)
        X_s = np.nan_to_num(X_s, nan=0.0, posinf=3.0, neginf=-3.0)

        self._model = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=30, subsample=0.7, random_state=42,
        )
        self._model.fit(X_s, y)
        self.is_trained = True

        # Calibrate threshold via time-series CV
        tscv = TimeSeriesSplit(n_splits=3)
        best_threshold = 0.50
        best_metric = -999

        for threshold in np.arange(0.45, 0.65, 0.02):
            cv_wrs = []
            for train_idx, val_idx in tscv.split(X_s):
                m = GradientBoostingClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.03,
                    min_samples_leaf=30, subsample=0.7, random_state=42)
                m.fit(X_s[train_idx], y[train_idx])
                proba = m.predict_proba(X_s[val_idx])
                classes = list(m.classes_)
                if 1 in classes:
                    p1 = proba[:, classes.index(1)]
                    accepted = p1 >= threshold
                    if accepted.sum() > 5:
                        cv_wrs.append(y[val_idx][accepted].mean())

            if cv_wrs:
                avg_wr = np.mean(cv_wrs)
                if avg_wr > best_metric:
                    best_metric = avg_wr
                    best_threshold = threshold

        self.calibrated_threshold = float(best_threshold)
        self.train_metrics = {'calibrated_wr': float(best_metric)}

    def predict_proba(self, features: Dict[str, float]) -> float:
        if not self.is_trained or self._model is None:
            return 0.5
        x = np.array([[features.get(f, 0.0) for f in self._feature_names]])
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = np.clip(x, -1e6, 1e6)
        x_s = self._scaler.transform(x)
        x_s = np.nan_to_num(x_s, nan=0.0, posinf=3.0, neginf=-3.0)
        proba = self._model.predict_proba(x_s)[0]
        classes = list(self._model.classes_)
        return float(proba[classes.index(1)]) if 1 in classes else 0.5

    def feature_importance(self) -> Dict[str, float]:
        if self._model is None:
            return {}
        return dict(zip(self._feature_names, self._model.feature_importances_))


class EnhancedMLFilterBacktester:
    """Backtest with enhanced ML filter + parquet features."""

    def __init__(
        self,
        fee_rate: float = 0.0002,
        slippage_rate: float = 0.0001,
        risk_pct: float = 0.02,
        initial_capital: float = 10_000.0,
        cooldown_bars: int = 32,
    ):
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.risk_pct = risk_pct
        self.initial_capital = initial_capital
        self.cooldown_bars = cooldown_bars

    def run(
        self,
        df_ohlcv: pd.DataFrame,
        df_features: pd.DataFrame,
        train_end: str = '2022-12-31',
        test_start: str = '2023-01-01',
        test_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Train
        ml = EnhancedMLFilter()
        ml.train(df_ohlcv.loc[:train_end], df_features.loc[:train_end])

        # Generate test candidates
        te = test_end or str(df_ohlcv.index[-1].date())
        candidates = generate_enhanced_candidates(
            df_ohlcv.loc[test_start:te], df_features.loc[test_start:te],
            fee_rate=self.fee_rate, slippage_rate=self.slippage_rate)

        # Score and filter with calibrated threshold
        threshold = ml.calibrated_threshold
        scored = [(c, ml.predict_proba(c['features'])) for c in candidates]
        accepted = [(c, p) for c, p in scored if p >= threshold]
        rejected = [(c, p) for c, p in scored if p < threshold]

        # Simulate with sizing
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        last_bar = -999

        for c, prob in accepted:
            if c['bar_idx'] - last_bar < self.cooldown_bars:
                continue
            atr_pct = c['features'].get('rule_regime', 0.005)
            atr_est = max(c['entry_price'] * atr_pct / 200, 1.0)

            # Size with soft gate: higher prob → bigger position
            ml_conf = prob
            dis = c['features'].get('disagreement', 0)
            rule_avg = np.mean([c['features'].get('rule_context', 0.5),
                                c['features'].get('rule_regime', 0.5),
                                c['features'].get('rule_setup', 0.5)])
            sf = compute_size_factor(ml_conf, dis, rule_avg)

            risk_dollars = capital * self.risk_pct * sf
            qty = min(risk_dollars / max(atr_est, 1), capital / max(c['entry_price'], 1))
            if qty <= 0 or capital <= 0:
                continue

            net_pnl = c['outcome_net'] * qty
            capital += net_pnl
            capital = max(capital, 0)
            equity_curve.append(capital)
            last_bar = c['bar_idx']

            trades.append({
                'timestamp': c['timestamp'], 'direction': c['direction'],
                'entry_price': c['entry_price'], 'net_pnl': net_pnl,
                'ml_prob': prob, 'size_factor': sf, 'reason': c['reason'],
            })

        # Metrics
        nt = len(trades)
        total_pnl = capital - self.initial_capital
        wins = [t for t in trades if t['net_pnl'] > 0]
        losses = [t for t in trades if t['net_pnl'] <= 0]
        wr = len(wins) / nt if nt > 0 else 0
        gw = sum(t['net_pnl'] for t in wins)
        gl = abs(sum(t['net_pnl'] for t in losses))
        pf = gw / gl if gl > 0 else float('inf')

        eq = np.array(equity_curve)
        peak = eq[0]; max_dd = 0
        for e in eq:
            peak = max(peak, e)
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        if nt > 5:
            rets = [t['net_pnl'] / self.initial_capital for t in trades]
            sharpe = float(np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(min(nt, 252)))
        else:
            sharpe = 0.0

        return {
            'n_trades': nt,
            'n_candidates': len(scored),
            'filter_reject_rate': len(rejected) / max(len(scored), 1),
            'calibrated_threshold': threshold,
            'win_rate': wr,
            'sharpe': sharpe,
            'total_pnl_dollars': total_pnl,
            'total_return_pct': total_pnl / self.initial_capital,
            'profit_factor': pf,
            'max_drawdown_pct': max_dd,
            'feature_importance': ml.feature_importance(),
            'trades': trades,
        }
