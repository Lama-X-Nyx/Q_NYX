"""
ML Trade Filter — Select top trades from hard gate candidates

The hard gate (trend + vol 3.0x + hours + cooldown) produces ~150/yr.
The ML filter scores each candidate and rejects the bottom ~40%.

Trains on net outcome (after fees), not raw labels.
Features: edge features + rule validator scores + disagreement.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from src.ml.jesse_features import _ema, _atr, _rsi, _adx, compute_stationary_features
from src.ml.soft_gate import ContextRuleValidator, RegimeRuleValidator, SetupRuleValidator, compute_disagreement


def generate_candidates(
    df: pd.DataFrame,
    vol_min: float = 3.0,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_bars: int = 50,
    fee_rate: float = 0.0002,
    slippage_rate: float = 0.0001,
) -> List[Dict]:
    """
    Generate hard gate candidate trades with features + net outcomes.
    Simulates every eligible entry, records what would have happened.
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    n = len(close)

    atr = np.nan_to_num(_atr(high, low, close, 14), nan=0.0)
    ema9 = _ema(close, 9); ema21 = _ema(close, 21); ema50 = _ema(close, 50)
    vol_ma = _ema(volume, 20)
    rsi = _rsi(close, 14)
    adx = _adx(high, low, close, 14)

    # Stationary features
    feat_df = compute_stationary_features(df, feature_set='core')
    feat_arr = np.nan_to_num(feat_df.values, nan=0.0)
    feat_cols = list(feat_df.columns)

    # Rule validators (vectorized)
    ctx_v = ContextRuleValidator()
    reg_v = RegimeRuleValidator()
    stp_v = SetupRuleValidator()

    candidates = []
    for i in range(60, n - max_bars):
        if np.isnan(ema9[i]) or np.isnan(ema50[i]) or atr[i] <= 0:
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
            continue

        # Hard gate: trend + volume + hours
        uptrend = ema9[i] > ema21[i] > ema50[i]
        downtrend = ema9[i] < ema21[i] < ema50[i]
        if not (uptrend or downtrend):
            continue
        if volume[i] / vol_ma[i] < vol_min:
            continue
        hour = df.index[i].hour if hasattr(df.index, 'hour') else 12
        if hour < 6 or hour > 20:
            continue

        direction = 1 if uptrend else -1

        # Simulate outcome
        entry_price = close[i] * (1 + direction * slippage_rate)
        tp = entry_price + direction * tp_mult * atr[i]
        sl = entry_price - direction * sl_mult * atr[i]

        outcome = 0.0
        reason = 'TIME'
        for j in range(i + 1, min(i + max_bars + 1, n)):
            hit_tp = (direction == 1 and high[j] >= tp) or (direction == -1 and low[j] <= tp)
            hit_sl = (direction == 1 and low[j] <= sl) or (direction == -1 and high[j] >= sl)
            if hit_tp:
                exit_p = tp * (1 - direction * slippage_rate)
                outcome = direction * (exit_p - entry_price)
                reason = 'TP'; break
            if hit_sl:
                exit_p = sl * (1 - direction * slippage_rate)
                outcome = direction * (exit_p - entry_price)
                reason = 'SL'; break
        else:
            exit_p = close[min(i + max_bars, n - 1)] * (1 - direction * slippage_rate)
            outcome = direction * (exit_p - entry_price)

        # Fees (both sides)
        fee = entry_price * fee_rate + abs(exit_p) * fee_rate
        outcome_net = outcome - fee

        # Features for ML
        features = {}
        for k, col in enumerate(feat_cols):
            features[col] = float(feat_arr[i, k])

        # Rule scores
        ctx_ratio = (ema9[i] - ema50[i]) / max(ema50[i], 1e-8)
        features['rule_context'] = float(np.clip(ctx_ratio * 20 + 0.5, 0, 1))
        features['rule_regime'] = float(np.clip(adx[i] / 50.0, 0, 1)) if not np.isnan(adx[i]) else 0.5
        rsi_s = 1.0 - abs(rsi[i] - 55) / 55 if not np.isnan(rsi[i]) else 0.5
        ema_a = float(ema9[i] > ema21[i]) if not np.isnan(ema21[i]) else 0.5
        features['rule_setup'] = float(np.clip(0.5 * rsi_s + 0.5 * ema_a, 0, 1))
        features['disagreement'] = compute_disagreement(direction, {
            'context': features['rule_context'],
            'regime': features['rule_regime'],
            'setup': features['rule_setup'],
        })

        # Extra ML features
        features['volume_spike'] = float(volume[i] / vol_ma[i])
        features['atr_pct'] = float(atr[i] / close[i])
        features['trend_strength'] = float(abs(ema9[i] - ema50[i]) / max(ema50[i], 1e-8))
        features['hour'] = float(hour)
        features['direction'] = float(direction)

        candidates.append({
            'bar_idx': i,
            'timestamp': df.index[i],
            'direction': direction,
            'entry_price': entry_price,
            'outcome_net': float(outcome_net),
            'reason': reason,
            'features': features,
        })

    return candidates


class MLTradeFilter:
    """ML model that scores hard gate candidates. Rejects bottom ~40%."""

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold
        self._model = None
        self._scaler = None
        self._feature_names: List[str] = []
        self.is_trained = False

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train on historical candidates with net outcomes."""
        candidates = generate_candidates(df)
        if len(candidates) < 50:
            self.is_trained = True
            return {'accuracy': 0.5, 'n_samples': len(candidates)}

        self._feature_names = sorted(candidates[0]['features'].keys())
        X = np.array([[c['features'][f] for f in self._feature_names] for c in candidates])
        y = np.array([1 if c['outcome_net'] > 0 else 0 for c in candidates])

        # Clean NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        X = np.clip(X, -1e6, 1e6)

        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X)
        X_s = np.nan_to_num(X_s, nan=0.0, posinf=3.0, neginf=-3.0)

        self._model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42,
        )
        self._model.fit(X_s, y)
        self.is_trained = True

        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y, self._model.predict(X_s)), 'n_samples': len(y)}

    def predict_proba(self, features: Dict[str, float]) -> float:
        """Return P(profitable) for a candidate."""
        if not self.is_trained or self._model is None or self._scaler is None:
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


class MLFilteredBacktester:
    """Hard gate + ML filter backtest with realistic fees."""

    def __init__(
        self,
        fee_rate: float = 0.0002,
        slippage_rate: float = 0.0001,
        risk_pct: float = 0.02,
        initial_capital: float = 10_000.0,
        ml_threshold: float = 0.52,  # slightly above random to filter
        cooldown_bars: int = 32,
    ):
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.risk_pct = risk_pct
        self.initial_capital = initial_capital
        self.ml_threshold = ml_threshold
        self.cooldown_bars = cooldown_bars

    def run(
        self,
        df: pd.DataFrame,
        train_end: str = '2022-12-31',
        test_start: str = '2023-01-01',
        test_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Train ML filter
        df_train = df.loc[:train_end]
        ml_filter = MLTradeFilter(threshold=self.ml_threshold)
        ml_filter.train(df_train)

        # Generate test candidates
        te = test_end or str(df.index[-1].date())
        df_test = df.loc[test_start:te]
        candidates = generate_candidates(df_test, fee_rate=self.fee_rate, slippage_rate=self.slippage_rate)

        # Score and filter
        scored = []
        for c in candidates:
            prob = ml_filter.predict_proba(c['features'])
            c['ml_prob'] = prob
            scored.append(c)

        accepted = [c for c in scored if c['ml_prob'] >= self.ml_threshold]
        rejected = [c for c in scored if c['ml_prob'] < self.ml_threshold]

        # Simulate with cooldown and sizing
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        last_bar = -999

        for c in accepted:
            if c['bar_idx'] - last_bar < self.cooldown_bars:
                continue
            # Position sizing
            close_approx = c['entry_price']
            atr_approx = c['features'].get('atr_pct', 0.005) * close_approx
            if atr_approx <= 0:
                continue
            risk_dollars = capital * self.risk_pct
            qty = min(risk_dollars / atr_approx, capital / close_approx)
            if qty <= 0:
                continue

            net_pnl = c['outcome_net'] * qty
            fee = (c['entry_price'] + abs(c['entry_price'] + c['outcome_net'])) * self.fee_rate * qty
            final_pnl = net_pnl  # outcome_net already includes fees from generate_candidates
            capital += final_pnl
            capital = max(capital, 0)
            equity_curve.append(capital)
            last_bar = c['bar_idx']

            trades.append({
                'timestamp': c['timestamp'],
                'direction': c['direction'],
                'entry_price': c['entry_price'],
                'net_pnl': final_pnl,
                'ml_prob': c['ml_prob'],
                'reason': c['reason'],
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

        # Sharpe from trade returns
        if nt > 5:
            trade_rets = [t['net_pnl'] / self.initial_capital for t in trades]
            sharpe = float(np.mean(trade_rets) / max(np.std(trade_rets), 1e-8) * np.sqrt(min(nt, 252)))
        else:
            sharpe = 0.0

        return {
            'n_trades': nt,
            'n_candidates': len(scored),
            'n_accepted': len(accepted),
            'n_rejected': len(rejected),
            'filter_reject_rate': len(rejected) / max(len(scored), 1),
            'win_rate': wr,
            'sharpe': sharpe,
            'total_pnl_dollars': total_pnl,
            'total_return_pct': total_pnl / self.initial_capital,
            'max_drawdown_pct': max_dd,
            'profit_factor': pf,
            'trades': trades,
            'feature_importance': ml_filter.feature_importance(),
        }
