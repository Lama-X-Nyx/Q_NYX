"""
NYX Unified Pipeline v0.3 — MTF + ML Filter + Soft Gate + Fees

Single entry point that merges:
  1. MTF data (15m execution + 1h regime + 1d context)
  2. Edge candidates (trend alignment + volume > 3x on 15m)
  3. 47 parquet features (15m) + MTF context from 1h/1d
  4. ML Filter GBM (threshold 0.60, trained on net outcomes)
  5. Soft gate sizing (rule scores → disagreement → size factor)
  6. Realistic execution (maker fees, slippage, cooldown)

Usage:
    pipe = NYXPipeline()
    result = pipe.run(mtf_data, mtf_features, train_end, test_start)
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from src.ml.jesse_features import _ema, _atr, _rsi, _adx
from src.ml.soft_gate import compute_disagreement, compute_size_factor
from src.ml.conditional_dial import should_activate_bear_dial, _compute_bar_signals
from src.ml.bear_risk_dial import get_risk_params


class NYXPipeline:
    """
    Unified MTF pipeline v0.3.1. One class, one call, everything merged.
    Includes conditional bear dial by default.
    """

    def __init__(
        self,
        # Edge parameters
        vol_min: float = 3.0,
        tp_mult: float = 1.5,
        sl_mult: float = 1.0,
        max_bars: int = 50,
        # ML filter
        ml_threshold: float = 0.60,
        # Execution
        fee_rate: float = 0.0002,       # maker
        slippage_rate: float = 0.0001,
        risk_pct: float = 0.02,
        initial_capital: float = 10_000.0,
        cooldown_bars: int = 32,
        max_daily_trades: int = 1,
        # Conditional bear dial
        use_conditional_dial: bool = True,
        # Execution policy
        use_execution_filter: bool = False,
    ):
        self.vol_min = vol_min
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.max_bars = max_bars
        self.ml_threshold = ml_threshold
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.risk_pct = risk_pct
        self.initial_capital = initial_capital
        self.cooldown_bars = cooldown_bars
        self.max_daily_trades = max_daily_trades
        self.use_conditional_dial = use_conditional_dial
        self.use_execution_filter = use_execution_filter

    def run(
        self,
        mtf_data: Dict[str, pd.DataFrame],
        mtf_features: Dict[str, pd.DataFrame],
        train_end: str,
        test_start: str,
        test_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: generate candidates → train ML → filter → execute.

        Args:
            mtf_data: {'15m': df, '1h': df, '1d': df} OHLCV DataFrames
            mtf_features: {'15m': df, '1h': df, '1d': df} parquet features
            train_end: last date of training data
            test_start: first date of test data
            test_end: optional last date of test
        """
        df_15m = mtf_data['15m']
        te = test_end or str(df_15m.index[-1].date())

        # --- Step 1: Build MTF context arrays ---
        ctx_1d = self._build_1d_context(mtf_data.get('1d', pd.DataFrame()),
                                         mtf_features.get('1d', pd.DataFrame()))
        ctx_1h = self._build_1h_context(mtf_data.get('1h', pd.DataFrame()),
                                         mtf_features.get('1h', pd.DataFrame()))

        # --- Step 2: Generate candidates with full MTF feature block ---
        feat_1h = mtf_features.get('1h', pd.DataFrame())
        feat_4h = mtf_features.get('4h', pd.DataFrame())
        feat_1d = mtf_features.get('1d', pd.DataFrame())
        train_cands = self._generate_candidates(
            df_15m.loc[:train_end], mtf_features['15m'].loc[:train_end],
            ctx_1d, ctx_1h,
            feat_1h=feat_1h, feat_4h=feat_4h, feat_1d=feat_1d)
        test_cands = self._generate_candidates(
            df_15m.loc[test_start:te], mtf_features['15m'].loc[test_start:te],
            ctx_1d, ctx_1h,
            feat_1h=feat_1h, feat_4h=feat_4h, feat_1d=feat_1d)

        if len(train_cands) < 30 or len(test_cands) == 0:
            return self._empty_result()

        # --- Step 3: Train ML filter ---
        feature_names = sorted(train_cands[0]['features'].keys())
        X_train = np.array([[c['features'].get(f, 0) for f in feature_names] for c in train_cands])
        y_train = np.array([1 if c['outcome_net'] > 0 else 0 for c in train_cands])
        X_train = np.clip(np.nan_to_num(X_train, nan=0.0, posinf=10, neginf=-10), -1e6, 1e6)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_tr_s = np.nan_to_num(X_tr_s, nan=0.0, posinf=3, neginf=-3)

        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=30, subsample=0.7, random_state=42,
        )
        model.fit(X_tr_s, y_train)

        # --- Step 4: Score test candidates ---
        X_test = np.array([[c['features'].get(f, 0) for f in feature_names] for c in test_cands])
        X_test = np.clip(np.nan_to_num(X_test, nan=0.0, posinf=10, neginf=-10), -1e6, 1e6)
        X_te_s = np.nan_to_num(scaler.transform(X_test), nan=0, posinf=3, neginf=-3)

        proba = model.predict_proba(X_te_s)
        classes = list(model.classes_)
        p1_idx = classes.index(1) if 1 in classes else 0
        scores = proba[:, p1_idx]

        # --- Step 5: Precompute conditional dial signals ---
        bear_dial_signals = None
        if self.use_conditional_dial:
            try:
                test_15m = mtf_data['15m'].loc[test_start:te]
                test_1h = mtf_data.get('1h', pd.DataFrame())
                if not test_1h.empty:
                    test_1h = test_1h.loc[test_start:te]
                bear_dial_signals = _compute_bar_signals(test_15m, test_1h)
            except Exception:
                bear_dial_signals = None

        bear_params = get_risk_params('bear')
        n_bear_active = 0
        n_exec_rejected = 0

        # --- Step 6: Filter + conditional dial + soft gate sizing + execute ---
        capital = self.initial_capital
        trades: List[Dict] = []
        equity_curve = [capital]
        last_bar = -999
        daily_counts: Dict[str, int] = {}
        total_fees = 0.0

        for i, (cand, score) in enumerate(zip(test_cands, scores)):
            # Conditional bear dial check
            bear_active = False
            if self.use_conditional_dial and bear_dial_signals is not None:
                idx = cand['bar_idx']
                if idx < len(bear_dial_signals.get('regime_1h', [])):
                    regime_1h = str(bear_dial_signals['regime_1h'][idx])
                    vol_r = float(bear_dial_signals['atr_ratio'][idx]) if idx < len(bear_dial_signals['atr_ratio']) else 1.0
                    tq = float(bear_dial_signals['trend_quality'][idx]) if idx < len(bear_dial_signals['trend_quality']) else 0.5
                    dis_check = cand['features'].get('disagreement', 0.1)
                    bear_active = should_activate_bear_dial(regime_1h, vol_r, tq, dis_check)

            if bear_active:
                n_bear_active += 1

            # ML filter — use bear threshold if dial active
            threshold = bear_params['ml_threshold'] if bear_active else self.ml_threshold
            if score < threshold:
                continue

            # Cooldown — use bear cooldown if dial active
            cooldown = bear_params['cooldown_bars'] if bear_active else self.cooldown_bars
            if cand['bar_idx'] - last_bar < cooldown:
                continue

            # Daily limit
            day_key = str(cand['timestamp'].date())
            if daily_counts.get(day_key, 0) >= self.max_daily_trades:
                continue

            # Soft gate sizing
            dis = cand['features'].get('disagreement', 0)
            rule_avg = np.mean([
                cand['features'].get('rule_context', 0.5),
                cand['features'].get('rule_regime', 0.5),
                cand['features'].get('rule_setup', 0.5),
            ])
            sf = compute_size_factor(float(score), dis, rule_avg)

            # Bear dial size reduction
            if bear_active:
                sf *= bear_params['size_mult']

            # Hour bonus
            hour = cand['timestamp'].hour if hasattr(cand['timestamp'], 'hour') else 12
            sf *= 1.1 if 8 <= hour <= 18 else 0.8

            # Execution filter: reject if cost > alpha
            if self.use_execution_filter:
                from src.ml.execution_policy import execution_check
                vol_r = cand['features'].get('volume_spike', 2.0)
                spread_est = cand['features'].get('atr_pct', 0.005) * 0.1
                mom_abs = abs(cand['features'].get('momentum_10', 0) if 'momentum_10' in cand['features'] else cand['features'].get('mom_4', 0))
                alpha_est = abs(cand['outcome_net']) / max(cand['entry_price'], 1) * 100
                ex = execution_check(vol_r, spread_est, cand['features'].get('atr_pct', 0.005),
                                     mom_abs, alpha_est)
                if not ex['should_trade']:
                    n_exec_rejected += 1
                    continue

            # Position sizing
            risk_pct = bear_params['risk_pct'] if bear_active else self.risk_pct
            atr_est = cand['features'].get('atr_pct', 0.005) * cand['entry_price']
            atr_est = max(atr_est, 1.0)
            risk_dollars = capital * risk_pct * sf
            qty = min(risk_dollars / atr_est, capital / cand['entry_price'])
            if qty <= 0 or capital <= 0:
                continue

            # Execute
            net_pnl = cand['outcome_net'] * qty
            fee = cand['entry_price'] * self.fee_rate * qty * 2
            total_fees += fee
            capital += net_pnl
            capital = max(capital, 0)
            equity_curve.append(capital)
            last_bar = cand['bar_idx']
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

            trades.append({
                'timestamp': cand['timestamp'],
                'direction': cand['direction'],
                'entry_price': cand['entry_price'],
                'net_pnl': net_pnl,
                'ml_score': float(score),
                'size_factor': sf,
                'disagreement': dis,
                'reason': cand['reason'],
            })

        # --- Metrics ---
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
            peak = max(peak, e); dd = (peak - e) / peak if peak > 0 else 0; max_dd = max(max_dd, dd)

        if nt > 5:
            rets = [t['net_pnl'] / self.initial_capital for t in trades]
            sharpe = float(np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(min(nt, 252)))
        else:
            sharpe = 0.0

        # Feature importance
        try:
            feat_imp = dict(zip(feature_names, model.feature_importances_))
        except Exception:
            feat_imp = {}

        return {
            'n_trades': nt,
            'n_candidates': len(test_cands),
            'filter_reject_rate': 1 - nt / max(len(test_cands), 1),
            'win_rate': wr,
            'sharpe': sharpe,
            'total_pnl_dollars': total_pnl,
            'total_return_pct': total_pnl / self.initial_capital,
            'max_drawdown_pct': max_dd,
            'profit_factor': pf,
            'total_fees': total_fees,
            'feature_names': feature_names,
            'feature_importance': feat_imp,
            'bear_dial_activation_rate': n_bear_active / max(len(test_cands), 1) if self.use_conditional_dial else 0,
            'execution_reject_rate': n_exec_rejected / max(len(test_cands), 1) if self.use_execution_filter else 0,
            'trades': trades,
        }

    # ------------------------------------------------------------------
    # MTF context builders
    # ------------------------------------------------------------------

    def _build_1d_context(self, df_1d: pd.DataFrame, feat_1d: pd.DataFrame) -> Dict[str, pd.Series]:
        """Build daily context signals: trend direction, momentum, vol."""
        if df_1d.empty:
            return {}
        close = df_1d['close'].values.astype(float)
        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        with np.errstate(divide='ignore', invalid='ignore'):
            trend = np.where(ema50 != 0, (ema20 - ema50) / np.abs(ema50), 0)
        mom20 = np.zeros_like(close)
        mom20[20:] = (close[20:] - close[:-20]) / np.maximum(close[:-20], 1e-8)

        return {
            'ctx_1d_trend': pd.Series(trend, index=df_1d.index),
            'ctx_1d_mom20': pd.Series(mom20, index=df_1d.index),
            'ctx_1d_bullish': pd.Series((trend > 0).astype(float), index=df_1d.index),
        }

    def _build_1h_context(self, df_1h: pd.DataFrame, feat_1h: pd.DataFrame) -> Dict[str, pd.Series]:
        """Build hourly regime signals: ADX, momentum, volatility."""
        if df_1h.empty:
            return {}
        close = df_1h['close'].values.astype(float)
        high = df_1h['high'].values.astype(float)
        low = df_1h['low'].values.astype(float)

        adx = np.nan_to_num(_adx(high, low, close, 14), nan=0) / 100.0
        atr = np.nan_to_num(_atr(high, low, close, 14), nan=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            atr_pct = np.where(close > 0, atr / close, 0)
        mom12 = np.zeros_like(close)
        mom12[12:] = (close[12:] - close[:-12]) / np.maximum(close[:-12], 1e-8)

        # Use parquet features if available
        regime_cols = {}
        if not feat_1h.empty:
            for col in ['buy_pressure', 'amihud', 'vol_surprise', 'ewma_vol_ratio']:
                if col in feat_1h.columns:
                    regime_cols[f'reg_1h_{col}'] = feat_1h[col]

        result = {
            'reg_1h_adx': pd.Series(adx, index=df_1h.index),
            'reg_1h_atr_pct': pd.Series(atr_pct, index=df_1h.index),
            'reg_1h_mom12': pd.Series(mom12, index=df_1h.index),
            'reg_1h_trending': pd.Series((adx > 0.25).astype(float), index=df_1h.index),
        }
        result.update(regime_cols)
        return result

    # ------------------------------------------------------------------
    # Candidate generation with MTF
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        df_15m: pd.DataFrame,
        feat_15m: pd.DataFrame,
        ctx_1d: Dict[str, pd.Series],
        ctx_1h: Dict[str, pd.Series],
        feat_1h: Optional[pd.DataFrame] = None,
        feat_1d: Optional[pd.DataFrame] = None,
        feat_4h: Optional[pd.DataFrame] = None,
    ) -> List[Dict]:
        """Generate edge candidates with full MTF features.

        PERMANENT RULE: every candidate must carry the 4 timeframes:
          no prefix  15m execution features
          h1_*       1h  stationary features
          h4_*       4h  stationary features
          d1_*       1d  stationary features
        """
        close = df_15m['close'].values.astype(float)
        high = df_15m['high'].values.astype(float)
        low = df_15m['low'].values.astype(float)
        volume = df_15m['volume'].values.astype(float)
        n = len(close)

        atr = np.nan_to_num(_atr(high, low, close, 14), nan=0)
        ema9 = _ema(close, 9); ema21 = _ema(close, 21); ema50 = _ema(close, 50)
        vol_ma = _ema(volume, 20)

        # Align parquet features (15m)
        feat_cols = [c for c in feat_15m.columns if feat_15m[c].nunique() > 2]
        feat_aligned = feat_15m.reindex(df_15m.index)
        feat_arr = np.nan_to_num(feat_aligned[feat_cols].values, nan=0)

        # Align full stationary feature vectors from 1h / 1d via ffill.
        # For a 15m bar at time T, we use the last completed 1h (or 1d)
        # bar's feature row — no look-ahead.
        h1_cols: List[str] = []
        h1_arr: np.ndarray = np.empty((n, 0), dtype=float)
        if feat_1h is not None and not feat_1h.empty:
            h1_cols = [c for c in feat_1h.columns if feat_1h[c].nunique() > 2]
            if h1_cols:
                h1_aligned = feat_1h[h1_cols].reindex(
                    df_15m.index, method='ffill'
                ).fillna(0.0)
                h1_arr = np.nan_to_num(h1_aligned.values, nan=0.0)

        d1_cols: List[str] = []
        d1_arr: np.ndarray = np.empty((n, 0), dtype=float)
        if feat_1d is not None and not feat_1d.empty:
            d1_cols = [c for c in feat_1d.columns if feat_1d[c].nunique() > 2]
            if d1_cols:
                d1_aligned = feat_1d[d1_cols].reindex(
                    df_15m.index, method='ffill'
                ).fillna(0.0)
                d1_arr = np.nan_to_num(d1_aligned.values, nan=0.0)

        h4_cols: List[str] = []
        h4_arr: np.ndarray = np.empty((n, 0), dtype=float)
        if feat_4h is not None and not feat_4h.empty:
            h4_cols = [c for c in feat_4h.columns if feat_4h[c].nunique() > 2]
            if h4_cols:
                h4_aligned = feat_4h[h4_cols].reindex(
                    df_15m.index, method='ffill'
                ).fillna(0.0)
                h4_arr = np.nan_to_num(h4_aligned.values, nan=0.0)

        # Align MTF context to 15m index
        ctx_1d_aligned = {}
        for name, series in ctx_1d.items():
            ctx_1d_aligned[name] = series.reindex(df_15m.index, method='ffill').fillna(0).values

        ctx_1h_aligned = {}
        for name, series in ctx_1h.items():
            ctx_1h_aligned[name] = series.reindex(df_15m.index, method='ffill').fillna(0).values

        candidates = []
        for i in range(60, n - self.max_bars):
            if np.isnan(ema9[i]) or np.isnan(ema50[i]) or atr[i] <= 0:
                continue
            if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
                continue

            uptrend = ema9[i] > ema21[i] > ema50[i]
            downtrend = ema9[i] < ema21[i] < ema50[i]
            if not (uptrend or downtrend):
                continue
            if volume[i] / vol_ma[i] < self.vol_min:
                continue
            hour = df_15m.index[i].hour if hasattr(df_15m.index, 'hour') else 12
            if hour < 6 or hour > 20:
                continue

            direction = 1 if uptrend else -1

            # Vectorized outcome
            entry_price = close[i] * (1 + direction * self.slippage_rate)
            tp = entry_price + direction * self.tp_mult * atr[i]
            sl = entry_price - direction * self.sl_mult * atr[i]
            end_j = min(i + self.max_bars + 1, n)
            fut_h = high[i + 1:end_j]; fut_l = low[i + 1:end_j]

            if direction == 1:
                tp_hits = np.where(fut_h >= tp)[0]
                sl_hits = np.where(fut_l <= sl)[0]
            else:
                tp_hits = np.where(fut_l <= tp)[0]
                sl_hits = np.where(fut_h >= sl)[0]

            tp_bar = tp_hits[0] if len(tp_hits) > 0 else self.max_bars + 1
            sl_bar = sl_hits[0] if len(sl_hits) > 0 else self.max_bars + 1

            if tp_bar <= sl_bar and tp_bar < self.max_bars:
                exit_p = tp * (1 - direction * self.slippage_rate)
                outcome = direction * (exit_p - entry_price); reason = 'TP'
            elif sl_bar < tp_bar and sl_bar < self.max_bars:
                exit_p = sl * (1 - direction * self.slippage_rate)
                outcome = direction * (exit_p - entry_price); reason = 'SL'
            else:
                exit_p = close[min(i + self.max_bars, n - 1)] * (1 - direction * self.slippage_rate)
                outcome = direction * (exit_p - entry_price); reason = 'TIME'

            fee = entry_price * self.fee_rate + abs(exit_p) * self.fee_rate
            outcome_net = outcome - fee

            # Build feature dict: parquet 15m + full h1_ + full d1_ + context + rules
            features = {}
            for k, col in enumerate(feat_cols):
                features[col] = float(feat_arr[i, k]) if i < len(feat_arr) else 0.0

            # Full 1h stationary features (h1_* prefix)
            for k, col in enumerate(h1_cols):
                features[f'h1_{col}'] = float(h1_arr[i, k]) if i < len(h1_arr) else 0.0

            # Full 4h stationary features (h4_* prefix)
            for k, col in enumerate(h4_cols):
                features[f'h4_{col}'] = float(h4_arr[i, k]) if i < len(h4_arr) else 0.0

            # Full 1d stationary features (d1_* prefix)
            for k, col in enumerate(d1_cols):
                features[f'd1_{col}'] = float(d1_arr[i, k]) if i < len(d1_arr) else 0.0

            # MTF context features (1D hand-crafted)
            for name, arr in ctx_1d_aligned.items():
                features[name] = float(arr[i]) if i < len(arr) else 0.0

            # MTF context features (1H hand-crafted)
            for name, arr in ctx_1h_aligned.items():
                features[name] = float(arr[i]) if i < len(arr) else 0.0

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

            # Extra
            features['volume_spike'] = float(volume[i] / vol_ma[i])
            features['atr_pct'] = float(atr[i] / close[i])
            features['trend_strength'] = float(np.clip(abs(ema9[i] - ema50[i]) / max(ema50[i], 1e-8) * 50, 0, 10))
            features['direction'] = float(direction)
            features['hour_norm'] = float(hour / 24.0)

            candidates.append({
                'bar_idx': i, 'timestamp': df_15m.index[i],
                'direction': direction, 'entry_price': entry_price,
                'outcome_net': float(outcome_net), 'reason': reason,
                'features': features,
            })

        return candidates

    def _empty_result(self):
        return {
            'n_trades': 0, 'n_candidates': 0, 'filter_reject_rate': 0,
            'win_rate': 0, 'sharpe': 0, 'total_pnl_dollars': 0,
            'total_return_pct': 0, 'max_drawdown_pct': 0, 'profit_factor': 0,
            'total_fees': 0, 'feature_names': [], 'feature_importance': {},
            'trades': [],
        }
