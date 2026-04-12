"""
ML Agents — LightGBM + River online learning replacements for rule-based agents.

Three agents:
  MLContextAgent  — 1D data, bull/bear/neutral bias
  MLRegimeAgent   — 1H data + HSMM probs, dominant market regime
  MLSetupAgent    — 15M data + HSMM probs + SMC patterns, trade setup quality

Each agent:
  - pretrain()  : walk-forward CV (WalkForwardSplitter), saves .pkl to data/pretrain_cache/
  - analyze()   : returns AgentResult (pass-through when not trained)
  - update_online(): River LR adapts after each outcome
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from src.agents.contracts import AgentResult
from src.ml.triple_barrier import triple_barrier_labels, label_with_context
from src.ml.walk_forward_splitter import WalkForwardSplitter

# River is optional — wrap all usage in try/except
compose: Any = None
preprocessing: Any = None
linear_model: Any = None
river_metrics: Any = None
try:
    from river import linear_model, preprocessing, compose, metrics as river_metrics
    _RIVER_OK = True
except ImportError:
    _RIVER_OK = False

CACHE_DIR = Path('data/pretrain_cache')
HSMM_STATE_NAMES = ['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation']
HSMM_COL_NAMES   = ['p_trend_plus', 'p_range', 'p_trend_minus',
                     'p_squeeze', 'p_distribution', 'p_liquidation']


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_l = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs    = avg_g / (avg_l + 1e-9)
    result: pd.Series = 100 - (100 / (1 + rs))  # type: ignore[assignment]
    return result


def _parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    log_hl = np.log(high / low.replace(0, np.nan))
    return np.sqrt((log_hl ** 2 / (4 * np.log(2))).rolling(window).mean())


def _build_river_pipeline():
    if not _RIVER_OK:
        return None
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression()
    )


def _river_predict(model, features: Dict) -> float:
    if model is None or not _RIVER_OK:
        return 0.5
    try:
        proba = model.predict_proba_one(features)
        return float(proba.get(True, 0.5))
    except Exception:
        return 0.5


def _river_update(model, features: Dict, label: int) -> None:
    if model is None or not _RIVER_OK:
        return
    try:
        model.learn_one(features, bool(label))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# MLContextAgent
# ---------------------------------------------------------------------------

class MLContextAgent:
    """
    1D context classifier: bullish / bearish / neutral.

    Two LGB binary classifiers:
      lgb_bull : P(1D trend is bullish in next 5 days)
      lgb_bear : P(1D trend is bearish in next 5 days)
    """

    MIN_BARS   = 250   # need at least SMA200 + buffer
    CACHE_FILE = CACHE_DIR / 'context_ml.pkl'

    def __init__(self):
        self._lgb_bull:   Optional[lgb.LGBMClassifier] = None
        self._lgb_bear:   Optional[lgb.LGBMClassifier] = None
        self._trained     = False
        self._feat_names: Optional[List[str]] = None
        self._online      = _build_river_pipeline()
        self._n_online    = 0

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def compute_features(self, df_1d: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df_1d.index)
        c: pd.Series = df_1d['close']  # type: ignore[assignment]
        h: pd.Series = df_1d['high']   # type: ignore[assignment]
        l: pd.Series = df_1d['low']    # type: ignore[assignment]
        v: pd.Series = df_1d['volume'] # type: ignore[assignment]
        ret = c.pct_change()

        # Momentum
        for w in [5, 20, 60, 200]:
            f[f'mom_{w}d'] = c / c.shift(w) - 1

        # Realized vol
        for w in [5, 20]:
            f[f'rv_{w}d'] = ret.rolling(w).std()

        # Parkinson vol
        f['parkinson_vol'] = _parkinson_vol(h, l, 20)

        # Trend indicators
        rsi14          = _safe_rsi(c, 14)
        f['rsi_14']    = rsi14

        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        f['macd']      = ema12 - ema26
        f['macd_sig']  = f['macd'].ewm(span=9).mean()
        f['macd_hist'] = f['macd'] - f['macd_sig']

        sma20  = c.rolling(20).mean()
        sma200 = c.rolling(200).mean()

        # Volume ratio
        vol_ma20       = v.rolling(20).mean()
        f['vol_ratio'] = v / (vol_ma20 + 1e-9)

        # Buy pressure
        denom = (h - l).replace(0, np.nan)
        f['buy_pressure'] = (c - l) / denom

        # Amihud illiquidity
        dollar_vol = c * v
        f['amihud'] = (ret.abs() / (dollar_vol.rolling(20).mean() + 1e-9)).rolling(20).mean()

        # Trend strength
        f['trend_strength'] = (c - sma200) / (sma200 + 1e-9)

        return f

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def create_labels(self, df_1d: pd.DataFrame,
                      pt_mult: float = 2.0,
                      sl_mult: float = 1.0,
                      num_bars: int = 5) -> pd.Series:
        """
        Triple-barrier labels on 1D data.
        Returns Series: 1=bullish (TP hit), -1=bearish (TP hit short), 0=neutral/SL/time.

        We run TB in both directions; bar gets:
          +1 if bullish TB hits TP first
          -1 if bearish TB hits TP first
           0 otherwise
        """
        lbl_bull = triple_barrier_labels(df_1d, context='bullish',
                                         pt_mult=pt_mult, sl_mult=sl_mult,
                                         num_bars=num_bars)
        lbl_bear = triple_barrier_labels(df_1d, context='bearish',
                                         pt_mult=pt_mult, sl_mult=sl_mult,
                                         num_bars=num_bars)
        labels = pd.Series(0, index=df_1d.index)
        labels[lbl_bull == 1] =  1
        labels[lbl_bear == 1] = -1
        return labels

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def pretrain(self, df_1d: pd.DataFrame, n_splits: int = 5) -> Dict:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if self.CACHE_FILE.exists():
            self._load()
            print('  [MLContextAgent] Loaded from cache.')
            return {'from_cache': True, 'deployed': True}

        try:
            X = self.compute_features(df_1d)
            y = self.create_labels(df_1d)

            mask = X.notna().all(axis=1) & y.notna()
            mask.iloc[-5:] = False
            X = pd.DataFrame(X[mask])
            y = pd.Series(y[mask])

            if len(X) < 300:
                print(f'  [MLContextAgent] Not enough data ({len(X)} rows)')
                return {'deployed': False, 'reason': 'not_enough_data'}

            y_bull = (y ==  1).astype(int)
            y_bear = (y == -1).astype(int)
            self._feat_names = X.columns.tolist()

            params = dict(num_leaves=15, learning_rate=0.05, n_estimators=200,
                          min_child_samples=20, verbose=-1, n_jobs=-1)

            # Walk-forward CV (1D bars: test_months=3, bars_per_month=30)
            splitter = WalkForwardSplitter(n_folds=n_splits, test_months=3,
                                           bars_per_month=30, embargo_bars=5,
                                           mode='expanding', min_train_bars=200)
            auc_bull, auc_bear = [], []

            for tr_idx, val_idx in splitter.split(X):
                Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
                for y_bin, auc_list in [
                    (y_bull, auc_bull),
                    (y_bear, auc_bear),
                ]:
                    ytr, yval = y_bin.iloc[tr_idx], y_bin.iloc[val_idx]
                    if ytr.sum() < 5 or yval.sum() < 2:
                        continue
                    m = lgb.LGBMClassifier(**params)
                    m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(-1)])
                    pred = m.predict_proba(Xval)[:, 1]
                    try:
                        auc_list.append(roc_auc_score(yval, pred))
                    except Exception:
                        pass

            # Final models on full data
            self._lgb_bull = lgb.LGBMClassifier(**params)
            if self._lgb_bull is not None:
                self._lgb_bull.fit(X, y_bull, callbacks=[lgb.log_evaluation(-1)])

            self._lgb_bear = lgb.LGBMClassifier(**params)
            if self._lgb_bear is not None:
                self._lgb_bear.fit(X, y_bear, callbacks=[lgb.log_evaluation(-1)])

            self._trained = True
            self._save()

            result = {
                'deployed': True,
                'auc_bull': float(np.mean(auc_bull)) if auc_bull else 0.5,
                'auc_bear': float(np.mean(auc_bear)) if auc_bear else 0.5,
                'n_samples': len(X),
            }
            print(f'  [MLContextAgent] Trained. AUC bull={result["auc_bull"]:.3f} '
                  f'bear={result["auc_bear"]:.3f}')
            return result

        except Exception as e:
            print(f'  [MLContextAgent] Training failed: {e}')
            return {'deployed': False, 'reason': str(e)}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def analyze(self, df_1d: pd.DataFrame) -> AgentResult:
        if len(df_1d) < self.MIN_BARS:
            return AgentResult(
                agent='context', state='not_ready', score=0.0,
                passed=False, ready=False, blocked_by_readiness=True,
                reason=f'MLContextAgent: need {self.MIN_BARS} bars, got {len(df_1d)}'
            )

        # --- pass-through: SMA200 rule-based ---
        close_s: pd.Series = df_1d['close']  # type: ignore[assignment]
        sma200 = close_s.rolling(200).mean()
        last_close = float(close_s.iloc[-1])
        last_sma   = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else last_close

        pt_state = 'bullish' if last_close > last_sma * 1.02 else \
                   'bearish' if last_close < last_sma * 0.98 else 'neutral'

        if not self._trained:
            return AgentResult(
                agent='context', state=pt_state, score=0.5,
                passed=(pt_state != 'neutral'), ready=True,
                reason='MLContextAgent: pass-through (not trained)',
                metadata={'p_bull': 0.5, 'p_bear': 0.5, 'p_neutral': 0.0}
            )

        try:
            X = self.compute_features(df_1d)
            last = X.iloc[[-1]]
            if last.isna().any().any():
                return AgentResult(
                    agent='context', state=pt_state, score=0.5,
                    passed=(pt_state != 'neutral'), ready=True,
                    reason='MLContextAgent: NaN features (warmup)',
                    metadata={'p_bull': 0.5, 'p_bear': 0.5, 'p_neutral': 0.0}
                )

            if self._lgb_bull is None or self._lgb_bear is None:
                raise RuntimeError('LGB models not initialized')
            p_bull = float(self._lgb_bull.predict_proba(last)[0, 1])
            p_bear = float(self._lgb_bear.predict_proba(last)[0, 1])

            # River blend
            feats_dict = last.iloc[0].to_dict()
            if self._n_online >= 20:
                p_bull = 0.7 * p_bull + 0.3 * _river_predict(self._online, feats_dict)

            p_neutral = max(0.0, 1.0 - p_bull - p_bear)
            # Renormalize
            total = p_bull + p_bear + p_neutral
            p_bull /= total; p_bear /= total; p_neutral /= total

            if p_bull > p_bear and p_bull > p_neutral:
                state, score = 'bullish', p_bull
            elif p_bear > p_bull and p_bear > p_neutral:
                state, score = 'bearish', p_bear
            else:
                state, score = 'neutral', p_neutral

            return AgentResult(
                agent='context', state=state, score=float(score),
                passed=(state != 'neutral'), ready=True,
                reason=f'MLContextAgent: P(bull)={p_bull:.3f} P(bear)={p_bear:.3f}',
                metadata={'p_bull': p_bull, 'p_bear': p_bear, 'p_neutral': p_neutral}
            )

        except Exception as e:
            return AgentResult(
                agent='context', state=pt_state, score=0.5,
                passed=(pt_state != 'neutral'), ready=True,
                reason=f'MLContextAgent: inference error ({e}), pass-through',
                metadata={'p_bull': 0.5, 'p_bear': 0.5, 'p_neutral': 0.0}
            )

    def update_online(self, features_dict: Dict, label: int) -> None:
        _river_update(self._online, features_dict, label)
        self._n_online += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        payload = {
            'lgb_bull': self._lgb_bull,
            'lgb_bear': self._lgb_bear,
            'feat_names': self._feat_names,
        }
        with open(self.CACHE_FILE, 'wb') as f:
            pickle.dump(payload, f, protocol=4)

    def _load(self):
        with open(self.CACHE_FILE, 'rb') as f:
            p = pickle.load(f)
        self._lgb_bull   = p['lgb_bull']
        self._lgb_bear   = p['lgb_bear']
        self._feat_names = p['feat_names']
        self._trained    = True


# ---------------------------------------------------------------------------
# MLRegimeAgent
# ---------------------------------------------------------------------------

class MLRegimeAgent:
    """
    1H regime classifier using HSMM probs + price micro-features.
    Binary: 1=bullish (price up >1% in next 4h), 0=bearish/neutral.
    """

    MIN_BARS   = 100
    CACHE_FILE = CACHE_DIR / 'regime_ml.pkl'

    def __init__(self):
        self._lgb:        Optional[lgb.LGBMClassifier] = None
        self._trained     = False
        self._feat_names: Optional[List[str]] = None
        self._online      = _build_river_pipeline()
        self._n_online    = 0

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def compute_features(self, df_1h: pd.DataFrame,
                         hsmm_probs: np.ndarray) -> pd.DataFrame:
        """
        hsmm_probs: (T, 6) array aligned with df_1h rows.
        """
        f  = pd.DataFrame(index=df_1h.index)
        c  = df_1h['close']
        h  = df_1h['high']
        l  = df_1h['low']
        v  = df_1h['volume']
        ret = c.pct_change()

        # Momentum (4h, 12h, 48h)
        for w in [4, 12, 48]:
            f[f'mom_{w}h'] = c / c.shift(w) - 1

        # Volatility (8h, 24h, 96h)
        for w in [8, 24, 96]:
            f[f'rv_{w}h'] = ret.rolling(w).std()

        # Vol ratio
        f['vol_ratio'] = f['rv_8h'] / (f['rv_96h'] + 1e-9)

        # Buy pressure
        denom = (h - l).replace(0, np.nan)
        f['buy_pressure'] = (c - l) / denom

        # Amihud
        dollar_vol = c * v
        f['amihud'] = (ret.abs() / (dollar_vol.rolling(20).mean() + 1e-9)).rolling(20).mean()

        # Kyle lambda: |ret| / sqrt(volume)
        f['kyle_lambda'] = ret.abs() / (np.sqrt(v + 1e-9))

        # Effective spread
        f['eff_spread'] = (h - l) / (c + 1e-9)

        # HSMM probabilities
        T = len(df_1h)
        for i, col in enumerate(HSMM_COL_NAMES):
            if hsmm_probs is not None and len(hsmm_probs) == T:
                f[col] = hsmm_probs[:, i]
            else:
                f[col] = 1.0 / 6.0

        return f

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def create_labels(self, df_1h: pd.DataFrame,
                      pt_mult: float = 1.5,
                      sl_mult: float = 1.0,
                      num_bars: int = 8) -> pd.Series:
        """
        Triple-barrier labels on 1H data.
        1 if bullish TP hit first within num_bars, else 0.
        """
        return triple_barrier_labels(df_1h, context='bullish',
                                     pt_mult=pt_mult, sl_mult=sl_mult,
                                     num_bars=num_bars).fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def pretrain(self, df_1h: pd.DataFrame,
                 hsmm_gamma: np.ndarray,
                 n_splits: int = 5) -> Dict:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if self.CACHE_FILE.exists():
            self._load()
            print('  [MLRegimeAgent] Loaded from cache.')
            return {'from_cache': True, 'deployed': True}

        try:
            X = self.compute_features(df_1h, hsmm_gamma)
            y = self.create_labels(df_1h)

            mask = X.notna().all(axis=1) & y.notna()
            mask.iloc[-8:] = False
            X = pd.DataFrame(X[mask])
            y = pd.Series(y[mask])

            if len(X) < 300:
                print(f'  [MLRegimeAgent] Not enough data ({len(X)} rows)')
                return {'deployed': False, 'reason': 'not_enough_data'}

            self._feat_names = X.columns.tolist()
            params = dict(num_leaves=31, learning_rate=0.05, n_estimators=300,
                          min_child_samples=50, verbose=-1, n_jobs=-1)

            # Walk-forward CV (1H bars: test_months=3, bars_per_month=720)
            splitter = WalkForwardSplitter(n_folds=n_splits, test_months=3,
                                           bars_per_month=720, embargo_bars=8,
                                           mode='expanding', min_train_bars=2000)
            fold_aucs = []

            for tr_idx, val_idx in splitter.split(X):
                Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
                Xval, yval = X.iloc[val_idx], y.iloc[val_idx]
                if ytr.sum() < 10 or yval.sum() < 5:
                    continue
                m = lgb.LGBMClassifier(**params)
                m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                      callbacks=[lgb.early_stopping(40, verbose=False),
                                 lgb.log_evaluation(-1)])
                pred = m.predict_proba(Xval)[:, 1]
                try:
                    fold_aucs.append(roc_auc_score(yval, pred))
                except Exception:
                    pass

            self._lgb = lgb.LGBMClassifier(**params)
            if self._lgb is not None:
                self._lgb.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
            self._trained = True
            self._save()

            mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
            print(f'  [MLRegimeAgent] Trained. AUC={mean_auc:.3f} n={len(X)}')
            return {'deployed': True, 'mean_auc': mean_auc, 'fold_aucs': fold_aucs}

        except Exception as e:
            print(f'  [MLRegimeAgent] Training failed: {e}')
            return {'deployed': False, 'reason': str(e)}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def analyze(self, df_1h: pd.DataFrame,
                hsmm_probs_row: np.ndarray) -> AgentResult:
        if len(df_1h) < self.MIN_BARS:
            return AgentResult(
                agent='regime', state='not_ready', score=0.0,
                passed=False, ready=False, blocked_by_readiness=True,
                reason=f'MLRegimeAgent: need {self.MIN_BARS} bars, got {len(df_1h)}'
            )

        # Pass-through: argmax of HSMM probs
        if hsmm_probs_row is not None and len(hsmm_probs_row) == 6:
            dom_idx   = int(np.argmax(hsmm_probs_row))
            dom_state = HSMM_STATE_NAMES[dom_idx].lower().replace('+', '_plus').replace('-', '_minus').replace(' ', '_')
            dom_prob  = float(hsmm_probs_row[dom_idx])
        else:
            dom_state, dom_prob = 'range', 0.5

        if not self._trained:
            passed = (dom_state not in ('liquidation',)) and dom_prob > 0.40
            return AgentResult(
                agent='regime', state=dom_state, score=dom_prob,
                passed=passed, ready=True,
                reason='MLRegimeAgent: pass-through (not trained)',
                metadata={'hsmm_probs': hsmm_probs_row.tolist() if hsmm_probs_row is not None else []}
            )

        try:
            # Build single-row feature with HSMM probs broadcast
            T = len(df_1h)
            hsmm_full = np.tile(hsmm_probs_row, (T, 1))
            X = self.compute_features(df_1h, hsmm_full)
            last = X.iloc[[-1]]
            if last.isna().any().any():
                passed = dom_prob > 0.40 and dom_state != 'liquidation'
                return AgentResult(
                    agent='regime', state=dom_state, score=dom_prob,
                    passed=passed, ready=True,
                    reason='MLRegimeAgent: NaN features, pass-through'
                )

            if self._lgb is None:
                raise RuntimeError('LGB model not initialized')
            p_bull = float(self._lgb.predict_proba(last)[0, 1])

            feats_dict = last.iloc[0].to_dict()
            if self._n_online >= 20:
                p_bull = 0.7 * p_bull + 0.3 * _river_predict(self._online, feats_dict)

            confidence = p_bull
            # Map to dominant state using HSMM for labelling
            state = dom_state
            passed = confidence > 0.55 and dom_state != 'liquidation'

            return AgentResult(
                agent='regime', state=state, score=float(confidence),
                passed=passed, ready=True,
                reason=f'MLRegimeAgent: P(bull)={p_bull:.3f} dom={dom_state}',
                metadata={
                    'p_bull': p_bull,
                    'hsmm_dominant': dom_state,
                    'hsmm_dominant_prob': dom_prob,
                    'hsmm_probs': hsmm_probs_row.tolist() if hsmm_probs_row is not None else []
                }
            )

        except Exception as e:
            passed = dom_prob > 0.40 and dom_state != 'liquidation'
            return AgentResult(
                agent='regime', state=dom_state, score=dom_prob,
                passed=passed, ready=True,
                reason=f'MLRegimeAgent: inference error ({e}), pass-through'
            )

    def update_online(self, features_dict: Dict, label: int) -> None:
        _river_update(self._online, features_dict, label)
        self._n_online += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        with open(self.CACHE_FILE, 'wb') as f:
            pickle.dump({'lgb': self._lgb, 'feat_names': self._feat_names}, f, protocol=4)

    def _load(self):
        with open(self.CACHE_FILE, 'rb') as f:
            p = pickle.load(f)
        self._lgb        = p['lgb']
        self._feat_names = p['feat_names']
        self._trained    = True


# ---------------------------------------------------------------------------
# MLSetupAgent
# ---------------------------------------------------------------------------

class MLSetupAgent:
    """
    15M setup quality classifier.
    Binary: 1 = valid setup (price moved in context direction by thr within forward_bars).
    """

    MIN_BARS   = 80
    CACHE_FILE = CACHE_DIR / 'setup_ml.pkl'

    def __init__(self):
        self._lgb:        Optional[lgb.LGBMClassifier] = None
        self._trained     = False
        self._feat_names: Optional[List[str]] = None
        self._online      = _build_river_pipeline()
        self._n_online    = 0

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def compute_features(self,
                         df_15m: pd.DataFrame,
                         hsmm_probs: np.ndarray,
                         smc_patterns: Any,   # list of dicts or single dict
                         regime_result: Optional['AgentResult'] = None,
                         context_result: Optional['AgentResult'] = None
                         ) -> pd.DataFrame:
        """
        hsmm_probs  : (T, 6) array aligned with df_15m
        smc_patterns: list[dict] aligned with df_15m, OR single dict for last row only
        """
        f   = pd.DataFrame(index=df_15m.index)
        c   = df_15m['close']
        h   = df_15m['high']
        l   = df_15m['low']
        v   = df_15m['volume']
        ret = c.pct_change()
        T   = len(df_15m)

        # Momentum (4, 8, 16, 32 bars in 15m = 1h, 2h, 4h, 8h)
        for w in [4, 8, 16, 32]:
            f[f'mom_{w}'] = c / c.shift(w) - 1

        # Vol
        for w in [4, 8, 16, 32]:
            f[f'rv_{w}'] = ret.rolling(w).std()

        f['vol_ratio'] = f['rv_4'] / (f['rv_32'] + 1e-9)

        # Buy pressure
        denom = (h - l).replace(0, np.nan)
        f['buy_pressure'] = (c - l) / denom

        # Amihud
        dollar_vol = c * v
        f['amihud'] = (ret.abs() / (dollar_vol.rolling(20).mean() + 1e-9)).rolling(20).mean()

        # Effective spread
        f['eff_spread'] = (h - l) / (c + 1e-9)

        # HSMM probs
        for i, col in enumerate(HSMM_COL_NAMES):
            if hsmm_probs is not None and len(hsmm_probs) == T:
                f[col] = hsmm_probs[:, i]
            else:
                f[col] = 1.0 / 6.0

        # SMC binary features
        smc_cols = ['bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg',
                    'bullish_choch', 'bearish_choch', 'smc_score_bullish', 'smc_score_bearish']

        if isinstance(smc_patterns, list) and len(smc_patterns) == T:
            for col in smc_cols:
                f[col] = [float(bool(p.get(col, False))) if col not in ('smc_score_bullish', 'smc_score_bearish')
                          else float(p.get(col, 0.0)) for p in smc_patterns]
        else:
            # Single dict (inference time) — broadcast to all rows
            single = smc_patterns if isinstance(smc_patterns, dict) else {}
            for col in smc_cols:
                val = float(bool(single.get(col, False))) if col not in ('smc_score_bullish', 'smc_score_bearish') \
                      else float(single.get(col, 0.0))
                f[col] = val

        # Regime score
        if regime_result is not None:
            if isinstance(regime_result, (list, np.ndarray)):
                f['regime_score'] = float(np.mean(regime_result))
            else:
                f['regime_score'] = float(getattr(regime_result, 'score', 0.5))
        else:
            f['regime_score'] = 0.5

        # Context direction
        if context_result is not None:
            if isinstance(context_result, str):
                ctx_str = context_result
            else:
                ctx_str = getattr(context_result, 'state', 'neutral')
            ctx_val = 1.0 if ctx_str == 'bullish' else -1.0 if ctx_str == 'bearish' else 0.0
            f['context_dir'] = ctx_val
        else:
            f['context_dir'] = 0.0

        return f

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def create_labels(self, df_15m: pd.DataFrame,
                      context_series: Optional[pd.Series] = None,
                      pt_mult: float = 2.0,
                      sl_mult: float = 1.0,
                      num_bars: int = 16) -> pd.Series:
        """
        Triple-barrier labels on 15M data, direction-aware.
        If context_series provided, uses label_with_context().
        Otherwise defaults to bullish.
        """
        if context_series is not None:
            return label_with_context(df_15m, context_series,
                                      pt_mult=pt_mult, sl_mult=sl_mult,
                                      num_bars=num_bars).fillna(0).astype(int)
        return triple_barrier_labels(df_15m, context='bullish',
                                     pt_mult=pt_mult, sl_mult=sl_mult,
                                     num_bars=num_bars).fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def pretrain(self, df_15m: pd.DataFrame,
                 hsmm_gamma_15m: np.ndarray,
                 smc_list: List[Dict],
                 context_series: pd.Series,
                 n_splits: int = 5) -> Dict:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if self.CACHE_FILE.exists():
            self._load()
            print('  [MLSetupAgent] Loaded from cache.')
            return {'from_cache': True, 'deployed': True}

        try:
            # Build context_dir series
            ctx_dir = context_series.map(
                lambda x: 1.0 if x == 'bullish' else -1.0 if x == 'bearish' else 0.0
            )

            X = self.compute_features(df_15m, hsmm_gamma_15m, smc_list,
                                      regime_result=None, context_result=None)
            # Overwrite context_dir with actual series
            if len(ctx_dir) == len(X):
                X['context_dir'] = ctx_dir.values

            # Triple-barrier labels: direction-aware using context_series
            y = self.create_labels(df_15m, context_series=context_series)

            mask = X.notna().all(axis=1) & y.notna()
            mask.iloc[-16:] = False   # embargo = num_bars of triple barrier
            X = pd.DataFrame(X[mask])
            y = pd.Series(y[mask])

            if len(X) < 300:
                print(f'  [MLSetupAgent] Not enough data ({len(X)} rows)')
                return {'deployed': False, 'reason': 'not_enough_data'}

            self._feat_names = X.columns.tolist()
            params = dict(num_leaves=31, learning_rate=0.05, n_estimators=300,
                          min_child_samples=50, verbose=-1, n_jobs=-1,
                          class_weight='balanced')

            # Walk-forward CV (15M bars: test_months=3, bars_per_month=2880)
            splitter = WalkForwardSplitter(n_folds=n_splits, test_months=3,
                                           bars_per_month=2_880, embargo_bars=16,
                                           mode='expanding', min_train_bars=10_000)
            fold_aucs = []

            for tr_idx, val_idx in splitter.split(X):
                Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
                Xval, yval = X.iloc[val_idx], y.iloc[val_idx]
                if ytr.sum() < 10 or yval.sum() < 5:
                    continue
                m = lgb.LGBMClassifier(**params)
                m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                      callbacks=[lgb.early_stopping(40, verbose=False),
                                 lgb.log_evaluation(-1)])
                pred = m.predict_proba(Xval)[:, 1]
                try:
                    fold_aucs.append(roc_auc_score(yval, pred))
                except Exception:
                    pass

            self._lgb = lgb.LGBMClassifier(**params)
            if self._lgb is not None:
                self._lgb.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
            self._trained = True
            self._save()

            mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
            print(f'  [MLSetupAgent] Trained. AUC={mean_auc:.3f} n={len(X)}')
            return {'deployed': True, 'mean_auc': mean_auc, 'fold_aucs': fold_aucs}

        except Exception as e:
            print(f'  [MLSetupAgent] Training failed: {e}')
            return {'deployed': False, 'reason': str(e)}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def analyze(self,
                df_15m: pd.DataFrame,
                hsmm_probs_row: np.ndarray,
                smc_patterns: Dict,
                regime_result: Optional[AgentResult] = None,
                context_result: Optional[AgentResult] = None) -> AgentResult:

        if len(df_15m) < self.MIN_BARS:
            return AgentResult(
                agent='setup', state='not_ready', score=0.0,
                passed=False, ready=False, blocked_by_readiness=True,
                reason=f'MLSetupAgent: need {self.MIN_BARS} bars, got {len(df_15m)}'
            )

        # Pass-through: linear combination of HSMM probs
        if hsmm_probs_row is not None and len(hsmm_probs_row) == 6:
            ctx_str = getattr(context_result, 'state', 'neutral') if context_result else 'neutral'
            p_tp   = float(hsmm_probs_row[0])
            p_rng  = float(hsmm_probs_row[1])
            p_tm   = float(hsmm_probs_row[2])
            p_sq   = float(hsmm_probs_row[3])
            p_dist = float(hsmm_probs_row[4])
            p_liq  = float(hsmm_probs_row[5])

            bull_score = p_tp + 0.5 * p_sq + 0.25 * p_rng
            bear_score = p_tm + p_dist
            pt_score   = bull_score if ctx_str == 'bullish' else \
                         bear_score if ctx_str == 'bearish' else \
                         max(bull_score, bear_score)
            pt_score   = float(np.clip(pt_score, 0.0, 1.0))
        else:
            pt_score = 0.5

        if not self._trained:
            if pt_score > 0.45:
                state = 'valid_setup'
            else:
                state = 'misaligned'
            return AgentResult(
                agent='setup', state=state, score=pt_score,
                passed=(pt_score > 0.45), ready=True,
                reason='MLSetupAgent: pass-through (not trained)'
            )

        try:
            T = len(df_15m)
            hsmm_full = np.tile(hsmm_probs_row, (T, 1))
            X = self.compute_features(df_15m, hsmm_full, smc_patterns,
                                      regime_result, context_result)
            last = X.iloc[[-1]]
            if last.isna().any().any():
                state = 'valid_setup' if pt_score > 0.45 else 'misaligned'
                return AgentResult(
                    agent='setup', state=state, score=pt_score,
                    passed=(pt_score > 0.45), ready=True,
                    reason='MLSetupAgent: NaN features, pass-through'
                )

            if self._lgb is None:
                raise RuntimeError('LGB model not initialized')
            score = float(self._lgb.predict_proba(last)[0, 1])
            feats_dict = last.iloc[0].to_dict()
            if self._n_online >= 20:
                score = 0.7 * score + 0.3 * _river_predict(self._online, feats_dict)

            if score > 0.6:
                state = 'valid_setup'
            elif score < 0.35:
                state = 'misaligned'
            else:
                state = 'no_pattern'

            return AgentResult(
                agent='setup', state=state, score=float(score),
                passed=(score > 0.45), ready=True,
                reason=f'MLSetupAgent: P(setup)={score:.3f}',
                metadata={'p_setup': score, 'hsmm_pt_score': pt_score}
            )

        except Exception as e:
            state = 'valid_setup' if pt_score > 0.45 else 'misaligned'
            return AgentResult(
                agent='setup', state=state, score=pt_score,
                passed=(pt_score > 0.45), ready=True,
                reason=f'MLSetupAgent: inference error ({e}), pass-through'
            )

    def update_online(self, features_dict: Dict, label: int) -> None:
        _river_update(self._online, features_dict, label)
        self._n_online += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        with open(self.CACHE_FILE, 'wb') as f:
            pickle.dump({'lgb': self._lgb, 'feat_names': self._feat_names}, f, protocol=4)

    def _load(self):
        with open(self.CACHE_FILE, 'rb') as f:
            p = pickle.load(f)
        self._lgb        = p['lgb']
        self._feat_names = p['feat_names']
        self._trained    = True
