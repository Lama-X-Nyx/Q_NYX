"""
ML Agents — LightGBM replacements for rule-based Context/Regime/Setup agents.

Architecture
------------
HSMM + SMA200 become FEATURE EXTRACTORS, not decision makers.
Each ML agent learns: "which combination of HSMM probs + microstructure
features actually predicts profitable direction?"

MLContextAgent  (1D daily)     → P(bull/bear/neutral)
MLRegimeAgent   (1H + HSMM)    → dominant regime + confidence
MLSetupAgent    (15M + HSMM + SMC) → P(valid setup)

All agents:
  - pretrain() : walk-forward CV with LightGBM (TimeSeriesSplit, 5 folds)
  - analyze()  : returns AgentResult (same contract as rule-based agents)
  - update_online() : River incremental update after trade resolves
  - Pass-through mode when not trained (preserves backward compatibility)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

try:
    from river import linear_model, preprocessing, compose
    _RIVER_OK = True
except ImportError:
    _RIVER_OK = False

from src.agents.contracts import AgentResult

CACHE_DIR = Path('data/pretrain_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# HSMM state ordering (must match precomputed_runner gamma columns)
HSMM_STATES = ['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation']


# ---------------------------------------------------------------------------
# Shared feature helpers
# ---------------------------------------------------------------------------

def _momentum_features(close: pd.Series, windows: List[int]) -> pd.DataFrame:
    f = pd.DataFrame(index=close.index)
    ret = close.pct_change()
    for w in windows:
        f[f'mom_{w}'] = close / close.shift(w) - 1
    return f


def _vol_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    ret = df['close'].pct_change()
    high, low, close, volume = df['high'], df['low'], df['close'], df['volume']

    for w in windows:
        f[f'rv_{w}'] = ret.rolling(w).std()

    # GARCH-like EWMA
    f['ewma_v94'] = np.sqrt(ret.ewm(alpha=0.06, adjust=False).var().clip(lower=0))
    f['ewma_v97'] = np.sqrt(ret.ewm(alpha=0.03, adjust=False).var().clip(lower=0))
    f['ewma_ratio'] = f['ewma_v94'] / (f['ewma_v97'] + 1e-9)

    # Buy pressure
    hl = (high - low).replace(0, np.nan)
    f['buy_pressure'] = (close - low) / hl
    f['buy_pressure_ma'] = f['buy_pressure'].rolling(min(8, windows[0])).mean()

    # Amihud illiquidity
    dollar_vol = volume * close
    f['amihud'] = (ret.abs() / (dollar_vol + 1e-9)).rolling(windows[-1]).mean()

    # Kyle's lambda
    f['kyle_lambda'] = (ret.abs() / (np.sqrt(volume) + 1e-9)).rolling(windows[0]).mean()

    # Effective spread
    f['eff_spread'] = (high - low) / (close + 1e-9)
    f['eff_spread_ratio'] = f['eff_spread'] / (f['eff_spread'].rolling(windows[-1]).mean() + 1e-9)

    return f


def _add_hsmm_features(f: pd.DataFrame, hsmm_probs: np.ndarray) -> pd.DataFrame:
    """Append HSMM probability columns (shape T×6) to feature DataFrame."""
    for i, state in enumerate(HSMM_STATES):
        col = f'hsmm_{state.replace("+","p").replace("-","m").lower()}'
        f[col] = hsmm_probs[:, i]
    # Dominant state index
    f['hsmm_dominant'] = np.argmax(hsmm_probs, axis=1).astype(float)
    f['hsmm_entropy'] = -(hsmm_probs * np.log(hsmm_probs + 1e-10)).sum(axis=1)
    return f


# ---------------------------------------------------------------------------
# MLContextAgent  (1D daily data)
# ---------------------------------------------------------------------------

class MLContextAgent:
    """
    Replaces SMA200 rule-based context with a LightGBM classifier.
    Predicts P(bullish) and P(bearish) from daily OHLCV features.
    """

    CACHE_PATH = CACHE_DIR / 'context_ml.pkl'

    def __init__(self):
        self.lgb_bull: Optional[lgb.LGBMClassifier] = None
        self.lgb_bear: Optional[lgb.LGBMClassifier] = None
        self._online  = None
        self._trained = False
        self._feat_cols: List[str] = []

        if _RIVER_OK:
            self._online = compose.Pipeline(
                preprocessing.StandardScaler(),
                linear_model.LogisticRegression()
            )

    # ------------------------------------------------------------------
    def compute_features(self, df_1d: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df_1d.index)
        close = df_1d['close']
        ret   = close.pct_change()

        # Multi-horizon momentum
        for w in [5, 20, 60, 200]:
            f[f'mom_{w}d'] = close / close.shift(w) - 1
        f['mom_acc_5'] = f['mom_5d'].diff()

        # Volatility
        f['rv_5']  = ret.rolling(5).std()
        f['rv_20'] = ret.rolling(20).std()
        f['rv_ratio'] = f['rv_5'] / (f['rv_20'] + 1e-9)
        f['ewma_v94'] = np.sqrt(ret.ewm(alpha=0.06, adjust=False).var().clip(lower=0))

        # RSI (Wilder's EMA)
        delta = close.diff()
        g = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        l = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        f['rsi_14'] = 100 - 100 / (1 + g / (l + 1e-9))

        # Trend position
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        f['vs_sma50']  = (close - sma50) / (sma50 + 1e-9)
        f['vs_sma200'] = (close - sma200) / (sma200 + 1e-9)
        f['sma50_200_ratio'] = sma50 / (sma200 + 1e-9) - 1

        # Garman-Klass vol
        hl = np.log(df_1d['high'] / df_1d['low']) ** 2
        co = np.log(close / df_1d['open']) ** 2
        gk = 0.5 * hl.rolling(20).mean() - (2 * np.log(2) - 1) * co.rolling(20).mean()
        f['gk_20d'] = np.sqrt(gk.clip(lower=0))

        # Buy pressure
        hl_rng = (df_1d['high'] - df_1d['low']).replace(0, np.nan)
        f['buy_pressure'] = (close - df_1d['low']) / hl_rng
        f['buy_pressure_ma'] = f['buy_pressure'].rolling(10).mean()

        # Amihud
        dollar_vol = df_1d['volume'] * close
        f['amihud'] = (ret.abs() / (dollar_vol + 1e-9)).rolling(20).mean()

        return f

    def create_labels(self, df_1d: pd.DataFrame,
                      forward_bars: int = 5,
                      bull_thr: float = 0.03,
                      bear_thr: float = -0.03) -> pd.Series:
        """
        1 = bullish (max future return > bull_thr)
        0 = bearish/neutral
        """
        close = df_1d['close']
        fut_max = close.rolling(forward_bars).max().shift(-forward_bars)
        fut_min = close.rolling(forward_bars).min().shift(-forward_bars)
        bull = (fut_max / close - 1 > bull_thr).astype(int)
        bear = (close / fut_min - 1 > abs(bear_thr)).astype(int)
        # For bull classifier: label=1 iff bullish AND NOT bearish
        y_bull = (bull & ~bear.astype(bool)).astype(int)
        y_bear = (bear & ~bull.astype(bool)).astype(int)
        return y_bull, y_bear

    def pretrain(self, df_1d: pd.DataFrame, n_splits: int = 5) -> Dict:
        X = self.compute_features(df_1d)
        y_bull, y_bear = self.create_labels(df_1d)

        valid = X.dropna().index.intersection(y_bull.dropna().index)
        X = X.loc[valid].dropna()
        y_bull = y_bull.loc[X.index]
        y_bear = y_bear.loc[X.index]
        self._feat_cols = list(X.columns)

        if len(X) < 100:
            print('  [ContextML] Not enough data for training')
            return {}

        tscv = TimeSeriesSplit(n_splits=n_splits)
        params = dict(num_leaves=15, learning_rate=0.05, n_estimators=200,
                      min_child_samples=20, verbose=-1, n_jobs=-1)

        scores_bull, scores_bear = [], []
        for tr, va in tscv.split(X):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            yb_tr, yb_va = y_bull.iloc[tr], y_bull.iloc[va]
            yz_tr, yz_va = y_bear.iloc[tr], y_bear.iloc[va]

            m_bull = lgb.LGBMClassifier(**params)
            m_bull.fit(Xtr, yb_tr)
            if yb_va.sum() > 0:
                scores_bull.append(roc_auc_score(yb_va, m_bull.predict_proba(Xva)[:, 1]))

            m_bear = lgb.LGBMClassifier(**params)
            m_bear.fit(Xtr, yz_tr)
            if yz_va.sum() > 0:
                scores_bear.append(roc_auc_score(yz_va, m_bear.predict_proba(Xva)[:, 1]))

        # Final fit on all data
        self.lgb_bull = lgb.LGBMClassifier(**params)
        self.lgb_bull.fit(X, y_bull)
        self.lgb_bear = lgb.LGBMClassifier(**params)
        self.lgb_bear.fit(X, y_bear)
        self._trained = True

        result = {
            'auc_bull': float(np.mean(scores_bull)) if scores_bull else 0.0,
            'auc_bear': float(np.mean(scores_bear)) if scores_bear else 0.0,
            'n_samples': len(X),
        }
        self._save()
        print(f'  [ContextML] AUC bull={result["auc_bull"]:.3f}  bear={result["auc_bear"]:.3f}  '
              f'n={result["n_samples"]}')
        return result

    def analyze(self, df_1d: pd.DataFrame) -> AgentResult:
        if not self._trained and not self._load():
            return self._passthrough(df_1d)

        X = self.compute_features(df_1d)
        last = X[self._feat_cols].iloc[-1:].fillna(0)

        p_bull = float(self.lgb_bull.predict_proba(last)[0, 1])
        p_bear = float(self.lgb_bear.predict_proba(last)[0, 1])
        p_neut = max(0.0, 1.0 - p_bull - p_bear)

        if p_bull > p_bear and p_bull > 0.40:
            state, score = 'bullish', p_bull
        elif p_bear > p_bull and p_bear > 0.40:
            state, score = 'bearish', p_bear
        else:
            state, score = 'neutral', p_neut

        return AgentResult(
            agent='context', state=state, score=min(score, 1.0),
            passed=(state != 'neutral'), ready=True,
            reason=f'MLContext: P(bull)={p_bull:.2f} P(bear)={p_bear:.2f}',
            metadata={'p_bull': p_bull, 'p_bear': p_bear, 'p_neutral': p_neut}
        )

    def update_online(self, features: Dict, label: int) -> None:
        if _RIVER_OK and self._online is not None:
            try:
                self._online.learn_one(features, label)
            except Exception:
                pass

    def _passthrough(self, df_1d: pd.DataFrame) -> AgentResult:
        """SMA200 fallback when not trained."""
        close = df_1d['close']
        sma200 = close.rolling(200).mean()
        if sma200.isna().iloc[-1]:
            return AgentResult(agent='context', state='not_ready', score=0.0,
                               passed=False, ready=False, blocked_by_readiness=True,
                               reason='SMA200 warmup')
        diff = (close.iloc[-1] - sma200.iloc[-1]) / sma200.iloc[-1]
        if diff > 0.02:
            state = 'bullish'
        elif diff < -0.02:
            state = 'bearish'
        else:
            state = 'neutral'
        return AgentResult(agent='context', state=state, score=0.5,
                           passed=(state != 'neutral'), ready=True,
                           reason=f'Passthrough SMA200 diff={diff:.2%}')

    def _save(self):
        with open(self.CACHE_PATH, 'wb') as f:
            pickle.dump({'lgb_bull': self.lgb_bull, 'lgb_bear': self.lgb_bear,
                         'feat_cols': self._feat_cols}, f)

    def _load(self) -> bool:
        if not self.CACHE_PATH.exists():
            return False
        try:
            p = pickle.load(open(self.CACHE_PATH, 'rb'))
            self.lgb_bull   = p['lgb_bull']
            self.lgb_bear   = p['lgb_bear']
            self._feat_cols = p['feat_cols']
            self._trained   = True
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# MLRegimeAgent  (1H data + HSMM probs)
# ---------------------------------------------------------------------------

class MLRegimeAgent:
    """
    Augments the HSMM regime signal with LightGBM.
    HSMM probabilities become features — the model learns which combinations
    of HSMM probs + microstructure actually predict profitable direction.
    """

    CACHE_PATH = CACHE_DIR / 'regime_ml.pkl'

    def __init__(self):
        self.lgb: Optional[lgb.LGBMClassifier] = None
        self._online  = None
        self._trained = False
        self._feat_cols: List[str] = []

        if _RIVER_OK:
            self._online = compose.Pipeline(
                preprocessing.StandardScaler(),
                linear_model.LogisticRegression()
            )

    def compute_features(self, df_1h: pd.DataFrame,
                         hsmm_probs: np.ndarray) -> pd.DataFrame:
        f = _momentum_features(df_1h['close'], [4, 12, 48])
        vol = _vol_features(df_1h, [8, 24, 96])
        f = pd.concat([f, vol], axis=1)
        f = _add_hsmm_features(f, hsmm_probs)

        # RSI 14
        delta = df_1h['close'].diff()
        g = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        l = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        f['rsi_14'] = 100 - 100 / (1 + g / (l + 1e-9))

        # Vol regime: vol of vol
        ret = df_1h['close'].pct_change()
        f['vov_24'] = ret.rolling(24).std().rolling(24).std()

        return f

    def create_labels(self, df_1h: pd.DataFrame,
                      forward_bars: int = 4,
                      thr: float = 0.01) -> pd.Series:
        """Binary: 1 = bullish (max future high > close*(1+thr) within 4h)"""
        close = df_1h['close']
        fut_high = df_1h['high'].rolling(forward_bars).max().shift(-forward_bars)
        return ((fut_high / close - 1) > thr).astype(int)

    def pretrain(self, df_1h: pd.DataFrame,
                 hsmm_gamma: np.ndarray,
                 n_splits: int = 5) -> Dict:
        X = self.compute_features(df_1h, hsmm_gamma)
        y = self.create_labels(df_1h)

        valid = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid].dropna()
        y = y.loc[X.index]
        self._feat_cols = list(X.columns)

        if len(X) < 200:
            print('  [RegimeML] Not enough data')
            return {}

        params = dict(num_leaves=31, learning_rate=0.05, n_estimators=300,
                      min_child_samples=50, verbose=-1, n_jobs=-1,
                      class_weight='balanced')

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for tr, va in tscv.split(X):
            m = lgb.LGBMClassifier(**params)
            m.fit(X.iloc[tr], y.iloc[tr])
            if y.iloc[va].sum() > 0:
                scores.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))

        self.lgb = lgb.LGBMClassifier(**params)
        self.lgb.fit(X, y)
        self._trained = True

        result = {'auc': float(np.mean(scores)) if scores else 0.0, 'n_samples': len(X)}
        self._save()
        print(f'  [RegimeML]  AUC={result["auc"]:.3f}  n={result["n_samples"]}')
        return result

    def analyze(self, df_1h: pd.DataFrame,
                hsmm_probs_row: np.ndarray) -> AgentResult:
        """
        hsmm_probs_row : (6,) array for the current 1H bar
        """
        if not self._trained and not self._load():
            return self._passthrough(hsmm_probs_row)

        # Build single-row feature vector from last bar
        if len(df_1h) < 100:
            return self._passthrough(hsmm_probs_row)

        hsmm_2d = np.tile(hsmm_probs_row, (len(df_1h), 1))
        X = self.compute_features(df_1h, hsmm_2d)
        last = X[self._feat_cols].iloc[-1:].fillna(0)
        p_bull = float(self.lgb.predict_proba(last)[0, 1])

        dom_idx  = int(np.argmax(hsmm_probs_row))
        dom_name = HSMM_STATES[dom_idx]
        dom_prob = float(hsmm_probs_row[dom_idx])

        state_map = {
            'Trend+': 'trend_plus', 'Range': 'range', 'Trend-': 'trend_minus',
            'Squeeze': 'squeeze', 'Distribution': 'distribution', 'Liquidation': 'liquidation'
        }
        state = state_map.get(dom_name, 'range')
        passed = p_bull > 0.50 and state != 'liquidation'

        return AgentResult(
            agent='regime', state=state,
            score=p_bull, passed=passed, ready=True,
            reason=f'MLRegime: P(bull)={p_bull:.2f} dom={dom_name}({dom_prob:.2f})',
            metadata={
                'p_bull': p_bull,
                'dominant_state': dom_name,
                'dominant_prob': dom_prob,
                'hsmm_probs': {s: float(hsmm_probs_row[i]) for i, s in enumerate(HSMM_STATES)},
            }
        )

    def update_online(self, features: Dict, label: int) -> None:
        if _RIVER_OK and self._online is not None:
            try:
                self._online.learn_one(features, label)
            except Exception:
                pass

    def _passthrough(self, hsmm_probs_row: np.ndarray) -> AgentResult:
        dom_idx  = int(np.argmax(hsmm_probs_row))
        dom_name = HSMM_STATES[dom_idx]
        dom_prob = float(hsmm_probs_row[dom_idx])
        state_map = {
            'Trend+': 'trend_plus', 'Range': 'range', 'Trend-': 'trend_minus',
            'Squeeze': 'squeeze', 'Distribution': 'distribution', 'Liquidation': 'liquidation'
        }
        state  = state_map.get(dom_name, 'range')
        passed = dom_prob > 0.45 and state != 'liquidation'
        return AgentResult(
            agent='regime', state=state, score=dom_prob,
            passed=passed, ready=True,
            reason=f'Passthrough HSMM dom={dom_name}({dom_prob:.2f})',
            metadata={'dominant_state': dom_name, 'dominant_prob': dom_prob}
        )

    def _save(self):
        with open(self.CACHE_PATH, 'wb') as f:
            pickle.dump({'lgb': self.lgb, 'feat_cols': self._feat_cols}, f)

    def _load(self) -> bool:
        if not self.CACHE_PATH.exists():
            return False
        try:
            p = pickle.load(open(self.CACHE_PATH, 'rb'))
            self.lgb        = p['lgb']
            self._feat_cols = p['feat_cols']
            self._trained   = True
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# MLSetupAgent  (15M data + HSMM probs + SMC patterns)
# ---------------------------------------------------------------------------

class MLSetupAgent:
    """
    Predicts P(valid_setup) from 15M features, HSMM probs, SMC patterns,
    and the outputs of the upstream agents (regime + context).
    """

    CACHE_PATH = CACHE_DIR / 'setup_ml.pkl'

    def __init__(self):
        self.lgb: Optional[lgb.LGBMClassifier] = None
        self._online  = None
        self._trained = False
        self._feat_cols: List[str] = []

        if _RIVER_OK:
            self._online = compose.Pipeline(
                preprocessing.StandardScaler(),
                linear_model.LogisticRegression()
            )

    def compute_features(self, df_15m: pd.DataFrame,
                         hsmm_probs: np.ndarray,
                         smc_patterns: Dict,
                         regime_score: float = 0.5,
                         context_dir: float = 0.0) -> pd.DataFrame:
        """
        context_dir : 1.0=bullish, -1.0=bearish, 0.0=neutral
        """
        f = _momentum_features(df_15m['close'], [4, 8, 16, 32])
        vol = _vol_features(df_15m, [8, 32, 96])
        f = pd.concat([f, vol], axis=1)
        f = _add_hsmm_features(f, hsmm_probs)

        # SMC pattern features (constant over the df, from the last bar's patterns)
        smc_keys = ['bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg',
                    'bullish_choch', 'bearish_choch', 'smc_score_bullish', 'smc_score_bearish']
        for k in smc_keys:
            val = float(smc_patterns.get(k, 0.0))
            if isinstance(val, bool):
                val = float(val)
            f[f'smc_{k}'] = val

        # Upstream agent scores (cross-agent features)
        f['regime_score']  = regime_score
        f['context_dir']   = context_dir

        # Interaction: regime × direction alignment
        f['regime_context_align'] = regime_score * context_dir

        return f

    def create_labels(self, df_15m: pd.DataFrame,
                      context: str = 'bullish',
                      forward_bars: int = 8,
                      thr: float = 0.008) -> pd.Series:
        """1 = price moved in context direction by thr within forward_bars"""
        close = df_15m['close']
        if context == 'bullish':
            fut = df_15m['high'].rolling(forward_bars).max().shift(-forward_bars)
            return ((fut / close - 1) > thr).astype(int)
        else:
            fut = df_15m['low'].rolling(forward_bars).min().shift(-forward_bars)
            return ((close / fut - 1) > thr).astype(int)

    def pretrain(self, df_15m: pd.DataFrame,
                 hsmm_gamma_15m: np.ndarray,
                 smc_list: List[Dict],
                 context_series: pd.Series,
                 smc_start_iloc: int = 0,
                 n_splits: int = 5) -> Dict:
        """
        hsmm_gamma_15m : (T, 6) array aligned with df_15m
        smc_list       : list[dict] from precomputed_runner
        smc_start_iloc : offset for smc_list indexing
        context_series : pd.Series of 'bullish'/'bearish'/'neutral' aligned to df_15m
        """
        all_X, all_y = [], []

        for ctx_str, ctx_dir in [('bullish', 1.0), ('bearish', -1.0)]:
            mask = (context_series == ctx_str)
            if mask.sum() < 200:
                continue

            df_ctx  = df_15m[mask]
            iloc_arr = np.where(mask.values)[0]

            rows = []
            for abs_iloc in iloc_arr:
                smc_idx = abs_iloc - smc_start_iloc
                smc_pat = smc_list[smc_idx] if 0 <= smc_idx < len(smc_list) else {}
                hsmm_row = hsmm_gamma_15m[abs_iloc]

                # Build feature vector for this bar
                start = max(0, abs_iloc - 200)
                df_sl = df_15m.iloc[start:abs_iloc + 1]
                hsmm_2d = np.tile(hsmm_row, (len(df_sl), 1))
                X_sl = self.compute_features(df_sl, hsmm_2d, smc_pat,
                                             regime_score=0.5, context_dir=ctx_dir)
                rows.append(X_sl.iloc[-1])

            if not rows:
                continue

            X_ctx = pd.DataFrame(rows).reset_index(drop=True)
            y_ctx = self.create_labels(df_ctx, context=ctx_str).reset_index(drop=True)

            valid = X_ctx.dropna().index
            all_X.append(X_ctx.loc[valid])
            all_y.append(y_ctx.iloc[valid])

        if not all_X:
            print('  [SetupML] Not enough data')
            return {}

        X = pd.concat(all_X, ignore_index=True)
        y = pd.concat(all_y, ignore_index=True)
        self._feat_cols = list(X.columns)

        # Sort by time (all_X rows are already time-ordered within each context)
        params = dict(num_leaves=31, learning_rate=0.05, n_estimators=400,
                      min_child_samples=50, verbose=-1, n_jobs=-1,
                      class_weight='balanced')

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for tr, va in tscv.split(X):
            m = lgb.LGBMClassifier(**params)
            m.fit(X.iloc[tr], y.iloc[tr])
            if y.iloc[va].sum() > 0:
                scores.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))

        self.lgb = lgb.LGBMClassifier(**params)
        self.lgb.fit(X, y)
        self._trained = True

        result = {'auc': float(np.mean(scores)) if scores else 0.0, 'n_samples': len(X)}
        self._save()
        print(f'  [SetupML]   AUC={result["auc"]:.3f}  n={result["n_samples"]}')
        return result

    def analyze(self, df_15m: pd.DataFrame,
                hsmm_probs_row: np.ndarray,
                smc_patterns: Dict,
                regime_result: AgentResult,
                context_result: AgentResult) -> AgentResult:
        if not self._trained and not self._load():
            return self._passthrough(hsmm_probs_row, smc_patterns, context_result)

        if len(df_15m) < 50:
            return self._passthrough(hsmm_probs_row, smc_patterns, context_result)

        ctx_dir = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0}.get(
            context_result.state, 0.0)

        hsmm_2d = np.tile(hsmm_probs_row, (len(df_15m), 1))
        X = self.compute_features(df_15m, hsmm_2d, smc_patterns,
                                  regime_score=regime_result.score,
                                  context_dir=ctx_dir)
        last = X[self._feat_cols].iloc[-1:].fillna(0)
        p_setup = float(self.lgb.predict_proba(last)[0, 1])

        passed = p_setup > 0.45
        has_pat = bool(
            smc_patterns.get('bullish_ob') or smc_patterns.get('bullish_fvg') or
            smc_patterns.get('bullish_choch') if ctx_dir > 0 else
            smc_patterns.get('bearish_ob') or smc_patterns.get('bearish_fvg') or
            smc_patterns.get('bearish_choch')
        )
        state = 'valid_setup' if (passed and has_pat) else \
                'misaligned' if not passed else 'no_pattern'

        return AgentResult(
            agent='setup', state=state,
            score=p_setup, passed=passed, ready=True,
            reason=f'MLSetup: P(setup)={p_setup:.2f} SMC={has_pat}',
            metadata={'p_setup': p_setup, 'has_pattern': has_pat, 'smc': smc_patterns}
        )

    def update_online(self, features: Dict, label: int) -> None:
        if _RIVER_OK and self._online is not None:
            try:
                self._online.learn_one(features, label)
            except Exception:
                pass

    def _passthrough(self, hsmm_probs_row: np.ndarray,
                     smc_patterns: Dict,
                     context_result: AgentResult) -> AgentResult:
        ctx = context_result.state if context_result else 'neutral'
        ctx_dir = {'bullish': 1.0, 'bearish': -1.0}.get(ctx, 0.0)

        p_tp   = float(hsmm_probs_row[0])
        p_rng  = float(hsmm_probs_row[1])
        p_tm   = float(hsmm_probs_row[2])
        p_sq   = float(hsmm_probs_row[3])

        bull_score = p_tp + 0.5 * p_sq + 0.25 * p_rng
        bear_score = p_tm + float(hsmm_probs_row[4])  # Trend- + Distribution

        alignment  = bull_score if ctx_dir > 0 else bear_score if ctx_dir < 0 else max(bull_score, bear_score)
        passed     = alignment > 0.40
        state      = 'valid_setup' if passed else 'misaligned'

        return AgentResult(
            agent='setup', state=state, score=alignment,
            passed=passed, ready=True,
            reason=f'Passthrough HSMM align={alignment:.2f}',
        )

    def _save(self):
        with open(self.CACHE_PATH, 'wb') as f:
            pickle.dump({'lgb': self.lgb, 'feat_cols': self._feat_cols}, f)

    def _load(self) -> bool:
        if not self.CACHE_PATH.exists():
            return False
        try:
            p = pickle.load(open(self.CACHE_PATH, 'rb'))
            self.lgb        = p['lgb']
            self._feat_cols = p['feat_cols']
            self._trained   = True
            return True
        except Exception:
            return False
