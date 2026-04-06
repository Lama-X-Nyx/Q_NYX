"""
MLOrchestrator — meta-learner that combines all 4 agent signals.

Key insight: agents may disagree in correlated ways that a rule-based
threshold (min_score=0.78) cannot capture. The meta-learner learns:
"when ContextML=0.8 AND RegimeML=0.6 AND SetupML=0.5 → actual WR=72%"

Training labels: direction outcome = did price move >0.8% in the right
direction within the next 8 bars?
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from src.agents.contracts import AgentResult, OrchestratorDecision

try:
    from river import linear_model, preprocessing, compose, metrics as river_metrics
    _RIVER_OK = True
except ImportError:
    _RIVER_OK = False

CACHE_DIR  = Path('data/pretrain_cache')
CACHE_FILE = CACHE_DIR / 'orchestrator_ml.pkl'


class MLOrchestrator:
    """
    Meta-learner that takes outputs from all 4 ML agents and learns the
    optimal non-linear combination to predict trade profitability.

    Key insight: agents may disagree in correlated ways that a rule-based
    threshold (min_score=0.78) cannot capture. The meta-learner learns:
    "when ContextML=0.8 AND RegimeML=0.6 AND SetupML=0.5 → actual WR=72%"

    Training labels: direction outcome = did price move >0.8% in the right
    direction within the next 8 bars? (bar-level, not trade-level → 50-100x
    more samples than actual trade outcomes)

    Features fed to meta-LGB:
      - p_context_bull, p_context_bear
      - p_regime_trend, p_regime_range, p_regime_squeeze
      - p_setup, p_entry
      - vol_ratio (short/long vol ratio)
      - buy_pressure (last bar)
      - amihud (rolling)
      - hour_of_day (sin/cos encoded, 0-23)
      - day_of_week (sin/cos encoded, 0-6)
      - agent_agreement: std of [p_context, p_regime, p_setup, p_entry] — low std = consensus
      - agent_max, agent_min
    """

    # Rule-based fallback threshold
    RULE_THRESHOLD  = 0.78
    # ML action threshold
    ML_THRESHOLD    = 0.55
    # Min forward return to be a positive label (0.8% in the right direction)
    LABEL_THR       = 0.008
    LABEL_BARS      = 8

    def __init__(self):
        self._lgb:        Optional[lgb.LGBMClassifier] = None
        self._trained     = False
        self._feat_names: Optional[List[str]] = None

        # River online adapter
        self._online = None
        self._n_online = 0
        if _RIVER_OK:
            try:
                self._online = compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LogisticRegression()
                )
                self._online_metric = river_metrics.ROCAUC()
            except Exception:
                self._online = None

    # ------------------------------------------------------------------
    # Meta-feature builder
    # ------------------------------------------------------------------

    def _build_meta_features(self,
                              context_result: AgentResult,
                              regime_result:  AgentResult,
                              setup_result:   AgentResult,
                              entry_result:   AgentResult,
                              df_15m_row:     pd.Series) -> Dict:
        """
        Builds the meta-feature dict from 4 agent results + last 15m bar.
        df_15m_row: a single-row Series with at minimum close, high, low, volume.
        """
        # Context probabilities
        ctx_meta  = context_result.metadata if context_result else {}
        p_ctx_bull = float(ctx_meta.get('p_bull', context_result.score if context_result and context_result.state == 'bullish' else 0.5))
        p_ctx_bear = float(ctx_meta.get('p_bear', context_result.score if context_result and context_result.state == 'bearish' else 0.5))

        # Regime probabilities
        reg_meta   = regime_result.metadata if regime_result else {}
        hsmm_probs = reg_meta.get('hsmm_probs', [1/6] * 6)
        if len(hsmm_probs) >= 6:
            p_reg_trend = float(hsmm_probs[0]) + float(hsmm_probs[2])   # Trend+ + Trend-
            p_reg_range = float(hsmm_probs[1])
            p_reg_squeeze = float(hsmm_probs[3])
        else:
            p_reg_trend, p_reg_range, p_reg_squeeze = 0.33, 0.33, 0.33

        # Setup / entry scores
        p_setup = float(setup_result.score) if setup_result else 0.5
        p_entry = float(entry_result.score) if entry_result else 0.5

        # 15m bar micro-features
        close  = float(df_15m_row.get('close', 1.0))
        high   = float(df_15m_row.get('high',  close))
        low    = float(df_15m_row.get('low',   close))
        volume = float(df_15m_row.get('volume', 1.0))

        denom        = (high - low) if (high - low) > 0 else 1e-9
        buy_pressure = (close - low) / denom

        # vol_ratio pulled from setup metadata if available
        setup_meta = setup_result.metadata if setup_result else {}
        vol_ratio  = float(setup_meta.get('vol_ratio', 1.0))
        amihud     = float(setup_meta.get('amihud', 0.0))

        # Time features (sin/cos encoding)
        ts = df_15m_row.name if hasattr(df_15m_row, 'name') and isinstance(df_15m_row.name, pd.Timestamp) else pd.Timestamp.now()
        hour = ts.hour
        dow  = ts.dayofweek
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin  = np.sin(2 * np.pi * dow  / 7.0)
        dow_cos  = np.cos(2 * np.pi * dow  / 7.0)

        # Agent agreement
        scores_vec  = [p_ctx_bull, p_reg_trend, p_setup, p_entry]
        agent_agree = float(np.std(scores_vec))
        agent_max   = float(np.max(scores_vec))
        agent_min   = float(np.min(scores_vec))

        return {
            'p_context_bull':  p_ctx_bull,
            'p_context_bear':  p_ctx_bear,
            'p_regime_trend':  p_reg_trend,
            'p_regime_range':  p_reg_range,
            'p_regime_squeeze': p_reg_squeeze,
            'p_setup':          p_setup,
            'p_entry':          p_entry,
            'vol_ratio':        vol_ratio,
            'buy_pressure':     buy_pressure,
            'amihud':           amihud,
            'hour_sin':         hour_sin,
            'hour_cos':         hour_cos,
            'dow_sin':          dow_sin,
            'dow_cos':          dow_cos,
            'agent_agreement':  agent_agree,
            'agent_max':        agent_max,
            'agent_min':        agent_min,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def pretrain(self, meta_dataset: List[Tuple[Dict, int, str]],
                 n_splits: int = 5) -> Dict:
        """
        meta_dataset: list of (features_dict, label, direction)
          - features_dict: output of _build_meta_features
          - label: 1 = price moved in direction by LABEL_THR within LABEL_BARS
          - direction: 'bullish' | 'bearish'
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if CACHE_FILE.exists():
            self._load()
            print('  [MLOrchestrator] Loaded from cache.')
            return {'from_cache': True, 'deployed': True}

        if len(meta_dataset) < 200:
            print(f'  [MLOrchestrator] Not enough meta-samples ({len(meta_dataset)})')
            return {'deployed': False, 'reason': 'not_enough_data'}

        try:
            records = [fd for fd, _, _ in meta_dataset]
            labels  = [lbl for _, lbl, _ in meta_dataset]

            X = pd.DataFrame(records)
            y = pd.Series(labels)

            # Drop rows with NaNs
            mask = X.notna().all(axis=1)
            X, y = X[mask], y[mask]

            self._feat_names = X.columns.tolist()
            params = dict(
                num_leaves=15, learning_rate=0.03, n_estimators=500,
                min_child_samples=30, verbose=-1, n_jobs=-1,
                class_weight='balanced'
            )

            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_aucs = []

            for tr_idx, val_idx in tscv.split(X):
                Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
                Xval, yval = X.iloc[val_idx], y.iloc[val_idx]
                if ytr.sum() < 10 or yval.sum() < 5:
                    continue
                m = lgb.LGBMClassifier(**params)
                m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
                pred = m.predict_proba(Xval)[:, 1]
                try:
                    fold_aucs.append(roc_auc_score(yval, pred))
                except Exception:
                    pass

            self._lgb = lgb.LGBMClassifier(**params)
            self._lgb.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
            self._trained = True
            self._save()

            mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
            print(f'  [MLOrchestrator] Trained. AUC={mean_auc:.3f} n={len(X)}')
            return {'deployed': True, 'mean_auc': mean_auc, 'fold_aucs': fold_aucs,
                    'n_samples': len(X)}

        except Exception as e:
            print(f'  [MLOrchestrator] Training failed: {e}')
            return {'deployed': False, 'reason': str(e)}

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self,
               context_result: AgentResult,
               regime_result:  AgentResult,
               setup_result:   AgentResult,
               entry_result:   AgentResult,
               df_15m_row:     pd.Series,
               current_price:  float) -> OrchestratorDecision:
        """
        Produce an OrchestratorDecision from all 4 agent results.
        Falls back to rule-based when not trained.
        """
        components = {
            'context': context_result,
            'regime':  regime_result,
            'setup':   setup_result,
            'entry':   entry_result,
        }

        # Collect blocked agents
        blocked_by = [k for k, r in components.items() if r is not None and not r.passed]
        not_ready  = [k for k, r in components.items() if r is not None and not r.ready]

        if not_ready:
            return OrchestratorDecision(
                action='WAIT', score=0.0,
                reason=f'Not ready: {not_ready}',
                blocked_by=['readiness'],
                components=components
            )

        # ---- Rule-based fallback ----
        if not self._trained:
            weights = {'context': 0.30, 'regime': 0.30, 'setup': 0.25, 'entry': 0.15}
            agg_score = sum(
                components[k].score * weights[k]
                for k in weights if components[k] is not None
            )
            agg_score = float(np.clip(agg_score, 0.0, 1.0))

            if blocked_by or agg_score < self.RULE_THRESHOLD:
                return OrchestratorDecision(
                    action='WAIT', score=agg_score,
                    reason=f'Rule-based WAIT: score={agg_score:.3f} blocked={blocked_by}',
                    blocked_by=blocked_by,
                    components=components
                )

            ctx_state = context_result.state if context_result else 'neutral'
            action    = 'BUY' if ctx_state == 'bullish' else \
                        'SELL' if ctx_state == 'bearish' else 'WAIT'
            return OrchestratorDecision(
                action=action, score=agg_score,
                reason=f'Rule-based {action}: score={agg_score:.3f}',
                blocked_by=[],
                components=components,
                risk_analysis={'size_factor': 1.0}
            )

        # ---- ML decision ----
        try:
            meta_feats = self._build_meta_features(
                context_result, regime_result, setup_result, entry_result, df_15m_row
            )

            # River blend
            if self._n_online >= 20 and self._online is not None and _RIVER_OK:
                try:
                    proba = self._online.predict_proba_one(meta_feats)
                    p_online = float(proba.get(True, 0.5))
                except Exception:
                    p_online = 0.5
            else:
                p_online = 0.5

            X_row = pd.DataFrame([meta_feats])[self._feat_names]
            p_lgb = float(self._lgb.predict_proba(X_row)[0, 1])

            if self._n_online >= 20:
                p_profit = 0.7 * p_lgb + 0.3 * p_online
            else:
                p_profit = p_lgb

            p_profit = float(np.clip(p_profit, 0.0, 1.0))

            # Direction from context
            ctx_state = context_result.state if context_result else 'neutral'

            if p_profit <= self.ML_THRESHOLD or ctx_state == 'neutral':
                return OrchestratorDecision(
                    action='WAIT', score=p_profit,
                    reason=f'ML WAIT: P(profit)={p_profit:.3f} ctx={ctx_state}',
                    blocked_by=blocked_by or (['ml_threshold'] if p_profit <= self.ML_THRESHOLD else []),
                    components=components,
                    risk_analysis={'p_profit': p_profit, 'size_factor': 0.0}
                )

            action = 'BUY' if ctx_state == 'bullish' else 'SELL'

            # Size factor
            if p_profit > 0.75:
                size_factor = 1.5
            elif p_profit > 0.60:
                size_factor = 1.0
            else:
                size_factor = 0.6

            return OrchestratorDecision(
                action=action, score=p_profit,
                reason=f'ML {action}: P(profit)={p_profit:.3f} ctx={ctx_state}',
                blocked_by=[],
                components=components,
                risk_analysis={'p_profit': p_profit, 'size_factor': size_factor,
                               'p_lgb': p_lgb, 'p_online': p_online}
            )

        except Exception as e:
            # Fall back to rule-based on inference error
            weights = {'context': 0.30, 'regime': 0.30, 'setup': 0.25, 'entry': 0.15}
            agg_score = float(np.clip(
                sum(components[k].score * weights[k] for k in weights if components[k] is not None),
                0.0, 1.0
            ))
            ctx_state = context_result.state if context_result else 'neutral'
            action = 'BUY' if (ctx_state == 'bullish' and agg_score >= self.RULE_THRESHOLD) else \
                     'SELL' if (ctx_state == 'bearish' and agg_score >= self.RULE_THRESHOLD) else 'WAIT'
            return OrchestratorDecision(
                action=action, score=agg_score,
                reason=f'ML fallback ({e}): rule-based {action}',
                blocked_by=blocked_by,
                components=components
            )

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update_online(self, features_dict: Dict, outcome: int) -> None:
        if self._online is None or not _RIVER_OK:
            return
        try:
            self._online.learn_one(features_dict, bool(outcome))
            self._n_online += 1
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Meta-dataset generation
    # ------------------------------------------------------------------

    def generate_training_data(self,
                                df_15m: pd.DataFrame,
                                context_arr: np.ndarray,
                                gamma_1h: np.ndarray,
                                gamma_15m: np.ndarray,
                                smc_list: List[Dict],
                                smc_start_iloc: int = 0,
                                warmup: int = 200) -> List[Tuple[Dict, int, str]]:
        """
        Generate meta-dataset from precomputed arrays.

        For each 15m bar (after warmup), builds a minimal features_dict
        using raw arrays and creates a forward-return label.

        Parameters
        ----------
        df_15m         : full 15m OHLCV DataFrame
        context_arr    : (T_1d,) string array of 1D context states
        gamma_1h       : (T_1h, 6) HSMM probability array for 1H
        gamma_15m      : (T_15m, 6) HSMM probability array for 15M
        smc_list       : list of SMC pattern dicts
        smc_start_iloc : absolute iloc offset for smc_list[0]
        warmup         : bars to skip at start
        """
        close = df_15m['close'].values
        high  = df_15m['high'].values
        low   = df_15m['low'].values
        vol   = df_15m['volume'].values
        n     = len(df_15m)

        # Build a 1D context index mapping: one context per day
        # We approximate by resampling context_arr to 15m length
        ctx_15m = self._broadcast_context(context_arr, n)

        records = []
        for i in range(warmup, n - self.LABEL_BARS):
            ctx = ctx_15m[i]
            if ctx not in ('bullish', 'bearish'):
                continue

            # Forward return label
            if ctx == 'bullish':
                future_max = high[i + 1: i + 1 + self.LABEL_BARS].max()
                label = 1 if future_max > close[i] * (1 + self.LABEL_THR) else 0
            else:
                future_min = low[i + 1: i + 1 + self.LABEL_BARS].min()
                label = 1 if future_min < close[i] * (1 - self.LABEL_THR) else 0

            # Build meta features from raw arrays
            hsmm_row_15m = gamma_15m[i] if gamma_15m is not None and i < len(gamma_15m) else np.ones(6) / 6
            hsmm_row_1h  = gamma_1h[min(i // 4, len(gamma_1h) - 1)] if gamma_1h is not None and len(gamma_1h) > 0 else np.ones(6) / 6

            smc_idx = i - smc_start_iloc
            smc_pat = smc_list[smc_idx] if 0 <= smc_idx < len(smc_list) else {}

            # Approximate agent scores from raw arrays
            p_trend  = float(hsmm_row_1h[0]) + float(hsmm_row_1h[2])
            p_range  = float(hsmm_row_1h[1])
            p_squeeze= float(hsmm_row_1h[3])
            p_setup  = float(hsmm_row_15m[0]) + 0.5 * float(hsmm_row_15m[3]) + 0.25 * float(hsmm_row_15m[1])
            p_setup  = float(np.clip(p_setup, 0.0, 1.0))

            p_ctx_bull = 0.8 if ctx == 'bullish' else 0.2
            p_ctx_bear = 0.8 if ctx == 'bearish' else 0.2

            # vol_ratio
            rv_short = float(np.std(close[max(0, i-4): i+1] / close[max(0, i-5): i] - 1)) if i >= 5 else 0.01
            rv_long  = float(np.std(close[max(0, i-32): i+1] / close[max(0, i-33): i] - 1)) if i >= 33 else 0.01
            vol_ratio = rv_short / (rv_long + 1e-9)

            denom        = (high[i] - low[i]) if (high[i] - low[i]) > 0 else 1e-9
            buy_pressure = (close[i] - low[i]) / denom

            ts = df_15m.index[i]
            if isinstance(ts, pd.Timestamp):
                hour_sin = np.sin(2 * np.pi * ts.hour / 24.0)
                hour_cos = np.cos(2 * np.pi * ts.hour / 24.0)
                dow_sin  = np.sin(2 * np.pi * ts.dayofweek / 7.0)
                dow_cos  = np.cos(2 * np.pi * ts.dayofweek / 7.0)
            else:
                hour_sin = hour_cos = dow_sin = dow_cos = 0.0

            scores_vec  = [p_ctx_bull, p_trend, p_setup, 0.5]
            agent_agree = float(np.std(scores_vec))

            feat = {
                'p_context_bull':   p_ctx_bull,
                'p_context_bear':   p_ctx_bear,
                'p_regime_trend':   p_trend,
                'p_regime_range':   p_range,
                'p_regime_squeeze': p_squeeze,
                'p_setup':          p_setup,
                'p_entry':          0.5,
                'vol_ratio':        float(np.clip(vol_ratio, 0.0, 10.0)),
                'buy_pressure':     float(np.clip(buy_pressure, 0.0, 1.0)),
                'amihud':           0.0,
                'hour_sin':         hour_sin,
                'hour_cos':         hour_cos,
                'dow_sin':          dow_sin,
                'dow_cos':          dow_cos,
                'agent_agreement':  agent_agree,
                'agent_max':        float(max(scores_vec)),
                'agent_min':        float(min(scores_vec)),
            }

            records.append((feat, label, ctx))

        print(f'  [MLOrchestrator] Generated {len(records)} meta-samples '
              f'({sum(1 for _, l, _ in records if l == 1)} positive)')
        return records

    @staticmethod
    def _broadcast_context(context_arr: np.ndarray, target_len: int) -> np.ndarray:
        """
        Broadcast a 1D context array (daily) to target_len (15m bars).
        Simple repeat: each day = 96 15m bars.
        """
        if context_arr is None or len(context_arr) == 0:
            return np.full(target_len, 'neutral', dtype=object)
        n_days = len(context_arr)
        bars_per_day = max(1, target_len // n_days)
        out = np.repeat(context_arr, bars_per_day)
        if len(out) < target_len:
            out = np.concatenate([out, np.full(target_len - len(out), out[-1], dtype=object)])
        return out[:target_len]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'lgb': self._lgb, 'feat_names': self._feat_names}, f, protocol=4)

    def _load(self):
        with open(CACHE_FILE, 'rb') as f:
            p = pickle.load(f)
        self._lgb        = p['lgb']
        self._feat_names = p['feat_names']
        self._trained    = True
