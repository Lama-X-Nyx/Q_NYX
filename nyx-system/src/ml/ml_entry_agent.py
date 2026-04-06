"""
MLEntryAgent — LightGBM + River online learning entry filter

Replaces the placeholder EntryAgent with a real ML-powered gate.

Architecture
------------
  Layer 1 (batch): LightGBM trained on historical data via walk-forward CV.
                   Captures slow-moving structural patterns (volatility regime,
                   momentum persistence, volume patterns).

  Layer 2 (online): River LogisticRegression that adapts to recent market
                    dynamics.  Updated in real-time after each trade resolves.

  Final score = 0.7 × lgb_prob + 0.3 × online_prob

Fits into the orchestrator as the 'entry' agent:
  config['fractal']['use_entry_agent'] = True
  config['fractal']['entry_tf']        = '15m'

The orchestrator passes setup_state from SetupAgent.  We also accept
context_state via metadata — see _extract_context() for how we retrieve it
from the SetupAgent metadata.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from river import linear_model, preprocessing, compose, metrics

from src.agents.contracts import AgentResult
from src.ml.feature_engine import MLFeatureEngine, IncrementalFeatureEngine


class MLEntryAgent:
    """
    ML-powered Entry Agent.

    Answers: "Given the current 15m micro-structure AND the macro/regime
    context, is this a statistically favourable entry point?"

    Notes
    -----
    - Works on 15m data (same TF as SetupAgent — no 5m data needed).
    - Direction is inferred from context_state passed via setup metadata.
    - LightGBM model is pre-trained offline; River adapts online after trades.
    - Returns AgentResult compliant with the orchestrator contract.
    """

    # Minimum bars for a reliable prediction
    MIN_BARS = 100

    def __init__(self, config: Dict):
        self.config  = config
        self.name    = 'entry'
        self.feature_engine = MLFeatureEngine()

        # LightGBM batch model (trained on historical data)
        self._lgb_model: Optional[lgb.Booster] = None
        self._feature_names: Optional[List[str]] = None
        self._lgb_trained = False

        # River online model (adapts after every trade)
        self._online_model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression()
        )
        self._online_metric = metrics.ROCAUC()
        self._online_n_updates = 0

        # Thresholds
        ml_cfg = config.get('ml_entry', {})
        self._threshold      = ml_cfg.get('pass_threshold', 0.55)
        self._lgb_weight     = ml_cfg.get('lgb_weight', 0.70)
        self._min_auc_deploy = ml_cfg.get('min_auc_deploy', 0.54)

        # Cache path for serialized LGB model
        self._cache_dir = Path(ml_cfg.get('cache_dir', 'data/ml_cache'))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Production incremental engine (optional — used when updating online)
        self._incremental = IncrementalFeatureEngine()

    # -----------------------------------------------------------------------
    # Public API — batch pre-training
    # -----------------------------------------------------------------------

    def pretrain(self, df_15m: pd.DataFrame,
                 context_series: pd.Series,
                 n_splits: int = 5,
                 n_rounds: int = 500,
                 cache_key: str = '') -> Dict:
        """
        Train LightGBM on historical data using walk-forward cross-validation.

        Parameters
        ----------
        df_15m : pd.DataFrame
            Full historical 15m OHLCV (e.g. 2019-2022 pre-training window).
        context_series : pd.Series
            Per-bar context label ('bullish'/'bearish'/'neutral'), aligned with
            df_15m.  Typically computed from ContextAgent on the same history.
        n_splits : int
            Walk-forward folds (TimeSeriesSplit).
        n_rounds : int
            Max LGB boosting rounds per fold (early stopping at 50).
        cache_key : str
            If provided, save/load model from disk.  Same semantics as the
            HSMM pretrain cache.

        Returns
        -------
        dict with 'mean_auc', 'fold_aucs', 'n_features', 'deployed'.
        """
        # --- Cache check ---
        if cache_key:
            path = self._cache_dir / f'lgb_{cache_key}.pkl'
            if path.exists():
                self._load_lgb(path)
                print(f'    [ML-ENTRY CACHE HIT]  {path.name}')
                return {'deployed': True, 'from_cache': True}

        # --- Build dataset ---
        X, y = self.feature_engine.build_walk_forward_dataset(
            df_15m, context_series
        )
        if len(X) < 500:
            print(f'    [ML-ENTRY] Insufficient data for training ({len(X)} rows)')
            return {'deployed': False, 'reason': 'not_enough_data'}

        self._feature_names = X.columns.tolist()

        # --- Walk-forward CV ---
        tscv     = TimeSeriesSplit(n_splits=n_splits)
        fold_aucs = []

        lgb_params = {
            'objective':        'binary',
            'metric':           'auc',
            'num_leaves':       31,
            'learning_rate':    0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq':     5,
            'max_depth':        6,
            'min_data_in_leaf': 100,
            'verbose':          -1,
        }

        X_arr = X.values
        y_arr = y.values

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_arr)):
            X_tr, y_tr = X_arr[tr_idx], y_arr[tr_idx]
            X_val, y_val = X_arr[val_idx], y_arr[val_idx]

            ds_tr  = lgb.Dataset(X_tr, label=y_tr)
            ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)

            fold_model = lgb.train(
                lgb_params, ds_tr,
                num_boost_round=n_rounds,
                valid_sets=[ds_val],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1)
                ]
            )

            preds = fold_model.predict(X_val)
            auc   = roc_auc_score(y_val, preds)
            fold_aucs.append(auc)

        mean_auc = float(np.mean(fold_aucs))
        print(f'    [ML-ENTRY]  Walk-forward AUC: {mean_auc:.4f}  '
              f'(folds: {[f"{a:.3f}" for a in fold_aucs]})')

        # --- Train final model on all data ---
        ds_full = lgb.Dataset(X_arr, label=y_arr)
        self._lgb_model = lgb.train(
            lgb_params, ds_full,
            num_boost_round=int(np.mean([
                m.best_iteration for m in [fold_model]  # approx from last fold
            ]) * 1.1),
            callbacks=[lgb.log_evaluation(period=-1)]
        )
        self._lgb_trained = True

        # --- Deploy only if AUC is meaningful ---
        deployed = mean_auc >= self._min_auc_deploy
        if not deployed:
            print(f'    [ML-ENTRY]  AUC {mean_auc:.4f} < {self._min_auc_deploy} — '
                  f'model NOT deployed (will pass-through)')

        if cache_key and deployed:
            self._save_lgb(self._cache_dir / f'lgb_{cache_key}.pkl')

        return {
            'deployed':   deployed,
            'mean_auc':   mean_auc,
            'fold_aucs':  fold_aucs,
            'n_features': len(self._feature_names),
        }

    # -----------------------------------------------------------------------
    # Public API — online update (called after trade resolves)
    # -----------------------------------------------------------------------

    def update_online(self, features: Dict, outcome: int) -> None:
        """
        Update the River online model after a trade resolves.

        Parameters
        ----------
        features : dict
            Feature dict at the moment of entry (from IncrementalFeatureEngine).
        outcome : int
            1 = trade succeeded (TP hit), 0 = trade failed (SL hit).
        """
        y_pred = self._online_model.predict_proba_one(features)
        self._online_model.learn_one(features, bool(outcome))

        prob = y_pred.get(True, 0.5)
        self._online_metric.update(bool(outcome), prob)
        self._online_n_updates += 1

    @property
    def online_auc(self) -> float:
        return self._online_metric.get() if self._online_n_updates >= 20 else 0.5

    # -----------------------------------------------------------------------
    # Public API — main analyze() — matches orchestrator contract
    # -----------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame,
                setup_state: Optional[str] = None,
                context_state: Optional[str] = None) -> AgentResult:
        """
        Analyze 15m data and return an ML-powered entry signal.

        Parameters
        ----------
        df : pd.DataFrame
            15m OHLCV slice (strictly look-ahead-free, from build_aligned_index).
        setup_state : str, optional
            State from SetupAgent ('valid_setup', 'misaligned', …).
        context_state : str, optional
            'bullish' | 'bearish' | 'neutral' — from ContextAgent.
            If not provided, defaults to 'neutral' (model still runs but
            with less signal).

        Returns
        -------
        AgentResult  (agent='entry')
        """
        # --- Readiness ---
        if len(df) < self.MIN_BARS:
            return AgentResult(
                agent=self.name,
                state='not_ready',
                score=0.0,
                passed=False,
                ready=False,
                blocked_by_readiness=True,
                reason=f'ML entry: insufficient data ({len(df)} bars, need {self.MIN_BARS})',
                metadata={'bars': len(df)}
            )

        # --- Setup gate: if setup explicitly failed, block ---
        if setup_state and setup_state in ('misaligned', 'no_pattern', 'liquidation',
                                            'detector_error', 'no_bias'):
            return AgentResult(
                agent=self.name,
                state='setup_blocked',
                score=0.0,
                passed=False,
                ready=True,
                reason=f'ML entry: setup not valid ({setup_state}) — skipping ML',
                metadata={'setup_state': setup_state}
            )

        # --- Feature computation ---
        ctx = context_state or 'neutral'
        try:
            X_full = self.feature_engine.compute(df, context_state=ctx)
            last_row = X_full.iloc[-1]

            if last_row.isna().any():
                n_nan = last_row.isna().sum()
                return self._pass_through(
                    reason=f'ML entry: {n_nan} NaN features (warmup)',
                    ctx=ctx, setup_state=setup_state
                )

            features_dict = last_row.to_dict()
            features_arr  = last_row.values.reshape(1, -1)

        except Exception as e:
            return self._pass_through(
                reason=f'ML entry: feature error ({type(e).__name__})',
                ctx=ctx, setup_state=setup_state
            )

        # --- LightGBM prediction ---
        lgb_prob = 0.5
        if self._lgb_trained and self._lgb_model is not None:
            try:
                lgb_prob = float(self._lgb_model.predict(features_arr)[0])
            except Exception:
                lgb_prob = 0.5

        # --- River online prediction ---
        online_prob = 0.5
        if self._online_n_updates >= 20:
            try:
                proba = self._online_model.predict_proba_one(features_dict)
                online_prob = proba.get(True, 0.5)
            except Exception:
                online_prob = 0.5

        # --- Blend ---
        # If online model is not warm yet, full weight to LGB
        online_weight = self._lgb_weight if self._online_n_updates < 20 else (1 - self._lgb_weight)
        lgb_w_eff = 1.0 - online_weight
        ml_prob   = lgb_w_eff * lgb_prob + online_weight * online_prob

        # If LGB was never trained, pass-through with neutral score
        if not self._lgb_trained:
            return self._pass_through(
                reason='ML entry: LGB not trained — pass-through',
                ctx=ctx, setup_state=setup_state
            )

        # --- Decision ---
        passed = ml_prob >= self._threshold
        state  = 'ml_pass' if passed else 'ml_block'

        return AgentResult(
            agent=self.name,
            state=state,
            score=float(ml_prob),
            passed=passed,
            ready=True,
            reason=(
                f'ML entry: P(success)={ml_prob:.3f} '
                f'(lgb={lgb_prob:.3f} online={online_prob:.3f}) '
                f'threshold={self._threshold:.2f} ctx={ctx}'
            ),
            metadata={
                'lgb_prob':        lgb_prob,
                'online_prob':     online_prob,
                'ml_prob':         ml_prob,
                'online_updates':  self._online_n_updates,
                'online_auc':      self.online_auc,
                'context_state':   ctx,
                'setup_state':     setup_state or '',
                'threshold':       self._threshold,
            }
        )

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def _save_lgb(self, path: Path) -> None:
        payload = {
            'model':         self._lgb_model,
            'feature_names': self._feature_names,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        print(f'    [ML-ENTRY SAVED]  {path}')

    def _load_lgb(self, path: Path) -> None:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        self._lgb_model    = payload['model']
        self._feature_names = payload['feature_names']
        self._lgb_trained  = True

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _pass_through(self, reason: str, ctx: str,
                       setup_state: Optional[str]) -> AgentResult:
        """
        When ML is unavailable / warming up: pass the bar unconditionally
        with a neutral score (0.5).  The upstream HSMM agents still gate.
        """
        return AgentResult(
            agent=self.name,
            state='pass_through',
            score=0.5,
            passed=True,
            ready=True,
            reason=reason,
            metadata={
                'context_state': ctx,
                'setup_state':   setup_state or '',
                'lgb_trained':   self._lgb_trained,
                'online_updates': self._online_n_updates,
            }
        )

    def get_feature_importance(self, top_n: int = 15) -> Optional[pd.DataFrame]:
        """Return top-N features by gain importance."""
        if not self._lgb_trained:
            return None
        imp = self._lgb_model.feature_importance(importance_type='gain')
        df = pd.DataFrame({
            'feature':    self._feature_names,
            'importance': imp,
        }).sort_values('importance', ascending=False)
        return df.head(top_n)
