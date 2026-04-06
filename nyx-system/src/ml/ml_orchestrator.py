"""
MLOrchestrator — Meta-learner that combines all 4 ML agent signals.

Key insight
-----------
A rule-based threshold (score > 0.78) treats all agent combinations
equally. The meta-learner discovers non-linear interactions:
  "Context=0.8 + Regime=0.6 + Setup=0.5 → WR=72%"  (different from)
  "Context=0.6 + Regime=0.8 + Setup=0.8 → WR=61%"

Training labels (bar-level, not trade-level)
--------------------------------------------
For every 15m bar above warmup: did price move >0.8% in the right
direction within the next 8 bars?  This gives ~5-10% positive rate
and 50-100x more samples than actual trade outcomes.

Features
--------
  4 agent probabilities + their statistics (agreement, entropy) +
  microstructure snapshot (vol, spread, buy_pressure) +
  time features (hour, day-of-week) +
  context direction
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

from src.agents.contracts import AgentResult, OrchestratorDecision

CACHE_DIR  = Path('data/pretrain_cache')
CACHE_PATH = CACHE_DIR / 'orchestrator_ml.pkl'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class MLOrchestrator:
    """
    Meta-learner orchestrator.

    Replaces the rule-based score threshold with a LightGBM model that
    learns the optimal combination of all agent signals.

    Usage
    -----
    orch = MLOrchestrator()
    orch.generate_training_data(...)   # build meta-dataset
    orch.pretrain(meta_dataset)        # train LGB

    # In backtest loop:
    decision = orch.decide(ctx, reg, setup, entry, df_15m_row, price)
    """

    MIN_SAMPLES = 200

    def __init__(self, min_score: float = 0.55):
        self.min_score  = min_score
        self.lgb: Optional[lgb.LGBMClassifier] = None
        self._online    = None
        self._trained   = False
        self._feat_cols: List[str] = []

        if _RIVER_OK:
            self._online = compose.Pipeline(
                preprocessing.StandardScaler(),
                linear_model.LogisticRegression()
            )

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_meta_features(self,
                              context_result: Optional[AgentResult],
                              regime_result:  Optional[AgentResult],
                              setup_result:   Optional[AgentResult],
                              entry_result:   Optional[AgentResult],
                              micro: Dict) -> Dict:
        """
        micro : dict with vol_ratio, buy_pressure, amihud, kyle_lambda,
                eff_spread_ratio, ewma_vol_ratio, hour_sin, hour_cos,
                dow_sin, dow_cos  (from MLFeatureEngine last bar)
        """
        ctx_meta = context_result.metadata if context_result else {}
        reg_meta = regime_result.metadata  if regime_result  else {}
        stp_meta = setup_result.metadata   if setup_result   else {}
        ent_meta = entry_result.metadata   if entry_result   else {}

        p_ctx   = ctx_meta.get('p_bull', 0.5)
        p_ctx_b = ctx_meta.get('p_bear', 0.5)
        p_reg   = float(reg_meta.get('p_bull', regime_result.score  if regime_result  else 0.5))
        p_stp   = float(stp_meta.get('p_setup', setup_result.score  if setup_result   else 0.5))
        p_ent   = float(ent_meta.get('ml_prob',  entry_result.score  if entry_result   else 0.5))

        ctx_dir = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0}.get(
            context_result.state if context_result else 'neutral', 0.0)

        # Agent agreement metrics
        probs  = np.array([p_ctx, p_reg, p_stp, p_ent])
        agree  = float(np.std(probs))        # low std = consensus
        agree2 = float(np.min(probs))        # weakest signal
        p_all  = float(np.prod(probs))       # joint probability (harsh)
        p_mean = float(np.mean(probs))

        feat = {
            # Individual agent signals
            'p_context_bull':  p_ctx,
            'p_context_bear':  p_ctx_b,
            'p_regime':        p_reg,
            'p_setup':         p_stp,
            'p_entry':         p_ent,
            # Agent agreement
            'agent_std':       agree,
            'agent_min':       agree2,
            'agent_mean':      p_mean,
            'agent_product':   p_all,
            # Context direction
            'context_dir':     ctx_dir,
            # Regime state one-hot lite
            'is_trending':     float(regime_result.state in ('trend_plus',) if regime_result else False),
            'is_range':        float(regime_result.state == 'range'         if regime_result else False),
            'is_squeeze':      float(regime_result.state == 'squeeze'       if regime_result else False),
            # Setup pattern presence
            'has_smc_pattern': float(stp_meta.get('has_pattern', False)),
            # Agent readiness flags
            'ctx_passed':      float(context_result.passed if context_result else False),
            'reg_passed':      float(regime_result.passed  if regime_result  else False),
            'stp_passed':      float(setup_result.passed   if setup_result   else False),
            'ent_passed':      float(entry_result.passed   if entry_result   else False),
        }

        # Microstructure features
        for k in ['vol_ratio_8_96', 'buy_pressure', 'amihud', 'kyle_lambda',
                  'eff_spread_ratio', 'ewma_vol_ratio',
                  'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']:
            feat[k] = float(micro.get(k, 0.0))

        return feat

    # ------------------------------------------------------------------
    # Training data generation (from precomputed arrays)
    # ------------------------------------------------------------------

    def generate_training_data(self,
                                df_15m: pd.DataFrame,
                                context_arr: np.ndarray,
                                gamma_1h: np.ndarray,
                                gamma_15m: np.ndarray,
                                smc_list: List[Dict],
                                smc_start_iloc: int,
                                df_1d: pd.DataFrame,
                                df_1h: pd.DataFrame,
                                forward_bars: int = 8,
                                thr: float = 0.008) -> List[Tuple[Dict, int, str]]:
        """
        Generate (features_dict, label, direction) tuples for every 15m bar.

        label = 1 if price moved >thr in context direction within forward_bars
        direction = 'bullish' | 'bearish'

        Parameters
        ----------
        context_arr    : (T_1d,) str array from _precompute_context
        gamma_1h       : (T_1h, 6) regime HSMM probs
        gamma_15m      : (T_15m, 6) setup HSMM probs
        smc_list       : precomputed SMC dicts
        smc_start_iloc : iloc offset for smc_list
        df_1d, df_1h   : raw data for alignment
        """
        from src.ml.feature_engine import MLFeatureEngine
        from src.ml.ml_agents import MLContextAgent, MLRegimeAgent, MLSetupAgent

        feat_eng = MLFeatureEngine()
        ctx_agent = MLContextAgent()
        reg_agent = MLRegimeAgent()
        stp_agent = MLSetupAgent()

        # Precompute microstructure features on full 15m dataset
        X_micro = feat_eng.compute(df_15m, context_state='neutral')

        # Alignment: 1D and 1H timestamps → iloc
        idx_1d  = df_1d.index.astype(np.int64)
        idx_1h  = df_1h.index.astype(np.int64)
        idx_15m = df_15m.index.astype(np.int64)

        close_15m = df_15m['close'].values
        high_15m  = df_15m['high'].values

        results = []
        warmup = max(200, smc_start_iloc)

        print(f'  [MLOrch] Generating training data: {len(df_15m) - warmup} bars…')

        for i in range(warmup, len(df_15m) - forward_bars):
            ts_ns = idx_15m[i]

            # 1D context
            i_1d = int(np.searchsorted(idx_1d, ts_ns, side='left')) - 1
            if i_1d < 0 or i_1d >= len(context_arr):
                continue
            ctx_str = str(context_arr[i_1d])
            if ctx_str in ('insufficient', 'neutral'):
                continue

            # Regime (1H)
            i_1h = int(np.searchsorted(idx_1h, ts_ns, side='left')) - 1
            if i_1h < 0 or i_1h >= len(gamma_1h):
                continue
            hsmm_1h = gamma_1h[i_1h]

            # Setup (15M)
            hsmm_15m = gamma_15m[i]

            # SMC
            smc_idx = i - smc_start_iloc
            smc_pat = smc_list[smc_idx] if 0 <= smc_idx < len(smc_list) else {}

            # Build mock agent results (passthrough)
            ctx_result = ctx_agent._passthrough(
                df_1d.iloc[max(0, i_1d - 200):i_1d + 1])
            reg_result = reg_agent._passthrough(hsmm_1h)
            stp_result = stp_agent._passthrough(hsmm_15m, smc_pat, ctx_result)

            # Micro features
            micro_row = X_micro.iloc[i].to_dict() if i < len(X_micro) else {}

            feat = self._build_meta_features(ctx_result, reg_result, stp_result,
                                             None, micro_row)

            # Label: did price move >thr in context direction within forward_bars?
            direction = ctx_str
            future_slice = slice(i + 1, i + 1 + forward_bars)
            if direction == 'bullish':
                fut_high = high_15m[future_slice].max() if len(high_15m[future_slice]) > 0 else close_15m[i]
                label = int((fut_high / close_15m[i] - 1) > thr)
            else:
                from src.ml.ml_agents import HSMM_STATES
                fut_low = df_15m['low'].values[future_slice].min() if i + 1 + forward_bars <= len(df_15m) else close_15m[i]
                label = int((close_15m[i] / (fut_low + 1e-9) - 1) > thr)

            results.append((feat, label, direction))

        pos_rate = sum(r[1] for r in results) / max(len(results), 1)
        print(f'  [MLOrch] Generated {len(results)} samples  positive_rate={pos_rate:.1%}')
        return results

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def pretrain(self, meta_dataset: List[Tuple[Dict, int, str]],
                 n_splits: int = 5) -> Dict:
        """
        meta_dataset : list of (features_dict, label, direction)
        """
        if len(meta_dataset) < self.MIN_SAMPLES:
            print(f'  [MLOrch] Not enough samples ({len(meta_dataset)} < {self.MIN_SAMPLES})')
            return {}

        feat_list   = [x[0] for x in meta_dataset]
        labels      = np.array([x[1] for x in meta_dataset])
        X = pd.DataFrame(feat_list).fillna(0.0)
        self._feat_cols = list(X.columns)

        params = dict(num_leaves=15, learning_rate=0.03, n_estimators=500,
                      min_child_samples=30, verbose=-1, n_jobs=-1,
                      class_weight='balanced')

        tscv   = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for tr, va in tscv.split(X):
            m = lgb.LGBMClassifier(**params)
            m.fit(X.iloc[tr], labels[tr])
            if labels[va].sum() > 0:
                scores.append(roc_auc_score(labels[va], m.predict_proba(X.iloc[va])[:, 1]))

        self.lgb = lgb.LGBMClassifier(**params)
        self.lgb.fit(X, labels)
        self._trained = True

        result = {
            'auc':       float(np.mean(scores)) if scores else 0.0,
            'n_samples': len(meta_dataset),
            'pos_rate':  float(labels.mean()),
        }
        self._save()
        print(f'  [MLOrch]    AUC={result["auc"]:.3f}  n={result["n_samples"]}  '
              f'pos={result["pos_rate"]:.1%}')
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def decide(self,
               context_result: Optional[AgentResult],
               regime_result:  Optional[AgentResult],
               setup_result:   Optional[AgentResult],
               entry_result:   Optional[AgentResult],
               micro: Dict,
               current_price: float) -> OrchestratorDecision:
        """
        micro : microstructure features dict (last bar from MLFeatureEngine)
        """
        components = {k: v for k, v in [
            ('context', context_result), ('regime', regime_result),
            ('setup', setup_result),     ('entry', entry_result),
        ] if v is not None}

        # Readiness gate
        for name, res in components.items():
            if not res.ready:
                return OrchestratorDecision(
                    action='WAIT', score=0.0,
                    reason=f'{name} not ready',
                    blocked_by=['readiness'], components=components
                )

        # Context gate — must have direction
        ctx   = context_result.state if context_result else 'neutral'
        if ctx == 'neutral':
            return OrchestratorDecision(
                action='WAIT', score=0.0, reason='Context neutral',
                blocked_by=['context'], components=components
            )
        action_dir = 'BUY' if ctx == 'bullish' else 'SELL'

        # All agents must pass individually (hard gate)
        blocked = [k for k, v in components.items() if not v.passed]
        if blocked:
            agg = self._rule_score(components)
            return OrchestratorDecision(
                action='WAIT', score=agg, reason=f'Blocked: {blocked}',
                blocked_by=blocked, components=components
            )

        # Meta-learner score
        if self._trained or self._load():
            feat = self._build_meta_features(
                context_result, regime_result, setup_result, entry_result, micro)
            X = pd.DataFrame([feat])[self._feat_cols].fillna(0.0)
            p_profit = float(self.lgb.predict_proba(X)[0, 1])
        else:
            # Fallback: weighted average of agent scores
            p_profit = self._rule_score(components)

        if p_profit < self.min_score:
            return OrchestratorDecision(
                action='WAIT', score=p_profit,
                reason=f'MLOrch: P(profit)={p_profit:.2f} < {self.min_score:.2f}',
                blocked_by=['orchestrator'], components=components
            )

        # Size factor: scale position with conviction
        if p_profit >= 0.75:
            size_factor = 1.5
        elif p_profit >= 0.65:
            size_factor = 1.2
        elif p_profit >= 0.60:
            size_factor = 1.0
        else:
            size_factor = 0.75   # 0.55–0.60 band — smaller size

        return OrchestratorDecision(
            action=action_dir,
            score=p_profit,
            reason=f'MLOrch: P(profit)={p_profit:.2f} ctx={ctx} '
                   f'reg={regime_result.state if regime_result else "?"} '
                   f'size×{size_factor:.2f}',
            blocked_by=[],
            components=components,
            risk_analysis={'ml_p_profit': p_profit, 'size_factor': size_factor}
        )

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update_online(self, features: Dict, outcome: int) -> None:
        if _RIVER_OK and self._online is not None:
            try:
                self._online.learn_one(features, outcome)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_score(components: Dict) -> float:
        w = {'context': 0.30, 'regime': 0.30, 'setup': 0.25, 'entry': 0.15}
        return sum(components[k].score * w.get(k, 0) for k in components if k in w)

    def _save(self) -> None:
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump({'lgb': self.lgb, 'feat_cols': self._feat_cols}, f)

    def _load(self) -> bool:
        if not CACHE_PATH.exists():
            return False
        try:
            p = pickle.load(open(CACHE_PATH, 'rb'))
            self.lgb        = p['lgb']
            self._feat_cols = p['feat_cols']
            self._trained   = True
            return True
        except Exception:
            return False
