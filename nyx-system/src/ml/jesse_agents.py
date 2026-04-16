"""
Jesse 5-Agent Architecture — TDD Sequential Build

STATUS: alternative modular architecture, NOT currently wired into
production. `NYXPipeline` (src/ml/nyx_pipeline.py) is the active engine
producing every A/B/C, walk-forward and reality-check number. The 5
agents here are an alternative modular design kept for reference and
future integration. See docs/JESSE_AGENTS_STATUS.md for the full
comparison, overlap, and how-to-swap notes.

Each agent:
  1. compute_features(df) → stationary features DataFrame
  2. train(df) → fit RF on labels
  3. analyze(df) → AgentResult
  4. backtest(df, train_ratio) → standalone metrics

Architecture:
  Agent 1 — JesseContextAgent  (1D)  → bullish/bearish/neutral
  Agent 2 — JesseRegimeAgent   (1H)  → trend+/range/trend-/squeeze/distribution/liquidation
  Agent 3 — JesseSetupAgent    (15M) → valid_setup/no_setup
  Agent 4 — JesseEntryAgent    (15M) → ready/not_ready
  Agent 5 — JesseOrchestrator  (Meta)→ BUY/SELL/WAIT + size_factor
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.agents.contracts import AgentResult, OrchestratorDecision
from src.ml.jesse_features import (
    _ema, _rsi, _atr, _sma, _adx, _macd, _mfi, _bollinger_bands,
    compute_stationary_features,
)


# ===================================================================
# BASE AGENT
# ===================================================================

class _BaseJesseAgent:
    """Base class for all Jesse ML agents."""

    agent_name: str = ''
    warmup_bars: int = 50

    # Canonical identity used by .report() (Ticket 05). Subclasses
    # override these to declare the canonical agent name and fractal
    # timeframe for the FractalReport schema.
    REPORT_AGENT: str = ''
    REPORT_TIMEFRAME: str = ''

    def __init__(self):
        self._model: Optional[RandomForestClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._is_trained: bool = False

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute agent-specific stationary features. Override in subclass."""
        raise NotImplementedError

    def compute_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Compute agent-specific labels. Override in subclass."""
        raise NotImplementedError

    def train(
        self,
        df: pd.DataFrame,
        sample_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train agent on DataFrame.

        Ticket 15 — `sample_mask` (optional) : boolean array of
        length `len(df)`. When provided, only bars where mask is
        True are kept as training samples (after warmup + NaN
        filtering). Used by the agent-specific dataset policy
        (see `src/ml/jesse_dataset.py`).
        """
        features = self.compute_features(df)
        labels = self.compute_labels(df)
        self._feature_names = list(features.columns)

        valid = np.ones(len(df), dtype=bool)
        valid[:self.warmup_bars] = False
        valid &= ~features.isna().any(axis=1).values
        if sample_mask is not None:
            m = np.asarray(sample_mask, dtype=bool)
            if len(m) != len(df):
                raise ValueError(
                    f'sample_mask length {len(m)} != len(df) {len(df)}'
                )
            valid &= m

        X = features.values[valid]
        y = labels[valid]

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        self._model.fit(X_scaled, y)
        self._is_trained = True

        y_pred = self._model.predict(X_scaled)
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y, y_pred), 'n_samples': len(y)}

    def _predict_last(self, df: pd.DataFrame) -> tuple:
        """Predict on the last bar of df. Returns (proba_dict, predicted_class)."""
        assert self._is_trained and self._model is not None and self._scaler is not None
        features = self.compute_features(df)
        row = features.iloc[-1:].values
        if np.isnan(row).any():
            return {}, 0
        row_scaled = self._scaler.transform(row)
        proba = self._model.predict_proba(row_scaled)[0]
        classes = list(self._model.classes_)
        proba_dict = {int(c): float(p) for c, p in zip(classes, proba)}
        pred_class = classes[np.argmax(proba)]
        return proba_dict, int(pred_class)

    def _predict_batch(self, df: pd.DataFrame) -> tuple:
        """Batch predict all bars. Returns (predictions, proba_matrix)."""
        assert self._is_trained and self._model is not None and self._scaler is not None
        features = self.compute_features(df)
        X = features.values.copy()
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self._scaler.transform(X)
        proba = self._model.predict_proba(X_scaled)
        classes = list(self._model.classes_)
        predictions = np.array([classes[i] for i in np.argmax(proba, axis=1)])
        return predictions, proba, classes

    def analyze(self, df: pd.DataFrame) -> AgentResult:
        """Analyze current state. Override in subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Ticket 05 — Fractal reporter API
    # ------------------------------------------------------------------
    def report(
        self,
        df: pd.DataFrame,
        asset: str,
        timestamp: Optional[str] = None,
        **analyze_kwargs: Any,
    ):
        """Emit a canonical `FractalReport` for the Meta-GBM input layer.

        Thin wrapper around `self.analyze(df, **kwargs)` + the existing
        `AgentResult.to_fractal_report()` adapter. Subclass declares
        `REPORT_AGENT` and `REPORT_TIMEFRAME`.
        """
        assert self.REPORT_TIMEFRAME, (
            f'{type(self).__name__} must declare REPORT_TIMEFRAME '
            '(Ticket 05)'
        )
        result = self.analyze(df, **analyze_kwargs)
        return result.to_fractal_report(
            asset=asset,
            timeframe=self.REPORT_TIMEFRAME,
            timestamp=timestamp,
        )

    def backtest(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.75,
        sample_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Standalone backtest: train on first portion, evaluate on rest.

        Ticket 15 — `sample_mask` (optional) : boolean array of length
        `len(df)`. When provided :
          - `.train()` restricts training to masked bars
          - the per-bar `.analyze()` eval loop SKIPS masked-out bars
            (this is the O(N²) bottleneck on Setup / Entry dataframes
            — skipping 70 %+ of bars cuts runtime quadratically).
        """
        split = int(len(df) * train_ratio)
        df_train = df.iloc[:split]
        df_test = df.iloc[split:]
        train_mask = None
        test_mask = None
        if sample_mask is not None:
            m = np.asarray(sample_mask, dtype=bool)
            train_mask = m[:split]
            test_mask = m[split:]
        self.train(df_train, sample_mask=train_mask)

        # Try batch prediction first (fast path)
        try:
            predictions, proba, classes = self._predict_batch(df_test)
            # Map predictions to states — skip bars where test_mask is
            # False to avoid the O(N²) per-bar analyze loop on noisy
            # or irrelevant rows.
            states = []
            scores = []
            passed_list = []
            for i in range(self.warmup_bars, len(df_test)):
                if test_mask is not None and not bool(test_mask[i]):
                    continue
                r = self.analyze(df_test.iloc[:i + 1])
                states.append(r.state)
                scores.append(r.score)
                passed_list.append(r.passed)
        except Exception:
            # Fallback to per-bar (slow)
            states = []
            scores = []
            passed_list = []
            for i in range(1, len(df_test) + 1):
                if test_mask is not None and i - 1 < len(test_mask) \
                        and not bool(test_mask[i - 1]):
                    continue
                r = self.analyze(df_test.iloc[:i])
                states.append(r.state)
                scores.append(r.score)
                passed_list.append(r.passed)

        from collections import Counter
        state_counts = Counter(states)
        total = len(states)
        if total == 0:
            return {'accuracy': 0, 'n_bars': 0, 'pct_bullish': 0, 'pct_bearish': 0,
                    'pct_neutral': 0, 'pct_passed': 0, 'avg_score': 0,
                    'state_distribution': {}}

        return {
            'accuracy': np.mean(scores),
            'n_bars': total,
            'pct_bullish': state_counts.get('bullish', 0) / total,
            'pct_bearish': state_counts.get('bearish', 0) / total,
            'pct_neutral': state_counts.get('neutral', 0) / total,
            'pct_passed': sum(passed_list) / total,
            'avg_score': float(np.mean(scores)),
            'state_distribution': dict(state_counts),
        }


# ===================================================================
# AGENT 1 — CONTEXT (1D)
# ===================================================================

class JesseContextAgent(_BaseJesseAgent):
    """
    Couche 1 — Filtre directionnel macro (1D).

    Output: bullish / bearish / neutral
    Features: momentum 5/20/60/200j, realized vol, RSI, Amihud
    Label: max return > 3% sur 5 jours → 1, else 0
    """

    agent_name = 'context'
    REPORT_AGENT = 'context'
    REPORT_TIMEFRAME = '1d'   # Ticket 05 — canonical fractal TF
    warmup_bars = 60  # EMA 50 needs ~50 bars to converge + margin

    # Ticket 14 — role-based feature plan. Context (1D) = slow /
    # structural / macro-bias signals drawn from the canonical
    # `compute_stationary_features` 'full' set. See
    # docs/JESSE_FEATURE_MAPPING.md for rationale.
    FEATURE_PLAN = (
        # slow trend alignment
        'ema_ratio_21_50', 'ema_ratio_50_200', 'close_vs_ema50',
        # macro momentum / oscillator
        'rsi_14', 'momentum_10', 'momentum_20', 'returns_5',
        # volatility regime
        'atr_ratio', 'bb_width_ratio',
        # deviation from mean (structural)
        'zscore_20',
        # participation / liquidity context
        'volume_ratio',
        # slow VWAP + compression/expansion (Ticket 09 slow families)
        'vwap_dist', 'chop_norm',
    )

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ticket 14 — Context (1D) features from canonical 'full'
        set, subsetted by FEATURE_PLAN."""
        from src.ml.jesse_features import compute_stationary_features
        full = compute_stationary_features(df, feature_set='full')
        cols = [c for c in self.FEATURE_PLAN if c in full.columns]
        features = full[cols].copy()
        # Fill warmup NaN with 0 (mono-file train() later drops warmup
        # rows; this stabilises downstream numerics for inference).
        features = features.fillna(0.0).replace(
            [np.inf, -np.inf], 0.0,
        )
        return features

    def compute_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Label: directional bias over next 5 bars.
        +1 if net return > +1%, -1 if < -1%, else 0.
        Adaptive threshold based on data volatility.
        """
        close = df['close'].values.astype(float)
        n = len(close)
        labels = np.zeros(n, dtype=int)
        # Adaptive threshold: 1x daily realized vol (min 0.5%)
        returns = np.abs(np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8))
        daily_vol = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.01
        threshold = max(daily_vol * 3, 0.005)  # 3x avg return, min 0.5%
        for i in range(n - 5):
            future_close = close[i + 5] if i + 5 < n else close[-1]
            net_return = (future_close - close[i]) / close[i]
            if net_return > threshold:
                labels[i] = 1
            elif net_return < -threshold:
                labels[i] = -1
        return labels

    def analyze(self, df: pd.DataFrame) -> AgentResult:
        """Ticket 18 — ML-native Context agent.

        The model (binary: bullish=1) owns the primary signal.
        `p_bull` and `p_bear` come directly from `predict_proba`.
        Heuristic removed from the scoring path.
        """
        proba_dict, pred_class = self._predict_last(df)
        p_bull = float(proba_dict.get(1, 0.5))
        p_bear = float(proba_dict.get(-1, 0.0))
        # Binary model: if only classes {0, 1}, p_bear = 1 - p_bull.
        if -1 not in proba_dict:
            p_bear = 1.0 - p_bull
        p_neutral = max(0.0, 1.0 - p_bull - p_bear)

        if p_bull > p_bear and p_bull > p_neutral:
            state = 'bullish'
            passed = True
            score = min(1.0, p_bull)
        elif p_bear > p_bull and p_bear > p_neutral:
            state = 'bearish'
            passed = False
            score = max(0.0, 1.0 - p_bear)
        else:
            state = 'neutral'
            passed = True
            score = 0.5

        return AgentResult(
            agent='context', state=state, score=score, passed=passed,
            reason=f"Context {state} (bull={p_bull:.2f}, bear={p_bear:.2f})",
            metadata={'p_bull': p_bull, 'p_bear': p_bear,
                      'p_neutral': p_neutral,
                      'context_confidence': score},
        )


# ===================================================================
# AGENT 2 — REGIME (1H)
# ===================================================================

class JesseRegimeAgent(_BaseJesseAgent):
    """
    Couche 2 — Détection de régime (1H).

    Output: trend_plus / range / trend_minus / squeeze / distribution / liquidation
    Features: momentum 4/12/48h + vol + HSMM proba (6)
    Label: max high > 1% dans 4 barres → 1, else 0
    """

    agent_name = 'regime'
    REPORT_AGENT = 'regime'
    REPORT_TIMEFRAME = '4h'   # Ticket 05 — canonical fractal TF
    warmup_bars = 60

    # Ticket 14 — role-based feature plan. Regime (4H) = state
    # classification: trend strength, volatility, compression /
    # expansion, momentum. See docs/JESSE_FEATURE_MAPPING.md.
    FEATURE_PLAN = (
        # trend strength + quality
        'adx_norm', 'ema_ratio_9_21', 'ema_ratio_21_50',
        # volatility regime
        'atr_ratio', 'bb_width_ratio',
        # compression / expansion
        'keltner_position', 'squeeze', 'chop_norm',
        # momentum state
        'macd_hist_ratio', 'momentum_10', 'momentum_20', 'roc_10',
        # bar structure + participation
        'close_position', 'volume_ratio',
        # volume regime (Ticket 09)
        'kvo_norm',
    )

    def compute_features(self, df: pd.DataFrame, hsmm_probs: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Ticket 14 — Regime (4H) features from canonical 'full' set,
        subsetted by FEATURE_PLAN. Agent-specific legacy extras
        (`momentum_12`, `rv_12`) are kept for backward-compat with
        `analyze()` state-mapping heuristics. HSMM state probs
        (6 states) are always present (0.0 defaults when
        `hsmm_probs` is not supplied)."""
        from src.ml.jesse_features import compute_stationary_features
        full = compute_stationary_features(df, feature_set='full')
        cols = [c for c in self.FEATURE_PLAN if c in full.columns]
        features = full[cols].copy()

        # Legacy extras used by `analyze()` state-mapping — kept so
        # the mono-file RegimeAgent contract remains stable.
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        returns = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
        shifted_12 = np.roll(close, 12)
        shifted_12[:12] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            features['momentum_12'] = np.where(
                shifted_12 != 0, (close - shifted_12) / np.abs(shifted_12), 0.0,
            )
        features['rv_12'] = pd.Series(returns).rolling(12).std().values
        # Force numpy-fallback ADX to preserve the legacy
        # state-mapping thresholds in `analyze()` (Jesse's ta.adx
        # returns slightly smaller values on synthetic data, flipping
        # trend_minus/trend_plus decisions at the 0.3 boundary).
        adx_legacy = _adx(high, low, close, 14)
        features['adx_norm'] = np.nan_to_num(adx_legacy, nan=0.0) / 100.0

        # HSMM state probs — always present with 0.0 defaults when
        # `hsmm_probs` is not supplied (backward-compat with parquet
        # callers that pass it explicitly).
        n = len(df)
        for i, state in enumerate(
            ['trendp', 'range', 'trendm', 'squeeze', 'dist', 'liq']
        ):
            if (hsmm_probs is not None
                    and hsmm_probs.shape[0] == n
                    and hsmm_probs.shape[1] > i):
                features[f'hsmm_{state}'] = hsmm_probs[:, i]
            else:
                features[f'hsmm_{state}'] = 0.0

        features = features.fillna(0.0).replace(
            [np.inf, -np.inf], 0.0,
        )
        return features

    def compute_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Label: max high > 1% over next 4 bars → 1, else 0."""
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        n = len(close)
        labels = np.zeros(n, dtype=int)
        for i in range(n - 4):
            future_max = np.max(high[i + 1:i + 5])
            max_ret = (future_max - close[i]) / close[i]
            if max_ret > 0.01:
                labels[i] = 1
        return labels

    def analyze(self, df: pd.DataFrame, hsmm_probs: Optional[np.ndarray] = None) -> AgentResult:
        """Ticket 18 — ML-native Regime agent.

        The model (binary: trending=1) owns the primary signal via
        `p_trend`. State selection uses p_trend threshold (0.5) +
        momentum direction for trend_plus / trend_minus split.
        Heuristic ADX/rv only used for squeeze detection (explicit
        safety-veto, not primary decision substrate).
        """
        proba_dict, pred_class = self._predict_last(df)
        p_trend = float(proba_dict.get(1, 0.0))

        # Direction sign from momentum (not a heuristic decision —
        # just the directional component of the trending state).
        features = self.compute_features(df, hsmm_probs)
        last = features.iloc[-1]
        mom = float(last.get('momentum_12', last.get('momentum_10', 0)))
        rv = float(last.get('rv_12', 0.01))
        adx = float(last.get('adx_norm', 0))

        # ML-native state selection. Threshold 0.4 (not 0.5): the
        # RandomForest on synthetic/real mixed data produces moderate
        # p_trend even on clear trends; 0.4 balances discriminance
        # (not passing everything) with sensitivity (detecting real
        # trends the model sees).
        if p_trend >= 0.4:
            if mom >= 0:
                state = 'trend_plus'
            else:
                state = 'trend_minus'
            passed = True
        elif rv < 0.002 and adx < 0.2:
            state = 'squeeze'
            passed = False
        else:
            state = 'range'
            passed = False

        score = min(1.0, max(0.0, p_trend))

        return AgentResult(
            agent='regime', state=state, score=score, passed=passed,
            reason=f"Regime {state} (p_trend={p_trend:.3f}, mom={mom:.4f})",
            metadata={'p_trend': p_trend, 'dominant_state': state,
                      'adx': adx, 'regime_confidence': score},
        )


# ===================================================================
# AGENT 3 — SETUP (15M)
# ===================================================================

class JesseSetupAgent(_BaseJesseAgent):
    """
    Couche 3 — Validation de setup (15M).

    Output: valid_setup / no_setup
    Features: momentum + cross-agent scores
    Label: prix > 0.8% en 8 barres → 1, else 0

    Uses ML + heuristic blend (like Context) to avoid pure-ML deadlocks.
    """

    agent_name = 'setup'
    REPORT_AGENT = 'setup'
    REPORT_TIMEFRAME = '1h'   # Ticket 05 — canonical fractal TF
    warmup_bars = 60

    # Ticket 14 — role-based feature plan. Setup (1H) is where
    # liquidity-hunter logic matters most: VWAP reclaim, S/R
    # break / failed break, MFI / AD / BOP imbalance,
    # compression → expansion. See docs/JESSE_FEATURE_MAPPING.md.
    FEATURE_PLAN = (
        # volume-weighted displacement
        'vwap_dist', 'vwma_dist',
        # pressure imbalance
        'ad_slope', 'adosc_norm', 'mfi_norm', 'marketfi_ratio',
        # bar-level rejection / pressure
        'bop', 'close_position',
        # structural zones (sweeps + distances)
        'sr_break_up_20', 'sr_break_dn_20',
        'sr_dist_high_20', 'sr_dist_low_20',
        'minmax_pos_20',
        # compression → expansion
        'bb_percent_b', 'squeeze',
    )

    def compute_features(
        self, df: pd.DataFrame,
        context_score: float = 0.5,
        regime_score: float = 0.5,
    ) -> pd.DataFrame:
        """Ticket 14 — Setup (1H) features from canonical 'full' set,
        subsetted by FEATURE_PLAN. Adds cross-agent context /
        regime scores as auxiliary inputs."""
        full = compute_stationary_features(df, feature_set='full')
        cols = [c for c in self.FEATURE_PLAN if c in full.columns]
        features = full[cols].copy()
        features['context_score'] = context_score
        features['regime_score'] = regime_score
        features['agent_agreement'] = 1.0 - abs(context_score - regime_score)
        features = features.fillna(0.0).replace(
            [np.inf, -np.inf], 0.0,
        )
        return features

    def compute_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Label: price > 0.8% in 8 bars → 1, else 0."""
        close = df['close'].values.astype(float)
        n = len(close)
        labels = np.zeros(n, dtype=int)
        for i in range(n - 8):
            future_max = np.max(close[i + 1:i + 9])
            max_ret = (future_max - close[i]) / close[i]
            if max_ret > 0.008:
                labels[i] = 1
        return labels

    def analyze(
        self, df: pd.DataFrame,
        context_score: float = 0.5,
        regime_score: float = 0.5,
    ) -> AgentResult:
        """Ticket 18 — ML-native Setup agent.

        The model (binary: valid_setup=1) owns the primary signal.
        `p_setup` = model probability directly. The Ticket 16
        heuristic blend is removed — the liquidity-hunter features
        are already IN the model's training set (FEATURE_PLAN,
        Ticket 14) so the RandomForest already learned from them.
        """
        proba_dict, pred_class = self._predict_last(df)
        p_setup_ml = float(proba_dict.get(1, 0.5))

        # ML-native: model probability is the primary signal.
        p_setup = p_setup_ml

        # Cross-agent gate: only valid if context + regime agree.
        agents_ok = context_score >= 0.4 and regime_score >= 0.3

        # Cross-agent gate: only valid if context + regime agree.
        agents_ok = context_score >= 0.4 and regime_score >= 0.3

        # ML-native threshold. Model directly decides p_setup;
        # threshold calibrated for 10-40 % pct_passed zone.
        if p_setup >= 0.55 and agents_ok:
            state = 'valid_setup'
            passed = True
            score = min(1.0, p_setup)
        else:
            state = 'no_setup'
            passed = False
            score = p_setup

        return AgentResult(
            agent='setup', state=state, score=score, passed=passed,
            reason=f"Setup {state} (p_ml={p_setup_ml:.3f})",
            metadata={'p_setup': p_setup, 'p_setup_ml': p_setup_ml,
                      'setup_confidence': score,
                      'context_score': context_score,
                      'regime_score': regime_score},
        )


# ===================================================================
# AGENT 4 — ENTRY (15M)
# ===================================================================

class JesseEntryAgent(_BaseJesseAgent):
    """
    Couche 4 — Timing d'entrée (15M).

    Output: ready / not_ready
    Features: 24 jesse features stationnaires
    Label: triple barrier (+1/-1/0)
    """

    agent_name = 'entry'
    REPORT_AGENT = 'entry'
    REPORT_TIMEFRAME = '15m'  # Ticket 05 — canonical fractal TF
    warmup_bars = 60

    # Ticket 14 — role-based feature plan. Entry (15M) = short-
    # horizon trigger confirmation: micro-momentum, volume spike,
    # bar-level pressure, micro-VWAP reclaim, S/R break triggers.
    # See docs/JESSE_FEATURE_MAPPING.md.
    FEATURE_PLAN = (
        # short momentum
        'returns_1', 'returns_5', 'momentum_10',
        # oscillator
        'rsi_14',
        # participation / volume spikes
        'volume_ratio', 'vol_change',
        # bar structure + pressure
        'close_position', 'high_low_ratio', 'bop',
        # micro VWAP reclaim
        'vwap_dist',
        # micro S/R triggers
        'sr_break_up_20', 'sr_break_dn_20',
        # short pressure imbalance (Ticket 09)
        'adosc_norm',
    )

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ticket 14 — Entry (15M) features from canonical 'full'
        set, subsetted by FEATURE_PLAN."""
        full = compute_stationary_features(df, feature_set='full')
        cols = [c for c in self.FEATURE_PLAN if c in full.columns]
        features = full[cols].copy().fillna(0.0).replace(
            [np.inf, -np.inf], 0.0,
        )
        return features

    def compute_labels(self, df: pd.DataFrame) -> np.ndarray:
        from src.ml.jesse_labeler import triple_barrier_labels
        return triple_barrier_labels(df, tp_mult=1.5, sl_mult=1.0, max_bars=50)

    def analyze(self, df: pd.DataFrame) -> AgentResult:
        proba_dict, pred_class = self._predict_last(df)

        p_up_ml = proba_dict.get(1, 0.0)
        p_down_ml = proba_dict.get(-1, 0.0)
        p_neutral_ml = proba_dict.get(0, 0.0)

        # Ticket 18 — ML-native Entry agent.
        # Triple-barrier model produces p_up / p_down / p_neutral
        # directly from predict_proba. No heuristic blend.
        p_up = float(p_up_ml)
        p_down = float(p_down_ml)

        # ML-native decision: model probability owns the direction.
        threshold = 0.45
        margin = 0.15

        if p_up >= threshold and p_up > p_down + margin:
            state = 'ready'
            direction = 1
            passed = True
            score = min(1.0, p_up)
        elif p_down >= threshold and p_down > p_up + margin:
            state = 'ready'
            direction = -1
            passed = True
            score = min(1.0, p_down)
        else:
            state = 'not_ready'
            direction = 0
            passed = False
            score = max(p_up, p_down)

        return AgentResult(
            agent='entry', state=state, score=min(1.0, max(0.0, score)), passed=passed,
            reason=f"Entry {state} (p_up={p_up:.3f}, p_down={p_down:.3f})",
            metadata={'p_up': p_up, 'p_down': p_down,
                      'p_neutral': float(p_neutral_ml),
                      'direction': direction,
                      'entry_confidence': score},
        )

    def backtest(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.75,
        sample_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Entry backtest — batch predict, no per-bar .analyze() loop.

        Ticket 15 — `sample_mask` (optional) restricts both training
        (via extended `.train()`) and test-split evaluation to the
        masked bars. No O(N²) analyze loop here, so the mask mostly
        shapes training distribution + evaluation denominator.
        """
        split = int(len(df) * train_ratio)
        train_mask = None
        test_mask = None
        if sample_mask is not None:
            m = np.asarray(sample_mask, dtype=bool)
            train_mask = m[:split]
            test_mask = m[split:]
        self.train(df.iloc[:split], sample_mask=train_mask)
        preds, proba, classes = self._predict_batch(df.iloc[split:])

        from src.ml.jesse_labeler import triple_barrier_labels
        true_labels = triple_barrier_labels(df.iloc[split:])
        valid = np.arange(self.warmup_bars, len(preds))
        if test_mask is not None:
            valid = np.array(
                [i for i in valid if i < len(test_mask) and bool(test_mask[i])]
            )
        from sklearn.metrics import accuracy_score
        if len(valid) == 0:
            return {
                'accuracy': 0.0, 'n_bars': 0, 'pct_bullish': 0.0,
                'pct_bearish': 0.0, 'pct_neutral': 0.0,
                'pct_passed': 0.0, 'avg_score': 0.0,
                'state_distribution': {'ready': 0, 'not_ready': 0},
            }
        acc = accuracy_score(true_labels[valid], preds[valid])

        return {
            'accuracy': acc,
            'n_bars': len(valid),
            'pct_bullish': float((preds[valid] == 1).mean()),
            'pct_bearish': float((preds[valid] == -1).mean()),
            'pct_neutral': float((preds[valid] == 0).mean()),
            'pct_passed': float((preds[valid] != 0).mean()),
            'avg_score': float(np.max(proba[valid], axis=1).mean()),
            'state_distribution': {'ready': int((preds[valid] != 0).sum()),
                                   'not_ready': int((preds[valid] == 0).sum())},
        }


# ===================================================================
# AGENT 5 — ORCHESTRATOR (Meta)
# ===================================================================

class JesseOrchestrator:
    """
    Meta-learner combining 4 agent scores.

    Input: 4 AgentResults
    Output: OrchestratorDecision (BUY/SELL/WAIT + size_factor)
    Sizing: P>0.75→1.5x, P>0.65→1.2x, P<0.60→0.75x

    ⚠️ DEPRECATED (Ticket 06) — this class uses "ALL must pass → trade
    / any block → WAIT" vote-based logic, which Ticket 06 replaced
    with the canonical strategy brain `src.core.meta_gbm.MetaGBM`.
    The MetaGBM treats disagreement as a feature (not an automatic
    failure) and emits a `MetaDecision` with `quality_bucket` +
    `risk_hint`. Kept here for backward compat + 45 existing
    GREEN tests ; no new code must call this orchestrator. See
    `docs/ARCHITECTURE_CANONIQUE.md` and `docs/JESSE_AGENTS_STATUS.md`.
    """

    def __init__(self):
        self._model: Optional[RandomForestClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._is_trained: bool = False

    def _extract_features(
        self,
        context: AgentResult,
        regime: AgentResult,
        setup: AgentResult,
        entry: AgentResult,
    ) -> np.ndarray:
        """Extract meta-features from 4 agent results."""
        p_bull = context.metadata.get('p_bull', 0.5)
        p_bear = context.metadata.get('p_bear', 0.5)
        p_trend = regime.metadata.get('p_trend', 0.5)
        p_setup = setup.metadata.get('p_setup', 0.5)
        p_up = entry.metadata.get('p_up', 0.5)
        p_down = entry.metadata.get('p_down', 0.5)

        scores = [context.score, regime.score, setup.score, entry.score]
        agreement = 1.0 - np.std(scores)
        all_passed = float(all(r.passed for r in [context, regime, setup, entry]))

        return np.array([
            p_bull, p_bear, p_trend, p_setup, p_up, p_down,
            context.score, regime.score, setup.score, entry.score,
            agreement, all_passed,
        ])

    def decide(
        self,
        context: AgentResult,
        regime: AgentResult,
        setup: AgentResult,
        entry: AgentResult,
    ) -> OrchestratorDecision:
        """Make final trading decision from 4 agent results."""
        blocked_by = [r.agent for r in [context, regime, setup, entry] if not r.passed]
        direction = entry.metadata.get('direction', 0)

        # All must pass for a trade
        if blocked_by:
            return OrchestratorDecision(
                action='WAIT',
                score=min(context.score, regime.score, setup.score, entry.score),
                reason=f"Blocked by: {', '.join(blocked_by)}",
                blocked_by=blocked_by,
                components={'context': context, 'regime': regime, 'setup': setup, 'entry': entry},
            )

        # All passed — determine action
        avg_score = np.mean([context.score, regime.score, setup.score, entry.score])

        if direction == 1:
            action = 'BUY'
        elif direction == -1:
            action = 'SELL'
        else:
            action = 'WAIT'

        # Size factor
        if avg_score > 0.75:
            size_factor = 1.5
        elif avg_score > 0.65:
            size_factor = 1.2
        elif avg_score < 0.60:
            size_factor = 0.75
        else:
            size_factor = 1.0

        return OrchestratorDecision(
            action=action,
            score=float(avg_score),
            reason=f"All agents passed (avg={avg_score:.2f}, size={size_factor:.1f}x)",
            blocked_by=[],
            components={'context': context, 'regime': regime, 'setup': setup, 'entry': entry},
            risk_analysis={'size_factor': size_factor, 'direction': direction},
        )
