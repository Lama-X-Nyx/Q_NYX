"""
Jesse 5-Agent Architecture — TDD Sequential Build

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

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train agent on DataFrame."""
        features = self.compute_features(df)
        labels = self.compute_labels(df)
        self._feature_names = list(features.columns)

        valid = np.ones(len(df), dtype=bool)
        valid[:self.warmup_bars] = False
        valid &= ~features.isna().any(axis=1).values

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

    def backtest(self, df: pd.DataFrame, train_ratio: float = 0.75) -> Dict[str, Any]:
        """Standalone backtest: train on first portion, evaluate on rest."""
        split = int(len(df) * train_ratio)
        df_train = df.iloc[:split]
        df_test = df.iloc[split:]
        self.train(df_train)

        results = []
        for i in range(1, len(df_test) + 1):
            result = self.analyze(df_test.iloc[:i])
            results.append(result)

        states = [r.state for r in results]
        scores = [r.score for r in results]
        passed = [r.passed for r in results]

        from collections import Counter
        state_counts = Counter(states)
        total = len(results)

        return {
            'accuracy': np.mean(scores),
            'n_bars': total,
            'pct_bullish': state_counts.get('bullish', 0) / total,
            'pct_bearish': state_counts.get('bearish', 0) / total,
            'pct_neutral': state_counts.get('neutral', 0) / total,
            'pct_passed': sum(passed) / total,
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
    warmup_bars = 60  # EMA50 + RSI14 convergence on daily

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        n = len(close)

        features = pd.DataFrame(index=df.index)

        # Momentum ratios: (close - close[N]) / close[N]
        # Use 5/20/60 for daily (200 needs too much warmup)
        for lookback in [5, 20, 60]:
            shifted = np.roll(close, lookback)
            shifted[:lookback] = np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
                features[f'momentum_{lookback}'] = np.where(
                    shifted != 0, (close - shifted) / np.abs(shifted), 0.0)

        # Realized volatility (20-day)
        returns = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
        rv = pd.Series(returns).rolling(20).std().values
        features['realized_vol'] = rv

        # RSI normalized [-1, 1]
        rsi_vals = _rsi(close, 14)
        features['rsi_14'] = (rsi_vals - 50.0) / 50.0

        # Amihud illiquidity: |return| / volume
        abs_ret = np.abs(returns)
        with np.errstate(divide='ignore', invalid='ignore'):
            amihud = np.where(volume > 0, abs_ret / volume * 1e6, 0.0)
        amihud_ma = _ema(amihud, 20)
        with np.errstate(divide='ignore', invalid='ignore'):
            features['amihud_ratio'] = np.where(
                amihud_ma != 0, amihud / amihud_ma, 1.0)

        # EMA ratios for trend confirmation
        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        with np.errstate(divide='ignore', invalid='ignore'):
            features['ema_ratio_20_50'] = np.where(
                ema50 != 0, (ema20 - ema50) / np.abs(ema50), 0.0)

        # ATR ratio
        atr_vals = _atr(high, low, close, 14)
        with np.errstate(divide='ignore', invalid='ignore'):
            features['atr_ratio'] = np.where(close != 0, atr_vals / close, 0.0)

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
        proba_dict, pred_class = self._predict_last(df)
        p_bull_ml = proba_dict.get(1, 0.0)
        p_bear_ml = proba_dict.get(-1, 0.0)

        # Heuristic signal from features (EMA + momentum)
        features = self.compute_features(df)
        last = features.iloc[-1]
        ema_ratio = last.get('ema_ratio_20_50', 0)
        mom_20 = last.get('momentum_20', 0)
        rsi = last.get('rsi_14', 0)

        # Combine ML + heuristic (0.5/0.5 blend)
        h_bull = float(ema_ratio > 0 and mom_20 > 0)
        h_bear = float(ema_ratio < 0 and mom_20 < 0)

        p_bull = 0.5 * p_bull_ml + 0.5 * h_bull
        p_bear = 0.5 * p_bear_ml + 0.5 * h_bear
        p_neutral = 1.0 - p_bull - p_bear

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
            reason=f"Context {state} (bull={p_bull:.2f}, bear={p_bear:.2f}, ema={ema_ratio:.4f})",
            metadata={'p_bull': p_bull, 'p_bear': p_bear, 'p_neutral': p_neutral},
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
    warmup_bars = 60

    def compute_features(self, df: pd.DataFrame, hsmm_probs: Optional[np.ndarray] = None) -> pd.DataFrame:
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        n = len(close)

        features = pd.DataFrame(index=df.index)

        # Momentum 4/12/48 bars
        for lb in [4, 12, 48]:
            shifted = np.roll(close, lb)
            shifted[:lb] = np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
                features[f'momentum_{lb}'] = np.where(
                    shifted != 0, (close - shifted) / np.abs(shifted), 0.0)

        # Realized vol
        returns = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
        features['rv_12'] = pd.Series(returns).rolling(12).std().values

        # Volume ratio
        vol_ma = _ema(volume, 20)
        with np.errstate(divide='ignore', invalid='ignore'):
            features['volume_ratio'] = np.where(vol_ma != 0, volume / vol_ma, 1.0)

        # Buy pressure
        bar_range = high - low
        with np.errstate(divide='ignore', invalid='ignore'):
            features['buy_pressure'] = np.where(
                bar_range != 0, (close - low) / bar_range, 0.5)

        # ATR ratio
        atr_vals = _atr(high, low, close, 14)
        with np.errstate(divide='ignore', invalid='ignore'):
            features['atr_ratio'] = np.where(close != 0, atr_vals / close, 0.0)

        # RSI
        rsi_vals = _rsi(close, 14)
        features['rsi_14'] = (rsi_vals - 50.0) / 50.0

        # ADX
        adx_vals = _adx(high, low, close, 14)
        features['adx_norm'] = np.nan_to_num(adx_vals, nan=0.0) / 100.0

        # HSMM probs (6 states) — from parquet or zeros
        if hsmm_probs is not None and hsmm_probs.shape[0] == n:
            for i, state in enumerate(['trendp', 'range', 'trendm', 'squeeze', 'dist', 'liq']):
                features[f'hsmm_{state}'] = hsmm_probs[:, i] if hsmm_probs.shape[1] > i else 0.0
        else:
            for state in ['trendp', 'range', 'trendm', 'squeeze', 'dist', 'liq']:
                features[f'hsmm_{state}'] = 0.0

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
        proba_dict, pred_class = self._predict_last(df)
        p_trend = proba_dict.get(1, 0.0)

        # Determine regime from features heuristic
        features = self.compute_features(df, hsmm_probs)
        last = features.iloc[-1]
        adx = last.get('adx_norm', 0)
        mom = last.get('momentum_12', 0)
        rv = last.get('rv_12', 0)

        if adx > 0.3 and mom > 0.005:
            state = 'trend_plus'
            passed = True
        elif adx > 0.3 and mom < -0.005:
            state = 'trend_minus'
            passed = True
        elif rv < 0.002 and adx < 0.2:
            state = 'squeeze'
            passed = False  # squeeze blocks entries
        else:
            state = 'range'
            passed = True

        score = min(1.0, max(0.0, p_trend))

        return AgentResult(
            agent='regime', state=state, score=score, passed=passed,
            reason=f"Regime {state} (adx={adx:.2f}, mom12={mom:.4f})",
            metadata={'p_trend': p_trend, 'dominant_state': state, 'adx': float(adx)},
        )


# ===================================================================
# AGENT 3 — SETUP (15M)
# ===================================================================

class JesseSetupAgent(_BaseJesseAgent):
    """
    Couche 3 — Validation de setup (15M).

    Output: valid_setup / no_setup
    Features: momentum + HSMM 15M + cross-agent scores
    Label: prix > 0.8% en 8 barres → 1, else 0
    """

    agent_name = 'setup'
    warmup_bars = 60

    def compute_features(
        self, df: pd.DataFrame,
        context_score: float = 0.5,
        regime_score: float = 0.5,
    ) -> pd.DataFrame:
        features = compute_stationary_features(df, feature_set='core')
        # Add cross-agent scores
        features['context_score'] = context_score
        features['regime_score'] = regime_score
        features['agent_agreement'] = 1.0 - abs(context_score - regime_score)
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
        proba_dict, pred_class = self._predict_last(df)
        p_setup = proba_dict.get(1, 0.0)

        if p_setup >= 0.55 and context_score >= 0.5 and regime_score >= 0.4:
            state = 'valid_setup'
            passed = True
            score = p_setup
        else:
            state = 'no_setup'
            passed = False
            score = p_setup

        return AgentResult(
            agent='setup', state=state, score=score, passed=passed,
            reason=f"Setup {state} (p={p_setup:.2f}, ctx={context_score:.2f}, reg={regime_score:.2f})",
            metadata={'p_setup': p_setup, 'context_score': context_score, 'regime_score': regime_score},
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
    warmup_bars = 60

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_stationary_features(df, feature_set='core')

    def compute_labels(self, df: pd.DataFrame) -> np.ndarray:
        from src.ml.jesse_labeler import triple_barrier_labels
        return triple_barrier_labels(df, tp_mult=1.5, sl_mult=1.0, max_bars=50)

    def analyze(self, df: pd.DataFrame) -> AgentResult:
        proba_dict, pred_class = self._predict_last(df)
        classes = list(proba_dict.keys())

        p_up = proba_dict.get(1, 0.0)
        p_down = proba_dict.get(-1, 0.0)
        p_neutral = proba_dict.get(0, 0.0)
        margin = 0.20

        if p_up >= 0.45 and p_up > p_down + margin:
            state = 'ready'
            direction = 1
            passed = True
            score = p_up
        elif p_down >= 0.45 and p_down > p_up + margin:
            state = 'ready'
            direction = -1
            passed = True
            score = p_down
        else:
            state = 'not_ready'
            direction = 0
            passed = False
            score = max(p_up, p_down)

        return AgentResult(
            agent='entry', state=state if state == 'not_ready' and not passed else state,
            score=min(1.0, max(0.0, score)), passed=passed,
            reason=f"Entry {state} (up={p_up:.2f}, down={p_down:.2f})",
            metadata={'p_up': p_up, 'p_down': p_down, 'p_neutral': p_neutral, 'direction': direction},
        )

    def backtest(self, df: pd.DataFrame, train_ratio: float = 0.75) -> Dict[str, Any]:
        split = int(len(df) * train_ratio)
        self.train(df.iloc[:split])
        preds, proba, classes = self._predict_batch(df.iloc[split:])

        from src.ml.jesse_labeler import triple_barrier_labels
        true_labels = triple_barrier_labels(df.iloc[split:])
        valid = np.arange(self.warmup_bars, len(preds))
        from sklearn.metrics import accuracy_score
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
