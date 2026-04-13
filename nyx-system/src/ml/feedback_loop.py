"""
Feedback Loop — Paper Trading → Evaluate → Retrain

Principe : on n'apprend pas dans le trade. On apprend après le trade.

Level 1: DecisionLogger   — log every bar decision (trades + WAITs)
Level 2: OutcomeEvaluator — relabel with real outcomes + error taxonomy
Level 3: ChampionChallenger — retrain + compare + promote only if better
"""
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.ml.jesse_features import compute_stationary_features
from src.ml.jesse_labeler import triple_barrier_labels


# ===================================================================
# LEVEL 1 — DecisionLogger
# ===================================================================

class DecisionLogger:
    """
    Logs every decision (BUY/SELL/WAIT) with full context.

    Each row = 1 bar decision with:
    - Agent outputs (state, score, passed, probabilities)
    - Orchestrator decision (action, score, size_factor)
    - Market context (price, ATR, volume)
    - Trade info if taken (entry, stop, TP)
    - Model version (frozen during session)
    """

    def __init__(self):
        self._rows: List[Dict[str, Any]] = []

    def log_decision(
        self,
        timestamp: pd.Timestamp,
        pair: str,
        features: Dict[str, float],
        agent_results: Dict[str, Dict],
        action: str,
        orch_score: float,
        size_factor: float,
        blocked_by: List[str],
        price: float,
        atr: float,
        volume_ratio: float,
        trade_info: Optional[Dict] = None,
        model_version: str = 'unknown',
    ) -> None:
        """Log a single decision."""
        row: Dict[str, Any] = {
            'timestamp': timestamp,
            'pair': pair,
            'action': action,
            'orch_score': orch_score,
            'size_factor': size_factor,
            'blocked_by': ','.join(blocked_by) if blocked_by else '',
            'price': price,
            'atr': atr,
            'volume_ratio': volume_ratio,
            'model_version': model_version,
            'features_json': json.dumps(features),
        }

        # Agent results
        for agent_name in ['context', 'regime', 'setup', 'entry']:
            ar = agent_results.get(agent_name, {})
            row[f'{agent_name}_state'] = ar.get('state', '')
            row[f'{agent_name}_score'] = ar.get('score', 0.0)
            row[f'{agent_name}_passed'] = ar.get('passed', False)
            if agent_name == 'context':
                row['context_p_bull'] = ar.get('p_bull', 0.0)
                row['context_p_bear'] = ar.get('p_bear', 0.0)
            if agent_name == 'entry':
                row['entry_direction'] = ar.get('direction', 0)
                row['entry_p_up'] = ar.get('p_up', 0.0)
                row['entry_p_down'] = ar.get('p_down', 0.0)

        # Trade info
        if trade_info:
            row['trade_entry_price'] = trade_info.get('entry_price', np.nan)
            row['trade_stop'] = trade_info.get('stop', np.nan)
            row['trade_tp'] = trade_info.get('tp', np.nan)
            row['trade_direction'] = trade_info.get('direction', 0)
            row['trade_size'] = trade_info.get('size', 0.0)
        else:
            row['trade_entry_price'] = np.nan
            row['trade_stop'] = np.nan
            row['trade_tp'] = np.nan
            row['trade_direction'] = 0
            row['trade_size'] = 0.0

        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows)

    def save(self, path: str) -> None:
        df = self.to_dataframe()
        df.to_parquet(path, index=False)

    @classmethod
    def load(cls, path: str) -> 'DecisionLogger':
        logger = cls()
        df = pd.read_parquet(path)
        logger._rows = df.to_dict('records')
        return logger


# ===================================================================
# LEVEL 2 — OutcomeEvaluator
# ===================================================================

class OutcomeEvaluator:
    """
    Takes logged decisions + OHLCV data, computes real outcomes.

    Adds:
    - realized returns (1h, 4h, 24h)
    - triple barrier label (recalculated)
    - decision type (TP/FP/TN/FN)
    - error taxonomy
    - max favorable/adverse excursion
    """

    def evaluate(
        self,
        logger: DecisionLogger,
        ohlcv: pd.DataFrame,
        tp_mult: float = 1.5,
        sl_mult: float = 1.0,
        max_bars: int = 50,
    ) -> pd.DataFrame:
        """Evaluate all logged decisions against actual outcomes."""
        df = logger.to_dataframe()
        if df.empty:
            return df

        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        ohlcv_index = ohlcv.index

        # Pre-compute triple barrier labels for the full OHLCV
        tb_labels = triple_barrier_labels(ohlcv, tp_mult=tp_mult, sl_mult=sl_mult, max_bars=max_bars)

        results = []
        for _, row in df.iterrows():
            ts = pd.Timestamp(row['timestamp'])
            # Find bar index in OHLCV
            idx_arr = np.where(ohlcv_index >= ts)[0]
            if len(idx_arr) == 0:
                results.append(self._empty_outcome())
                continue
            idx = idx_arr[0]

            # Realized returns
            r1h = self._realized_return(close, idx, 4)    # 4 bars = 1h at 15m
            r4h = self._realized_return(close, idx, 16)   # 16 bars = 4h
            r24h = self._realized_return(close, idx, 96)  # 96 bars = 24h

            # Triple barrier label at this bar
            tb_label = int(tb_labels[idx]) if idx < len(tb_labels) else 0

            # Max favorable/adverse excursion
            mfe, mae = self._excursions(close, high, low, idx, max_bars, int(row.get('trade_direction', 0)))

            # Decision classification
            action = row['action']
            is_trade = action in ('BUY', 'SELL')

            if is_trade:
                # Trade was taken
                trade_won = (tb_label == 1 and action == 'BUY') or \
                            (tb_label == -1 and action == 'SELL')
                decision_type = 'TRUE_POSITIVE' if trade_won else 'FALSE_POSITIVE'
            else:
                # WAIT — was it correct?
                move_existed = abs(r4h) > 0.005  # >0.5% move in 4h
                if move_existed:
                    decision_type = 'FALSE_NEGATIVE'  # missed a move
                else:
                    decision_type = 'TRUE_NEGATIVE'  # correct wait

            # Error taxonomy
            error_type = self._classify_error(row, decision_type, tb_label, r4h)

            results.append({
                'realized_return_1h': r1h,
                'realized_return_4h': r4h,
                'realized_return_24h': r24h,
                'triple_barrier_label': tb_label,
                'max_favorable_excursion': mfe,
                'max_adverse_excursion': mae,
                'decision_type': decision_type,
                'error_type': error_type,
            })

        outcome_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), outcome_df], axis=1)

    def summary(self, evaluated_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary report from evaluated decisions."""
        if evaluated_df.empty:
            return {'total_decisions': 0}

        trades = evaluated_df[evaluated_df['action'].isin(['BUY', 'SELL'])]
        waits = evaluated_df[evaluated_df['action'] == 'WAIT']

        from collections import Counter
        dt_counts = Counter(evaluated_df['decision_type'].tolist())
        et_counts = Counter(evaluated_df['error_type'].tolist())

        return {
            'total_decisions': len(evaluated_df),
            'trades_taken': len(trades),
            'waits': len(waits),
            'decision_type_counts': dict(dt_counts),
            'error_type_counts': dict(et_counts),
            'tp_rate': dt_counts.get('TRUE_POSITIVE', 0) / max(len(trades), 1),
            'fp_rate': dt_counts.get('FALSE_POSITIVE', 0) / max(len(trades), 1),
            'tn_rate': dt_counts.get('TRUE_NEGATIVE', 0) / max(len(waits), 1),
            'fn_rate': dt_counts.get('FALSE_NEGATIVE', 0) / max(len(waits), 1),
            'avg_return_1h': float(evaluated_df['realized_return_1h'].mean()),
            'avg_return_4h': float(evaluated_df['realized_return_4h'].mean()),
        }

    def _realized_return(self, close: np.ndarray, idx: int, bars_ahead: int) -> float:
        if idx + bars_ahead >= len(close):
            return 0.0
        return (close[idx + bars_ahead] - close[idx]) / close[idx]

    def _excursions(self, close, high, low, idx, max_bars, direction):
        end = min(idx + max_bars, len(close))
        if direction == 1:
            mfe = (np.max(high[idx:end]) - close[idx]) / close[idx] if end > idx else 0
            mae = (close[idx] - np.min(low[idx:end])) / close[idx] if end > idx else 0
        elif direction == -1:
            mfe = (close[idx] - np.min(low[idx:end])) / close[idx] if end > idx else 0
            mae = (np.max(high[idx:end]) - close[idx]) / close[idx] if end > idx else 0
        else:
            mfe = mae = 0.0
        return mfe, mae

    def _classify_error(self, row, decision_type, tb_label, r4h) -> str:
        if decision_type in ('TRUE_POSITIVE', 'TRUE_NEGATIVE'):
            return 'none'

        action = row['action']

        if decision_type == 'FALSE_POSITIVE':
            # Trade taken but lost — why?
            ctx_state = row.get('context_state', '')
            if ctx_state == 'bearish' and action == 'BUY':
                return 'bad_context'
            regime = row.get('regime_state', '')
            if regime in ('squeeze', 'distribution', 'liquidation'):
                return 'bad_regime'
            setup_score = row.get('setup_score', 0)
            if setup_score < 0.4:
                return 'setup_false_positive'
            entry_score = row.get('entry_score', 0)
            if entry_score < 0.5:
                return 'entry_too_early'
            sf = row.get('size_factor', 1.0)
            if sf > 1.2:
                return 'size_too_large'
            return 'traded_should_have_waited'

        if decision_type == 'FALSE_NEGATIVE':
            # Waited but missed a move — why?
            blocked = row.get('blocked_by', '')
            if 'setup' in blocked:
                return 'setup_false_negative'
            if 'entry' in blocked:
                return 'entry_too_late'
            return 'wait_should_have_traded'

        return 'none'

    def _empty_outcome(self):
        return {
            'realized_return_1h': 0.0, 'realized_return_4h': 0.0,
            'realized_return_24h': 0.0, 'triple_barrier_label': 0,
            'max_favorable_excursion': 0.0, 'max_adverse_excursion': 0.0,
            'decision_type': 'TRUE_NEGATIVE', 'error_type': 'none',
        }


# ===================================================================
# LEVEL 3 — ChampionChallenger
# ===================================================================

class ChampionChallenger:
    """
    Maintains champion (production) and challenger (candidate) models.
    Challenger replaces champion ONLY if strictly better on OOS.
    """

    def __init__(self):
        self.champion: Optional[Dict] = None
        self.challenger: Optional[Dict] = None
        self.champion_metrics: Optional[Dict] = None
        self.challenger_metrics: Optional[Dict] = None

    def train_champion(self, df: pd.DataFrame, train_ratio: float = 0.75) -> Dict:
        """Train the initial champion model."""
        model, metrics = self._train_model(df, train_ratio)
        self.champion = model
        self.champion_metrics = metrics
        return metrics

    def train_challenger(self, df: pd.DataFrame, train_ratio: float = 0.75) -> Dict:
        """Train a challenger model on new/updated data."""
        model, metrics = self._train_model(df, train_ratio)
        self.challenger = model
        self.challenger_metrics = metrics
        return metrics

    def promote_if_better(self, min_improvement: float = 0.02) -> bool:
        """
        Promote challenger to champion if strictly better.

        Args:
            min_improvement: Minimum accuracy improvement required.

        Returns:
            True if promoted, False otherwise.
        """
        if self.champion_metrics is None or self.challenger_metrics is None:
            return False

        champ_acc = self.champion_metrics.get('accuracy', 0)
        chall_acc = self.challenger_metrics.get('accuracy', 0)
        champ_dd = self.champion_metrics.get('max_drawdown', 1.0)
        chall_dd = self.challenger_metrics.get('max_drawdown', 1.0)

        better_accuracy = chall_acc > champ_acc + min_improvement
        not_worse_dd = chall_dd <= champ_dd * 1.1  # max 10% worse drawdown

        if better_accuracy and not_worse_dd:
            self.champion = self.challenger
            self.champion_metrics = self.challenger_metrics
            self.challenger = None
            self.challenger_metrics = None
            return True

        return False

    def comparison_report(self) -> Dict[str, Any]:
        """Generate comparison report between champion and challenger."""
        report = {
            'champion': self.champion_metrics or {},
            'challenger': self.challenger_metrics or {},
        }

        if self.champion_metrics and self.challenger_metrics:
            champ_acc = self.champion_metrics.get('accuracy', 0)
            chall_acc = self.challenger_metrics.get('accuracy', 0)
            diff = chall_acc - champ_acc

            if diff > 0.02:
                report['recommendation'] = 'PROMOTE (challenger is better)'
            elif diff < -0.02:
                report['recommendation'] = 'KEEP_CHAMPION (challenger is worse)'
            else:
                report['recommendation'] = 'KEEP_CHAMPION (no significant difference)'
        else:
            report['recommendation'] = 'INSUFFICIENT_DATA'

        return report

    def _train_model(self, df: pd.DataFrame, train_ratio: float) -> tuple:
        """Train a RF model and return (model_dict, metrics)."""
        features = compute_stationary_features(df, feature_set='core')
        labels = triple_barrier_labels(df)
        warmup = 60

        valid = np.ones(len(df), dtype=bool)
        valid[:warmup] = False
        valid &= ~features.isna().any(axis=1).values
        X = features.values[valid]
        y = labels[valid]
        X = np.nan_to_num(X, nan=0.0)

        split = int(len(X) * train_ratio)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_test, y_pred)

        # Simple max drawdown estimate from predictions
        equity = 10000.0
        peak = equity
        max_dd = 0.0
        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                equity += 10
            else:
                equity -= 10
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

        model_dict = {
            'model': model, 'scaler': scaler,
            'feature_names': list(features.columns),
        }
        metrics = {
            'accuracy': acc,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'max_drawdown': max_dd,
        }

        return model_dict, metrics
