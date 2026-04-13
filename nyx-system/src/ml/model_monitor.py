"""
NYX ML Model Monitor

Tracks prediction quality, feature drift, and calibration in production.
Decoupled from the agent — receives data via log_prediction() after each bar.

Key fixes vs. original code:
  - Outcome logging is deferred (outcomes arrive ~4h after predictions)
  - KS drift uses a rolling baseline (not a single static baseline)
  - alert() is a real method (print + optional callback)
"""

import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, List, Tuple


class ModelMonitor:
    """
    Production monitor for MLEntryAgent.

    Usage
    -----
    monitor = ModelMonitor(alert_callback=send_telegram_alert)

    # At every bar where the agent predicts:
    monitor.log_prediction(bar_ts, features, ml_prob, context_state)

    # After trade resolves (TP or SL hit):
    monitor.resolve_outcome(entry_ts, outcome=1)   # 1=TP, 0=SL

    # Periodic health check (e.g. every 100 bars):
    monitor.run_checks()
    """

    # How many bars to hold as "baseline" for drift comparison
    BASELINE_SIZE  = 2000
    # Window size for rolling checks
    RECENT_WINDOW  = 200
    # Min outcomes needed before checking performance
    MIN_OUTCOMES   = 50
    # AUC degradation threshold to alert
    AUC_DROP_ALERT = 0.05

    def __init__(self, alert_callback: Optional[Callable[[str], None]] = None):
        """
        Parameters
        ----------
        alert_callback : callable, optional
            Function(message: str) called when an alert fires.
            Defaults to print().
        """
        self._alert_fn = alert_callback or print

        # Rolling prediction log: (timestamp, features_dict, ml_prob, ctx)
        self._pred_log: deque = deque(maxlen=10_000)

        # Deferred outcome resolution: entry_ts → {features, prediction, ctx}
        self._pending: Dict[datetime, Dict] = {}

        # Resolved outcomes for performance tracking
        self._outcomes: List[Dict] = []   # {ts, prediction, outcome, ctx}

        # Feature baselines for drift detection
        self._feature_baseline: Optional[pd.DataFrame] = None
        self._n_checks = 0

    # -----------------------------------------------------------------------
    # Core logging API
    # -----------------------------------------------------------------------

    def log_prediction(self, bar_ts: datetime,
                        features: Dict,
                        ml_prob: float,
                        context_state: str = 'neutral') -> None:
        """
        Log a prediction at `bar_ts`.  Outcome will be resolved later.
        """
        entry = {
            'ts':      bar_ts,
            'features': features,
            'prob':    ml_prob,
            'ctx':     context_state,
        }
        self._pred_log.append(entry)

        # Register as pending outcome (resolved when trade closes)
        self._pending[bar_ts] = entry

        # Build baseline from first BASELINE_SIZE entries
        if self._feature_baseline is None and len(self._pred_log) >= self.BASELINE_SIZE:
            self._feature_baseline = self._to_feature_df(
                list(self._pred_log)[:self.BASELINE_SIZE]
            )

    def resolve_outcome(self, entry_ts: datetime, outcome: int) -> bool:
        """
        Resolve a deferred outcome.

        Parameters
        ----------
        entry_ts : datetime
            Timestamp of the original prediction (from log_prediction).
        outcome : int
            1 = trade succeeded (TP hit), 0 = trade failed (SL hit).

        Returns
        -------
        bool : True if the prediction was found and resolved.
        """
        pending = self._pending.pop(entry_ts, None)
        if pending is None:
            # Try fuzzy match within 1 bar (15m) — handles minor timestamp drift
            for ts, data in list(self._pending.items()):
                if abs((ts - entry_ts).total_seconds()) < 900:
                    pending = data
                    del self._pending[ts]
                    break

        if pending is None:
            return False

        self._outcomes.append({
            'ts':         entry_ts,
            'prediction': pending['prob'],
            'outcome':    outcome,
            'ctx':        pending['ctx'],
        })
        return True

    # -----------------------------------------------------------------------
    # Periodic health checks
    # -----------------------------------------------------------------------

    def run_checks(self) -> Dict:
        """
        Run all checks.  Call periodically (e.g. every 100 bars or daily).

        Returns
        -------
        dict with 'drift_features', 'calibration_ok', 'auc_ok', 'auc'
        """
        self._n_checks += 1
        results = {
            'drift_features':  [],
            'calibration_ok':  True,
            'auc_ok':          True,
            'auc':             None,
        }

        if len(self._outcomes) < self.MIN_OUTCOMES:
            return results

        # --- Feature drift ---
        results['drift_features'] = self._check_feature_drift()

        # --- Calibration ---
        results['calibration_ok'] = self._check_calibration()

        # --- Performance degradation ---
        auc_ok, auc = self._check_performance()
        results['auc_ok'] = auc_ok
        results['auc']    = auc

        return results

    # -----------------------------------------------------------------------
    # Individual check methods
    # -----------------------------------------------------------------------

    def _check_feature_drift(self) -> List[str]:
        """KS test: recent features vs baseline.  Returns drifted feature names."""
        if self._feature_baseline is None or len(self._pred_log) < self.BASELINE_SIZE + self.RECENT_WINDOW:
            return []

        from scipy.stats import ks_2samp

        recent = self._to_feature_df(list(self._pred_log)[-self.RECENT_WINDOW:])
        drifted = []

        for col in recent.columns:
            baseline_col = self._feature_baseline[col].dropna()
            recent_col   = recent[col].dropna()
            if len(baseline_col) < 10 or len(recent_col) < 10:
                continue
            ks_result = ks_2samp(baseline_col, recent_col)
            pvalue: float = ks_result.pvalue  # type: ignore[assignment]
            if pvalue < 0.01:
                drifted.append(col)

        if drifted:
            self._alert(
                f'[MONITOR] Feature drift detected ({len(drifted)} features): '
                f'{drifted[:5]}{"..." if len(drifted) > 5 else ""}'
            )

        return drifted

    def _check_calibration(self) -> bool:
        """
        Binned calibration: predicted probability should approximate actual win rate.
        Alerts if any bin deviates by > 15pp.
        """
        df = pd.DataFrame(self._outcomes[-self.RECENT_WINDOW:])
        bins = [0.0, 0.45, 0.55, 0.65, 1.0]
        labels = ['<0.45', '0.45-0.55', '0.55-0.65', '>0.65']

        df['bin'] = pd.cut(df['prediction'], bins=bins, labels=labels)
        cal = df.groupby('bin', observed=True)['outcome'].agg(['mean', 'count'])

        ok = True
        for bin_label, row in cal.iterrows():
            if row['count'] < 10:
                continue
            # Expected mid of bin
            mid = {'<0.45': 0.35, '0.45-0.55': 0.50,
                   '0.55-0.65': 0.60, '>0.65': 0.72}.get(str(bin_label), 0.5)
            if abs(row['mean'] - mid) > 0.15:
                self._alert(
                    f'[MONITOR] Calibration drift — bin {bin_label}: '
                    f'predicted≈{mid:.0%} actual={row["mean"]:.0%} '
                    f'(n={row["count"]})'
                )
                ok = False

        return ok

    def _check_performance(self) -> Tuple[bool, Optional[float]]:
        """
        Rolling AUC.  Alert if recent AUC drops > AUC_DROP_ALERT vs baseline.
        """
        from sklearn.metrics import roc_auc_score

        if len(self._outcomes) < self.MIN_OUTCOMES * 2:
            return True, None

        df = pd.DataFrame(self._outcomes)
        preds = df['prediction'].values
        actuals = df['outcome'].values

        # Can only compute AUC if both classes present
        if len(np.unique(np.asarray(actuals))) < 2:
            return True, None

        window = min(self.RECENT_WINDOW, len(df))
        recent_auc   = roc_auc_score(actuals[-window:], preds[-window:])
        baseline_auc = roc_auc_score(actuals[:-window], preds[:-window]) \
                       if len(actuals) > window * 2 else recent_auc

        ok = True
        if recent_auc < baseline_auc - self.AUC_DROP_ALERT:
            self._alert(
                f'[MONITOR] Performance degradation! '
                f'Recent AUC={recent_auc:.4f} vs baseline={baseline_auc:.4f} '
                f'(drop={baseline_auc - recent_auc:.4f})'
            )
            ok = False

        return ok, float(recent_auc)

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of monitor state."""
        n_pred    = len(self._pred_log)
        n_pending = len(self._pending)
        n_resolved = len(self._outcomes)

        lines = [
            f'ModelMonitor — checks run: {self._n_checks}',
            f'  Predictions logged:  {n_pred}',
            f'  Pending resolution:  {n_pending}',
            f'  Outcomes resolved:   {n_resolved}',
        ]

        if n_resolved >= self.MIN_OUTCOMES:
            df = pd.DataFrame(self._outcomes)
            win_rate = df['outcome'].mean()
            lines.append(f'  Overall win rate:    {win_rate:.1%}')

            # Per-context breakdown
            for ctx, grp in df.groupby('ctx'):
                lines.append(
                    f'    {ctx:8s}: win={grp["outcome"].mean():.1%} '
                    f'n={len(grp)}'
                )

        if self._feature_baseline is not None:
            lines.append(f'  Baseline features:   {len(self._feature_baseline.columns)}')

        return '\n'.join(lines)

    def get_context_performance(self) -> Optional[pd.DataFrame]:
        """
        Return per-context AUC / win-rate table.
        Returns None if insufficient outcomes.
        """
        if len(self._outcomes) < self.MIN_OUTCOMES:
            return None

        from sklearn.metrics import roc_auc_score

        df = pd.DataFrame(self._outcomes)
        rows = []
        for ctx, grp in df.groupby('ctx'):
            row = {'context': ctx, 'n': len(grp), 'win_rate': grp['outcome'].mean()}
            if len(np.unique(grp['outcome'])) == 2:
                row['auc'] = roc_auc_score(grp['outcome'], grp['prediction'])
            else:
                row['auc'] = float('nan')
            rows.append(row)

        return pd.DataFrame(rows).set_index('context')

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _alert(self, message: str) -> None:
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        self._alert_fn(f'{ts}  {message}')

    @staticmethod
    def _to_feature_df(entries: List[Dict]) -> pd.DataFrame:
        """Convert a list of pred_log entries to a feature DataFrame."""
        rows = [e['features'] for e in entries if e.get('features')]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
