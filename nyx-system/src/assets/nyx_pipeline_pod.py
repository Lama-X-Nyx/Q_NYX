"""
NYXPipelinePod — replay SignalPod wrapping NYXPipeline batch output.

Task (b.A) of the corrective plan
(see `/root/.claude/plans/generic-seeking-bird.md`).

This pod pre-computes all trades by running `NYXPipeline.run()` batch
on a historical window, indexes them by timestamp, and exposes them
bar-by-bar via `on_bar(bar) -> Signal`. It satisfies the `SignalPod`
protocol expected by `HubSpokeRunner`, so A/B/C can be re-validated
through the hub-and-spoke + PostOnlyPaperBroker + PortfolioAllocator
stack without touching NYXPipeline itself.

**Scope**: HISTORICAL REPLAY only.

For live real-time multi-TF inference, see task (b.B) and
`src/ml/nyx_live_decider.py` / `src/assets/nyx_live_pod.py`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.assets.signal import Signal


class NYXPipelinePod:
    """Historical-replay SignalPod for NYXPipeline.

    On construction, runs NYXPipeline.run() batch on [test_start,
    test_end], indexes resulting trades by timestamp. On each
    `on_bar(bar)`, looks up the timestamp and returns an actionable
    `Signal` if a trade was pre-computed at that bar, else FLAT.
    """

    def __init__(
        self,
        symbol: str,
        mtf_data: Dict[str, pd.DataFrame],
        mtf_features: Dict[str, pd.DataFrame],
        train_end: str,
        test_start: str,
        test_end: str,
        cluster_group: str = 'majors',
    ):
        self.symbol = symbol
        self.cluster_group = cluster_group

        # Lazy import to avoid circular imports.
        from src.ml.nyx_pipeline import NYXPipeline
        pipe = NYXPipeline()
        result = pipe.run(
            mtf_data, mtf_features,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        # Index precomputed trades by timestamp (str keys for robust lookup).
        trades: List[Dict[str, Any]] = list(result.get('trades', []))
        self._trade_by_ts: Dict[str, Dict[str, Any]] = {}
        for t in trades:
            ts_key = self._ts_key(t.get('timestamp'))
            if ts_key is None:
                continue
            self._trade_by_ts[ts_key] = t

        self._precomputed_trade_timestamps: List[str] = list(
            self._trade_by_ts.keys()
        )
        self.n_precomputed_trades: int = len(self._trade_by_ts)

    # ------------------------------------------------------------------
    @staticmethod
    def _ts_key(ts: Any) -> Optional[str]:
        """Canonicalise a timestamp-like value to a string key."""
        if ts is None:
            return None
        if isinstance(ts, pd.Timestamp):
            return ts.isoformat()
        try:
            return pd.Timestamp(ts).isoformat()
        except (ValueError, TypeError):
            return str(ts)

    # ------------------------------------------------------------------
    def on_bar(self, bar: Dict[str, Any]) -> Signal:
        """Return an actionable Signal if this bar matches a pre-computed
        trade, FLAT Signal otherwise."""
        ts_key = self._ts_key(bar.get('timestamp'))
        ts_iso = ts_key if ts_key is not None else ''
        trade = self._trade_by_ts.get(ts_key) if ts_key is not None else None

        if trade is None:
            return Signal(
                symbol=self.symbol,
                timestamp=ts_iso,
                direction=0,
                conviction=0.0,
                expected_edge_net=0.0,
                maker_viability=0.0,
                regime_tag='range',
                bull_bear_tag='range',
                size_suggestion=0.0,
                cluster_group=self.cluster_group,
            )

        direction = int(trade.get('direction', 0))
        if direction not in (-1, 0, 1):
            direction = 0

        ml_score = float(trade.get('ml_score', 0.5))
        size_factor = float(trade.get('size_factor', 1.0))
        # NYXPipeline stores the realised net PnL under 'net_pnl' (dollars).
        # A forward decider cannot see the outcome's sign — at emission
        # time it only has the ML proba (ml_score). We therefore use
        # |net_pnl| as a proxy for the MAGNITUDE of expected edge for
        # ranking purposes, and let the downstream broker record the
        # actual outcome after replay.
        edge_net_abs = abs(float(trade.get('net_pnl',
                                             trade.get('outcome_net', 0.0))))

        return Signal(
            symbol=self.symbol,
            timestamp=ts_iso,
            direction=direction,
            conviction=max(0.0, min(1.0, ml_score)),
            expected_edge_net=edge_net_abs,
            maker_viability=0.7,  # default; NYXPipeline uses maker execution
            regime_tag='bull' if direction > 0 else 'bear' if direction < 0 else 'range',
            bull_bear_tag='bull' if direction > 0 else 'bear' if direction < 0 else 'range',
            size_suggestion=max(0.0, size_factor),
            cluster_group=self.cluster_group,
        )
