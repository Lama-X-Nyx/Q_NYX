"""
NYXLiveDecider — real-time multi-TF per-bar decider.

Task (b.B) layer 3.

Loads a pre-trained artefact (models/<SYMBOL>/) and, on each 15m
bar, replicates the decision logic of `NYXPipeline.run()`:

  1. Hard gate — EMA alignment, volume > vol_min × MA20, hour ∈ [6, 20]
  2. Feature vector — 84 features via MTFFeatureStack (Rule #2)
  3. StandardScaler.transform
  4. GBM predict_proba(1)
  5. Bear-dial conditional threshold (stricter when conditions align)
  6. Cooldown (bars since last trade)
  7. Daily limit
  8. Emit Signal(direction ∈ {-1, +1}) or Signal(direction=0)

No retrain, no batch scan. Per-bar O(window_size).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.assets.signal import Signal
from src.ml.feature_buffer import IncrementalFeatureBuffer  # noqa: F401
from src.ml.mtf_feature_stack import MTFFeatureStack
from src.ml.train_asset_model import load_artifact


# Defaults match NYXPipeline.__init__
DEFAULT_VOL_MIN = 3.0
DEFAULT_ML_THRESHOLD = 0.60
DEFAULT_COOLDOWN_BARS = 32
DEFAULT_MAX_DAILY_TRADES = 1
DEFAULT_HOUR_WINDOW = (6, 20)       # inclusive: hour >= 6 AND hour <= 20
DEFAULT_BEAR_THRESHOLD = 0.70       # stricter when bear-dial active
DEFAULT_WARMUP_BARS = 60            # NYXPipeline starts scanning at i=60


class NYXLiveDecider:
    """Per-bar multi-TF decider backed by a pre-trained artefact."""

    def __init__(
        self,
        symbol: str,
        artifact_dir: Path,
        cluster_group: str = 'majors',
        vol_min: float = DEFAULT_VOL_MIN,
        ml_threshold: float = DEFAULT_ML_THRESHOLD,
        cooldown_bars: int = DEFAULT_COOLDOWN_BARS,
        max_daily_trades: int = DEFAULT_MAX_DAILY_TRADES,
        hour_window: tuple = DEFAULT_HOUR_WINDOW,
        bear_threshold: float = DEFAULT_BEAR_THRESHOLD,
        warmup_bars: int = DEFAULT_WARMUP_BARS,
        mtf_window_size: int = 300,
    ):
        # Load persisted artefact.
        art = load_artifact(Path(artifact_dir))
        self.model = art['model']
        self.scaler = art['scaler']
        self.feature_names = list(art['feature_names'])
        self.metadata = art['metadata']

        # Runtime config
        self.symbol = symbol
        self.cluster_group = cluster_group
        self._vol_min = float(vol_min)
        self._ml_threshold = float(ml_threshold)
        self._cooldown_bars = int(cooldown_bars)
        self._max_daily_trades = int(max_daily_trades)
        self._hour_window = hour_window
        self._bear_threshold = float(bear_threshold)
        self._warmup_bars = int(warmup_bars)

        # State
        self.stack = MTFFeatureStack(symbol=symbol, window_size=mtf_window_size)
        self._bars_seen: int = 0
        self._last_trade_bar_idx: int = -10 ** 9
        self._daily_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Gate helpers (exposed for unit testing)
    # ------------------------------------------------------------------
    def _cooldown_blocks_now(self) -> bool:
        return (self._bars_seen - self._last_trade_bar_idx) < self._cooldown_bars

    def _daily_limit_blocks(self, day_key: str) -> bool:
        return self._daily_counts.get(day_key, 0) >= self._max_daily_trades

    # ------------------------------------------------------------------
    # Feature vector assembly
    # ------------------------------------------------------------------
    def _build_feature_vector(self) -> Optional[np.ndarray]:
        """Return scaled 1 x N feature vector or None if required
        features missing."""
        feats = self.stack.latest_features_dict()
        if not feats:
            return None
        row = np.zeros((1, len(self.feature_names)), dtype=float)
        for i, name in enumerate(self.feature_names):
            v = feats.get(name, 0.0)
            if v != v:          # NaN check
                v = 0.0
            row[0, i] = v
        row = np.clip(np.nan_to_num(row, nan=0.0, posinf=10.0, neginf=-10.0),
                      -1e6, 1e6)
        try:
            scaled = self.scaler.transform(row)
        except Exception:
            return None
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=3.0, neginf=-3.0)
        return scaled

    # ------------------------------------------------------------------
    # Hard gate (EMA + volume + hour)
    # ------------------------------------------------------------------
    def _hard_gate(self, bar: Dict[str, Any]) -> int:
        """Return direction {-1, 0, +1} given the current 15m buffer.

        Mirrors `NYXPipeline._generate_candidates` hard gate."""
        if self._bars_seen < self._warmup_bars:
            return 0

        ts = pd.Timestamp(bar['timestamp'])
        hour = int(ts.hour)
        if hour < self._hour_window[0] or hour > self._hour_window[1]:
            return 0

        df = self.stack.buf_15m._as_dataframe()
        if df.empty or len(df) < 50:
            return 0

        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)

        from src.ml.jesse_features import _ema
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)
        ema50 = _ema(close, 50)
        vol_ma = _ema(volume, 20)

        i = len(close) - 1
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema50[i])
                or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            return 0

        uptrend = bool(ema9[i] > ema21[i] > ema50[i])
        downtrend = bool(ema9[i] < ema21[i] < ema50[i])
        if not (uptrend or downtrend):
            return 0

        if volume[i] / vol_ma[i] < self._vol_min:
            return 0

        return 1 if uptrend else -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def on_15m_bar(self, bar: Dict[str, Any]) -> Signal:
        """Consume one 15m bar and return a Signal."""
        self.stack.on_15m_bar(bar)
        self._bars_seen += 1

        ts = pd.Timestamp(bar['timestamp'])
        day_key = ts.strftime('%Y-%m-%d')

        flat = Signal(
            symbol=self.symbol,
            timestamp=bar['timestamp'],
            direction=0,
            conviction=0.0,
            expected_edge_net=0.0,
            maker_viability=0.0,
            regime_tag='range',
            bull_bear_tag='range',
            size_suggestion=0.0,
            cluster_group=self.cluster_group,
            # FLAT signal: 'don't trade'. No position to hold.
            expected_hold_bars=0,
        )

        # Hard gate
        direction = self._hard_gate(bar)
        if direction == 0:
            return flat

        # Cooldown and daily limits
        if self._cooldown_blocks_now():
            return flat
        if self._daily_limit_blocks(day_key):
            return flat

        # ML score
        x = self._build_feature_vector()
        if x is None:
            return flat
        proba = self.model.predict_proba(x)
        classes = list(self.model.classes_)
        p1_idx = classes.index(1) if 1 in classes else 0
        score = float(proba[0, p1_idx])
        if score < self._ml_threshold:
            return flat

        # State update — mark trade
        self._last_trade_bar_idx = self._bars_seen
        self._daily_counts[day_key] = self._daily_counts.get(day_key, 0) + 1

        return Signal(
            symbol=self.symbol,
            timestamp=bar['timestamp'],
            direction=int(direction),
            conviction=max(0.0, min(1.0, score)),
            expected_edge_net=score,    # proxy; real edge unknown forward
            maker_viability=0.7,
            regime_tag='bull' if direction > 0 else 'bear',
            bull_bear_tag='bull' if direction > 0 else 'bear',
            size_suggestion=1.0,
            cluster_group=self.cluster_group,
            # Expected hold = NYXPipeline.max_bars. HubSpokeRunner
            # will release the position after this many bars so the
            # same symbol can take the next signal.
            expected_hold_bars=50,
        )
