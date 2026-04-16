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


# Defaults match NYXPipeline.__init__ and bear_risk_dial.get_risk_params.
DEFAULT_VOL_MIN = 3.0
DEFAULT_ML_THRESHOLD = 0.60
DEFAULT_COOLDOWN_BARS = 32
DEFAULT_MAX_DAILY_TRADES = 1
DEFAULT_HOUR_WINDOW = (6, 20)       # inclusive: hour >= 6 AND hour <= 20
DEFAULT_BEAR_THRESHOLD = 0.68       # get_risk_params('bear')['ml_threshold']
DEFAULT_BEAR_COOLDOWN_BARS = 64     # get_risk_params('bear')['cooldown_bars']
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
        bear_cooldown_bars: int = DEFAULT_BEAR_COOLDOWN_BARS,
        warmup_bars: int = DEFAULT_WARMUP_BARS,
        mtf_window_size: int = 300,
        use_bear_dial: bool = True,
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
        self._bear_cooldown_bars = int(bear_cooldown_bars)
        self._warmup_bars = int(warmup_bars)
        self._use_bear_dial = bool(use_bear_dial)

        # State
        self.stack = MTFFeatureStack(symbol=symbol, window_size=mtf_window_size)
        self._bars_seen: int = 0
        self._last_trade_bar_idx: int = -10 ** 9
        self._daily_counts: Dict[str, int] = {}
        # Observability: effective threshold + bear-active flag of the
        # most recent on_15m_bar() call.
        self._last_bear_active: bool = False
        self._last_effective_threshold: float = float(ml_threshold)

        # Ticket 07 — MetaGBM ownership wrapper. The trained GBM +
        # scaler + feature_names are encapsulated by MetaGBM, which
        # is now the canonical strategy brain. The live decider
        # delegates per-bar scoring to MetaGBM.decide().
        from src.core.meta_gbm import MetaGBM
        self._meta = MetaGBM(
            threshold=self._ml_threshold,
            model=self.model,
            scaler=self.scaler,
            feature_names=self.feature_names,
        )

    # ------------------------------------------------------------------
    # Gate helpers (exposed for unit testing)
    # ------------------------------------------------------------------
    def _cooldown_blocks_now(self, cooldown: Optional[int] = None) -> bool:
        c = self._cooldown_bars if cooldown is None else int(cooldown)
        return (self._bars_seen - self._last_trade_bar_idx) < c

    def _daily_limit_blocks(self, day_key: str) -> bool:
        return self._daily_counts.get(day_key, 0) >= self._max_daily_trades

    # ------------------------------------------------------------------
    # Feature vector assembly
    # ------------------------------------------------------------------
    def _build_rule_and_extra_features(
        self, direction: int, bar_hour: int
    ) -> Dict[str, float]:
        """Build the 15m rule_* / disagreement / extras features.

        Exact replicas of `NYXPipeline._generate_candidates` inline
        scalars so the GBM sees the same inputs at inference as at
        training time.
        """
        import numpy as np
        from src.ml.jesse_features import _ema, _atr
        from src.ml.soft_gate import compute_disagreement

        df = self.stack.buf_15m._as_dataframe()
        if df.empty or len(df) < 50:
            return {}
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)
        ema50 = _ema(close, 50)
        vol_ma = _ema(volume, 20)
        atr = np.nan_to_num(_atr(high, low, close, 14), nan=0.0)

        i = len(close) - 1
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema50[i])
                or np.isnan(vol_ma[i]) or vol_ma[i] <= 0
                or close[i] <= 0 or ema50[i] == 0 or ema21[i] == 0):
            return {}

        ctx_ratio = (ema9[i] - ema50[i]) / max(abs(ema50[i]), 1e-8)
        rule_context = float(np.clip(ctx_ratio * 20 + 0.5, 0.0, 1.0))
        rule_regime = float(np.clip(atr[i] / close[i] * 200, 0.0, 1.0))
        rule_setup = float(np.clip(
            abs(ema9[i] - ema21[i]) / max(abs(ema21[i]), 1e-8) * 100,
            0.0, 1.0,
        ))
        disagreement = compute_disagreement(direction, {
            'context': rule_context,
            'regime':  rule_regime,
            'setup':   rule_setup,
        })
        volume_spike = float(volume[i] / vol_ma[i])
        atr_pct = float(atr[i] / close[i])
        trend_strength = float(np.clip(
            abs(ema9[i] - ema50[i]) / max(abs(ema50[i]), 1e-8) * 50,
            0.0, 10.0,
        ))

        return {
            'rule_context':   rule_context,
            'rule_regime':    rule_regime,
            'rule_setup':     rule_setup,
            'disagreement':   float(disagreement),
            'volume_spike':   volume_spike,
            'atr_pct':        atr_pct,
            'trend_strength': trend_strength,
            'direction':      float(direction),
            'hour_norm':      float(bar_hour / 24.0),
        }

    def _build_feature_vector(
        self,
        direction: int = 0,
        bar_hour: int = 12,
    ) -> Optional[np.ndarray]:
        """Return scaled 1 x N feature vector or None if any critical
        feature is missing.

        Merges (a) MTFFeatureStack (15m + h1_ + h4_ + d1_ + ctx_1d_*
        + reg_1h_*) with (b) per-bar rule/extra scalars built from the
        15m buffer. Together they cover the full 84-feature vector
        the trained model expects.
        """
        feats = dict(self.stack.latest_features_dict())
        if not feats:
            return None
        feats.update(self._build_rule_and_extra_features(direction, bar_hour))

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
    # Bear-dial conditional helpers
    # ------------------------------------------------------------------
    def _compute_bear_dial_signals(self) -> Optional[Dict[str, Any]]:
        """Per-bar equivalent of `conditional_dial._compute_bar_signals`
        on the CURRENT buffer tails (15m + 1h).

        Returns a dict with scalar values at the tail:
          - regime_1h: 'trending_up' / 'trending_down' / 'ranging'
          - atr_ratio: atr[-1] / ema(atr, 50)[-1] on 15m
          - trend_quality: EMA alignment × spread × 50, clipped to 1

        Returns None when buffers are too short for any of the metrics.
        """
        import numpy as np
        from src.ml.jesse_features import _adx, _atr, _ema

        df_15 = self.stack.buf_15m._as_dataframe()
        df_1h = self.stack.buf_1h._as_dataframe()
        if df_15.empty or len(df_15) < 60 or df_1h.empty or len(df_1h) < 22:
            return None

        close_15 = df_15['close'].values.astype(float)
        high_15 = df_15['high'].values.astype(float)
        low_15 = df_15['low'].values.astype(float)

        # Trend quality — last bar only (tail).
        ema9 = _ema(close_15, 9)
        ema21 = _ema(close_15, 21)
        ema50 = _ema(close_15, 50)
        tq = 0.0
        if not (np.isnan(ema9[-1]) or np.isnan(ema50[-1])) and ema50[-1] != 0:
            spread = abs(ema9[-1] - ema50[-1]) / ema50[-1]
            aligned = (
                (ema9[-1] > ema21[-1] > ema50[-1])
                or (ema9[-1] < ema21[-1] < ema50[-1])
            )
            tq = float(min(1.0, spread * 50)) if aligned else float(spread * 10)

        # ATR ratio (stress indicator) — atr[-1] / ema(atr, 50)[-1].
        atr = np.nan_to_num(_atr(high_15, low_15, close_15, 14), nan=0.0)
        atr_ma = _ema(atr, 50)
        atr_ratio = 0.0
        if not np.isnan(atr_ma[-1]) and atr_ma[-1] > 0:
            atr_ratio = float(atr[-1] / atr_ma[-1])

        # 1h regime string from last 1h bar (ADX + EMA alignment).
        close_1h = df_1h['close'].values.astype(float)
        high_1h = df_1h['high'].values.astype(float)
        low_1h = df_1h['low'].values.astype(float)
        adx_1h = np.nan_to_num(_adx(high_1h, low_1h, close_1h, 14), nan=0.0)
        ema9_1h = _ema(close_1h, 9)
        ema21_1h = _ema(close_1h, 21)
        regime_1h = 'ranging'
        if not (np.isnan(ema9_1h[-1]) or np.isnan(ema21_1h[-1])):
            if adx_1h[-1] > 25 and ema9_1h[-1] > ema21_1h[-1]:
                regime_1h = 'trending_up'
            elif adx_1h[-1] > 25 and ema9_1h[-1] < ema21_1h[-1]:
                regime_1h = 'trending_down'

        return {
            'regime_1h':     regime_1h,
            'atr_ratio':     atr_ratio,
            'trend_quality': tq,
        }

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

        # Reset observability flags (updated below once we reach the
        # bear-dial gate).
        self._last_bear_active = False
        self._last_effective_threshold = self._ml_threshold

        # Hard gate
        direction = self._hard_gate(bar)
        if direction == 0:
            return flat

        # ML score via MetaGBM (Ticket 07) — the canonical strategy
        # brain owns the decision. The underlying trained GBM is
        # unchanged; we route the per-bar scoring through MetaGBM so
        # NYXLiveDecider no longer owns the model-based decision path.
        x = self._build_feature_vector(direction=direction, bar_hour=int(ts.hour))
        if x is None:
            return flat
        base_dec = self._meta.decide(
            fractal_reports={},   # Jesse agents not yet wired in
                                  # live path (future ticket).
            features={},
            asset=self.symbol,
            timestamp=bar['timestamp'],
            timeframe='15m',
            hint_direction=int(direction),
            feature_vector=x,
            already_scaled=True,   # `x` is already scaler.transform()'d
        )
        score = base_dec.probability
        self._last_meta_decision = base_dec

        # Bear-dial conditional: read regime/atr_ratio/trend_quality/
        # disagreement from the CURRENT buffer tail. If triggered, use
        # stricter threshold + longer cooldown. Mirrors NYXPipeline
        # batch logic so live and batch stay in sync.
        threshold = self._ml_threshold
        cooldown = self._cooldown_bars
        if self._use_bear_dial:
            sig = self._compute_bear_dial_signals()
            if sig is not None:
                from src.ml.conditional_dial import should_activate_bear_dial
                feats_for_dis = self._build_rule_and_extra_features(
                    direction=direction, bar_hour=int(ts.hour),
                )
                dis = float(feats_for_dis.get('disagreement', 0.1))
                bear_active = should_activate_bear_dial(
                    regime_1h=str(sig['regime_1h']),
                    vol_ratio=float(sig['atr_ratio']),
                    trend_quality=float(sig['trend_quality']),
                    disagreement=dis,
                )
                self._last_bear_active = bool(bear_active)
                if bear_active:
                    threshold = self._bear_threshold
                    cooldown = self._bear_cooldown_bars

        self._last_effective_threshold = float(threshold)

        # Cooldown (uses effective cooldown — longer when bear-dial
        # active) and daily limits.
        if self._cooldown_blocks_now(cooldown):
            return flat
        if self._daily_limit_blocks(day_key):
            return flat

        if score < threshold:
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
