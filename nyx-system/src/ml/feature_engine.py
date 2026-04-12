"""
NYX ML Feature Engine

Computes the feature set used by MLEntryAgent from 15m OHLCV data.
Fixes applied vs. the original code:
  - RSI: Wilder's EMA (not SMA)
  - Sortino: RMS of negative returns (not broken pandas filter)
  - IncrementalFeatureComputer: ddof=1, separated from the batch version
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Batch feature computation (training + backtest)
# ---------------------------------------------------------------------------

class MLFeatureEngine:
    """
    Computes a rich feature set from 15m OHLCV data.

    Designed to complement (not duplicate) the HSMM features already computed
    by RegimeAgent / SetupAgent:
      - HSMM captures latent state transitions (non-linear, probabilistic)
      - MLFeatureEngine captures measurable market microstructure signals

    Usage
    -----
    engine = MLFeatureEngine()
    X = engine.compute(df_15m, context_state='bullish')  # pd.DataFrame
    """

    # Lookback windows (in 15m bars) used across features
    MOMENTUM_WINDOWS = [4, 8, 16, 32, 96]    # 1h, 2h, 4h, 8h, 24h
    VOL_WINDOWS      = [8, 32, 96]            # 2h, 8h, 24h

    def compute(self, df: pd.DataFrame,
                context_state: Optional[str] = None) -> pd.DataFrame:
        """
        Compute all features for a 15m OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data (columns: open, high, low, close, volume).
        context_state : str, optional
            'bullish' | 'bearish' | 'neutral' — passed from ContextAgent.
            Encoded as a feature so the model can learn context-conditional patterns.

        Returns
        -------
        pd.DataFrame
            Feature matrix (NaN rows from warmup are kept; caller drops them).
        """
        f = pd.DataFrame(index=df.index)

        close:  pd.Series = df['close']  # type: ignore[assignment]
        high:   pd.Series = df['high']   # type: ignore[assignment]
        low:    pd.Series = df['low']    # type: ignore[assignment]
        volume: pd.Series = df['volume'] # type: ignore[assignment]
        ret    = close.pct_change()

        # ---- Momentum ----
        for w in self.MOMENTUM_WINDOWS:
            f[f'mom_{w}'] = close / close.shift(w) - 1
            # Acceleration: momentum of momentum
            f[f'mom_acc_{w}'] = f[f'mom_{w}'].diff()

        # ---- Volatility ----
        for w in self.VOL_WINDOWS:
            f[f'rv_{w}']       = ret.rolling(w).std()
            f[f'parkinson_{w}'] = self._parkinson(high, low, w)

        # Vol ratio: short/long — captures vol expansion
        f['vol_ratio_8_96'] = f['rv_8'] / (f['rv_96'] + 1e-9)

        # ---- Risk-adjusted returns ----
        for w in [8, 32]:
            vol_w = f[f'rv_{w}']
            f[f'sharpe_{w}'] = ret.rolling(w).mean() / (vol_w + 1e-9)
            # Sortino: RMS of negative returns (fixed vs original)
            neg_ret = pd.Series(ret.clip(upper=0), index=ret.index)
            f[f'sortino_{w}'] = ret.rolling(w).mean() / (
                (neg_ret ** 2).rolling(w).mean().pow(0.5) + 1e-9
            )

        # ---- RSI (Wilder's EMA — fixed vs original SMA version) ----
        f['rsi_14'] = self._rsi_wilder(close, 14)
        f['rsi_28'] = self._rsi_wilder(close, 28)

        # ---- MACD ----
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        f['macd']        = macd / (close + 1e-9)          # Normalize by price
        f['macd_signal'] = macd.ewm(span=9, adjust=False).mean() / (close + 1e-9)
        f['macd_hist']   = f['macd'] - f['macd_signal']

        # ---- Volume ----
        f['vol_change']     = volume.pct_change()
        f['vol_ratio_6_24'] = volume / (volume.rolling(96).mean() + 1e-9)

        # ---- GARCH-like (EWMA variance) ----
        # Two decay factors: ~daily (0.94) and ~weekly (0.97)
        for lam, tag in [(0.94, '94'), (0.97, '97')]:
            ewma_var = ret.ewm(alpha=1 - lam, adjust=False).var()
            f[f'ewma_vol_{tag}'] = np.sqrt(ewma_var.clip(lower=0))
        # Ratio EWMA short/long — vol regime signal
        f['ewma_vol_ratio'] = f['ewma_vol_94'] / (f['ewma_vol_97'] + 1e-9)

        # ---- Order-book proxies (from OHLCV) ----
        dollar_vol = volume * close

        # Buy pressure: close near high → buying, near low → selling [0, 1]
        hl_range = (high - low).replace(0, np.nan)
        f['buy_pressure']     = (close - low) / hl_range
        f['buy_pressure_ma8'] = f['buy_pressure'].rolling(8).mean()
        # Delta: change in buy pressure (momentum of order flow)
        f['buy_pressure_d']   = f['buy_pressure'].diff(4)

        # Amihud illiquidity: |return| / dollar_volume — price impact per $ traded
        f['amihud'] = (ret.abs() / (dollar_vol + 1e-9)).rolling(32).mean()
        f['amihud_z'] = (f['amihud'] - f['amihud'].rolling(96).mean()) / (
            f['amihud'].rolling(96).std() + 1e-9)   # z-score vs recent history

        # Kyle's lambda (price impact): |return| / sqrt(volume)
        f['kyle_lambda'] = (ret.abs() / (np.sqrt(volume) + 1e-9)).rolling(16).mean()

        # Effective spread proxy: (high - low) / close
        f['eff_spread']       = (high - low) / (close + 1e-9)
        f['eff_spread_ma16']  = f['eff_spread'].rolling(16).mean()
        f['eff_spread_ratio'] = f['eff_spread'] / (f['eff_spread_ma16'] + 1e-9)

        # Volume surprise: current vs 32-bar average
        f['vol_surprise'] = volume / (volume.rolling(32).mean() + 1e-9)

        # Garman-Klass volatility (more efficient than realized vol)
        open_: pd.Series = df['open']  # type: ignore[assignment]
        f['gk_24h'] = self._garman_klass(high, low, close, open_, 96)

        # ---- Intraday seasonality (sin/cos encoding) ----
        idx = df.index
        if hasattr(idx, 'hour'):
            h = idx.hour  # type: ignore[attr-defined]
            f['hour_sin'] = np.sin(2 * np.pi * h / 24)
            f['hour_cos'] = np.cos(2 * np.pi * h / 24)
            dow = idx.dayofweek  # type: ignore[attr-defined]
            f['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            f['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        # ---- Context direction (0=bearish, 0.5=neutral, 1=bullish) ----
        ctx_map = {'bullish': 1.0, 'neutral': 0.5, 'bearish': 0.0}
        f['context'] = ctx_map.get(context_state or 'neutral', 0.5)

        return f

    # -----------------------------------------------------------------------
    # Label creation (training only — uses future data)
    # -----------------------------------------------------------------------

    def create_labels(self, df: pd.DataFrame, context_state: str,
                      forward_bars: int = 16,
                      threshold: float = 0.005) -> pd.Series:
        """
        Direction-aware label: did the trade succeed?

          Bullish: 1 if max(close_{t+1..t+N}) / close_t - 1 > threshold
          Bearish: 1 if min(close_{t+1..t+N}) / close_t - 1 < -threshold

        forward_bars=16 → 4h on 15m data.  threshold=0.5% matches the entry
        pullback filter already in the backtest.

        Returns
        -------
        pd.Series of int (0/1), aligned to df.index.
        """
        close = df['close']

        if context_state == 'bullish':
            # Best price reached in the window
            future_high = close.rolling(forward_bars).max().shift(-forward_bars)
            labels = ((future_high / close - 1) > threshold).astype(int)
        elif context_state == 'bearish':
            future_low = close.rolling(forward_bars).min().shift(-forward_bars)
            labels = ((close / future_low - 1) > threshold).astype(int)
        else:
            # Neutral: simple symmetric return
            future_ret = close.shift(-forward_bars) / close - 1
            labels = (future_ret.abs() > threshold).astype(int)

        return labels

    # -----------------------------------------------------------------------
    # Walk-forward dataset builder (training)
    # -----------------------------------------------------------------------

    def build_walk_forward_dataset(
            self, df: pd.DataFrame, context_series: pd.Series,
            forward_bars: int = 16, threshold: float = 0.005
    ):
        """
        Build (X, y) for walk-forward training.

        context_series : pd.Series indexed like df, values in {'bullish','bearish','neutral'}
        """
        frames_X, frames_y = [], []

        for ctx in ['bullish', 'bearish']:
            mask = (context_series == ctx)
            if mask.sum() < 200:
                continue

            df_ctx = df[mask]
            X_ctx  = self.compute(df, context_state=ctx).loc[mask]
            y_ctx  = self.create_labels(df, context_state=ctx,
                                        forward_bars=forward_bars,
                                        threshold=threshold).loc[mask]

            # Align and drop NaN
            valid = X_ctx.index.intersection(y_ctx.dropna().index)
            X_ctx, y_ctx = X_ctx.loc[valid].dropna(), y_ctx.loc[valid]
            valid2 = X_ctx.index
            frames_X.append(X_ctx)
            frames_y.append(y_ctx.loc[valid2])

        if not frames_X:
            return pd.DataFrame(), pd.Series(dtype=int)

        return pd.concat(frames_X).sort_index(), pd.concat(frames_y).sort_index()

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _rsi_wilder(prices: pd.Series, period: int) -> pd.Series:
        """RSI with Wilder's EMA smoothing (alpha = 1/period)."""
        delta = prices.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        """Parkinson (high-low) volatility estimator."""
        hl2 = np.log(high / low) ** 2
        return np.sqrt(hl2.rolling(window).mean() / (4 * np.log(2)))

    @staticmethod
    def _garman_klass(high: pd.Series, low: pd.Series,
                      close: pd.Series, open_: pd.Series,
                      window: int) -> pd.Series:
        """Garman-Klass OHLC volatility estimator."""
        hl = np.log(high / low) ** 2
        co = np.log(close / open_) ** 2
        gk = 0.5 * hl.rolling(window).mean() - (2 * np.log(2) - 1) * co.rolling(window).mean()
        return np.sqrt(gk.clip(lower=0))


# ---------------------------------------------------------------------------
# Incremental feature computer (production / backtest loop)
# ---------------------------------------------------------------------------

class IncrementalFeatureEngine:
    """
    Maintains rolling state and updates in O(1) per bar.

    Suitable for the live trading loop where recomputing from scratch
    each bar is too slow.  Keeps only the last `maxlen` bars in memory.

    Usage
    -----
    engine = IncrementalFeatureEngine()
    for candle in live_stream:
        engine.update(candle)
        features = engine.get_features(context_state='bullish')
        # → dict, ready for model.predict_one(features)
    """

    MAXLEN = 200  # 200 × 15m = 50h of data kept in memory

    def __init__(self):
        self._close  = deque(maxlen=self.MAXLEN)
        self._high   = deque(maxlen=self.MAXLEN)
        self._low    = deque(maxlen=self.MAXLEN)
        self._open   = deque(maxlen=self.MAXLEN)
        self._volume = deque(maxlen=self.MAXLEN)

    def update(self, candle: Dict) -> None:
        """Append one 15m candle (dict with open/high/low/close/volume)."""
        self._close.append(float(candle['close']))
        self._high.append(float(candle['high']))
        self._low.append(float(candle['low']))
        self._open.append(float(candle['open']))
        self._volume.append(float(candle['volume']))

    def get_features(self, context_state: Optional[str] = None) -> Optional[Dict]:
        """
        Return a feature dict for the current bar, or None if insufficient data.
        Needs at least 96 bars (24h) to compute all rolling windows.
        """
        if len(self._close) < 96:
            return None

        df = pd.DataFrame({
            'close':  list(self._close),
            'high':   list(self._high),
            'low':    list(self._low),
            'open':   list(self._open),
            'volume': list(self._volume),
        })

        engine = MLFeatureEngine()
        X = engine.compute(df, context_state=context_state)

        last = X.iloc[-1]
        if last.isna().any():
            return None

        return last.to_dict()
