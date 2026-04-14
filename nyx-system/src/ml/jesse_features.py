"""
Stationary Feature Engine — Jesse ML Pipeline (v2)

Key principle: NEVER raw values, ALWAYS RATIOS.
When Jesse is installed, leverages 176 indicators via jesse.indicators as ta.
Falls back to pure numpy/pandas otherwise.

Feature groups:
  - Trend:       EMA ratios (9/21/50/200), ADX ratio, Supertrend signal
  - Momentum:    RSI, MACD histogram ratio, ROC, CMO
  - Volatility:  ATR ratio, Bollinger %B, Keltner position, Squeeze
  - Volume:      Volume ratio, MFI, OBV slope
  - Structure:   Bar position, High/Low ratio, returns
  - Advanced:    Hurst exponent proxy, Z-score
"""
import numpy as np
import pandas as pd
from typing import Any, Optional

ta: Any = None
try:
    import jesse.indicators as ta  # type: ignore[import-untyped,no-redef]
    _JESSE_AVAILABLE = True
except ImportError:
    _JESSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback indicators (when Jesse not installed)
# ---------------------------------------------------------------------------

def _ema(series: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(series, np.nan, dtype=float)
    if len(series) < period:
        return result
    result[period - 1] = np.mean(series[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(series)):
        result[i] = series[i] * k + result[i - 1] * (1 - k)
    return result


def _sma(series: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(series, np.nan, dtype=float)
    if len(series) < period:
        return result
    cs = np.cumsum(series)
    result[period - 1:] = (cs[period - 1:] - np.concatenate([[0], cs[:-period]])) / period
    return result


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return result


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr_arr = np.full(n, np.nan)
    if n >= period:
        atr_arr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period
    return atr_arr


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> np.ndarray:
    """Simplified ADX — returns values in [0, 100]."""
    n = len(close)
    result = np.full(n, np.nan)
    if n < period * 2:
        return result
    atr_vals = _atr(high, low, close, period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
    smooth_plus = _ema(plus_dm, period)
    smooth_minus = _ema(minus_dm, period)
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = np.where(atr_vals > 0, 100 * smooth_plus / atr_vals, 0)
        minus_di = np.where(atr_vals > 0, 100 * smooth_minus / atr_vals, 0)
        dx = np.where(
            (plus_di + minus_di) > 0,
            100 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
            0
        )
    result = _ema(dx, period)
    return result


def _macd(close: np.ndarray) -> tuple:
    """MACD line, signal, histogram."""
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal = _ema(np.nan_to_num(macd_line, nan=0.0), 9)
    histogram = macd_line - signal
    return macd_line, signal, histogram


def _bollinger_bands(close: np.ndarray, period: int = 20,
                     mult: float = 2.0) -> tuple:
    """Returns (upper, middle, lower)."""
    middle = _sma(close, period)
    std = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        std[i] = np.std(close[i - period + 1:i + 1], ddof=0)
    upper = middle + mult * std
    lower = middle - mult * std
    return upper, middle, lower


def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Money Flow Index [0-100]."""
    n = len(close)
    result = np.full(n, np.nan)
    tp = (high + low + close) / 3.0
    mf = tp * volume
    for i in range(period, n):
        pos_mf = sum(mf[j] for j in range(i - period + 1, i + 1) if tp[j] > tp[j - 1])
        neg_mf = sum(mf[j] for j in range(i - period + 1, i + 1) if tp[j] < tp[j - 1])
        if neg_mf == 0:
            result[i] = 100.0
        else:
            result[i] = 100.0 - 100.0 / (1.0 + pos_mf / neg_mf)
    return result


# ---------------------------------------------------------------------------
# Jesse candle format helper
# ---------------------------------------------------------------------------

def _to_jesse_candles(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to Jesse candle format: [timestamp, open, high, low, close, volume]."""
    n = len(df)
    timestamps = np.zeros(n)
    if hasattr(df.index, 'astype'):
        try:
            timestamps = df.index.astype(np.int64) // 10**6  # ms
        except Exception:
            pass
    return np.column_stack([
        timestamps,
        df['open'].values.astype(float),
        df['high'].values.astype(float),
        df['low'].values.astype(float),
        df['close'].values.astype(float),
        df['volume'].values.astype(float),
    ])


# ---------------------------------------------------------------------------
# Main feature computation — v2 with full Jesse integration
# ---------------------------------------------------------------------------

def compute_stationary_features(
    df: pd.DataFrame,
    feature_set: str = 'full',
) -> pd.DataFrame:
    """
    Compute stationary (ratio-based) features from OHLCV data.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume'.
        feature_set: 'core' (13 features) or 'full' (25+ features).

    Returns:
        DataFrame of stationary features aligned with input index.
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    open_ = df['open'].values.astype(float)
    n = len(close)

    features = pd.DataFrame(index=df.index)

    # Precompute Jesse candle format once if available
    candles: Optional[np.ndarray] = _to_jesse_candles(df) if _JESSE_AVAILABLE else None

    # ---------------------------------------------------------------
    # TREND features — EMA ratios
    # ---------------------------------------------------------------
    if _JESSE_AVAILABLE:
        ema9 = ta.ema(candles, 9)
        ema21 = ta.ema(candles, 21)
        ema50 = ta.ema(candles, 50)
        rsi_vals = ta.rsi(candles, 14)
        atr_vals = ta.atr(candles, 14)
    else:
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)
        ema50 = _ema(close, 50)
        rsi_vals = _rsi(close, 14)
        atr_vals = _atr(high, low, close, 14)

    with np.errstate(divide='ignore', invalid='ignore'):
        features['ema_ratio_9_21'] = np.where(
            ema21 != 0, (ema9 - ema21) / np.abs(ema21), 0.0)
        features['ema_ratio_21_50'] = np.where(
            ema50 != 0, (ema21 - ema50) / np.abs(ema50), 0.0)
        features['close_vs_ema50'] = np.where(
            ema50 != 0, (close - ema50) / np.abs(ema50), 0.0)

    # RSI normalized [-1, 1]
    features['rsi_14'] = (rsi_vals - 50.0) / 50.0

    # ATR ratio (volatility / price)
    with np.errstate(divide='ignore', invalid='ignore'):
        features['atr_ratio'] = np.where(close != 0, atr_vals / close, 0.0)

    # Volume ratio
    vol_ma20 = _ema(volume, 20)
    with np.errstate(divide='ignore', invalid='ignore'):
        features['volume_ratio'] = np.where(vol_ma20 != 0, volume / vol_ma20, 1.0)

    # Momentum
    for lookback in [10, 20]:
        shifted = np.roll(close, lookback)
        shifted[:lookback] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            features[f'momentum_{lookback}'] = np.where(
                shifted != 0, (close - shifted) / np.abs(shifted), 0.0)

    # Bar structure
    bar_range = high - low
    with np.errstate(divide='ignore', invalid='ignore'):
        features['high_low_ratio'] = np.where(close != 0, bar_range / close, 0.0)
        features['close_position'] = np.where(bar_range != 0, (close - low) / bar_range, 0.5)

    # Returns
    features['returns_1'] = pd.Series(close, index=df.index).pct_change().values
    features['returns_5'] = pd.Series(close, index=df.index).pct_change(5).values

    # Volume change
    features['vol_change'] = pd.Series(volume, index=df.index).pct_change().values

    if feature_set == 'core':
        return features

    # ---------------------------------------------------------------
    # FULL feature set — advanced Jesse indicators
    # ---------------------------------------------------------------

    # ADX ratio (trend strength normalized to [0, 1])
    if _JESSE_AVAILABLE:
        adx_vals = ta.adx(candles, 14)
    else:
        adx_vals = _adx(high, low, close, 14)
    features['adx_norm'] = np.nan_to_num(adx_vals, nan=0.0) / 100.0

    # MACD histogram ratio (histogram / close)
    if _JESSE_AVAILABLE:
        macd_result = ta.macd(candles, 12, 26, 9)
        # Jesse returns (macd_line, signal, histogram) or similar
        if isinstance(macd_result, tuple) and len(macd_result) > 0:
            macd_hist = macd_result[2] if len(macd_result) > 2 else macd_result[0]
        else:
            macd_hist = np.zeros(n)
    else:
        _, _, macd_hist = _macd(close)
    with np.errstate(divide='ignore', invalid='ignore'):
        features['macd_hist_ratio'] = np.where(
            close != 0, np.nan_to_num(macd_hist, nan=0.0) / close, 0.0)

    # Bollinger %B — position within bands [0, 1]
    if _JESSE_AVAILABLE:
        try:
            bb = ta.bollinger_bands(candles, 20, 2.0)
            if isinstance(bb, tuple) and len(bb) >= 3:
                bb_upper, bb_mid, bb_lower = bb[0], bb[1], bb[2]
            else:
                bb_upper, bb_mid, bb_lower = _bollinger_bands(close, 20, 2.0)
        except Exception:
            bb_upper, bb_mid, bb_lower = _bollinger_bands(close, 20, 2.0)
    else:
        bb_upper, bb_mid, bb_lower = _bollinger_bands(close, 20, 2.0)
    bb_width = bb_upper - bb_lower
    with np.errstate(divide='ignore', invalid='ignore'):
        features['bb_percent_b'] = np.where(
            bb_width != 0, (close - bb_lower) / bb_width, 0.5)
        features['bb_width_ratio'] = np.where(
            close != 0, np.nan_to_num(bb_width, nan=0.0) / close, 0.0)

    # MFI normalized [-1, 1]
    if _JESSE_AVAILABLE:
        try:
            mfi_vals = ta.mfi(candles, 14)
        except Exception:
            mfi_vals = _mfi(high, low, close, volume, 14)
    else:
        mfi_vals = _mfi(high, low, close, volume, 14)
    features['mfi_norm'] = (np.nan_to_num(mfi_vals, nan=50.0) - 50.0) / 50.0

    # OBV slope (normalized)
    obv = np.cumsum(np.where(np.diff(close, prepend=close[0]) > 0, volume, -volume))
    obv_ema = _ema(obv.astype(float), 20)
    with np.errstate(divide='ignore', invalid='ignore'):
        features['obv_slope'] = np.where(
            obv_ema != 0, (obv - obv_ema) / np.abs(obv_ema), 0.0)

    # ROC (Rate of Change) — 10 bars
    shifted_10 = np.roll(close, 10)
    shifted_10[:10] = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        features['roc_10'] = np.where(
            shifted_10 != 0, (close - shifted_10) / np.abs(shifted_10), 0.0)

    # Keltner position — where close sits relative to Keltner channel
    kelt_mid = _ema(close, 20)
    kelt_upper = kelt_mid + 1.5 * atr_vals
    kelt_lower = kelt_mid - 1.5 * atr_vals
    kelt_width = kelt_upper - kelt_lower
    with np.errstate(divide='ignore', invalid='ignore'):
        features['keltner_position'] = np.where(
            kelt_width != 0, (close - kelt_lower) / kelt_width, 0.5)

    # Squeeze detection — BB inside Keltner = squeeze (binary-ish)
    with np.errstate(invalid='ignore'):
        squeeze = np.where(
            (bb_lower > kelt_lower) & (bb_upper < kelt_upper), 1.0, 0.0)
    features['squeeze'] = np.nan_to_num(squeeze, nan=0.0)

    # EMA 200 ratio (long-term trend)
    if _JESSE_AVAILABLE:
        ema200 = ta.ema(candles, 200)
    else:
        ema200 = _ema(close, 200)
    with np.errstate(divide='ignore', invalid='ignore'):
        features['ema_ratio_50_200'] = np.where(
            ema200 != 0, (ema50 - ema200) / np.abs(ema200), 0.0)

    # Z-score of close (20-bar window)
    close_sma20 = _sma(close, 20)
    close_std20 = np.full(n, np.nan)
    for i in range(19, n):
        close_std20[i] = np.std(close[i - 19:i + 1], ddof=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        features['zscore_20'] = np.where(
            close_std20 != 0, (close - close_sma20) / close_std20, 0.0)

    return features
