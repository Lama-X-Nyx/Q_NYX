"""
Conditional Bear Dial — Activate ONLY when market signals stress

Not "always adaptive". Bear dial is a CONDITIONAL switch:
  ON  when: regime hostile, vol elevated, trend weak, disagreement high
  OFF when: clear bull trend → full static params for max capture

Triggers (ANY one activates):
  1. regime_1h = trending_down or mixed with weak momentum
  2. vol_ratio > 1.5x (elevated stress)
  3. trend_quality < 0.3 (ambiguous direction)
  4. disagreement > 0.4 (ML vs rules conflict)
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from src.ml.jesse_features import _ema, _atr, _adx
from src.ml.bear_risk_dial import get_risk_params, detect_regime
from src.ml.soft_gate import compute_disagreement


def should_activate_bear_dial(
    regime_1h: str,
    vol_ratio: float,
    trend_quality: float,
    disagreement: float,
) -> bool:
    """
    Conditional activation. Returns True if ANY trigger fires.

    Triggers:
      1. regime_1h is bearish/hostile
      2. vol_ratio > 3.0 (abnormally elevated — not just above average)
      3. trend_quality < 0.3 (weak/ambiguous)
      4. disagreement > 0.4 (ML vs rules conflict)
    """
    # Count triggers — need 2+ for activation (not just 1)
    triggers = 0
    if regime_1h in ('trending_down', 'bear', 'distribution', 'liquidation'):
        triggers += 2  # regime is strong signal, counts double
    if vol_ratio > 1.5:
        triggers += 1
    if trend_quality < 0.3:
        triggers += 1
    if disagreement > 0.4:
        triggers += 1
    return triggers >= 2


def _compute_bar_signals(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Precompute per-bar signals for conditional activation."""
    close_15m = df_15m['close'].values.astype(float)
    high_15m = df_15m['high'].values.astype(float)
    low_15m = df_15m['low'].values.astype(float)
    volume_15m = df_15m['volume'].values.astype(float)
    n = len(close_15m)

    # 15m indicators
    ema9 = _ema(close_15m, 9)
    ema21 = _ema(close_15m, 21)
    ema50 = _ema(close_15m, 50)
    atr = np.nan_to_num(_atr(high_15m, low_15m, close_15m, 14), nan=0)
    vol_ma = _ema(volume_15m, 20)

    # Trend quality: how aligned are EMAs (0=chaos, 1=perfect alignment)
    trend_quality = np.zeros(n)
    for i in range(60, n):
        if np.isnan(ema9[i]) or np.isnan(ema50[i]) or ema50[i] == 0:
            continue
        spread = abs(ema9[i] - ema50[i]) / ema50[i]
        aligned = (ema9[i] > ema21[i] > ema50[i]) or (ema9[i] < ema21[i] < ema50[i])
        trend_quality[i] = min(1.0, spread * 50) if aligned else spread * 10

    # Vol ratio
    vol_ratio = np.ones(n)
    for i in range(20, n):
        if vol_ma[i] > 0:
            vol_ratio[i] = volume_15m[i] / vol_ma[i]

    # ATR ratio (proxy for regime stress)
    atr_ratio = np.zeros(n)
    atr_ma = _ema(atr, 50)
    for i in range(60, n):
        if atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]

    # 1H regime (mapped to 15m via forward fill)
    close_1h = df_1h['close'].values.astype(float)
    high_1h = df_1h['high'].values.astype(float)
    low_1h = df_1h['low'].values.astype(float)
    adx_1h = np.nan_to_num(_adx(high_1h, low_1h, close_1h, 14), nan=0)
    ema9_1h = _ema(close_1h, 9)
    ema21_1h = _ema(close_1h, 21)

    # Map 1H regime to string per hour
    regime_1h_arr = []
    for i in range(len(close_1h)):
        if np.isnan(ema9_1h[i]) or np.isnan(ema21_1h[i]):
            regime_1h_arr.append('ranging')
        elif adx_1h[i] > 25 and ema9_1h[i] > ema21_1h[i]:
            regime_1h_arr.append('trending_up')
        elif adx_1h[i] > 25 and ema9_1h[i] < ema21_1h[i]:
            regime_1h_arr.append('trending_down')
        else:
            regime_1h_arr.append('ranging')

    # Forward fill to 15m
    regime_1h_series = pd.Series(regime_1h_arr, index=df_1h.index)
    regime_15m = regime_1h_series.reindex(df_15m.index, method='ffill').fillna('ranging').values

    return {
        'trend_quality': trend_quality,
        'vol_ratio': vol_ratio,
        'atr_ratio': atr_ratio,
        'regime_1h': regime_15m,
    }


class ConditionalNYXPipeline:
    """
    NYX Pipeline with conditional bear dial.

    Full static params when market is clear.
    Bear dial activates only when conditions are hostile/ambiguous.
    """

    def __init__(self):
        self.fee_rate = 0.0002
        self.slippage_rate = 0.0001

    def run(
        self,
        mtf_data: Dict[str, pd.DataFrame],
        mtf_features: Dict[str, pd.DataFrame],
        train_end: str,
        test_start: str,
        test_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        te = test_end or str(mtf_data['15m'].index[-1].date())

        # Run base pipeline with BULL params (most permissive) to get all trades
        bull_params = get_risk_params('bull')
        from src.ml.nyx_pipeline import NYXPipeline  # lazy import to avoid circular
        base_pipe = NYXPipeline(
            risk_pct=bull_params['risk_pct'],
            ml_threshold=bull_params['ml_threshold'],
            cooldown_bars=bull_params['cooldown_bars'],
            max_daily_trades=bull_params['max_daily_trades'],
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
        )
        base_result = base_pipe.run(mtf_data, mtf_features,
                                     train_end=train_end, test_start=test_start, test_end=te)

        if not base_result['trades']:
            base_result['bear_dial_activation_rate'] = 0.0
            return base_result

        # Precompute bar signals for conditional check
        signals = _compute_bar_signals(
            mtf_data['15m'].loc[test_start:te],
            mtf_data.get('1h', pd.DataFrame()).loc[test_start:te],
        )
        test_15m = mtf_data['15m'].loc[test_start:te]

        # Replay trades with conditional dial
        bear_params = get_risk_params('bear')
        initial_capital = 10_000.0
        capital = initial_capital
        trades_out = []
        n_bear_active = 0

        for t in base_result['trades']:
            ts = t['timestamp']
            # Find bar index in test data
            idx_arr = np.where(test_15m.index >= ts)[0]
            if len(idx_arr) == 0:
                continue
            idx = idx_arr[0]

            # Read signals at this bar
            regime_1h = str(signals['regime_1h'][idx]) if idx < len(signals['regime_1h']) else 'ranging'
            # Use ATR ratio (stress indicator) not volume ratio (already filtered high)
            vol_r = float(signals['atr_ratio'][idx]) if idx < len(signals['atr_ratio']) else 1.0
            tq = float(signals['trend_quality'][idx]) if idx < len(signals['trend_quality']) else 0.5
            dis = t.get('disagreement', 0.1)

            # Conditional check
            bear_active = should_activate_bear_dial(regime_1h, vol_r, tq, dis)

            if bear_active:
                n_bear_active += 1
                # Apply bear threshold filter
                if t.get('ml_score', 0) < bear_params['ml_threshold']:
                    continue
                size_mult = bear_params['size_mult']
            else:
                size_mult = bull_params.get('size_mult', 1.0)

            adjusted_pnl = t['net_pnl'] * size_mult
            capital += adjusted_pnl
            capital = max(capital, 0)

            trades_out.append({
                **t,
                'bear_dial_active': bear_active,
                'size_mult': size_mult,
                'net_pnl': adjusted_pnl,
            })

        # Metrics
        nt = len(trades_out)
        total_pnl = capital - initial_capital
        wins = [t for t in trades_out if t['net_pnl'] > 0]
        losses = [t for t in trades_out if t['net_pnl'] <= 0]
        wr = len(wins) / nt if nt > 0 else 0
        gw = sum(t['net_pnl'] for t in wins)
        gl = abs(sum(t['net_pnl'] for t in losses))

        eq = [initial_capital]
        for t in trades_out:
            eq.append(eq[-1] + t['net_pnl'])
        eq_arr = np.array(eq)
        peak = eq_arr[0]; max_dd = 0
        for e in eq_arr:
            peak = max(peak, e); dd = (peak - e) / peak if peak > 0 else 0; max_dd = max(max_dd, dd)

        if nt > 5:
            rets = [t['net_pnl'] / initial_capital for t in trades_out]
            sharpe = float(np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(min(nt, 252)))
        else:
            sharpe = 0.0

        total_base = len(base_result['trades'])
        activation_rate = n_bear_active / total_base if total_base > 0 else 0

        return {
            'n_trades': nt,
            'n_candidates': base_result['n_candidates'],
            'filter_reject_rate': 1 - nt / max(total_base, 1),
            'win_rate': wr,
            'sharpe': sharpe,
            'total_pnl_dollars': total_pnl,
            'total_return_pct': total_pnl / initial_capital,
            'max_drawdown_pct': max_dd,
            'profit_factor': gw / gl if gl > 0 else float('inf'),
            'total_fees': base_result.get('total_fees', 0),
            'bear_dial_activation_rate': activation_rate,
            'feature_names': base_result.get('feature_names', []),
            'trades': trades_out,
        }
