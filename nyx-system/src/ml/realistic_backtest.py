"""
Realistic Backtester — Fees, Slippage, Sizing, Sharpe

All the things the edge backtest was missing:
  - Fees: 0.04% taker per side (Binance Futures)
  - Slippage: 0.02% per side
  - Position sizing: risk_pct of capital / ATR = qty
  - Max N trades/day
  - Cooldown bars between trades
  - PnL in dollars with compounding
  - Sharpe annualized from daily returns
  - Equity curve with max drawdown
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Set
from src.ml.jesse_features import _ema, _atr


class RealisticBacktester:

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        risk_pct: float = 0.02,
        tp_mult: float = 1.5,
        sl_mult: float = 1.0,
        max_bars: int = 50,
        fee_rate: float = 0.0004,      # 0.04% per side (taker)
        slippage_rate: float = 0.0002,  # 0.02% per side
        vol_min: float = 1.5,
        use_hours: bool = False,
        good_hours: Optional[Set[int]] = None,
        cooldown_bars: int = 4,
        max_daily_trades: int = 3,
    ):
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.max_bars = max_bars
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.vol_min = vol_min
        self.use_hours = use_hours
        self.good_hours = good_hours or {8, 9, 10, 14, 15, 16, 17, 18}
        self.cooldown_bars = cooldown_bars
        self.max_daily_trades = max_daily_trades

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        n = len(close)

        atr = np.nan_to_num(_atr(high, low, close, 14), nan=0.0)
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)
        ema50 = _ema(close, 50)
        vol_ma = _ema(volume, 20)

        # State
        capital = self.initial_capital
        equity_curve = [capital]
        trades: List[Dict] = []
        pos = None
        last_exit_bar = -999
        daily_trade_count: Dict[str, int] = {}
        total_fees = 0.0

        for i in range(60, n):
            # --- Position management ---
            if pos is not None:
                d = pos['d']
                hit_tp = (d == 1 and high[i] >= pos['tp']) or (d == -1 and low[i] <= pos['tp'])
                hit_sl = (d == 1 and low[i] <= pos['sl']) or (d == -1 and high[i] >= pos['sl'])
                expired = (i - pos['bar']) >= self.max_bars

                if hit_tp or hit_sl or expired:
                    # Determine raw exit price
                    if hit_tp:
                        raw_exit = pos['tp']
                        reason = 'TP'
                    elif hit_sl:
                        raw_exit = pos['sl']
                        reason = 'SL'
                    else:
                        raw_exit = close[i]
                        reason = 'TIME'

                    # Apply slippage on exit (worse price)
                    exit_price = raw_exit * (1 - d * self.slippage_rate)

                    # PnL
                    gross_pnl = d * (exit_price - pos['entry_price']) * pos['qty']
                    exit_fee = abs(pos['qty']) * exit_price * self.fee_rate
                    net_pnl = gross_pnl - exit_fee
                    total_fees += pos['entry_fee'] + exit_fee

                    capital += net_pnl
                    if capital < 0:
                        capital = 0

                    trades.append({
                        'entry_time': df.index[pos['bar']],
                        'exit_time': df.index[i],
                        'entry_bar': pos['bar'],
                        'exit_bar': i,
                        'direction': d,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'qty': pos['qty'],
                        'gross_pnl': gross_pnl,
                        'fees': pos['entry_fee'] + exit_fee,
                        'net_pnl': net_pnl,
                        'risk_dollars': pos['risk_dollars'],
                        'capital_at_entry': pos['capital_at_entry'],
                        'reason': reason,
                    })
                    last_exit_bar = i
                    pos = None

            # --- Entry logic ---
            if pos is None and capital > 0:
                # Cooldown check
                if i - last_exit_bar < self.cooldown_bars:
                    equity_curve.append(capital)
                    continue

                # Guard conditions
                if (np.isnan(ema9[i]) or np.isnan(ema50[i]) or atr[i] <= 0
                        or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
                    equity_curve.append(capital)
                    continue

                # Trend alignment
                uptrend = ema9[i] > ema21[i] > ema50[i]
                downtrend = ema9[i] < ema21[i] < ema50[i]
                if not (uptrend or downtrend):
                    equity_curve.append(capital)
                    continue
                direction = 1 if uptrend else -1

                # Volume filter
                if volume[i] / vol_ma[i] < self.vol_min:
                    equity_curve.append(capital)
                    continue

                # Hour filter
                if self.use_hours and hasattr(df.index, 'hour'):
                    if df.index[i].hour not in self.good_hours:
                        equity_curve.append(capital)
                        continue

                # Max daily trades
                day_key = str(df.index[i].date())
                today_count = daily_trade_count.get(day_key, 0)
                if today_count >= self.max_daily_trades:
                    equity_curve.append(capital)
                    continue

                # Position sizing: risk_pct of capital, capped by notional
                risk_dollars = capital * self.risk_pct
                stop_distance = self.sl_mult * atr[i]
                if stop_distance <= 0:
                    equity_curve.append(capital)
                    continue
                qty_from_risk = risk_dollars / stop_distance
                # Cap notional at 1x capital (no leverage beyond 1x)
                max_qty = capital / close[i]
                qty = min(qty_from_risk, max_qty)

                # Apply slippage on entry (worse price)
                entry_price = close[i] * (1 + direction * self.slippage_rate)
                entry_fee = qty * entry_price * self.fee_rate

                # Set TP/SL from actual entry price
                tp = entry_price + direction * self.tp_mult * atr[i]
                sl = entry_price - direction * self.sl_mult * atr[i]

                pos = {
                    'd': direction,
                    'entry_price': entry_price,
                    'bar': i,
                    'qty': qty,
                    'tp': tp,
                    'sl': sl,
                    'entry_fee': entry_fee,
                    'risk_dollars': risk_dollars,
                    'capital_at_entry': capital,
                }

                daily_trade_count[day_key] = today_count + 1

            equity_curve.append(capital)

        # Close any open position at end
        if pos is not None:
            d = pos['d']
            exit_price = close[-1] * (1 - d * self.slippage_rate)
            gross_pnl = d * (exit_price - pos['entry_price']) * pos['qty']
            exit_fee = abs(pos['qty']) * exit_price * self.fee_rate
            net_pnl = gross_pnl - exit_fee
            total_fees += pos['entry_fee'] + exit_fee
            capital += net_pnl
            trades.append({
                'entry_time': df.index[pos['bar']],
                'exit_time': df.index[-1],
                'entry_bar': pos['bar'],
                'exit_bar': len(df) - 1,
                'direction': d,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'qty': pos['qty'],
                'gross_pnl': gross_pnl,
                'fees': pos['entry_fee'] + exit_fee,
                'net_pnl': net_pnl,
                'risk_dollars': pos['risk_dollars'],
                'capital_at_entry': pos['capital_at_entry'],
                'reason': 'EOD',
            })
            equity_curve[-1] = capital

        # --- Metrics ---
        eq = np.array(equity_curve)
        total_pnl = capital - self.initial_capital
        n_trades = len(trades)
        wins = [t for t in trades if t['net_pnl'] > 0]
        losses = [t for t in trades if t['net_pnl'] <= 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0

        # Profit factor
        gross_wins = sum(t['net_pnl'] for t in wins)
        gross_losses = abs(sum(t['net_pnl'] for t in losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        # Max drawdown
        peak = eq[0]
        max_dd = 0.0
        for e in eq:
            peak = max(peak, e)
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe — from daily equity returns
        daily_eq = pd.Series(eq, index=df.index[:len(eq)]).resample('1D').last().dropna()
        daily_returns = daily_eq.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(365))
        else:
            sharpe = 0.0

        # Trades per day
        n_days = (df.index[-1] - df.index[0]).days
        trades_per_day = n_trades / max(n_days, 1)

        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_pnl_dollars': total_pnl,
            'total_return_pct': total_pnl / self.initial_capital,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_pnl_per_trade': total_pnl / n_trades if n_trades > 0 else 0,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'max_drawdown_pct': max_dd,
            'total_fees': total_fees,
            'trades_per_day': trades_per_day,
            'equity_curve': eq.tolist(),
            'trades': trades,
        }
