"""
Soft Gate Architecture — ML decides, rules validate

ML = moteur principal (edge decision)
Non-ML = structure, audit, garde-fou

3 niveaux:
  1. Soft validation: rule scores → features pour meta-ML
  2. Size/risk adjustment: disagreement → reduce size
  3. Hard veto: RARE (data invalide, spread extrême)
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from src.ml.jesse_features import _ema, _atr, _rsi, _adx, _sma
from src.ml.realistic_backtest import RealisticBacktester


# ===================================================================
# RULE VALIDATORS — produce scores [0,1], NOT booleans
# ===================================================================

class ContextRuleValidator:
    """Trend structure validator. Returns 0-1 score (1 = strong trend aligned)."""

    def score(self, df: pd.DataFrame) -> float:
        close = df['close'].values.astype(float)
        if len(close) < 50:
            return 0.5  # neutral if insufficient data
        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        if np.isnan(ema20[-1]) or np.isnan(ema50[-1]):
            return 0.5
        # Score based on EMA alignment strength
        ratio = (ema20[-1] - ema50[-1]) / max(ema50[-1], 1e-8)
        # Clip and normalize: strong uptrend → 1.0, strong downtrend → 0.0
        return float(np.clip(ratio * 20 + 0.5, 0.0, 1.0))


class RegimeRuleValidator:
    """ADX-based regime validator. High ADX = trending (good for our edge)."""

    def score(self, df: pd.DataFrame) -> float:
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        if len(close) < 30:
            return 0.5
        adx = _adx(high, low, close, 14)
        if np.isnan(adx[-1]):
            return 0.5
        # ADX > 25 = trending, > 40 = strong trend
        return float(np.clip(adx[-1] / 50.0, 0.0, 1.0))


class SetupRuleValidator:
    """RSI + momentum alignment. Is the setup structurally valid?"""

    def score(self, df: pd.DataFrame) -> float:
        close = df['close'].values.astype(float)
        if len(close) < 20:
            return 0.5
        rsi = _rsi(close, 14)
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)
        if np.isnan(rsi[-1]) or np.isnan(ema9[-1]) or np.isnan(ema21[-1]):
            return 0.5
        # RSI in trending zone (40-70 for bull) and EMAs aligned
        rsi_score = 1.0 - abs(rsi[-1] - 55) / 55  # peak at 55
        ema_aligned = float(ema9[-1] > ema21[-1])
        return float(np.clip(0.5 * rsi_score + 0.5 * ema_aligned, 0.0, 1.0))


# ===================================================================
# DISAGREEMENT SIGNAL
# ===================================================================

def compute_disagreement(ml_direction: int, rule_scores: Dict[str, float]) -> float:
    """
    Compute disagreement between ML signal and rule validators.

    ml_direction: +1 (BUY) or -1 (SELL)
    rule_scores: {'context': 0-1, 'regime': 0-1, 'setup': 0-1}
      where 0.5 = neutral, >0.5 = bullish, <0.5 = bearish

    Returns: 0.0 (full agreement) to 1.0 (full disagreement)
    """
    if ml_direction == 0:
        return 0.0

    scores = list(rule_scores.values())
    avg_rule = np.mean(scores)

    if ml_direction == 1:
        # ML wants long, rules > 0.5 = agree
        disagreement = max(0, 0.5 - avg_rule) * 2  # 0 when avg=0.5, 1 when avg=0
    else:
        # ML wants short, rules < 0.5 = agree
        disagreement = max(0, avg_rule - 0.5) * 2  # 0 when avg=0.5, 1 when avg=1

    return float(np.clip(disagreement, 0.0, 1.0))


# ===================================================================
# SIZE ADJUSTMENT
# ===================================================================

def compute_size_factor(
    ml_confidence: float,
    disagreement: float,
    rule_avg: float,
) -> float:
    """
    Compute position size factor based on ML confidence and rule agreement.

    Returns: 0.25 to 1.5
      Full agreement + high confidence → 1.5
      Disagreement → 0.25-0.75
      Never 0 (that's hard veto territory)
    """
    # Base from ML confidence
    if ml_confidence >= 0.75:
        base = 1.5
    elif ml_confidence >= 0.65:
        base = 1.2
    elif ml_confidence >= 0.55:
        base = 1.0
    else:
        base = 0.75

    # Disagreement penalty
    penalty = disagreement * 0.6  # max 60% reduction
    adjusted = base * (1.0 - penalty)

    # Floor at 0.25 (never zero from soft gate)
    return float(max(0.25, adjusted))


# ===================================================================
# HARD VETO — RARE
# ===================================================================

def check_hard_veto(
    price: float,
    atr: float,
    spread_pct: float,
    volume_ratio: float,
) -> Tuple[bool, str]:
    """
    Hard veto for extreme conditions only.
    Should trigger < 1% of the time.
    """
    if atr <= 0 or np.isnan(atr):
        return True, "ATR invalid (no volatility data)"
    if spread_pct > 0.5:
        return True, f"Spread {spread_pct:.2%} too wide (> 0.5%)"
    if volume_ratio <= 0.01:
        return True, "Volume essentially zero"
    if price <= 0 or np.isnan(price):
        return True, "Price invalid"
    return False, ""


# ===================================================================
# SOFT GATE ORCHESTRATOR
# ===================================================================

class SoftGateOrchestrator:
    """
    ML-driven orchestrator with soft rule validation.

    ML edge (trend + volume) makes the core decision.
    Rule validators provide scores as additional context.
    Disagreement adjusts position size, doesn't block trades.
    Hard veto only for data integrity issues.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        risk_pct: float = 0.02,
        tp_mult: float = 1.5,
        sl_mult: float = 1.0,
        max_bars: int = 50,
        fee_rate: float = 0.0002,
        slippage_rate: float = 0.0001,
        vol_min: float = 3.0,          # same base edge quality, soft gate modulates size
        cooldown_bars: int = 24,
        max_daily_trades: int = 1,
    ):
        self.capital = initial_capital
        self.risk_pct = risk_pct
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.max_bars = max_bars
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        # Volume filter: soft min 2.0, but size scales with volume strength
        self.vol_min = vol_min
        self.cooldown_bars = cooldown_bars
        self.max_daily_trades = max_daily_trades

        self.ctx_validator = ContextRuleValidator()
        self.reg_validator = RegimeRuleValidator()
        self.stp_validator = SetupRuleValidator()

    def compute_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute ML edge signal + rule scores + disagreement."""
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)

        if len(close) < 60:
            return {'direction': 0, 'rule_context': 0.5, 'rule_regime': 0.5,
                    'rule_setup': 0.5, 'disagreement': 0.0, 'size_factor': 0.0}

        ema9 = _ema(close, 9); ema21 = _ema(close, 21); ema50 = _ema(close, 50)
        vol_ma = _ema(volume, 20)
        atr = np.nan_to_num(_atr(high, low, close, 14), nan=0.0)

        # ML edge signal: trend + volume
        direction = 0
        if not (np.isnan(ema9[-1]) or np.isnan(ema50[-1])):
            if ema9[-1] > ema21[-1] > ema50[-1]:
                direction = 1
            elif ema9[-1] < ema21[-1] < ema50[-1]:
                direction = -1

        if direction != 0 and vol_ma[-1] > 0 and volume[-1] / vol_ma[-1] < self.vol_min:
            direction = 0  # volume too low

        # Rule validator scores
        rc = self.ctx_validator.score(df)
        rr = self.reg_validator.score(df)
        rs = self.stp_validator.score(df)

        # Disagreement
        rule_scores = {'context': rc, 'regime': rr, 'setup': rs}
        disagreement = compute_disagreement(direction, rule_scores)

        # ML confidence proxy: how aligned is the trend?
        if direction != 0 and not np.isnan(ema50[-1]) and ema50[-1] > 0:
            trend_strength = abs(ema9[-1] - ema50[-1]) / ema50[-1]
            ml_confidence = min(1.0, trend_strength * 50)
        else:
            ml_confidence = 0.0

        # Size factor
        rule_avg = np.mean([rc, rr, rs])
        sf = compute_size_factor(ml_confidence, disagreement, rule_avg)

        return {
            'direction': direction,
            'ml_confidence': ml_confidence,
            'rule_context': rc,
            'rule_regime': rr,
            'rule_setup': rs,
            'disagreement': disagreement,
            'size_factor': sf,
            'atr': float(atr[-1]),
        }

    def backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run realistic backtest with soft gate logic."""
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        n = len(close)

        atr = np.nan_to_num(_atr(high, low, close, 14), nan=0.0)
        vol_ma = _ema(volume, 20)

        # Pre-compute EMAs for speed
        ema9 = _ema(close, 9); ema21 = _ema(close, 21); ema50 = _ema(close, 50)

        # Pre-compute rule scores (vectorized approximation)
        ctx_scores = np.zeros(n)
        reg_scores = np.zeros(n)
        stp_scores = np.zeros(n)
        rsi = _rsi(close, 14)
        adx = _adx(high, low, close, 14)

        for i in range(60, n):
            if not np.isnan(ema50[i]) and ema50[i] > 0:
                ratio = (ema9[i] - ema50[i]) / ema50[i]  # simplified
                ctx_scores[i] = np.clip(ratio * 20 + 0.5, 0.0, 1.0)
            else:
                ctx_scores[i] = 0.5
            reg_scores[i] = np.clip(adx[i] / 50.0, 0.0, 1.0) if not np.isnan(adx[i]) else 0.5
            if not np.isnan(rsi[i]) and not np.isnan(ema9[i]):
                rsi_s = 1.0 - abs(rsi[i] - 55) / 55
                ema_a = float(ema9[i] > ema21[i]) if not np.isnan(ema21[i]) else 0.5
                stp_scores[i] = np.clip(0.5 * rsi_s + 0.5 * ema_a, 0.0, 1.0)
            else:
                stp_scores[i] = 0.5

        # Backtest loop
        capital = float(self.capital)
        equity_curve = [capital]
        trades: List[Dict] = []
        pos = None
        last_exit = -999
        daily_counts: Dict[str, int] = {}
        total_fees = 0.0

        for i in range(60, n):
            # Position management
            if pos is not None:
                d = pos['d']
                hit_tp = (d == 1 and high[i] >= pos['tp']) or (d == -1 and low[i] <= pos['tp'])
                hit_sl = (d == 1 and low[i] <= pos['sl']) or (d == -1 and high[i] >= pos['sl'])
                expired = (i - pos['bar']) >= self.max_bars

                if hit_tp or hit_sl or expired:
                    raw_exit = pos['tp'] if hit_tp else (pos['sl'] if hit_sl else close[i])
                    exit_price = raw_exit * (1 - d * self.slippage_rate)
                    reason = 'TP' if hit_tp else ('SL' if hit_sl else 'TIME')

                    gross_pnl = d * (exit_price - pos['ep']) * pos['qty']
                    exit_fee = abs(pos['qty']) * exit_price * self.fee_rate
                    net_pnl = gross_pnl - exit_fee
                    total_fees += pos['ef'] + exit_fee
                    capital += net_pnl
                    capital = max(capital, 0)

                    trades.append({
                        'entry_time': df.index[pos['bar']], 'exit_time': df.index[i],
                        'entry_bar': pos['bar'], 'exit_bar': i,
                        'direction': d, 'entry_price': pos['ep'], 'exit_price': exit_price,
                        'qty': pos['qty'], 'net_pnl': net_pnl, 'fees': pos['ef'] + exit_fee,
                        'size_factor': pos['sf'], 'disagreement': pos['dis'],
                        'reason': reason,
                    })
                    last_exit = i
                    pos = None

            # Entry
            if pos is None and capital > 0:
                if i - last_exit < self.cooldown_bars:
                    equity_curve.append(capital); continue
                if np.isnan(ema9[i]) or np.isnan(ema50[i]) or atr[i] <= 0:
                    equity_curve.append(capital); continue
                if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
                    equity_curve.append(capital); continue

                # Hard veto check
                spread_est = atr[i] / close[i] * 0.1  # rough spread estimate
                veto, _ = check_hard_veto(close[i], atr[i], spread_est, volume[i]/max(vol_ma[i],1))
                if veto:
                    equity_curve.append(capital); continue

                # ML edge: trend + volume
                direction = 0
                if ema9[i] > ema21[i] > ema50[i]: direction = 1
                elif ema9[i] < ema21[i] < ema50[i]: direction = -1
                if direction == 0:
                    equity_curve.append(capital); continue
                if volume[i] / vol_ma[i] < self.vol_min:
                    equity_curve.append(capital); continue

                # Daily limit
                day_key = str(df.index[i].date())
                if daily_counts.get(day_key, 0) >= self.max_daily_trades:
                    equity_curve.append(capital); continue

                # Hour filter: strong penalty for off-session, bonus for peak hours
                hour = df.index[i].hour if hasattr(df.index, 'hour') else 12
                if 8 <= hour <= 18:
                    hour_bonus = 1.15  # EU/US session
                elif 6 <= hour <= 20:
                    hour_bonus = 0.7   # fringe hours — significant penalty
                else:
                    hour_bonus = 0.0   # overnight → skip (effective soft block via size=0)
                if hour_bonus == 0.0:
                    equity_curve.append(capital); continue

                # Soft gate: rule scores + disagreement
                rule_scores = {'context': ctx_scores[i], 'regime': reg_scores[i], 'setup': stp_scores[i]}
                disagreement = compute_disagreement(direction, rule_scores)
                trend_strength = abs(ema9[i] - ema50[i]) / max(ema50[i], 1e-8)
                ml_confidence = min(1.0, trend_strength * 50)
                rule_avg = np.mean(list(rule_scores.values()))
                size_factor = compute_size_factor(ml_confidence, disagreement, rule_avg)
                size_factor *= hour_bonus

                # Position sizing
                risk_dollars = capital * self.risk_pct * size_factor
                stop_distance = self.sl_mult * atr[i]
                qty = min(risk_dollars / stop_distance, capital / close[i])
                if qty <= 0:
                    equity_curve.append(capital); continue

                entry_price = close[i] * (1 + direction * self.slippage_rate)
                entry_fee = qty * entry_price * self.fee_rate
                tp = entry_price + direction * self.tp_mult * atr[i]
                sl = entry_price - direction * self.sl_mult * atr[i]

                pos = {'d': direction, 'ep': entry_price, 'bar': i, 'qty': qty,
                       'tp': tp, 'sl': sl, 'ef': entry_fee, 'sf': size_factor,
                       'dis': disagreement}
                daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

            equity_curve.append(capital)

        # Close open position
        if pos:
            d = pos['d']
            ep = close[-1] * (1 - d * self.slippage_rate)
            gpnl = d * (ep - pos['ep']) * pos['qty']
            ef = abs(pos['qty']) * ep * self.fee_rate
            capital += gpnl - ef
            total_fees += pos['ef'] + ef
            trades.append({'entry_time': df.index[pos['bar']], 'exit_time': df.index[-1],
                           'entry_bar': pos['bar'], 'exit_bar': n-1,
                           'direction': d, 'entry_price': pos['ep'], 'exit_price': ep,
                           'qty': pos['qty'], 'net_pnl': gpnl-ef, 'fees': pos['ef']+ef,
                           'size_factor': pos['sf'], 'disagreement': pos['dis'], 'reason': 'EOD'})
            equity_curve[-1] = capital

        # Metrics
        eq = np.array(equity_curve)
        nt = len(trades)
        total_pnl = capital - self.capital
        wins = [t for t in trades if t['net_pnl'] > 0]
        losses = [t for t in trades if t['net_pnl'] <= 0]

        peak = eq[0]; max_dd = 0.0
        for e in eq:
            peak = max(peak, e)
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        daily_eq = pd.Series(eq, index=df.index[:len(eq)]).resample('1D').last().dropna()
        daily_ret = daily_eq.pct_change().dropna()
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0

        gw = sum(t['net_pnl'] for t in wins)
        gl = abs(sum(t['net_pnl'] for t in losses))

        return {
            'initial_capital': self.capital,
            'final_capital': capital,
            'total_pnl_dollars': total_pnl,
            'total_return_pct': total_pnl / self.capital,
            'n_trades': nt,
            'win_rate': len(wins) / nt if nt > 0 else 0,
            'profit_factor': gw / gl if gl > 0 else float('inf'),
            'sharpe': sharpe,
            'max_drawdown_pct': max_dd,
            'total_fees': total_fees,
            'trades_per_day': nt / max((df.index[-1] - df.index[0]).days, 1),
            'avg_disagreement': np.mean([t['disagreement'] for t in trades]) if trades else 0,
            'avg_size_factor': np.mean([t['size_factor'] for t in trades]) if trades else 0,
            'equity_curve': eq.tolist(),
            'trades': trades,
        }
