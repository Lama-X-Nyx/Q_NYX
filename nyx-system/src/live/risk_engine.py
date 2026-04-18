"""
Risk Engine — sovereign, no bypass (Ticket 25 + Ticket 40).

Sits BETWEEN decision and execution. Can BLOCK any trade.
NYXEngine → risk_engine.validate_trade() → OMS.

Cannot be bypassed. No risk logic inside GBM or Jesse.

Ticket 40: All primary limits are %-based. They scale with equity.
Optional hard caps (absolute notional) act as secondary safety nets.

Rules :
  Trade-level :
    - max risk per trade (% of available balance)
    - max position (% of equity → computed dynamically)
  Portfolio-level :
    - max total exposure (% of capital)
    - max concurrent positions
  Drawdown protection :
    - max daily loss (% of capital)
    - max weekly loss (% of capital)
    - max drawdown from peak (% of peak equity)
  VaR / CVaR (Ticket 31) :
    - max_var_95_pct / max_cvar_95_pct (% of equity)
  Kill switch :
    - manually activated on anomaly
    - blocks ALL trades until reset
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class RiskEngine:
    """Sovereign risk gate — no trade passes without validation."""

    def __init__(
        self,
        max_risk_per_trade_pct: float = 2.0,
        max_position_pct: float = 5.0,
        max_total_exposure_pct: float = 100.0,
        max_concurrent_positions: int = 5,
        max_daily_loss_pct: float = 1.0,
        max_weekly_loss_pct: float = 3.0,
        max_drawdown_pct: float = 5.0,
        max_var_95_pct: Optional[float] = None,
        max_cvar_95_pct: Optional[float] = None,
        max_position_notional_hard_cap: Optional[float] = None,
        max_order_notional_hard_cap: Optional[float] = None,
        # Legacy backward-compat aliases
        max_position_notional: Optional[float] = None,
        max_var_95: Optional[float] = None,
        max_cvar_95: Optional[float] = None,
    ) -> None:
        self.max_risk_per_trade_pct = float(max_risk_per_trade_pct)
        self.max_position_pct = float(max_position_pct)
        self.max_total_exposure_pct = float(max_total_exposure_pct)
        self.max_concurrent_positions = int(max_concurrent_positions)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_weekly_loss_pct = float(max_weekly_loss_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)

        self.max_var_95_pct = float(max_var_95_pct) if max_var_95_pct is not None else None
        self.max_cvar_95_pct = float(max_cvar_95_pct) if max_cvar_95_pct is not None else None

        if max_position_notional is not None and max_position_notional_hard_cap is None:
            max_position_notional_hard_cap = max_position_notional
        self.max_position_notional_hard_cap = (
            float(max_position_notional_hard_cap)
            if max_position_notional_hard_cap is not None else None
        )
        self.max_order_notional_hard_cap = (
            float(max_order_notional_hard_cap)
            if max_order_notional_hard_cap is not None else None
        )

        self._legacy_var_95 = float(max_var_95) if max_var_95 is not None else None
        self._legacy_cvar_95 = float(max_cvar_95) if max_cvar_95 is not None else None

        self._kill_switch_active = False
        self._kill_switch_reason = ''

    @property
    def is_killed(self) -> bool:
        return self._kill_switch_active

    def activate_kill_switch(self, reason: str = '') -> None:
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        log.critical('KILL SWITCH ACTIVATED: %s', reason)

    def reset_kill_switch(self) -> None:
        self._kill_switch_active = False
        self._kill_switch_reason = ''
        log.info('kill switch reset')

    def validate_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio: Dict[str, Any],
        rolling_risk: Any = None,
    ) -> Dict[str, Any]:
        """Validate a trade intent BEFORE it reaches OMS.

        Returns {'allowed': bool, 'reason': str, 'risk_meta': dict}.
        """
        if self._kill_switch_active:
            return self._block(
                f'kill_switch: {self._kill_switch_reason}',
                portfolio=portfolio,
            )

        notional = float(quantity) * float(price)
        balance = float(portfolio.get('available_balance', 0.0))
        exposure = float(portfolio.get('total_exposure', 0.0))

        meta = {
            'equity': balance,
            'notional': notional,
            'computed_max_position': balance * self.max_position_pct / 100.0 if balance > 0 else 0.0,
            'hard_cap': self.max_position_notional_hard_cap,
        }

        if balance <= 0:
            return self._block('equity <= 0', risk_meta=meta)

        risk_pct = (notional / balance) * 100.0
        if risk_pct > self.max_risk_per_trade_pct:
            return self._block(
                f'max_risk_per_trade: {risk_pct:.1f}% > '
                f'{self.max_risk_per_trade_pct}%',
                risk_meta=meta,
            )

        max_pos_notional = balance * self.max_position_pct / 100.0
        if notional > max_pos_notional:
            return self._block(
                f'max_position_pct: ${notional:,.0f} > '
                f'${max_pos_notional:,.0f} ({self.max_position_pct}%)',
                risk_meta=meta,
            )

        if self.max_position_notional_hard_cap is not None:
            if notional > self.max_position_notional_hard_cap:
                return self._block(
                    f'hard_cap: ${notional:,.0f} > '
                    f'${self.max_position_notional_hard_cap:,.0f}',
                    risk_meta=meta,
                )

        if self.max_order_notional_hard_cap is not None:
            if notional > self.max_order_notional_hard_cap:
                return self._block(
                    f'order_hard_cap: ${notional:,.0f} > '
                    f'${self.max_order_notional_hard_cap:,.0f}',
                    risk_meta=meta,
                )

        new_exposure_pct = ((exposure + notional) / balance) * 100.0
        if new_exposure_pct > self.max_total_exposure_pct:
            return self._block(
                f'exposure: {new_exposure_pct:.1f}% > '
                f'{self.max_total_exposure_pct}%',
                risk_meta=meta,
            )

        n_open = int(portfolio.get('open_position_count', 0))
        if n_open >= self.max_concurrent_positions:
            return self._block(
                f'concurrent: {n_open} >= {self.max_concurrent_positions}',
                risk_meta=meta,
            )

        daily_pnl = float(portfolio.get('daily_realized_pnl', 0.0))
        if daily_pnl < 0:
            daily_loss_pct = abs(daily_pnl) / balance * 100.0
            if daily_loss_pct >= self.max_daily_loss_pct:
                return self._block(
                    f'daily_loss: {daily_loss_pct:.2f}% >= '
                    f'{self.max_daily_loss_pct}%',
                    risk_meta=meta,
                )

        weekly_pnl = float(portfolio.get('weekly_realized_pnl', 0.0))
        if weekly_pnl < 0:
            weekly_loss_pct = abs(weekly_pnl) / balance * 100.0
            if weekly_loss_pct >= self.max_weekly_loss_pct:
                return self._block(
                    f'weekly_loss: {weekly_loss_pct:.2f}% >= '
                    f'{self.max_weekly_loss_pct}%',
                    risk_meta=meta,
                )

        dd = float(portfolio.get('max_drawdown_from_peak', 0.0))
        if dd * 100.0 >= self.max_drawdown_pct:
            return self._block(
                f'drawdown: {dd * 100:.2f}% >= {self.max_drawdown_pct}%',
                risk_meta=meta,
            )

        if rolling_risk is not None:
            var95 = getattr(rolling_risk, 'var_95', None)
            cvar95 = getattr(rolling_risk, 'cvar_95', None)

            if var95 is not None and self.max_var_95_pct is not None:
                var_limit = -(balance * self.max_var_95_pct / 100.0)
                if var95 <= var_limit:
                    return self._block(
                        f'var_95: {var95:,.2f} <= limit {var_limit:,.2f} '
                        f'({self.max_var_95_pct}%)',
                        risk_meta=meta,
                    )
            elif var95 is not None and self._legacy_var_95 is not None:
                if var95 <= self._legacy_var_95:
                    return self._block(
                        f'var_95: {var95:.2f} <= limit {self._legacy_var_95:.2f}',
                        risk_meta=meta,
                    )

            if cvar95 is not None and self.max_cvar_95_pct is not None:
                cvar_limit = -(balance * self.max_cvar_95_pct / 100.0)
                if cvar95 <= cvar_limit:
                    return self._block(
                        f'cvar_95: {cvar95:,.2f} <= limit {cvar_limit:,.2f} '
                        f'({self.max_cvar_95_pct}%)',
                        risk_meta=meta,
                    )
            elif cvar95 is not None and self._legacy_cvar_95 is not None:
                if cvar95 <= self._legacy_cvar_95:
                    return self._block(
                        f'cvar_95: {cvar95:.2f} <= limit {self._legacy_cvar_95:.2f}',
                        risk_meta=meta,
                    )

        return {'allowed': True, 'reason': '', 'risk_meta': meta}

    @staticmethod
    def _block(reason: str, risk_meta: dict = None,
               portfolio: dict = None) -> Dict[str, Any]:
        log.warning('RISK BLOCKED: %s', reason)
        if risk_meta is None and portfolio is not None:
            risk_meta = {'equity': float(portfolio.get('available_balance', 0))}
        return {
            'allowed': False,
            'reason': reason,
            'risk_meta': risk_meta or {},
        }
