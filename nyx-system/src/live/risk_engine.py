"""
Risk Engine — sovereign, no bypass (Ticket 25).

Sits BETWEEN decision and execution. Can BLOCK any trade.
NYXEngine → risk_engine.validate_trade() → OMS.

Cannot be bypassed. No risk logic inside GBM or Jesse.

Rules :
  Trade-level :
    - max risk per trade (notional as % of available balance)
    - max position notional
  Portfolio-level :
    - max total exposure (% of capital)
    - max concurrent positions
  Drawdown protection :
    - max daily loss (% of capital)
    - max weekly loss (% of capital)
    - max drawdown from peak (% of peak equity)
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
        max_risk_per_trade_pct: float = 2.0,      # % of available balance
        max_position_notional: float = 50_000.0,
        max_total_exposure_pct: float = 100.0,     # % of capital
        max_concurrent_positions: int = 5,
        max_daily_loss_pct: float = 1.0,           # % of capital
        max_weekly_loss_pct: float = 3.0,
        max_drawdown_pct: float = 5.0,             # % of peak equity
        # Ticket 31 — VaR / CVaR thresholds (distribution-aware).
        max_var_95: Optional[float] = None,        # max VaR (negative = loss)
        max_cvar_95: Optional[float] = None,
    ) -> None:
        self.max_risk_per_trade_pct = float(max_risk_per_trade_pct)
        self.max_position_notional = float(max_position_notional)
        self.max_total_exposure_pct = float(max_total_exposure_pct)
        self.max_concurrent_positions = int(max_concurrent_positions)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_weekly_loss_pct = float(max_weekly_loss_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.max_var_95 = float(max_var_95) if max_var_95 is not None else None
        self.max_cvar_95 = float(max_cvar_95) if max_cvar_95 is not None else None
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
        rolling_risk: Any = None,  # Ticket 31 — RollingRiskMetrics
    ) -> Dict[str, Any]:
        """Validate a trade intent BEFORE it reaches OMS.

        Returns {'allowed': bool, 'reason': str}.
        If allowed=False, the trade MUST NOT be submitted.
        """
        # Kill switch — absolute override.
        if self._kill_switch_active:
            return self._block(f'kill_switch: {self._kill_switch_reason}')

        notional = float(quantity) * float(price)
        balance = float(portfolio.get('available_balance', 0.0))
        exposure = float(portfolio.get('total_exposure', 0.0))

        # Trade-level: max risk per trade.
        if balance > 0:
            risk_pct = (notional / balance) * 100.0
            if risk_pct > self.max_risk_per_trade_pct:
                return self._block(
                    f'max_risk_per_trade: {risk_pct:.1f}% > '
                    f'{self.max_risk_per_trade_pct}%'
                )

        # Trade-level: max position notional.
        if notional > self.max_position_notional:
            return self._block(
                f'max_position: ${notional:.0f} > '
                f'${self.max_position_notional:.0f}'
            )

        # Portfolio-level: max total exposure.
        if balance > 0:
            new_exposure_pct = ((exposure + notional) / balance) * 100.0
            if new_exposure_pct > self.max_total_exposure_pct:
                return self._block(
                    f'exposure: {new_exposure_pct:.1f}% > '
                    f'{self.max_total_exposure_pct}%'
                )

        # Portfolio-level: max concurrent positions.
        n_open = int(portfolio.get('open_position_count', 0))
        if n_open >= self.max_concurrent_positions:
            return self._block(
                f'concurrent: {n_open} >= '
                f'{self.max_concurrent_positions}'
            )

        # Drawdown: daily loss.
        daily_pnl = float(portfolio.get('daily_realized_pnl', 0.0))
        if balance > 0 and daily_pnl < 0:
            daily_loss_pct = abs(daily_pnl) / balance * 100.0
            if daily_loss_pct >= self.max_daily_loss_pct:
                return self._block(
                    f'daily_loss: {daily_loss_pct:.2f}% >= '
                    f'{self.max_daily_loss_pct}%'
                )

        # Drawdown: weekly loss.
        weekly_pnl = float(portfolio.get('weekly_realized_pnl', 0.0))
        if balance > 0 and weekly_pnl < 0:
            weekly_loss_pct = abs(weekly_pnl) / balance * 100.0
            if weekly_loss_pct >= self.max_weekly_loss_pct:
                return self._block(
                    f'weekly_loss: {weekly_loss_pct:.2f}% >= '
                    f'{self.max_weekly_loss_pct}%'
                )

        # Drawdown: max from peak.
        dd = float(portfolio.get('max_drawdown_from_peak', 0.0))
        if dd * 100.0 >= self.max_drawdown_pct:
            return self._block(
                f'drawdown: {dd * 100:.2f}% >= '
                f'{self.max_drawdown_pct}%'
            )

        # Ticket 31 — VaR / CVaR (distribution-aware, AFTER static rules).
        if rolling_risk is not None:
            var95 = getattr(rolling_risk, 'var_95', None)
            if var95 is not None and self.max_var_95 is not None:
                if var95 <= self.max_var_95:
                    return self._block(
                        f'var_95: {var95:.2f} <= limit {self.max_var_95:.2f}'
                    )
            cvar95 = getattr(rolling_risk, 'cvar_95', None)
            if cvar95 is not None and self.max_cvar_95 is not None:
                if cvar95 <= self.max_cvar_95:
                    return self._block(
                        f'cvar_95: {cvar95:.2f} <= limit {self.max_cvar_95:.2f}'
                    )

        return {'allowed': True, 'reason': ''}

    @staticmethod
    def _block(reason: str) -> Dict[str, Any]:
        log.warning('RISK BLOCKED: %s', reason)
        return {'allowed': False, 'reason': reason}
