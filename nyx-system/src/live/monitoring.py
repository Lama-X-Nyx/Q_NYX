"""
Monitoring & Observability — no blind trading (Ticket 27).

MetricsCollector : real-time trading + system metrics.
AlertManager : threshold-based alert triggers.
"""
from __future__ import annotations

from typing import Any, Dict, List


class MetricsCollector:
    """Real-time metrics for trading + system health."""

    def __init__(self) -> None:
        self.total_trades: int = 0
        self.total_pnl: float = 0.0
        self._wins: int = 0
        self._losses: int = 0
        self._attempts: int = 0
        self._fills: int = 0
        self._rejects: int = 0
        self._peak_equity: float = 0.0
        self._max_dd_pct: float = 0.0
        self._ws_latencies: List[float] = []
        self.reconnect_count: int = 0

    def record_trade(self, pnl: float, side: str = '',
                     filled: bool = True) -> None:
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self._wins += 1
        else:
            self._losses += 1

    @property
    def win_rate(self) -> float:
        return self._wins / max(self.total_trades, 1)

    def record_order_attempt(self, filled: bool) -> None:
        self._attempts += 1
        if filled:
            self._fills += 1

    def record_order_reject(self) -> None:
        self._attempts += 1
        self._rejects += 1

    @property
    def fill_rate(self) -> float:
        return self._fills / max(self._attempts, 1)

    @property
    def miss_rate(self) -> float:
        misses = self._attempts - self._fills - self._rejects
        return misses / max(self._attempts, 1)

    @property
    def reject_rate(self) -> float:
        return self._rejects / max(self._attempts, 1)

    def update_equity(self, equity: float) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity * 100.0
            if dd > self._max_dd_pct:
                self._max_dd_pct = dd

    @property
    def max_drawdown_pct(self) -> float:
        return self._max_dd_pct

    def record_ws_latency_ms(self, latency_ms: float) -> None:
        self._ws_latencies.append(float(latency_ms))

    @property
    def avg_ws_latency_ms(self) -> float:
        if not self._ws_latencies:
            return 0.0
        return sum(self._ws_latencies) / len(self._ws_latencies)

    def record_reconnect(self) -> None:
        self.reconnect_count += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            'total_trades': self.total_trades,
            'total_pnl': round(self.total_pnl, 2),
            'win_rate': round(self.win_rate, 4),
            'fill_rate': round(self.fill_rate, 4),
            'miss_rate': round(self.miss_rate, 4),
            'reject_rate': round(self.reject_rate, 4),
            'max_drawdown_pct': round(self.max_drawdown_pct, 4),
            'avg_ws_latency_ms': round(self.avg_ws_latency_ms, 2),
            'reconnect_count': self.reconnect_count,
        }


class AlertManager:
    """Threshold-based alert triggers."""

    def __init__(
        self,
        max_daily_loss_pct: float = 1.0,
        max_drawdown_alert_pct: float = 3.0,
    ) -> None:
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_drawdown_alert_pct = float(max_drawdown_alert_pct)

    def check(self, **kwargs: Any) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        daily_loss = float(kwargs.get('daily_loss_pct', 0.0))
        if daily_loss >= self.max_daily_loss_pct:
            alerts.append({
                'type': 'daily_loss',
                'severity': 'critical',
                'message': f'Daily loss {daily_loss:.2f}% >= {self.max_daily_loss_pct}%',
            })
        dd = float(kwargs.get('drawdown_pct', 0.0))
        if dd >= self.max_drawdown_alert_pct:
            alerts.append({
                'type': 'drawdown',
                'severity': 'critical',
                'message': f'Drawdown {dd:.2f}% >= {self.max_drawdown_alert_pct}%',
            })
        if not kwargs.get('ws_connected', True):
            alerts.append({
                'type': 'disconnect',
                'severity': 'warning',
                'message': 'WebSocket disconnected',
            })
        return alerts
