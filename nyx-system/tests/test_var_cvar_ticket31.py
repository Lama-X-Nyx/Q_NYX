"""
TDD Tests — Ticket 31 — Advanced Risk Engine (VaR / CVaR).

VaR/CVaR extend the existing RiskEngine (T25) — not replace it.
Computed from realized PnL history. Integrated into validate_trade()
as an ADDITIONAL gate after the existing static rules.
"""
from __future__ import annotations

import numpy as np
import pytest


# ===========================================================================
class TestVaRComputation:

    def test_var_95_on_known_distribution(self):
        from src.live.var_cvar import compute_var
        pnl = np.array([-100, -80, -60, -40, -20, 0, 20, 40, 60, 80,
                         100, 120, 140, 160, 180, 200, 220, 240, 260, 280])
        var95 = compute_var(pnl, percentile=5)
        # 5th percentile of this sorted array = -100 + 0.05*(280-(-100)) ≈ -81
        assert var95 < 0, 'VaR should be negative (loss)'
        assert var95 == pytest.approx(np.percentile(pnl, 5))

    def test_var_99_stricter_than_95(self):
        from src.live.var_cvar import compute_var
        np.random.seed(42)
        pnl = np.random.randn(200) * 100
        var95 = compute_var(pnl, percentile=5)
        var99 = compute_var(pnl, percentile=1)
        assert var99 < var95, 'VaR 99% must be stricter (more negative)'

    def test_var_empty_returns_zero(self):
        from src.live.var_cvar import compute_var
        assert compute_var(np.array([]), percentile=5) == 0.0


# ===========================================================================
class TestCVaRComputation:

    def test_cvar_is_mean_of_tail(self):
        from src.live.var_cvar import compute_cvar
        pnl = np.array([-200, -150, -100, -50, 0, 50, 100, 150, 200, 250])
        cvar95 = compute_cvar(pnl, percentile=5)
        # Tail below 5th percentile. On 10 values, the 5th percentile
        # is near -200. CVaR = mean of values <= VaR threshold.
        assert cvar95 < 0

    def test_cvar_stricter_than_var(self):
        from src.live.var_cvar import compute_var, compute_cvar
        np.random.seed(42)
        pnl = np.random.randn(200) * 100
        var95 = compute_var(pnl, percentile=5)
        cvar95 = compute_cvar(pnl, percentile=5)
        assert cvar95 <= var95, 'CVaR must be <= VaR (deeper in tail)'

    def test_cvar_empty_returns_zero(self):
        from src.live.var_cvar import compute_cvar
        assert compute_cvar(np.array([]), percentile=5) == 0.0


# ===========================================================================
class TestRollingWindow:

    def test_rolling_var(self):
        from src.live.var_cvar import RollingRiskMetrics
        rm = RollingRiskMetrics(window=10)
        for pnl in [-50, -30, -10, 10, 30, 50, 70, 90, 110, 130]:
            rm.add_trade_pnl(pnl)
        assert rm.var_95 is not None
        assert rm.var_95 < 0

    def test_rolling_cvar(self):
        from src.live.var_cvar import RollingRiskMetrics
        rm = RollingRiskMetrics(window=10)
        for pnl in [-50, -30, -10, 10, 30, 50, 70, 90, 110, 130]:
            rm.add_trade_pnl(pnl)
        assert rm.cvar_95 is not None
        assert rm.cvar_95 <= rm.var_95

    def test_window_respects_size(self):
        from src.live.var_cvar import RollingRiskMetrics
        rm = RollingRiskMetrics(window=5)
        for i in range(20):
            rm.add_trade_pnl(float(i))
        assert len(rm._pnl_buffer) == 5


# ===========================================================================
class TestRiskEngineVaRIntegration:

    def test_high_var_blocks_trade(self):
        from src.live.risk_engine import RiskEngine
        from src.live.var_cvar import RollingRiskMetrics
        re = RiskEngine(max_var_95=-100.0)  # block when VaR ≤ -100
        rm = RollingRiskMetrics(window=10)
        for pnl in [-200, -180, -150, -120, -100, -80, -60, -40, -20, -10]:
            rm.add_trade_pnl(pnl)
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.01,
            price=16500.0,
            portfolio={'available_balance': 10_000, 'total_exposure': 0,
                       'daily_realized_pnl': 0, 'weekly_realized_pnl': 0,
                       'max_drawdown_from_peak': 0, 'open_position_count': 0},
            rolling_risk=rm,
        )
        assert result['allowed'] is False
        assert 'var' in result['reason'].lower()

    def test_normal_var_allows_trade(self):
        from src.live.risk_engine import RiskEngine
        from src.live.var_cvar import RollingRiskMetrics
        re = RiskEngine(max_var_95=-500.0)
        rm = RollingRiskMetrics(window=10)
        for pnl in [50, 30, 10, -10, 20, 40, 60, 80, 100, 120]:
            rm.add_trade_pnl(pnl)
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.01,
            price=16500.0,
            portfolio={'available_balance': 10_000, 'total_exposure': 0,
                       'daily_realized_pnl': 0, 'weekly_realized_pnl': 0,
                       'max_drawdown_from_peak': 0, 'open_position_count': 0},
            rolling_risk=rm,
        )
        assert result['allowed'] is True

    def test_risk_scaling_from_cvar(self):
        from src.live.var_cvar import RollingRiskMetrics
        rm = RollingRiskMetrics(window=10)
        for pnl in [-100, -80, -60, 10, 20, 30, 40, 50, 60, 70]:
            rm.add_trade_pnl(pnl)
        scale = rm.risk_scale_factor(max_cvar=-200.0)
        assert 0.0 < scale <= 1.0
