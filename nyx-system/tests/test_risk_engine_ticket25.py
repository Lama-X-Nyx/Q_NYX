"""
TDD Tests — Ticket 25 — Risk Engine (sovereign, no bypass).

The risk engine sits BETWEEN decision and execution. It can BLOCK
any trade. NYXEngine → risk_engine.validate() → OMS.

No trade reaches OMS without passing risk validation.
"""
from __future__ import annotations

import pytest


def _portfolio(capital=10_000.0, exposure=0.0, daily_loss=0.0):
    """Mock portfolio snapshot for risk checks."""
    return {
        'available_balance': capital,
        'total_exposure': exposure,
        'daily_realized_pnl': daily_loss,
        'weekly_realized_pnl': daily_loss,
        'max_drawdown_from_peak': abs(daily_loss) / max(capital, 1),
        'open_position_count': 1 if exposure > 0 else 0,
    }


# ===========================================================================
class TestTradeLevelRules:

    def test_trade_within_limits_passes(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine()
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.01,
            price=16500.0, portfolio=_portfolio(),
        )
        assert result['allowed'] is True

    def test_trade_exceeds_max_risk_blocked(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_risk_per_trade_pct=0.25)
        # 1 BTC × $16500 = $16500 notional > 0.25% of $10k = $25
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=1.0,
            price=16500.0, portfolio=_portfolio(),
        )
        assert result['allowed'] is False
        assert 'max_risk_per_trade' in result['reason']

    def test_trade_exceeds_max_position_size_blocked(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_notional=5_000.0,
                        max_risk_per_trade_pct=100.0)  # disable risk-pct
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.5,
            price=16500.0, portfolio=_portfolio(),
        )
        assert result['allowed'] is False
        assert 'max_position' in result['reason']


# ===========================================================================
class TestPortfolioLevelRules:

    def test_max_total_exposure_blocked(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_total_exposure_pct=50.0)
        # Already 60% exposed
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.01,
            price=16500.0,
            portfolio=_portfolio(capital=10_000, exposure=6_000),
        )
        assert result['allowed'] is False
        assert 'exposure' in result['reason']

    def test_max_concurrent_trades_blocked(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_concurrent_positions=1,
                        max_risk_per_trade_pct=100.0,
                        max_total_exposure_pct=200.0)
        pf = _portfolio(capital=50_000, exposure=1000)
        pf['open_position_count'] = 1
        result = re.validate_trade(
            symbol='ETHUSDT', side='buy', quantity=1.0,
            price=1200.0, portfolio=pf,
        )
        assert result['allowed'] is False
        assert 'concurrent' in result['reason']


# ===========================================================================
class TestDrawdownProtection:

    def test_daily_loss_limit_blocks(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_daily_loss_pct=1.0)
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.01,
            price=16500.0,
            portfolio=_portfolio(daily_loss=-150.0),  # -1.5% on $10k
        )
        assert result['allowed'] is False
        assert 'daily_loss' in result['reason']

    def test_weekly_loss_limit_blocks(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_weekly_loss_pct=3.0, max_daily_loss_pct=10.0)
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.01,
            price=16500.0,
            portfolio=_portfolio(daily_loss=-400.0),  # -4% weekly
        )
        assert result['allowed'] is False
        assert 'weekly_loss' in result['reason']

    def test_max_drawdown_stop(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_drawdown_pct=5.0)
        pf = _portfolio()
        pf['max_drawdown_from_peak'] = 0.06  # 6% DD
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.01,
            price=16500.0, portfolio=pf,
        )
        assert result['allowed'] is False
        assert 'drawdown' in result['reason']


# ===========================================================================
class TestKillSwitch:

    def test_kill_switch_blocks_everything(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine()
        re.activate_kill_switch(reason='anomaly detected')
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.01,
            price=16500.0, portfolio=_portfolio(),
        )
        assert result['allowed'] is False
        assert 'kill_switch' in result['reason']

    def test_kill_switch_can_be_reset(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine()
        re.activate_kill_switch(reason='test')
        re.reset_kill_switch()
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.01,
            price=16500.0, portfolio=_portfolio(),
        )
        assert result['allowed'] is True

    def test_kill_switch_status_exposed(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine()
        assert re.is_killed is False
        re.activate_kill_switch(reason='test')
        assert re.is_killed is True


# ===========================================================================
class TestValidateReturnsStructuredResult:

    def test_result_has_required_keys(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine()
        result = re.validate_trade(
            symbol='BTCUSDT', side='buy', quantity=0.01,
            price=16500.0, portfolio=_portfolio(),
        )
        assert 'allowed' in result
        assert 'reason' in result
        assert isinstance(result['allowed'], bool)
        assert isinstance(result['reason'], str)
