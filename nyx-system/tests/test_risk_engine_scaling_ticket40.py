"""
TDD Tests — Ticket 40 — Capital-Aware Risk Engine Scaling.

All primary limits expressed in % of equity. No hardcoded notional
dominates. Behavior consistent across $10k → $10M.
"""
from __future__ import annotations

import pytest


def _make_portfolio(balance: float = 10_000.0, exposure: float = 0.0,
                    n_open: int = 0) -> dict:
    return {
        'available_balance': balance,
        'total_exposure': exposure,
        'daily_realized_pnl': 0.0,
        'weekly_realized_pnl': 0.0,
        'max_drawdown_from_peak': 0.0,
        'open_position_count': n_open,
    }


class TestPositionLimitScaling:

    def test_default_max_position_is_pct_based(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine()
        assert hasattr(re, 'max_position_pct')

    def test_position_scales_with_equity_10k(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_pct=5.0, max_risk_per_trade_pct=10.0)
        pf = _make_portfolio(balance=10_000.0)
        # 0.01 * 30000 = $300 = 3% of $10k → within 5% position + 10% risk
        r = re.validate_trade('BTCUSDT', 'buy', quantity=0.01, price=30000.0,
                              portfolio=pf)
        assert r['allowed'] is True
        # 0.02 * 30000 = $600 = 6% of $10k → exceeds 5% position
        r2 = re.validate_trade('BTCUSDT', 'buy', quantity=0.02, price=30000.0,
                               portfolio=pf)
        assert r2['allowed'] is False

    def test_position_scales_with_equity_10M(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_pct=5.0, max_risk_per_trade_pct=10.0)
        pf = _make_portfolio(balance=10_000_000.0)
        # 15 * 30000 = $450k = 4.5% of $10M → within 5%
        r = re.validate_trade('BTCUSDT', 'buy', quantity=15.0, price=30000.0,
                              portfolio=pf)
        assert r['allowed'] is True

    def test_same_pct_same_result_different_capitals(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_pct=5.0, max_risk_per_trade_pct=10.0)
        for capital in (10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0):
            notional = capital * 0.04
            price = 30000.0
            qty = notional / price
            pf = _make_portfolio(balance=capital)
            r = re.validate_trade('BTCUSDT', 'buy', quantity=qty, price=price, portfolio=pf)
            assert r['allowed'] is True, f'Blocked at capital=${capital:,.0f}'

    def test_exceeding_pct_blocked_at_any_capital(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_pct=5.0, max_risk_per_trade_pct=10.0)
        for capital in (10_000.0, 100_000.0, 10_000_000.0):
            notional = capital * 0.06
            price = 30000.0
            qty = notional / price
            pf = _make_portfolio(balance=capital)
            r = re.validate_trade('BTCUSDT', 'buy', quantity=qty, price=price, portfolio=pf)
            assert r['allowed'] is False, f'Allowed at capital=${capital:,.0f}'


class TestHardCapAsSecondary:

    def test_hard_cap_blocks_even_if_pct_allows(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_pct=5.0,
                        max_position_notional_hard_cap=100_000.0)
        pf = _make_portfolio(balance=10_000_000.0)
        qty = 4.0
        r = re.validate_trade('BTCUSDT', 'buy', quantity=qty, price=30000.0,
                              portfolio=pf)
        assert r['allowed'] is False
        assert 'hard_cap' in r['reason']

    def test_no_hard_cap_by_default(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_pct=5.0)
        assert re.max_position_notional_hard_cap is None


class TestVarCvarScaling:

    def test_var_pct_based(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_var_95_pct=3.0, max_cvar_95_pct=5.0)
        assert re.max_var_95_pct == 3.0
        assert re.max_cvar_95_pct == 5.0

    def test_var_computed_from_equity(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_var_95_pct=3.0)
        pf = _make_portfolio(balance=1_000_000.0)

        class FakeRolling:
            var_95 = -25_000.0
            cvar_95 = -40_000.0

        r = re.validate_trade('BTCUSDT', 'buy', quantity=0.001, price=30000.0,
                              portfolio=pf, rolling_risk=FakeRolling())
        assert r['allowed'] is True

    def test_var_blocks_when_exceeded(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_var_95_pct=3.0)
        pf = _make_portfolio(balance=1_000_000.0)

        class FakeRolling:
            var_95 = -35_000.0
            cvar_95 = None

        r = re.validate_trade('BTCUSDT', 'buy', quantity=0.001, price=30000.0,
                              portfolio=pf, rolling_risk=FakeRolling())
        assert r['allowed'] is False


class TestRiskMetadata:

    def test_allowed_includes_metadata(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_pct=5.0, max_risk_per_trade_pct=10.0)
        pf = _make_portfolio(balance=100_000.0)
        r = re.validate_trade('BTCUSDT', 'buy', quantity=0.1, price=30000.0,
                              portfolio=pf)
        assert r['allowed'] is True
        assert 'risk_meta' in r
        meta = r['risk_meta']
        assert 'equity' in meta
        assert 'computed_max_position' in meta
        assert meta['equity'] == 100_000.0
        assert meta['computed_max_position'] == 5_000.0

    def test_blocked_includes_metadata(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_pct=5.0)
        pf = _make_portfolio(balance=10_000.0)
        r = re.validate_trade('BTCUSDT', 'buy', quantity=0.02, price=30000.0,
                              portfolio=pf)
        assert r['allowed'] is False
        assert 'risk_meta' in r


class TestBackwardCompatibility:

    def test_old_max_position_notional_still_works(self):
        """Legacy callers passing max_position_notional get it as hard cap."""
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_notional=50_000.0)
        assert re.max_position_notional_hard_cap == 50_000.0

    def test_default_small_capital_behavior_preserved(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine()
        pf = _make_portfolio(balance=10_000.0)
        r = re.validate_trade('BTCUSDT', 'buy', quantity=0.005, price=30000.0,
                              portfolio=pf)
        assert r['allowed'] is True


class TestEdgeCases:

    def test_zero_equity(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_pct=5.0)
        pf = _make_portfolio(balance=0.0)
        r = re.validate_trade('BTCUSDT', 'buy', quantity=0.001, price=30000.0,
                              portfolio=pf)
        assert r['allowed'] is False

    def test_negative_equity(self):
        from src.live.risk_engine import RiskEngine
        re = RiskEngine(max_position_pct=5.0)
        pf = _make_portfolio(balance=-1000.0)
        r = re.validate_trade('BTCUSDT', 'buy', quantity=0.001, price=30000.0,
                              portfolio=pf)
        assert r['allowed'] is False
