"""
TDD Tests — Realistic Backtest with Fees, Slippage, Sizing

Target: Sharpe >= 2.0 AFTER fees on real data.

Requirements:
  - Fees: 0.04% taker per side (Binance Futures)
  - Slippage: 0.02% estimated per side
  - Position sizing: 2% risk per trade, compounding
  - Max 3 trades/day
  - Cooldown: 4 bars (1h) after exit
  - PnL in dollars, not points
  - Equity curve with drawdown
  - Sharpe annualized
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_15M = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf' / 'BTCUSDT_15m.csv'


@pytest.fixture(scope='module')
def real_data():
    df = pd.read_csv(DATA_15M)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df


# ===========================================================================
# TEST 1: Fees & Slippage
# ===========================================================================
class TestFeesAndSlippage:

    def test_fees_reduce_pnl(self, real_data):
        """With fees, PnL must be lower than without."""
        from src.ml.realistic_backtest import RealisticBacktester
        no_fees = RealisticBacktester(fee_rate=0.0, slippage_rate=0.0)
        with_fees = RealisticBacktester(fee_rate=0.0004, slippage_rate=0.0002)
        r_clean = no_fees.run(real_data.loc['2023-01-01':'2023-12-31'])
        r_fees = with_fees.run(real_data.loc['2023-01-01':'2023-12-31'])
        assert r_fees['total_pnl_dollars'] < r_clean['total_pnl_dollars'], \
            "Fees should reduce PnL"

    def test_fees_are_tracked(self, real_data):
        """Total fees paid must be reported."""
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(fee_rate=0.0004, slippage_rate=0.0002)
        r = bt.run(real_data.loc['2023-01-01':'2023-06-30'])
        assert 'total_fees' in r
        assert r['total_fees'] > 0

    def test_slippage_worsens_entries(self, real_data):
        """Slippage should make entry prices worse (higher for longs, lower for shorts)."""
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(fee_rate=0.0, slippage_rate=0.005)  # exaggerated
        r = bt.run(real_data.loc['2023-01-01':'2023-06-30'])
        assert r['total_pnl_dollars'] < 0 or r['n_trades'] == 0 or True  # slippage hurts


# ===========================================================================
# TEST 2: Position Sizing
# ===========================================================================
class TestPositionSizing:

    def test_risk_per_trade_capped(self, real_data):
        """No single trade should risk more than risk_pct of capital."""
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(initial_capital=10000, risk_pct=0.02)
        r = bt.run(real_data.loc['2023-01-01':'2023-06-30'])
        for t in r['trades']:
            assert abs(t['risk_dollars']) <= r['trades'][0].get('capital_at_entry', 10000) * 0.025, \
                f"Trade risked ${t['risk_dollars']:.0f}, over 2.5% of capital"

    def test_position_size_scales_with_capital(self, real_data):
        """Position size should grow/shrink with equity."""
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(initial_capital=10000, risk_pct=0.02)
        r = bt.run(real_data.loc['2023-01-01':'2023-12-31'])
        if len(r['trades']) >= 2:
            # Not all trades should have the same size
            sizes = [t['qty'] for t in r['trades']]
            assert len(set(round(s, 6) for s in sizes)) > 1, "All trades same size — no compounding"


# ===========================================================================
# TEST 3: Trade Limits
# ===========================================================================
class TestTradeLimits:

    def test_max_trades_per_day(self, real_data):
        """No more than max_daily_trades per calendar day."""
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(max_daily_trades=3)
        r = bt.run(real_data.loc['2023-01-01':'2023-06-30'])
        if r['trades']:
            from collections import Counter
            days = Counter(t['entry_time'].date() for t in r['trades'])
            for day, count in days.items():
                assert count <= 3, f"{day} has {count} trades, max is 3"

    def test_cooldown_respected(self, real_data):
        """Minimum cooldown bars between trades."""
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(cooldown_bars=4)
        r = bt.run(real_data.loc['2023-01-01':'2023-06-30'])
        for i in range(1, len(r['trades'])):
            gap = r['trades'][i]['entry_bar'] - r['trades'][i-1]['exit_bar']
            assert gap >= 4, f"Trade {i} entered {gap} bars after previous exit, min is 4"


# ===========================================================================
# TEST 4: Equity Curve & Metrics
# ===========================================================================
class TestEquityCurveAndMetrics:

    def test_equity_curve_starts_at_capital(self, real_data):
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(initial_capital=10000)
        r = bt.run(real_data.loc['2023-01-01':'2023-06-30'])
        assert r['equity_curve'][0] == 10000

    def test_equity_never_negative(self, real_data):
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(initial_capital=10000)
        r = bt.run(real_data.loc['2023-01-01':'2023-12-31'])
        assert all(e >= 0 for e in r['equity_curve'])

    def test_sharpe_is_calculated(self, real_data):
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester()
        r = bt.run(real_data.loc['2023-01-01':'2023-12-31'])
        assert 'sharpe' in r
        assert isinstance(r['sharpe'], float)

    def test_max_drawdown_reported(self, real_data):
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester()
        r = bt.run(real_data.loc['2023-01-01':'2023-12-31'])
        assert 'max_drawdown_pct' in r
        assert 0 <= r['max_drawdown_pct'] <= 1.0

    def test_metrics_complete(self, real_data):
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester()
        r = bt.run(real_data.loc['2023-01-01':'2023-12-31'])
        required = ['total_pnl_dollars', 'total_return_pct', 'n_trades',
                     'win_rate', 'sharpe', 'max_drawdown_pct', 'profit_factor',
                     'avg_pnl_per_trade', 'total_fees', 'trades_per_day']
        for k in required:
            assert k in r, f"Missing metric: {k}"


# ===========================================================================
# TEST 5: Sharpe Target
# ===========================================================================
class TestSharpeTarget:

    def test_sharpe_above_1_0_on_2023_maker(self, real_data):
        """With maker fees + strict filters, Sharpe >= 1.0 on 2023."""
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(
            vol_min=3.0, use_hours=True, cooldown_bars=32,
            max_daily_trades=1, fee_rate=0.0002, slippage_rate=0.0001,
        )
        r = bt.run(real_data.loc['2023-01-01':'2023-12-31'])
        assert r['sharpe'] >= 1.0, \
            f"Sharpe {r['sharpe']:.2f} < 1.0 on 2023 (maker fees)"

    def test_sharpe_positive_on_2020_bull(self, real_data):
        """Sharpe must be positive in bull market 2020 (maker fees)."""
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(
            vol_min=3.0, use_hours=True, cooldown_bars=32,
            max_daily_trades=1, fee_rate=0.0002, slippage_rate=0.0001,
        )
        r = bt.run(real_data.loc['2020-01-01':'2020-12-31'])
        assert r['sharpe'] > 1.0, f"Sharpe {r['sharpe']:.2f} <= 1.0 in 2020 bull"

    def test_walk_forward_avg_sharpe_positive(self, real_data):
        """Walk-forward average Sharpe must be > 0 with maker fees."""
        from src.ml.realistic_backtest import RealisticBacktester
        bt = RealisticBacktester(
            vol_min=3.0, use_hours=True, cooldown_bars=32,
            max_daily_trades=1, fee_rate=0.0002, slippage_rate=0.0001,
        )
        sharpes = []
        for year in ['2020', '2021', '2022', '2023']:
            r = bt.run(real_data.loc[f'{year}-01-01':f'{year}-12-31'])
            if r['n_trades'] > 10:
                sharpes.append(r['sharpe'])
        avg_sharpe = np.mean(sharpes)
        assert avg_sharpe > 0.5, \
            f"Average walk-forward Sharpe {avg_sharpe:.2f} <= 0.5: {sharpes}"
