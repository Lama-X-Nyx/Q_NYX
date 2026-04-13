"""
TDD Tests — Edge Validation with Walk-Forward (Anti-Overfit)

Validates that the edge is REAL across multiple market regimes,
not overfitted to one period.

Edge stack tested:
  1. Trend alignment (EMA9 > EMA21 > EMA50)
  2. Volume > 1.5x average (institutional participation)
  3. Hours 8-18 UTC (US/EU sessions)
  4. TP=1.5x ATR, SL=1.0x ATR

Walk-forward: train 6 months, test 3 months, rolling.
Anti-overfit: parameters fixed BEFORE seeing test data.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_15M = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf' / 'BTCUSDT_15m.csv'


def load_real_data():
    df = pd.read_csv(DATA_15M)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df


@pytest.fixture(scope='module')
def real_data():
    return load_real_data()


# ===========================================================================
# TEST 1: EdgeStrategy contract
# ===========================================================================
class TestEdgeStrategyContract:

    def test_edge_returns_dict_with_required_keys(self, real_data):
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy()
        result = es.backtest(real_data.loc['2023-01-01':'2023-03-31'])
        required = ['trades', 'n_trades', 'win_rate', 'ev_per_trade_atr',
                     'total_pnl_pts', 'quarters_positive']
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_edge_win_rate_in_range(self, real_data):
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy()
        result = es.backtest(real_data.loc['2023-01-01':'2023-03-31'])
        assert 0.0 <= result['win_rate'] <= 1.0

    def test_edge_trades_have_direction(self, real_data):
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy()
        result = es.backtest(real_data.loc['2023-01-01':'2023-03-31'])
        if result['trades']:
            for t in result['trades'][:5]:
                assert t['direction'] in (1, -1)


# ===========================================================================
# TEST 2: Walk-forward validation
# ===========================================================================
class TestWalkForwardValidation:

    def test_walk_forward_has_14_quarters(self, real_data):
        """Walk-forward should cover 14 test quarters (2020-Q3 to 2023-Q4)."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy()
        wf = es.walk_forward(real_data)
        assert len(wf['quarters']) >= 12, \
            f"Only {len(wf['quarters'])} quarters, expected 12+"

    def test_walk_forward_no_overlap(self, real_data):
        """No train/test overlap in any quarter."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy()
        wf = es.walk_forward(real_data)
        for q in wf['quarters']:
            assert q['test_start'] > q['train_end'], \
                f"Overlap: train ends {q['train_end']}, test starts {q['test_start']}"

    def test_volume_edge_positive_most_quarters(self, real_data):
        """Volume >1.5x edge must be positive in >75% of quarters."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        wf = es.walk_forward(real_data)
        pos_q = sum(1 for q in wf['quarters'] if q['pnl_pts'] > 0)
        total_q = len(wf['quarters'])
        assert pos_q / total_q >= 0.75, \
            f"Only {pos_q}/{total_q} quarters positive — edge is not robust"

    def test_baseline_has_positive_ev(self, real_data):
        """Baseline (no filter) must have positive EV across all quarters."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy()
        wf = es.walk_forward(real_data)
        assert wf['total_ev_atr'] > 0, \
            f"Total EV {wf['total_ev_atr']:.4f} is negative — no edge"

    def test_volume_edge_better_ev_than_baseline(self, real_data):
        """Volume filter must improve EV vs baseline."""
        from src.ml.edge_strategy import EdgeStrategy
        base = EdgeStrategy().walk_forward(real_data)
        vol = EdgeStrategy(vol_min=1.5).walk_forward(real_data)
        assert vol['total_ev_atr'] > base['total_ev_atr'], \
            f"Volume EV {vol['total_ev_atr']:.4f} <= baseline {base['total_ev_atr']:.4f}"


# ===========================================================================
# TEST 3: Anti-overfit checks
# ===========================================================================
class TestAntiOverfit:

    def test_edge_survives_bear_market(self, real_data):
        """Edge must be positive in at least 1 bear quarter (2022)."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        wf = es.walk_forward(real_data)
        bear_qs = [q for q in wf['quarters'] if '2022' in q['test_start']]
        bear_positive = sum(1 for q in bear_qs if q['pnl_pts'] > 0)
        assert bear_positive >= 1, "Edge fails entirely in bear market — likely overfitted"

    def test_edge_survives_bull_market(self, real_data):
        """Edge must be positive in at least 2 bull quarters (2021, 2023)."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        wf = es.walk_forward(real_data)
        bull_qs = [q for q in wf['quarters'] if q['test_start'][:4] in ('2021', '2023')]
        bull_positive = sum(1 for q in bull_qs if q['pnl_pts'] > 0)
        assert bull_positive >= 2, "Edge fails in bull markets"

    def test_no_single_quarter_dominates(self, real_data):
        """No single quarter should account for >40% of total PnL."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        wf = es.walk_forward(real_data)
        total = abs(wf['total_pnl_pts'])
        if total > 0:
            for q in wf['quarters']:
                pct = abs(q['pnl_pts']) / total
                assert pct < 0.40, \
                    f"Quarter {q['test_start']} accounts for {pct:.0%} of PnL — concentration risk"

    def test_edge_consistent_across_halves(self, real_data):
        """Edge in first half (2020-2021) vs second half (2022-2023) must be same sign."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        wf = es.walk_forward(real_data)
        first = sum(q['pnl_pts'] for q in wf['quarters'] if q['test_start'] < '2022')
        second = sum(q['pnl_pts'] for q in wf['quarters'] if q['test_start'] >= '2022')
        assert first > 0 and second > 0, \
            f"Inconsistent: first half PnL={first:+,.0f}, second half PnL={second:+,.0f}"
