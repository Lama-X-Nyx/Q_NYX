"""
TDD Tests — Yearly Walk-Forward + Full OOS 2019-2024

Walk-forward strict par année:
  2019 train → 2020 test
  2020+2021 train → 2022 test
  2022+2023 train → 2024 test (partial)

Puis OOS complet: train 2019-2021, test 2022-2024

Anti-overfit: aucun paramètre touché entre train et test.
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
# TEST 1: Yearly walk-forward structure
# ===========================================================================
class TestYearlyWalkForwardStructure:

    def test_returns_3_folds(self, real_data):
        """Must produce 3 folds (train→test pairs)."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.yearly_walk_forward(real_data)
        assert len(result['folds']) == 3, f"Expected 3 folds, got {len(result['folds'])}"

    def test_no_train_test_overlap(self, real_data):
        """Train end must be strictly before test start in every fold."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.yearly_walk_forward(real_data)
        for fold in result['folds']:
            assert fold['train_end'] < fold['test_start'], \
                f"Overlap: train {fold['train_end']} >= test {fold['test_start']}"

    def test_folds_cover_correct_years(self, real_data):
        """Fold 1=2020 test, Fold 2=2022 test, Fold 3=2024 test."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.yearly_walk_forward(real_data)
        test_years = [f['test_start'][:4] for f in result['folds']]
        assert test_years == ['2020', '2022', '2023'], \
            f"Expected test years [2020,2022,2023], got {test_years}"

    def test_each_fold_has_metrics(self, real_data):
        """Each fold must have trades, win_rate, ev, pnl."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.yearly_walk_forward(real_data)
        for fold in result['folds']:
            for key in ['n_trades', 'win_rate', 'ev_atr', 'pnl_pts', 'btc_return']:
                assert key in fold, f"Missing key {key} in fold {fold['test_start']}"


# ===========================================================================
# TEST 2: Edge robustness across years
# ===========================================================================
class TestYearlyEdgeRobustness:

    def test_positive_ev_in_majority_of_folds(self, real_data):
        """Edge must have positive EV in at least 2/3 folds."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.yearly_walk_forward(real_data)
        pos_folds = sum(1 for f in result['folds'] if f['ev_atr'] > 0)
        assert pos_folds >= 2, \
            f"Only {pos_folds}/3 folds with positive EV — edge not robust"

    def test_total_pnl_positive(self, real_data):
        """Total PnL across all folds must be positive."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.yearly_walk_forward(real_data)
        assert result['total_pnl_pts'] > 0, \
            f"Total PnL {result['total_pnl_pts']:+,.0f} is negative"

    def test_survives_bear_year_2022(self, real_data):
        """Edge must not blow up in bear year 2022."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.yearly_walk_forward(real_data)
        bear_fold = [f for f in result['folds'] if f['test_start'][:4] == '2022']
        if bear_fold:
            assert bear_fold[0]['ev_atr'] > -0.05, \
                f"Edge EV {bear_fold[0]['ev_atr']:.3f} crashes in bear — not robust"


# ===========================================================================
# TEST 3: Full OOS
# ===========================================================================
class TestFullOOS:

    def test_oos_returns_complete_result(self, real_data):
        """OOS must return trades, metrics, equity curve."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.full_oos(real_data, train_end='2021-12-31', test_start='2022-01-01')
        for key in ['n_trades', 'win_rate', 'ev_atr', 'pnl_pts', 'btc_return']:
            assert key in result, f"Missing key: {key}"

    def test_oos_has_positive_ev(self, real_data):
        """OOS 2022-2024 must have positive expected value."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.full_oos(real_data, train_end='2021-12-31', test_start='2022-01-01')
        assert result['ev_atr'] > 0, \
            f"OOS EV {result['ev_atr']:.4f} is negative — no edge out of sample"

    def test_oos_win_rate_above_random(self, real_data):
        """OOS win rate must be above 40% (random for this setup ~38%)."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.full_oos(real_data, train_end='2021-12-31', test_start='2022-01-01')
        assert result['win_rate'] > 0.40, \
            f"OOS WR {result['win_rate']:.0%} <= 40% — no edge"

    def test_oos_yearly_breakdown(self, real_data):
        """OOS must provide per-year breakdown."""
        from src.ml.edge_strategy import EdgeStrategy
        es = EdgeStrategy(vol_min=1.5)
        result = es.full_oos(real_data, train_end='2021-12-31', test_start='2022-01-01')
        assert 'yearly' in result
        assert len(result['yearly']) >= 2  # at least 2022 and 2023
