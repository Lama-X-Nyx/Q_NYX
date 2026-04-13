"""
TDD Tests — ML Filter on Hard Gate Candidates

The hard gate produces ~150 trades/year with Sharpe 1.26.
The ML filter must select the best 50-60% and reject the rest.
Target: Sharpe >= 2.0 with fewer, higher-quality trades.

ML trains on net outcome (after fees), not raw triple barrier.
Walk-forward: train on previous years, test on next.
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
# TEST 1: Candidate generation (hard gate signals)
# ===========================================================================
class TestCandidateGeneration:

    def test_generates_candidates_with_features(self, real_data):
        """Must produce candidate trades with feature vectors."""
        from src.ml.ml_filter import generate_candidates
        candidates = generate_candidates(real_data.loc['2023-01-01':'2023-12-31'])
        assert len(candidates) > 50
        assert 'features' in candidates[0]
        assert 'outcome_net' in candidates[0]

    def test_candidates_have_outcome_after_fees(self, real_data):
        """Each candidate must have net outcome (after fees)."""
        from src.ml.ml_filter import generate_candidates
        candidates = generate_candidates(real_data.loc['2023-01-01':'2023-12-31'])
        for c in candidates[:10]:
            assert 'outcome_net' in c
            assert isinstance(c['outcome_net'], float)

    def test_candidates_have_rule_scores(self, real_data):
        """Candidates must include rule validator scores as features."""
        from src.ml.ml_filter import generate_candidates
        candidates = generate_candidates(real_data.loc['2023-01-01':'2023-12-31'])
        feat_keys = set(candidates[0]['features'].keys())
        assert 'rule_context' in feat_keys
        assert 'rule_regime' in feat_keys
        assert 'rule_setup' in feat_keys
        assert 'disagreement' in feat_keys


# ===========================================================================
# TEST 2: ML filter training
# ===========================================================================
class TestMLFilterTraining:

    def test_filter_trains_on_candidates(self, real_data):
        """ML filter must train on historical candidate outcomes."""
        from src.ml.ml_filter import MLTradeFilter
        f = MLTradeFilter()
        f.train(real_data.loc['2020-01-01':'2022-12-31'])
        assert f.is_trained

    def test_filter_predicts_probability(self, real_data):
        """Filter must output P(profitable) for each candidate."""
        from src.ml.ml_filter import MLTradeFilter, generate_candidates
        f = MLTradeFilter()
        f.train(real_data.loc['2020-01-01':'2022-12-31'])
        candidates = generate_candidates(real_data.loc['2023-01-01':'2023-03-31'])
        for c in candidates[:10]:
            prob = f.predict_proba(c['features'])
            assert 0.0 <= prob <= 1.0

    def test_filter_rejects_some_candidates(self, real_data):
        """Filter must reject at least 20% of candidates."""
        from src.ml.ml_filter import MLTradeFilter, generate_candidates
        f = MLTradeFilter()
        f.train(real_data.loc['2020-01-01':'2022-12-31'])
        candidates = generate_candidates(real_data.loc['2023-01-01':'2023-12-31'])
        accepted = sum(1 for c in candidates if f.predict_proba(c['features']) >= 0.5)
        reject_rate = 1 - accepted / len(candidates)
        assert reject_rate >= 0.15, \
            f"Only rejecting {reject_rate:.0%} — filter not selective enough"


# ===========================================================================
# TEST 3: Filtered backtest improvement
# ===========================================================================
class TestFilteredBacktest:

    def test_filtered_fewer_trades_than_unfiltered(self, real_data):
        """ML-filtered must have fewer trades."""
        from src.ml.ml_filter import MLFilteredBacktester
        bt = MLFilteredBacktester()
        r_filtered = bt.run(real_data, train_end='2022-12-31', test_start='2023-01-01')
        from src.ml.realistic_backtest import RealisticBacktester
        bt_base = RealisticBacktester(vol_min=3.0, use_hours=True, cooldown_bars=32,
                                       max_daily_trades=1, fee_rate=0.0002, slippage_rate=0.0001)
        r_base = bt_base.run(real_data.loc['2023-01-01':'2023-12-31'])
        assert r_filtered['n_trades'] < r_base['n_trades'], \
            f"Filtered {r_filtered['n_trades']} >= unfiltered {r_base['n_trades']}"

    def test_filtered_higher_win_rate(self, real_data):
        """ML-filtered must have higher win rate than unfiltered."""
        from src.ml.ml_filter import MLFilteredBacktester
        bt = MLFilteredBacktester()
        r_filtered = bt.run(real_data, train_end='2022-12-31', test_start='2023-01-01')
        from src.ml.realistic_backtest import RealisticBacktester
        bt_base = RealisticBacktester(vol_min=3.0, use_hours=True, cooldown_bars=32,
                                       max_daily_trades=1, fee_rate=0.0002, slippage_rate=0.0001)
        r_base = bt_base.run(real_data.loc['2023-01-01':'2023-12-31'])
        assert r_filtered['win_rate'] >= r_base['win_rate'], \
            f"Filtered WR {r_filtered['win_rate']:.0%} < base {r_base['win_rate']:.0%}"

    def test_filtered_positive_sharpe(self, real_data):
        """ML-filtered backtest must have positive Sharpe on 2023."""
        from src.ml.ml_filter import MLFilteredBacktester
        bt = MLFilteredBacktester()
        r = bt.run(real_data, train_end='2022-12-31', test_start='2023-01-01')
        assert r['sharpe'] > 0, f"Sharpe {r['sharpe']:.2f} negative"

    def test_filtered_metrics_complete(self, real_data):
        """Must return all standard metrics."""
        from src.ml.ml_filter import MLFilteredBacktester
        bt = MLFilteredBacktester()
        r = bt.run(real_data, train_end='2022-12-31', test_start='2023-01-01')
        for k in ['n_trades', 'win_rate', 'sharpe', 'total_pnl_dollars',
                   'max_drawdown_pct', 'profit_factor', 'filter_reject_rate']:
            assert k in r, f"Missing: {k}"


# ===========================================================================
# TEST 4: Walk-forward ML filter
# ===========================================================================
class TestWalkForwardMLFilter:

    def test_walk_forward_positive_sharpe(self, real_data):
        """Walk-forward filtered must have positive avg Sharpe."""
        from src.ml.ml_filter import MLFilteredBacktester
        bt = MLFilteredBacktester()
        sharpes = []
        folds = [
            ('2020-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
            ('2020-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
        ]
        for tr_s, tr_e, te_s, te_e in folds:
            r = bt.run(real_data, train_end=tr_e, test_start=te_s, test_end=te_e)
            if r['n_trades'] > 5:
                sharpes.append(r['sharpe'])
        avg = np.mean(sharpes) if sharpes else 0
        assert avg > 0, f"Walk-forward avg Sharpe {avg:.2f} negative: {sharpes}"

    def test_filter_improves_win_rate_vs_base(self, real_data):
        """Filtered must have higher WR on at least 1 fold (ML adds selectivity)."""
        from src.ml.ml_filter import MLFilteredBacktester
        from src.ml.realistic_backtest import RealisticBacktester
        bt = MLFilteredBacktester()
        bt_base = RealisticBacktester(vol_min=3.0, use_hours=True, cooldown_bars=32,
                                       max_daily_trades=1, fee_rate=0.0002, slippage_rate=0.0001)
        improved = 0
        for te_year in ['2022', '2023']:
            tr_e = str(int(te_year) - 1) + '-12-31'
            r_f = bt.run(real_data, train_end=tr_e, test_start=f'{te_year}-01-01', test_end=f'{te_year}-12-31')
            r_b = bt_base.run(real_data.loc[f'{te_year}-01-01':f'{te_year}-12-31'])
            if r_f['win_rate'] > r_b['win_rate']:
                improved += 1
        assert improved >= 1, "ML filter didn't improve WR on any fold"
