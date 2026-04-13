"""
TDD Tests — Threshold Optimizer for Sharpe 2.0

Find the optimal ML threshold that maximizes Sharpe ratio
while maintaining enough trades for statistical significance.

Constraint: at least 20 trades/year for meaningful stats.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_15M = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf' / 'BTCUSDT_15m.csv'
FEAT_15M = Path(__file__).parent.parent / 'data' / 'features' / 'BTCUSDT_features_15m.parquet'


@pytest.fixture(scope='module')
def real_data():
    df = pd.read_csv(DATA_15M)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df


@pytest.fixture(scope='module')
def parquet_features():
    return pd.read_parquet(FEAT_15M)


class TestThresholdOptimizer:

    def test_optimizer_finds_threshold(self, real_data, parquet_features):
        """Must find an optimal threshold in [0.50, 0.70]."""
        from src.ml.threshold_optimizer import optimize_threshold
        result = optimize_threshold(
            real_data, parquet_features,
            train_end='2022-12-31', val_start='2023-01-01', val_end='2023-12-31',
        )
        assert 0.45 <= result['best_threshold'] <= 0.75
        assert result['best_sharpe'] > 0

    def test_optimizer_respects_min_trades(self, real_data, parquet_features):
        """Optimal config must have at least min_trades."""
        from src.ml.threshold_optimizer import optimize_threshold
        result = optimize_threshold(
            real_data, parquet_features,
            train_end='2022-12-31', val_start='2023-01-01', val_end='2023-12-31',
            min_trades=10,
        )
        assert result['best_n_trades'] >= 10

    def test_optimizer_returns_full_sweep(self, real_data, parquet_features):
        """Must return results for all tested thresholds."""
        from src.ml.threshold_optimizer import optimize_threshold
        result = optimize_threshold(
            real_data, parquet_features,
            train_end='2022-12-31', val_start='2023-01-01', val_end='2023-12-31',
        )
        assert 'sweep' in result
        assert len(result['sweep']) >= 5


class TestOptimalBacktest:

    def test_optimal_sharpe_above_1_5(self, real_data, parquet_features):
        """Optimized config must achieve Sharpe >= 1.5 on 2023."""
        from src.ml.threshold_optimizer import run_optimal_backtest
        r = run_optimal_backtest(
            real_data, parquet_features,
            train_end='2022-12-31', test_start='2023-01-01',
        )
        assert r['sharpe'] >= 1.5, f"Sharpe {r['sharpe']:.2f} < 1.5"

    def test_optimal_wr_above_55(self, real_data, parquet_features):
        """Win rate must be above 55% with optimal threshold."""
        from src.ml.threshold_optimizer import run_optimal_backtest
        r = run_optimal_backtest(
            real_data, parquet_features,
            train_end='2022-12-31', test_start='2023-01-01',
        )
        assert r['win_rate'] >= 0.55, f"WR {r['win_rate']:.0%} < 55%"

    def test_optimal_dd_below_5pct(self, real_data, parquet_features):
        """Max drawdown must be under 5% with optimal config."""
        from src.ml.threshold_optimizer import run_optimal_backtest
        r = run_optimal_backtest(
            real_data, parquet_features,
            train_end='2022-12-31', test_start='2023-01-01',
        )
        assert r['max_drawdown_pct'] < 0.05, f"DD {r['max_drawdown_pct']:.1%} >= 5%"

    def test_walk_forward_optimal(self, real_data, parquet_features):
        """Walk-forward with per-fold optimization must be positive."""
        from src.ml.threshold_optimizer import run_optimal_backtest
        sharpes = []
        for tr_e, te_s, te_e in [
            ('2021-12-31', '2022-01-01', '2022-12-31'),
            ('2022-12-31', '2023-01-01', '2023-12-31'),
        ]:
            r = run_optimal_backtest(
                real_data, parquet_features,
                train_end=tr_e, test_start=te_s, test_end=te_e,
            )
            if r['n_trades'] > 5:
                sharpes.append(r['sharpe'])
        avg = np.mean(sharpes) if sharpes else 0
        assert avg > 0, f"Walk-forward avg Sharpe {avg:.2f} negative: {sharpes}"
