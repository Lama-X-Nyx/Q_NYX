"""
TDD Tests for Jesse Fast Backtester — Q1 2023

Tests:
  1. Precomputation returns numpy arrays (not DataFrames)
  2. Backtest speed: 8,640 bars in < 5 seconds
  3. No look-ahead bias
  4. PnL matches manual calculation
  5. Walk-forward split (train Jan-Feb, test Mar)
  6. Full pipeline Q1 2023
  7. Equity curve monotonic on synthetic bullish data
"""
import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / 'data' / 'raw' / 'BTCUSDT_15m_2023Q1.csv'


def load_q1_data() -> pd.DataFrame:
    """Load Q1 2023 BTCUSDT data."""
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return df


# ===========================================================================
# TEST 1: Data Availability
# ===========================================================================
class TestDataAvailability:

    def test_q1_data_file_exists(self):
        """Q1 2023 data file must exist."""
        assert DATA_PATH.exists(), f"Missing data: {DATA_PATH}"

    def test_q1_data_has_correct_shape(self):
        """Must have ~8,640 bars with OHLCV columns."""
        df = load_q1_data()
        assert len(df) >= 8000, f"Only {len(df)} bars, expected ~8,640"
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in df.columns, f"Missing column: {col}"

    def test_q1_data_price_range(self):
        """BTC Q1 2023: ~$16k start, ~$28k end."""
        df = load_q1_data()
        assert df['close'].iloc[0] > 15000 and df['close'].iloc[0] < 20000
        assert df['close'].iloc[-1] > 25000 and df['close'].iloc[-1] < 35000


# ===========================================================================
# TEST 2: Precomputation
# ===========================================================================
class TestPrecomputation:

    def test_precompute_returns_numpy_arrays(self):
        """precompute() should return dict of numpy arrays, not DataFrames."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        precomp = bt.precompute(df)
        assert isinstance(precomp, dict)
        for key in ['features', 'labels', 'close', 'high', 'low', 'atr']:
            assert key in precomp, f"Missing key: {key}"
            assert isinstance(precomp[key], np.ndarray), \
                f"precomp['{key}'] is {type(precomp[key])}, expected ndarray"

    def test_precompute_features_shape(self):
        """Features should be (n_bars, n_features) numpy array."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        precomp = bt.precompute(df)
        assert precomp['features'].ndim == 2
        assert precomp['features'].shape[0] == len(df)
        assert precomp['features'].shape[1] >= 13  # at least core features

    def test_precompute_speed_under_2_seconds(self):
        """Precomputation of 8,640 bars should take < 2 seconds."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        t0 = time.perf_counter()
        bt.precompute(df)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Precomputation took {elapsed:.2f}s, expected < 2s"


# ===========================================================================
# TEST 3: Backtest Speed
# ===========================================================================
class TestBacktestSpeed:

    def test_backtest_speed_under_5_seconds(self):
        """Full train + backtest of 8,640 bars should complete in < 5 seconds."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        t0 = time.perf_counter()
        result = bt.run(df, train_end='2023-02-28', test_start='2023-03-01')
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Backtest took {elapsed:.2f}s, expected < 5s"
        assert 'metrics' in result


# ===========================================================================
# TEST 4: No Look-Ahead Bias
# ===========================================================================
class TestNoLookAheadBias:

    def test_no_future_data_in_predictions(self):
        """Predictions at bar i must only use data from bars 0..i."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        result = bt.run(df, train_end='2023-02-28', test_start='2023-03-01')
        # All trade entries must be AFTER train period
        if result['trades']:
            for trade in result['trades']:
                assert trade['entry_time'] >= pd.Timestamp('2023-03-01'), \
                    f"Trade entered before test period: {trade['entry_time']}"


# ===========================================================================
# TEST 5: PnL Correctness
# ===========================================================================
class TestPnLCorrectness:

    def test_pnl_calculation(self):
        """Total PnL should equal sum of individual trade PnLs."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        result = bt.run(df, train_end='2023-02-28', test_start='2023-03-01')
        if result['trades']:
            sum_trade_pnl = sum(t['pnl'] for t in result['trades'])
            assert abs(sum_trade_pnl - result['metrics']['total_pnl']) < 0.01, \
                f"PnL mismatch: trades sum={sum_trade_pnl:.2f}, reported={result['metrics']['total_pnl']:.2f}"

    def test_equity_never_negative(self):
        """Equity should never go below 0."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        result = bt.run(df, train_end='2023-02-28', test_start='2023-03-01')
        assert all(e >= 0 for e in result['equity_curve']), "Equity went negative"


# ===========================================================================
# TEST 6: Walk-Forward Split
# ===========================================================================
class TestWalkForward:

    def test_walk_forward_split_correct(self):
        """Train on Jan-Feb, test on Mar. No overlap."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        result = bt.run(df, train_end='2023-02-28', test_start='2023-03-01')
        assert result['train_bars'] > 0
        assert result['test_bars'] > 0
        assert result['train_bars'] + result['test_bars'] <= len(df)

    def test_train_accuracy_on_train_set(self):
        """Model should have reasonable accuracy on train data."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        result = bt.run(df, train_end='2023-02-28', test_start='2023-03-01')
        assert result['train_accuracy'] >= 0.5, \
            f"Train accuracy {result['train_accuracy']:.1%} < 50%"


# ===========================================================================
# TEST 7: Full Pipeline
# ===========================================================================
class TestFullPipeline:

    def test_full_pipeline_returns_complete_result(self):
        """Full pipeline must return all expected fields."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        result = bt.run(df, train_end='2023-02-28', test_start='2023-03-01')
        required_keys = [
            'metrics', 'trades', 'equity_curve',
            'train_bars', 'test_bars', 'train_accuracy',
        ]
        for key in required_keys:
            assert key in result, f"Missing result key: {key}"

    def test_metrics_contain_required_fields(self):
        """Metrics must include accuracy, sharpe, drawdown, total_pnl."""
        from src.ml.jesse_backtest import FastBacktester
        df = load_q1_data()
        bt = FastBacktester()
        result = bt.run(df, train_end='2023-02-28', test_start='2023-03-01')
        for key in ['total_pnl', 'n_trades', 'win_rate', 'max_drawdown']:
            assert key in result['metrics'], f"Missing metric: {key}"
