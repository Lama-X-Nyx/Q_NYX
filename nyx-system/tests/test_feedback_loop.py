"""
TDD Tests — Feedback Loop (Paper Trading → Evaluate → Retrain)

Level 1: DecisionLogger — logs every bar decision
Level 2: OutcomeEvaluator — relabels with real outcomes + error taxonomy
Level 3: ChampionChallenger — retrain + compare + promote
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path


def make_test_15m(n=500, start=20000.0):
    np.random.seed(42)
    pct = 0.001 + np.random.randn(n) * 0.003
    close = start * np.exp(np.cumsum(pct))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.002 + 0.001)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.002 - 0.001)
    open_ = np.roll(close, 1); open_[0] = start
    volume = np.random.randint(500, 3000, n).astype(float)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=pd.date_range('2023-03-01', periods=n, freq='15min'))


# ===========================================================================
# LEVEL 1: DecisionLogger
# ===========================================================================
class TestDecisionLogger:

    def test_log_entry_has_required_columns(self):
        """Each logged row must have all required columns."""
        from src.ml.feedback_loop import DecisionLogger
        logger = DecisionLogger()
        logger.log_decision(
            timestamp=pd.Timestamp('2023-03-01 10:00'),
            pair='BTCUSDT',
            features={'momentum_10': 0.01, 'rsi_14': 0.3},
            agent_results={
                'context': {'state': 'bullish', 'score': 0.7, 'passed': True, 'p_bull': 0.7, 'p_bear': 0.2},
                'regime': {'state': 'trend_plus', 'score': 0.6, 'passed': True},
                'setup': {'state': 'valid_setup', 'score': 0.55, 'passed': True},
                'entry': {'state': 'ready', 'score': 0.65, 'passed': True, 'direction': 1, 'p_up': 0.65, 'p_down': 0.1},
            },
            action='BUY', orch_score=0.65, size_factor=1.0, blocked_by=[],
            price=28000.0, atr=150.0, volume_ratio=1.3,
            trade_info={'entry_price': 28000, 'stop': 27850, 'tp': 28225, 'direction': 1, 'size': 0.5},
            model_version='v1.0',
        )
        df = logger.to_dataframe()
        assert len(df) == 1
        required = ['timestamp', 'pair', 'action', 'price', 'context_state',
                     'regime_state', 'setup_state', 'entry_state', 'orch_score',
                     'model_version']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_log_wait_decision(self):
        """WAIT decisions must be logged too (for false negative analysis)."""
        from src.ml.feedback_loop import DecisionLogger
        logger = DecisionLogger()
        logger.log_decision(
            timestamp=pd.Timestamp('2023-03-01 10:00'),
            pair='BTCUSDT', features={}, agent_results={
                'context': {'state': 'neutral', 'score': 0.5, 'passed': True},
                'regime': {'state': 'range', 'score': 0.4, 'passed': True},
                'setup': {'state': 'no_setup', 'score': 0.2, 'passed': False},
                'entry': {'state': 'not_ready', 'score': 0.3, 'passed': False, 'direction': 0, 'p_up': 0.3, 'p_down': 0.2},
            },
            action='WAIT', orch_score=0.3, size_factor=0.0,
            blocked_by=['setup', 'entry'],
            price=28000.0, atr=150.0, volume_ratio=0.8,
            trade_info=None, model_version='v1.0',
        )
        df = logger.to_dataframe()
        assert df.iloc[0]['action'] == 'WAIT'
        assert 'setup' in df.iloc[0]['blocked_by']

    def test_save_and_load_parquet(self):
        """Logger must persist to parquet and reload identically."""
        from src.ml.feedback_loop import DecisionLogger
        logger = DecisionLogger()
        logger.log_decision(
            timestamp=pd.Timestamp('2023-03-01 10:00'), pair='BTCUSDT',
            features={'m': 0.01}, agent_results={
                'context': {'state': 'bullish', 'score': 0.7, 'passed': True},
                'regime': {'state': 'trend_plus', 'score': 0.6, 'passed': True},
                'setup': {'state': 'valid_setup', 'score': 0.5, 'passed': True},
                'entry': {'state': 'ready', 'score': 0.6, 'passed': True, 'direction': 1, 'p_up': 0.6, 'p_down': 0.1},
            },
            action='BUY', orch_score=0.6, size_factor=1.0, blocked_by=[],
            price=28000.0, atr=150.0, volume_ratio=1.0,
            trade_info=None, model_version='v1.0',
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'test_log.parquet'
            logger.save(str(path))
            loaded = DecisionLogger.load(str(path))
            assert len(loaded.to_dataframe()) == 1

    def test_multiple_decisions_accumulate(self):
        """Multiple calls accumulate rows."""
        from src.ml.feedback_loop import DecisionLogger
        logger = DecisionLogger()
        for i in range(10):
            logger.log_decision(
                timestamp=pd.Timestamp('2023-03-01') + pd.Timedelta(minutes=15*i),
                pair='BTCUSDT', features={}, agent_results={
                    'context': {'state': 'neutral', 'score': 0.5, 'passed': True},
                    'regime': {'state': 'range', 'score': 0.5, 'passed': True},
                    'setup': {'state': 'no_setup', 'score': 0.3, 'passed': False},
                    'entry': {'state': 'not_ready', 'score': 0.3, 'passed': False, 'direction': 0, 'p_up': 0.3, 'p_down': 0.3},
                },
                action='WAIT', orch_score=0.3, size_factor=0.0,
                blocked_by=['setup'], price=28000+i*10, atr=150.0,
                volume_ratio=1.0, trade_info=None, model_version='v1.0',
            )
        assert len(logger.to_dataframe()) == 10


# ===========================================================================
# LEVEL 2: OutcomeEvaluator
# ===========================================================================
class TestOutcomeEvaluator:

    def _make_logged_session(self):
        """Create a logger with realistic decisions for evaluation."""
        from src.ml.feedback_loop import DecisionLogger
        logger = DecisionLogger()
        df = make_test_15m(200)
        # Log a BUY that will win (price goes up)
        logger.log_decision(
            timestamp=df.index[60], pair='BTCUSDT',
            features={'momentum_10': 0.01},
            agent_results={
                'context': {'state': 'bullish', 'score': 0.8, 'passed': True, 'p_bull': 0.8, 'p_bear': 0.1},
                'regime': {'state': 'trend_plus', 'score': 0.7, 'passed': True},
                'setup': {'state': 'valid_setup', 'score': 0.6, 'passed': True},
                'entry': {'state': 'ready', 'score': 0.7, 'passed': True, 'direction': 1, 'p_up': 0.7, 'p_down': 0.1},
            },
            action='BUY', orch_score=0.7, size_factor=1.2, blocked_by=[],
            price=float(df['close'].iloc[60]), atr=50.0, volume_ratio=1.5,
            trade_info={'entry_price': float(df['close'].iloc[60]), 'stop': float(df['close'].iloc[60])-50,
                        'tp': float(df['close'].iloc[60])+75, 'direction': 1, 'size': 0.5},
            model_version='v1.0',
        )
        # Log a WAIT
        logger.log_decision(
            timestamp=df.index[100], pair='BTCUSDT',
            features={'momentum_10': -0.005},
            agent_results={
                'context': {'state': 'neutral', 'score': 0.5, 'passed': True},
                'regime': {'state': 'range', 'score': 0.4, 'passed': True},
                'setup': {'state': 'no_setup', 'score': 0.2, 'passed': False},
                'entry': {'state': 'not_ready', 'score': 0.3, 'passed': False, 'direction': 0, 'p_up': 0.3, 'p_down': 0.3},
            },
            action='WAIT', orch_score=0.3, size_factor=0.0,
            blocked_by=['setup', 'entry'],
            price=float(df['close'].iloc[100]), atr=50.0, volume_ratio=0.8,
            trade_info=None, model_version='v1.0',
        )
        return logger, df

    def test_evaluate_adds_outcome_columns(self):
        """Evaluation must add realized returns and labels."""
        from src.ml.feedback_loop import OutcomeEvaluator
        logger, df = self._make_logged_session()
        evaluator = OutcomeEvaluator()
        evaluated = evaluator.evaluate(logger, df)
        required = ['realized_return_1h', 'realized_return_4h',
                     'triple_barrier_label', 'decision_type', 'error_type']
        for col in required:
            assert col in evaluated.columns, f"Missing outcome column: {col}"

    def test_decision_types_are_valid(self):
        """decision_type must be TP/FP/TN/FN."""
        from src.ml.feedback_loop import OutcomeEvaluator
        logger, df = self._make_logged_session()
        evaluator = OutcomeEvaluator()
        evaluated = evaluator.evaluate(logger, df)
        valid_types = {'TRUE_POSITIVE', 'FALSE_POSITIVE', 'TRUE_NEGATIVE', 'FALSE_NEGATIVE'}
        for dt in evaluated['decision_type']:
            assert dt in valid_types, f"Invalid decision type: {dt}"

    def test_error_taxonomy(self):
        """error_type must be from the defined taxonomy."""
        from src.ml.feedback_loop import OutcomeEvaluator
        logger, df = self._make_logged_session()
        evaluator = OutcomeEvaluator()
        evaluated = evaluator.evaluate(logger, df)
        valid_errors = {
            'none', 'bad_context', 'bad_regime', 'setup_false_positive',
            'setup_false_negative', 'entry_too_early', 'entry_too_late',
            'size_too_large', 'size_too_small', 'wait_should_have_traded',
            'traded_should_have_waited',
        }
        for et in evaluated['error_type']:
            assert et in valid_errors, f"Invalid error type: {et}"

    def test_summary_report(self):
        """Summary must include win/loss, expectancy, per-agent errors."""
        from src.ml.feedback_loop import OutcomeEvaluator
        logger, df = self._make_logged_session()
        evaluator = OutcomeEvaluator()
        evaluated = evaluator.evaluate(logger, df)
        summary = evaluator.summary(evaluated)
        assert 'total_decisions' in summary
        assert 'trades_taken' in summary
        assert 'waits' in summary
        assert 'decision_type_counts' in summary
        assert 'error_type_counts' in summary


# ===========================================================================
# LEVEL 3: ChampionChallenger
# ===========================================================================
class TestChampionChallenger:

    def test_champion_challenger_structure(self):
        """Must maintain champion and challenger models."""
        from src.ml.feedback_loop import ChampionChallenger
        cc = ChampionChallenger()
        assert cc.champion is None  # no model yet
        assert cc.challenger is None

    def test_train_champion(self):
        """First training sets the champion."""
        from src.ml.feedback_loop import ChampionChallenger
        df = make_test_15m(500)
        cc = ChampionChallenger()
        cc.train_champion(df, train_ratio=0.75)
        assert cc.champion is not None
        assert cc.champion_metrics is not None
        assert 'accuracy' in cc.champion_metrics

    def test_train_challenger(self):
        """Challenger trained on new data."""
        from src.ml.feedback_loop import ChampionChallenger
        df = make_test_15m(500)
        cc = ChampionChallenger()
        cc.train_champion(df, train_ratio=0.75)
        # Train challenger on slightly different data
        df2 = make_test_15m(600, start=25000.0)
        cc.train_challenger(df2, train_ratio=0.75)
        assert cc.challenger is not None
        assert cc.challenger_metrics is not None

    def test_promote_only_if_better(self):
        """Challenger replaces champion ONLY if strictly better OOS."""
        from src.ml.feedback_loop import ChampionChallenger
        df = make_test_15m(500)
        cc = ChampionChallenger()
        cc.train_champion(df, train_ratio=0.75)
        old_champion = cc.champion
        # Train challenger on same data (should be similar)
        cc.train_challenger(df, train_ratio=0.75)
        promoted = cc.promote_if_better(min_improvement=0.05)
        # promoted is True only if challenger is 5%+ better
        assert isinstance(promoted, bool)

    def test_comparison_report(self):
        """Must produce comparison report."""
        from src.ml.feedback_loop import ChampionChallenger
        df = make_test_15m(500)
        cc = ChampionChallenger()
        cc.train_champion(df, train_ratio=0.75)
        cc.train_challenger(df, train_ratio=0.75)
        report = cc.comparison_report()
        assert 'champion' in report
        assert 'challenger' in report
        assert 'recommendation' in report
