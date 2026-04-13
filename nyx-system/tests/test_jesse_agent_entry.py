"""
TDD Tests — Agent 4: JesseEntryAgent (15M)

Rôle: timing précis d'entrée.
  - ready → go
  - not_ready → block

Specs:
  - Input:    OHLCV 15M
  - Features: 13 jesse features stationnaires
  - Label:    triple barrier (+1/-1/0)
  - Confidence: prob_up > 0.45 AND prob_up > prob_down + 0.20
"""
import pytest
import numpy as np
import pandas as pd
from tests.test_jesse_ml_tdd import make_bullish_candles, make_bearish_candles, make_mixed_synthetic


class TestEntryContract:
    def test_returns_agent_result(self):
        from src.ml.jesse_agents import JesseEntryAgent
        from src.agents.contracts import AgentResult
        agent = JesseEntryAgent()
        agent.train(make_mixed_synthetic(500))
        result = agent.analyze(make_bullish_candles(200))
        assert isinstance(result, AgentResult)
        assert result.agent == 'entry'

    def test_state_is_valid(self):
        from src.ml.jesse_agents import JesseEntryAgent
        agent = JesseEntryAgent()
        agent.train(make_mixed_synthetic(500))
        result = agent.analyze(make_bullish_candles(200))
        assert result.state in ('ready', 'not_ready')

    def test_metadata_has_direction(self):
        from src.ml.jesse_agents import JesseEntryAgent
        agent = JesseEntryAgent()
        agent.train(make_mixed_synthetic(500))
        result = agent.analyze(make_bullish_candles(200))
        assert 'direction' in result.metadata
        assert result.metadata['direction'] in (-1, 0, 1)

    def test_metadata_has_probabilities(self):
        from src.ml.jesse_agents import JesseEntryAgent
        agent = JesseEntryAgent()
        agent.train(make_mixed_synthetic(500))
        result = agent.analyze(make_bullish_candles(200))
        assert 'p_up' in result.metadata
        assert 'p_down' in result.metadata


class TestEntryDetection:
    def test_ready_on_strong_trend(self):
        from src.ml.jesse_agents import JesseEntryAgent
        agent = JesseEntryAgent()
        agent.train(make_mixed_synthetic(600))
        result = agent.analyze(make_bullish_candles(200))
        # On strong bull, entry should have an opinion
        assert result.score > 0.0

    def test_confidence_threshold_respected(self):
        from src.ml.jesse_agents import JesseEntryAgent
        agent = JesseEntryAgent()
        agent.train(make_mixed_synthetic(600))
        result = agent.analyze(make_bullish_candles(200))
        if result.state == 'ready':
            # If ready, score must be meaningful
            assert result.score >= 0.3


class TestEntryBacktest:
    def test_standalone_backtest(self):
        from src.ml.jesse_agents import JesseEntryAgent
        agent = JesseEntryAgent()
        mixed = make_mixed_synthetic(500)
        metrics = agent.backtest(mixed, train_ratio=0.75)
        assert 'accuracy' in metrics
        assert 'n_bars' in metrics
        assert metrics['n_bars'] > 0

    def test_backtest_has_direction_distribution(self):
        from src.ml.jesse_agents import JesseEntryAgent
        agent = JesseEntryAgent()
        mixed = make_mixed_synthetic(500)
        metrics = agent.backtest(mixed, train_ratio=0.75)
        assert 'pct_bullish' in metrics
        assert 'pct_bearish' in metrics
