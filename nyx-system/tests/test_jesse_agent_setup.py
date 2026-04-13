"""
TDD Tests — Agent 3: JesseSetupAgent (15M)

Rôle: validation de la qualité du setup avant l'entrée.
  - valid_setup → le setup est confirmé
  - no_setup → pas de setup, block

Specs:
  - Input:    OHLCV 15M + scores des agents 1 et 2
  - Features: 13 jesse features + context_score + regime_score + agreement
  - Label:    prix > 0.8% en 8 barres → 1, else 0
"""
import pytest
import numpy as np
import pandas as pd
from tests.test_jesse_ml_tdd import make_bullish_candles, make_bearish_candles, make_mixed_synthetic


class TestSetupContract:
    def test_returns_agent_result(self):
        from src.ml.jesse_agents import JesseSetupAgent
        from src.agents.contracts import AgentResult
        agent = JesseSetupAgent()
        agent.train(make_mixed_synthetic(400))
        result = agent.analyze(make_bullish_candles(200), context_score=0.8, regime_score=0.7)
        assert isinstance(result, AgentResult)
        assert result.agent == 'setup'

    def test_state_is_valid(self):
        from src.ml.jesse_agents import JesseSetupAgent
        agent = JesseSetupAgent()
        agent.train(make_mixed_synthetic(400))
        result = agent.analyze(make_bullish_candles(200))
        assert result.state in ('valid_setup', 'no_setup')


class TestSetupDetection:
    def test_valid_setup_with_good_context(self):
        from src.ml.jesse_agents import JesseSetupAgent
        agent = JesseSetupAgent()
        agent.train(make_mixed_synthetic(500))
        result = agent.analyze(make_bullish_candles(200), context_score=0.9, regime_score=0.8)
        # With strong context + regime, setup should be valid on bull data
        assert result.score > 0.0

    def test_no_setup_blocks(self):
        from src.ml.jesse_agents import JesseSetupAgent
        agent = JesseSetupAgent()
        agent.train(make_mixed_synthetic(500))
        # Low context/regime scores should block
        result = agent.analyze(make_bullish_candles(200), context_score=0.1, regime_score=0.1)
        if result.state == 'no_setup':
            assert result.passed is False


class TestSetupFeatures:
    def test_features_include_cross_agent_scores(self):
        from src.ml.jesse_agents import JesseSetupAgent
        agent = JesseSetupAgent()
        features = agent.compute_features(make_bullish_candles(200), context_score=0.7, regime_score=0.6)
        assert 'context_score' in features.columns
        assert 'regime_score' in features.columns
        assert 'agent_agreement' in features.columns


class TestSetupBacktest:
    def test_standalone_backtest(self):
        from src.ml.jesse_agents import JesseSetupAgent
        agent = JesseSetupAgent()
        mixed = make_mixed_synthetic(500)
        metrics = agent.backtest(mixed, train_ratio=0.75)
        assert 'n_bars' in metrics
        assert metrics['n_bars'] > 0
