"""
Test Context Agent
"""

import sys
sys.path.insert(0, '.')

import pytest
import pandas as pd
import numpy as np
from src.agents.context_agent import ContextAgent
from src.agents.contracts import AgentResult


class TestContextAgent:
    """Test suite for Context Agent"""
    
    @pytest.fixture
    def config(self):
        return {
            'fractal': {
                'context_tf': '1d'
            }
        }
    
    @pytest.fixture
    def agent(self, config):
        return ContextAgent(config)
    
    @pytest.fixture
    def bullish_data(self):
        """Create uptrend data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='1D')
        prices = np.linspace(40000, 50000, 100)
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.rand(100) * 1000
        }, index=dates)
    
    @pytest.fixture
    def bearish_data(self):
        """Create downtrend data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='1D')
        prices = np.linspace(50000, 40000, 100)
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.rand(100) * 1000
        }, index=dates)
    
    @pytest.fixture
    def neutral_data(self):
        """Create sideways data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='1D')
        prices = np.ones(100) * 45000 + np.random.randn(100) * 100
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.rand(100) * 1000
        }, index=dates)
    
    def test_returns_valid_agent_result(self, agent, bullish_data):
        """Test that agent returns valid AgentResult"""
        result = agent.analyze(bullish_data)
        
        assert isinstance(result, AgentResult)
        assert result.agent == 'context'
        assert isinstance(result.state, str)
        assert isinstance(result.score, float)
        assert isinstance(result.passed, bool)
        assert isinstance(result.reason, str)
        assert isinstance(result.metadata, dict)
    
    def test_score_bounded(self, agent, bullish_data):
        """Test that score is in [0, 1]"""
        result = agent.analyze(bullish_data)
        
        assert 0 <= result.score <= 1
    
    def test_detects_bullish(self, agent, bullish_data):
        """Test that agent detects bullish trend"""
        result = agent.analyze(bullish_data)
        
        assert result.state == 'bullish'
        assert result.passed == True
        assert result.score > 0.5
    
    def test_detects_bearish(self, agent, bearish_data):
        """Test that agent detects bearish trend"""
        result = agent.analyze(bearish_data)
        
        assert result.state == 'bearish'
        assert result.passed == True
        assert result.score > 0.5
    
    def test_detects_neutral(self, agent, neutral_data):
        """Test that agent detects neutral/range"""
        result = agent.analyze(neutral_data)
        
        assert result.state == 'neutral'
        assert result.passed == False  # Neutral blocks
    
    def test_insufficient_data(self, agent):
        """Test handling of insufficient data"""
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        result = agent.analyze(small_data)
        
        assert result.passed == False
        assert 'Insufficient' in result.reason


    # ------------------------------------------------------------------
    # P1 tests: HSMM projection
    # ------------------------------------------------------------------

    @pytest.fixture
    def bullish_regime_4h_result(self):
        """Fake 4H regime result with Trend+ dominant"""
        return AgentResult(
            agent='regime',
            state='trend_plus',
            score=0.72,
            passed=True,
            reason='test',
            metadata={
                'hsmm_states': {'Trend+': 0.72, 'Range': 0.18, 'Trend-': 0.10},
                'transition_matrix': [
                    [0.75, 0.125, 0.125],
                    [0.125, 0.75, 0.125],
                    [0.125, 0.125, 0.75]
                ]
            }
        )

    @pytest.fixture
    def bearish_regime_4h_result(self):
        """Fake 4H regime result with Trend- dominant"""
        return AgentResult(
            agent='regime',
            state='trend_minus',
            score=0.68,
            passed=True,
            reason='test',
            metadata={
                'hsmm_states': {'Trend+': 0.08, 'Range': 0.22, 'Trend-': 0.70},
                'transition_matrix': [
                    [0.75, 0.125, 0.125],
                    [0.125, 0.75, 0.125],
                    [0.125, 0.125, 0.75]
                ]
            }
        )

    def test_hsmm_projection_used_when_regime_provided(
        self, agent, bullish_data, bullish_regime_4h_result
    ):
        """When 4H regime result provided, use HSMM projection for Intent_1D"""
        result = agent.analyze(bullish_data, regime_4h_result=bullish_regime_4h_result)

        assert result.metadata.get('intent_method') == 'hsmm_projection', \
            f"Expected 'hsmm_projection', got '{result.metadata.get('intent_method')}'"

    def test_fallback_to_sma_when_no_regime(self, agent, bullish_data):
        """Without regime_4h_result, fall back to SMA heuristic"""
        result = agent.analyze(bullish_data, regime_4h_result=None)

        assert result.metadata.get('intent_method') == 'sma_heuristic'

    def test_hsmm_projection_bullish(self, agent, bullish_data, bullish_regime_4h_result):
        """Trend+ dominant in 4H → bullish Intent_1D"""
        result = agent.analyze(bullish_data, regime_4h_result=bullish_regime_4h_result)

        assert result.state == 'bullish'
        assert result.passed is True

    def test_hsmm_projection_bearish(self, agent, bullish_data, bearish_regime_4h_result):
        """Trend- dominant in 4H → bearish Intent_1D"""
        result = agent.analyze(bullish_data, regime_4h_result=bearish_regime_4h_result)

        assert result.state == 'bearish'
        assert result.passed is True

    def test_projected_distribution_in_metadata(
        self, agent, bullish_data, bullish_regime_4h_result
    ):
        """Projected distribution must appear in metadata"""
        result = agent.analyze(bullish_data, regime_4h_result=bullish_regime_4h_result)

        assert 'projected_distribution' in result.metadata
        pd_ = result.metadata['projected_distribution']
        assert set(pd_.keys()) == {'Trend+', 'Range', 'Trend-'}
        prob_sum = sum(pd_.values())
        assert abs(prob_sum - 1.0) < 0.01, f"Projected distribution sums to {prob_sum:.4f}"

    def test_hsmm_score_not_hardcoded(self, agent, bullish_data, bullish_regime_4h_result):
        """Score must be derived from projected probability, not a fixed constant"""
        result = agent.analyze(bullish_data, regime_4h_result=bullish_regime_4h_result)
        hardcoded = {0.5, 0.4, 0.72}
        if result.metadata.get('intent_method') == 'hsmm_projection':
            assert result.score not in hardcoded, \
                f"Score {result.score} looks like a hardcoded value"

    def test_projection_k_metadata(self, agent, bullish_data, bullish_regime_4h_result):
        """projection_k must appear in metadata"""
        result = agent.analyze(bullish_data, regime_4h_result=bullish_regime_4h_result)
        if result.metadata.get('intent_method') == 'hsmm_projection':
            assert 'projection_k' in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
