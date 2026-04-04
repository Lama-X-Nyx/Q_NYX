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


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
