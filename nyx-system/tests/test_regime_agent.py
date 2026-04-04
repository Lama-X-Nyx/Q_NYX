"""
Test Regime Agent
"""

import sys
sys.path.insert(0, '.')

import pytest
import pandas as pd
import numpy as np
from src.agents.regime_agent import RegimeAgent
from src.agents.contracts import AgentResult


class TestRegimeAgent:
    """Test suite for Regime Agent"""
    
    @pytest.fixture
    def config(self):
        return {
            'fractal': {
                'regime_tf': '1h'
            },
            'strategy': {
                'mtf_conditions': {
                    'sdc_min': 5.0,
                    'stability_4h_min': 0.60
                }
            }
        }
    
    @pytest.fixture
    def agent(self, config):
        return RegimeAgent(config)
    
    @pytest.fixture
    def trend_data(self):
        """Create trend data for HSMM"""
        dates = pd.date_range('2023-01-01', periods=200, freq='1h')
        prices = np.linspace(40000, 45000, 200)
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.rand(200) * 1000
        }, index=dates)
    
    def test_returns_valid_agent_result(self, agent, trend_data):
        """Test that agent returns valid AgentResult"""
        result = agent.analyze(trend_data, context_state='bullish')
        
        assert isinstance(result, AgentResult)
        assert result.agent == 'regime'
        assert isinstance(result.state, str)
        assert isinstance(result.score, float)
        assert isinstance(result.passed, bool)
    
    def test_uses_hsmm(self, agent, trend_data):
        """Test that agent uses real HSMM"""
        result = agent.analyze(trend_data)
        
        # Check HSMM metadata
        assert 'sdc' in result.metadata
        assert 'stability' in result.metadata
        assert 'hsmm_states' in result.metadata
        
        # Check HSMM states structure
        hsmm_states = result.metadata['hsmm_states']
        assert 'Trend+' in hsmm_states
        assert 'Range' in hsmm_states
        assert 'Trend-' in hsmm_states
    
    def test_state_valid(self, agent, trend_data):
        """Test that state is valid"""
        result = agent.analyze(trend_data)
        
        assert result.state in ['trend_plus', 'range', 'trend_minus']
    
    def test_score_bounded(self, agent, trend_data):
        """Test that score is in [0, 1]"""
        result = agent.analyze(trend_data)
        
        assert 0 <= result.score <= 1
    
    def test_context_alignment(self, agent, trend_data):
        """Test context alignment check"""
        # Bullish context should align with trend_plus
        result = agent.analyze(trend_data, context_state='bullish')
        
        assert 'context_aligned' in result.metadata
    
    def test_insufficient_data(self, agent):
        """Test handling of insufficient data"""
        small_data = pd.DataFrame({
            'open': [100] * 50,
            'high': [102] * 50,
            'low': [99] * 50,
            'close': [101] * 50,
            'volume': [1000] * 50
        })
        
        result = agent.analyze(small_data)
        
        assert result.passed == False
        assert 'Insufficient' in result.reason


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
