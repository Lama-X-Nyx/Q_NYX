"""
Test Setup Agent
"""

import sys
sys.path.insert(0, '.')

import pytest
import pandas as pd
import numpy as np
from src.agents.setup_agent import SetupAgent
from src.agents.contracts import AgentResult


class TestSetupAgent:
    """Test suite for Setup Agent"""
    
    @pytest.fixture
    def config(self):
        return {
            'fractal': {
                'setup_tf': '15m'
            },
            'strategy': {
                'mtf_conditions': {
                    'alignment_15m_min': 0.60
                },
                'smc': {
                    'ob_range_threshold': 0.015,
                    'fvg_threshold': 0.01
                }
            }
        }
    
    @pytest.fixture
    def agent(self, config):
        return SetupAgent(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        dates = pd.date_range('2023-01-01', periods=200, freq='15min')
        prices = np.linspace(40000, 41000, 200)
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.rand(200) * 1000
        }, index=dates)
    
    def test_returns_valid_agent_result(self, agent, sample_data):
        """Test that agent returns valid AgentResult"""
        result = agent.analyze(sample_data, context_state='bullish')
        
        assert isinstance(result, AgentResult)
        assert result.agent == 'setup'
        assert isinstance(result.state, str)
        assert isinstance(result.score, float)
        assert isinstance(result.passed, bool)
    
    def test_uses_smc_detector(self, agent, sample_data):
        """Test that agent uses real SMCDetector"""
        result = agent.analyze(sample_data, context_state='bullish')
        
        # Check SMC patterns in metadata
        assert 'patterns' in result.metadata
        patterns = result.metadata['patterns']
        
        assert 'bullish_ob' in patterns
        assert 'bearish_ob' in patterns
        assert 'bullish_fvg' in patterns
        assert 'bearish_fvg' in patterns
    
    def test_state_valid(self, agent, sample_data):
        """Test that state is valid"""
        result = agent.analyze(sample_data, context_state='bullish')
        
        valid_states = ['valid_setup', 'no_pattern', 'misaligned', 'mixed_signals', 
                       'pattern_found', 'no_bias']
        assert result.state in valid_states
    
    def test_score_bounded(self, agent, sample_data):
        """Test that score is in [0, 1]"""
        result = agent.analyze(sample_data, context_state='bullish')
        
        assert 0 <= result.score <= 1
    
    def test_alignment_check(self, agent, sample_data):
        """Test alignment with context"""
        result = agent.analyze(sample_data, context_state='bullish')
        
        assert 'alignment' in result.metadata
        assert 'context_state' in result.metadata
        assert result.metadata['context_state'] == 'bullish'
    
    def test_no_context_handling(self, agent, sample_data):
        """Test handling when no context provided"""
        result = agent.analyze(sample_data, context_state=None)
        
        # Should return result but may not pass
        assert isinstance(result, AgentResult)
    
    def test_insufficient_data(self, agent):
        """Test handling of insufficient data"""
        small_data = pd.DataFrame({
            'open': [100] * 20,
            'high': [102] * 20,
            'low': [99] * 20,
            'close': [101] * 20,
            'volume': [1000] * 20
        })
        
        result = agent.analyze(small_data, context_state='bullish')
        
        assert result.passed == False
        assert 'Insufficient' in result.reason


    def test_hsmm_alignment_not_hardcoded(self, agent, sample_data):
        """Score must come from HSMM, not a hardcoded constant (0.75, 0.55, or 0.30)"""
        hardcoded = {0.75, 0.55, 0.30}
        for ctx in ('bullish', 'bearish'):
            result = agent.analyze(sample_data, context_state=ctx)
            if result.ready:
                assert result.score not in hardcoded, (
                    f"Score {result.score} is a hardcoded constant for context={ctx}. "
                    "Alignment must come from HSMM Forward-Backward."
                )

    def test_hsmm_metadata_present(self, agent, sample_data):
        """HSMM state probabilities (all 5 states) must appear in metadata."""
        for ctx in ('bullish', 'bearish'):
            result = agent.analyze(sample_data, context_state=ctx)
            if result.ready:
                for key in ('hsmm_p_trend_plus', 'hsmm_p_trend_minus', 'hsmm_p_range',
                            'hsmm_p_squeeze', 'hsmm_p_distribution'):
                    assert key in result.metadata, \
                        f"Missing {key} for context={ctx}"
                # All 5 state probabilities must sum to ~1
                p_sum = (
                    result.metadata['hsmm_p_trend_plus']
                    + result.metadata['hsmm_p_trend_minus']
                    + result.metadata['hsmm_p_range']
                    + result.metadata['hsmm_p_squeeze']
                    + result.metadata['hsmm_p_distribution']
                )
                assert abs(p_sum - 1.0) < 0.01, \
                    f"HSMM 5-state probabilities do not sum to 1: {p_sum:.4f}"

    def test_hsmm_alignment_is_directional(self, agent, sample_data):
        """Alignment score uses 5-state formula:
           bullish = P(Trend+) + 0.5×P(Squeeze)
           bearish = P(Trend-) + P(Distribution)
        """
        for ctx in ('bullish', 'bearish'):
            result = agent.analyze(sample_data, context_state=ctx)
            if result.ready and result.metadata.get('hsmm_ok', False):
                m = result.metadata
                if ctx == 'bullish':
                    expected = m['hsmm_p_trend_plus'] + 0.5 * m['hsmm_p_squeeze']
                else:
                    expected = m['hsmm_p_trend_minus'] + m['hsmm_p_distribution']
                assert abs(result.score - expected) < 1e-9, (
                    f"For context={ctx}, score {result.score:.6f} != "
                    f"formula {expected:.6f}"
                )


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
