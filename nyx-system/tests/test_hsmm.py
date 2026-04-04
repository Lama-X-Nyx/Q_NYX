"""
Unit Tests for Semi-Markov HMM Module

Run:
    pytest tests/test_hsmm.py -v
    pytest tests/test_hsmm.py::test_initialization -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.hsmm import SemiMarkovHMM


@pytest.fixture
def sample_data():
    """Create sample price data for testing"""
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range('2020-01-01', periods=n, freq='1h')
    
    df = pd.DataFrame({
        'close': 10000 + np.cumsum(np.random.randn(n) * 100),
        'returns': np.random.randn(n) * 0.02,
        'atr_14': 100 + np.random.randn(n) * 20,
        'sma_20': 10000 + np.cumsum(np.random.randn(n) * 50),
        'sma_50': 10000 + np.cumsum(np.random.randn(n) * 30)
    }, index=dates)
    
    return df


class TestHSMMInitialization:
    """Test HSMM initialization"""
    
    def test_default_states(self):
        """Test default state initialization"""
        hsmm = SemiMarkovHMM()
        
        assert hsmm.states == ['Trend+', 'Range', 'Trend-']
        assert hsmm.n_states == 3
        assert len(hsmm.state_to_idx) == 3
    
    def test_custom_states(self):
        """Test custom state initialization"""
        custom_states = ['Bull', 'Neutral', 'Bear']
        hsmm = SemiMarkovHMM(states=custom_states)
        
        assert hsmm.states == custom_states
        assert hsmm.n_states == 3
        assert hsmm.state_to_idx['Bull'] == 0
        assert hsmm.state_to_idx['Neutral'] == 1
        assert hsmm.state_to_idx['Bear'] == 2
    
    def test_initial_parameters_none(self):
        """Test that parameters are None before initialization"""
        hsmm = SemiMarkovHMM()
        
        assert hsmm.initial_probs is None
        assert hsmm.transition_matrix is None
        assert hsmm.emission_params is None
        assert hsmm.duration_params is None


class TestHSMMParameterLearning:
    """Test parameter learning"""
    
    def test_initialize_parameters(self, sample_data):
        """Test parameter initialization from data"""
        hsmm = SemiMarkovHMM()
        hsmm.initialize_parameters(sample_data)
        
        # Check initial probabilities
        assert hsmm.initial_probs is not None
        assert len(hsmm.initial_probs) == 3
        assert np.allclose(hsmm.initial_probs.sum(), 1.0)
        
        # Check transition matrix
        assert hsmm.transition_matrix is not None
        assert hsmm.transition_matrix.shape == (3, 3)
        
        # Each row should sum to 1
        for i in range(3):
            assert np.allclose(hsmm.transition_matrix[i].sum(), 1.0)
        
        # Check emission parameters
        assert hsmm.emission_params is not None
        assert len(hsmm.emission_params) == 3
        
        for state in hsmm.states:
            assert 'price_mu' in hsmm.emission_params[state]
            assert 'price_sigma' in hsmm.emission_params[state]
            assert 'atr_mu' in hsmm.emission_params[state]
            assert 'atr_sigma' in hsmm.emission_params[state]
        
        # Check duration parameters
        assert hsmm.duration_params is not None
        assert len(hsmm.duration_params) == 3
        
        for state in hsmm.states:
            assert 'p' in hsmm.duration_params[state]
            assert 'mean' in hsmm.duration_params[state]
            assert hsmm.duration_params[state]['p'] > 0
            assert hsmm.duration_params[state]['mean'] > 0
    
    def test_transition_matrix_persistence(self, sample_data):
        """Test that transition matrix has persistence bias"""
        hsmm = SemiMarkovHMM()
        hsmm.initialize_parameters(sample_data)
        
        # Diagonal elements should be larger (persistence)
        for i in range(3):
            assert hsmm.transition_matrix[i, i] >= 0.5


class TestHSMMInference:
    """Test inference algorithms"""
    
    def test_emission_probability(self, sample_data):
        """Test emission probability computation"""
        hsmm = SemiMarkovHMM()
        hsmm.initialize_parameters(sample_data)
        
        observation = {
            'price': 0.01,  # 1% return
            'atr': 100
        }
        
        for state in hsmm.states:
            log_prob = hsmm.emission_probability(observation, state)
            assert isinstance(log_prob, float)
            assert log_prob <= 0  # Log probabilities are negative
    
    def test_forward_backward(self, sample_data):
        """Test Forward-Backward algorithm"""
        hsmm = SemiMarkovHMM()
        hsmm.initialize_parameters(sample_data)
        
        # Create observations
        observations = []
        for i in range(100):
            obs = {
                'price': sample_data['returns'].iloc[i],
                'atr': sample_data['atr_14'].iloc[i]
            }
            observations.append(obs)
        
        # Run Forward-Backward
        gamma = hsmm.forward_backward(observations)
        
        # Check output shape
        assert gamma.shape == (100, 3)
        
        # Each row should sum to ~1 (probabilities)
        for t in range(100):
            assert np.allclose(gamma[t].sum(), 1.0, atol=0.01)
        
        # All values should be between 0 and 1
        assert np.all(gamma >= 0)
        assert np.all(gamma <= 1)
    
    def test_viterbi(self, sample_data):
        """Test Viterbi algorithm"""
        hsmm = SemiMarkovHMM()
        hsmm.initialize_parameters(sample_data)
        
        # Create observations
        observations = []
        for i in range(50):
            obs = {
                'price': sample_data['returns'].iloc[i],
                'atr': sample_data['atr_14'].iloc[i]
            }
            observations.append(obs)
        
        # Run Viterbi
        states = hsmm.viterbi(observations)
        
        # Check output
        assert len(states) == 50
        assert all(s in hsmm.states for s in states)
    
    def test_confidence_score(self, sample_data):
        """Test confidence score calculation"""
        hsmm = SemiMarkovHMM()
        hsmm.initialize_parameters(sample_data)
        
        # Test with different probability distributions
        state_probs = np.array([0.8, 0.15, 0.05])
        confidence = hsmm.get_confidence_score(state_probs)
        
        assert 0 <= confidence <= 10
        assert np.isclose(confidence, 8.0)
        
        # Low confidence case
        state_probs = np.array([0.4, 0.3, 0.3])
        confidence = hsmm.get_confidence_score(state_probs)
        
        assert confidence < 5.0
    
    def test_dominant_state(self, sample_data):
        """Test dominant state identification"""
        hsmm = SemiMarkovHMM()
        hsmm.initialize_parameters(sample_data)
        
        state_probs = np.array([0.65, 0.25, 0.10])
        dominant_state, prob = hsmm.get_dominant_state(state_probs)
        
        assert dominant_state == 'Trend+'
        assert np.isclose(prob, 0.65)


class TestHSMMEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_observations(self):
        """Test with empty observations"""
        hsmm = SemiMarkovHMM()
        
        # Should handle gracefully
        observations = []
        
        # This might raise an error or return empty result
        # Depending on implementation
        try:
            result = hsmm.forward_backward(observations)
            assert result.shape[0] == 0
        except (ValueError, IndexError):
            pass  # Expected behavior
    
    def test_nan_values(self, sample_data):
        """Test handling of NaN values"""
        hsmm = SemiMarkovHMM()
        hsmm.initialize_parameters(sample_data)
        
        observation = {
            'price': np.nan,
            'atr': 100
        }
        
        # Should handle NaN gracefully
        log_prob = hsmm.emission_probability(observation, 'Trend+')
        assert isinstance(log_prob, float)
    
    def test_extreme_values(self, sample_data):
        """Test with extreme observation values"""
        hsmm = SemiMarkovHMM()
        hsmm.initialize_parameters(sample_data)
        
        observation = {
            'price': 1000.0,  # Extreme return
            'atr': 10000
        }
        
        # Should not crash
        log_prob = hsmm.emission_probability(observation, 'Trend+')
        assert isinstance(log_prob, float)


class TestHSMMIntegration:
    """Integration tests"""
    
    def test_full_workflow(self, sample_data):
        """Test complete workflow"""
        # Initialize
        hsmm = SemiMarkovHMM()
        
        # Train
        hsmm.initialize_parameters(sample_data)
        
        # Prepare observations
        observations = []
        for i in range(100):
            obs = {
                'price': sample_data['returns'].iloc[i],
                'atr': sample_data['atr_14'].iloc[i]
            }
            observations.append(obs)
        
        # Infer states
        gamma = hsmm.forward_backward(observations)
        
        # Get latest state
        latest_probs = gamma[-1]
        confidence = hsmm.get_confidence_score(latest_probs)
        dominant_state, prob = hsmm.get_dominant_state(latest_probs)
        
        # Validate results
        assert 0 <= confidence <= 10
        assert dominant_state in hsmm.states
        assert 0 <= prob <= 1
        
        print(f"\nIntegration Test Results:")
        print(f"  Dominant State: {dominant_state} ({prob*100:.1f}%)")
        print(f"  Confidence (SdC): {confidence:.2f}/10")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
