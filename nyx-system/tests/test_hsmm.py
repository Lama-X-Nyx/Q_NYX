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

from src.core.hsmm import SemiMarkovHMM, _logsumexp


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
        assert hsmm.transition_matrix is not None
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


# ------------------------------------------------------------------
# P4 — 5-state and 6-state HSMM tests
# ------------------------------------------------------------------

@pytest.fixture
def sample_data_5state():
    """Sample data with extra features needed for 5-state HSMM."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='1h')
    close = 10000 + np.cumsum(np.random.randn(n) * 100)
    atr_14 = np.abs(100 + np.random.randn(n) * 20)
    # atr_50 is smoothed version of atr_14
    atr_50 = pd.Series(atr_14).rolling(50, min_periods=1).mean().values
    volume = np.abs(1000 + np.random.randn(n) * 200)
    volume_ma20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
    sma_20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    sma_50 = pd.Series(close).rolling(50, min_periods=1).mean().values

    return pd.DataFrame({
        'close':       close,
        'returns':     np.concatenate([[0], np.diff(close) / close[:-1]]),
        'atr_14':      atr_14,
        'atr_50':      atr_50,
        'volume':      volume,
        'volume_ma20': volume_ma20,
        'sma_20':      sma_20,
        'sma_50':      sma_50,
    }, index=dates)


class TestHSMMP4FiveState:
    """P4a — 5-state HSMM tests (Squeeze + Distribution)."""

    def test_5state_init(self):
        """SemiMarkovHMM accepts 5-state list."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        assert hsmm.n_states == 5
        assert 'Squeeze' in hsmm.states
        assert 'Distribution' in hsmm.states

    def test_5state_transition_matrix_shape(self, sample_data_5state):
        """Transition matrix is 5×5 for 5-state model."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        assert hsmm.transition_matrix is not None
        assert hsmm.transition_matrix.shape == (5, 5)

    def test_5state_transition_rows_sum_to_one(self, sample_data_5state):
        """Every row of the 5-state transition matrix sums to 1."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        assert hsmm.transition_matrix is not None
        row_sums = hsmm.transition_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_5state_forward_backward_shape(self, sample_data_5state):
        """Forward-Backward returns (T, 5) for 5-state model."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        observations = [
            {'price': sample_data_5state['returns'].iloc[i],
             'atr':   sample_data_5state['atr_14'].iloc[i]}
            for i in range(50)
        ]
        gamma = hsmm.forward_backward(observations)
        assert gamma.shape == (50, 5)

    def test_5state_posteriors_sum_to_one(self, sample_data_5state):
        """Each timestep posterior sums to 1 in 5-state model."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        observations = [
            {'price': sample_data_5state['returns'].iloc[i],
             'atr':   sample_data_5state['atr_14'].iloc[i]}
            for i in range(50)
        ]
        gamma = hsmm.forward_backward(observations)
        assert np.allclose(gamma.sum(axis=1), 1.0, atol=0.01)

    def test_5state_emission_params_present(self, sample_data_5state):
        """All 5 states have emission params after init."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        assert hsmm.emission_params is not None
        for state in hsmm.states:
            assert state in hsmm.emission_params
            assert 'price_mu' in hsmm.emission_params[state]
            assert 'price_sigma' in hsmm.emission_params[state]

    def test_squeeze_prior_row_sums_to_one(self):
        """Squeeze row in 5-state prior sums to 1."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        A = hsmm._build_transition_prior()
        squeeze_idx = hsmm.state_to_idx['Squeeze']
        assert np.isclose(A[squeeze_idx].sum(), 1.0, atol=1e-6)

    def test_distribution_leads_to_trend_minus(self):
        """Prior: P(Trend- | Distribution) > P(Trend+ | Distribution)."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        A = hsmm._build_transition_prior()
        dist_idx  = hsmm.state_to_idx['Distribution']
        tp_idx    = hsmm.state_to_idx['Trend+']
        tm_idx    = hsmm.state_to_idx['Trend-']
        assert A[dist_idx, tm_idx] > A[dist_idx, tp_idx]

    def test_squeeze_exits_to_both_trends(self):
        """Prior: Squeeze exits to Trend+ and Trend- with equal probability."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        A = hsmm._build_transition_prior()
        sqz_idx = hsmm.state_to_idx['Squeeze']
        tp_idx  = hsmm.state_to_idx['Trend+']
        tm_idx  = hsmm.state_to_idx['Trend-']
        # Both exit probabilities should be substantial (> 15%)
        assert A[sqz_idx, tp_idx] > 0.15
        assert A[sqz_idx, tm_idx] > 0.15


class TestHSMMP4SixState:
    """P4b — 6-state HSMM tests (Liquidation)."""

    def test_6state_init(self):
        """SemiMarkovHMM accepts 6-state list."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation'])
        assert hsmm.n_states == 6
        assert 'Liquidation' in hsmm.states

    def test_liquidation_nearly_absorbing(self):
        """Liquidation self-transition ≥ 0.90 (nearly absorbing)."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation'])
        A = hsmm._build_transition_prior()
        liq_idx = hsmm.state_to_idx['Liquidation']
        assert A[liq_idx, liq_idx] >= 0.90, \
            f"Liquidation self-transition {A[liq_idx, liq_idx]:.3f} < 0.90"

    def test_6state_rows_sum_to_one(self):
        """All rows of 6-state prior sum to 1."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation'])
        A = hsmm._build_transition_prior()
        assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)

    def test_liquidation_default_emission_negative_mu(self):
        """Liquidation default emission has negative price_mu."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation'])
        em = hsmm._default_emission('Liquidation')
        assert em['price_mu'] < 0, "Liquidation should have negative expected return"

    def test_6state_forward_backward_shape(self, sample_data_5state):
        """Forward-Backward returns (T, 6) for 6-state model."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation'])
        hsmm.initialize_parameters(sample_data_5state)
        observations = [
            {'price': sample_data_5state['returns'].iloc[i],
             'atr':   sample_data_5state['atr_14'].iloc[i]}
            for i in range(30)
        ]
        gamma = hsmm.forward_backward(observations)
        assert gamma.shape == (30, 6)
        assert np.allclose(gamma.sum(axis=1), 1.0, atol=0.01)


# ------------------------------------------------------------------
# Baum-Welch EM tests
# ------------------------------------------------------------------

class TestBaumWelchEM:
    """Tests for the Baum-Welch EM implementation."""

    def test_fit_returns_ll_history(self, sample_data_5state):
        """fit() returns a non-empty list of log-likelihoods."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        observations = [
            {'price': sample_data_5state['returns'].iloc[i],
             'atr':   sample_data_5state['atr_14'].iloc[i]}
            for i in range(200)
        ]
        ll_hist = hsmm.fit(observations, n_iter=5)
        assert isinstance(ll_hist, list)
        assert len(ll_hist) > 0

    def test_fit_ll_non_decreasing(self, sample_data_5state):
        """EM should not decrease log-likelihood (up to floating-point noise)."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        observations = [
            {'price': sample_data_5state['returns'].iloc[i],
             'atr':   sample_data_5state['atr_14'].iloc[i]}
            for i in range(300)
        ]
        ll_hist = hsmm.fit(observations, n_iter=20, tol=1e-6)
        # Each step should not drop by more than a tiny numerical tolerance
        for prev, curr in zip(ll_hist[:-1], ll_hist[1:]):
            assert curr >= prev - 0.5, f"LL decreased: {prev:.2f} → {curr:.2f}"

    def test_transition_matrix_rows_sum_after_em(self, sample_data_5state):
        """After EM, transition matrix rows still sum to 1."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        observations = [
            {'price': sample_data_5state['returns'].iloc[i],
             'atr':   sample_data_5state['atr_14'].iloc[i]}
            for i in range(200)
        ]
        hsmm.fit(observations, n_iter=10)
        assert hsmm.transition_matrix is not None
        row_sums = hsmm.transition_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_emission_sigma_positive_after_em(self, sample_data_5state):
        """After EM, all emission sigmas are strictly positive."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        observations = [
            {'price': sample_data_5state['returns'].iloc[i],
             'atr':   sample_data_5state['atr_14'].iloc[i]}
            for i in range(200)
        ]
        hsmm.fit(observations, n_iter=10)
        assert hsmm.emission_params is not None
        for state, params in hsmm.emission_params.items():
            assert params['price_sigma'] > 0, f"{state} price_sigma not positive"
            assert params['atr_sigma']   > 0, f"{state} atr_sigma not positive"

    def test_initialize_with_em_public_api(self, sample_data_5state):
        """initialize_parameters_with_em() runs warm-start + EM and returns LL history."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        ll_hist = hsmm.initialize_parameters_with_em(sample_data_5state, n_iter=5)
        assert isinstance(ll_hist, list)
        assert len(ll_hist) > 0
        # Model should be usable after EM
        observations = [
            {'price': sample_data_5state['returns'].iloc[i],
             'atr':   sample_data_5state['atr_14'].iloc[i]}
            for i in range(30)
        ]
        gamma = hsmm.forward_backward(observations)
        assert gamma.shape == (30, 5)

    def test_compute_xi_shape(self, sample_data_5state):
        """_compute_xi() returns (T-1, n, n) tensor."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        T, n = 40, hsmm.n_states
        observations = [
            {'price': sample_data_5state['returns'].iloc[i],
             'atr':   sample_data_5state['atr_14'].iloc[i]}
            for i in range(T)
        ]
        log_B = hsmm._compute_log_B(observations)
        log_alpha = np.full((T, n), -np.inf)
        assert hsmm.transition_matrix is not None
        assert hsmm.initial_probs is not None
        log_A     = np.log(hsmm.transition_matrix + 1e-10)
        log_alpha[0] = np.log(hsmm.initial_probs + 1e-10) + log_B[0]
        for t in range(1, T):
            log_alpha[t] = _logsumexp(log_alpha[t-1, :, np.newaxis] + log_A, axis=0) + log_B[t]
        log_beta = np.zeros((T, n))
        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(log_A + log_B[t+1] + log_beta[t+1], axis=1)
        xi = hsmm._compute_xi(log_alpha, log_beta, log_B)
        assert xi.shape == (T - 1, n, n)

    def test_compute_xi_sums_to_one(self, sample_data_5state):
        """Each ξ(t) slice should sum to 1 over (i, j)."""
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'])
        hsmm.initialize_parameters(sample_data_5state)
        T, n = 40, hsmm.n_states
        observations = [
            {'price': sample_data_5state['returns'].iloc[i],
             'atr':   sample_data_5state['atr_14'].iloc[i]}
            for i in range(T)
        ]
        assert hsmm.transition_matrix is not None
        assert hsmm.initial_probs is not None
        log_B    = hsmm._compute_log_B(observations)
        log_A    = np.log(hsmm.transition_matrix + 1e-10)
        log_alpha = np.full((T, n), -np.inf)
        log_alpha[0] = np.log(hsmm.initial_probs + 1e-10) + log_B[0]
        for t in range(1, T):
            log_alpha[t] = _logsumexp(log_alpha[t-1, :, np.newaxis] + log_A, axis=0) + log_B[t]
        log_beta = np.zeros((T, n))
        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(log_A + log_B[t+1] + log_beta[t+1], axis=1)
        xi = hsmm._compute_xi(log_alpha, log_beta, log_B)
        slice_sums = xi.sum(axis=(1, 2))
        assert np.allclose(slice_sums, 1.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
