"""
Semi-Markov Hidden Markov Model (HSMM)
Advanced state detection with duration modeling
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from typing import List, Dict, Optional, Tuple

class SemiMarkovHMM:
    """
    Semi-Markov Hidden Markov Model for market regime detection
    
    Features:
    - State detection (Trend+, Range, Trend-)
    - Duration modeling
    - Transition probabilities
    - Emission parameters (multi-variate)
    - Viterbi & Forward-Backward algorithms
    
    Usage:
        hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-'])
        hsmm.initialize_parameters(data)
        probs = hsmm.forward_backward(observations)
        confidence = max(probs[-1]) * 10  # SdC score
    """
    
    def __init__(self, states: List[str] = None):
        """
        Initialize HSMM
        
        Args:
            states: List of state names (default: ['Trend+', 'Range', 'Trend-'])
        """
        self.states = states or ['Trend+', 'Range', 'Trend-']
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        
        # Model parameters (learned from data)
        self.initial_probs = None
        self.transition_matrix = None
        self.emission_params = None
        self.duration_params = None
    
    def initialize_parameters(self, data: pd.DataFrame):
        """
        Initialize/learn model parameters from data
        
        Args:
            data: DataFrame with OHLCV + indicators
        """
        # Initial state probabilities (uniform)
        self.initial_probs = np.ones(self.n_states) / self.n_states
        
        # Transition matrix (with persistence bias)
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i == j:
                    self.transition_matrix[i, j] = 0.75  # Persistence
                else:
                    self.transition_matrix[i, j] = 0.25 / (self.n_states - 1)
        
        # Label states using simple heuristic
        labeled_states = self._label_states_heuristic(data)
        
        # Learn emission parameters (per state)
        self.emission_params = {}
        for state in self.states:
            state_data = data[labeled_states == state]
            
            if len(state_data) > 10:
                returns = state_data['returns'].dropna()
                
                self.emission_params[state] = {
                    'price_mu': returns.mean(),
                    'price_sigma': max(returns.std(), 0.001),
                    'atr_mu': state_data['atr_14'].mean() if 'atr_14' in state_data else 100,
                    'atr_sigma': max(state_data['atr_14'].std() if 'atr_14' in state_data else 50, 0.001)
                }
            else:
                # Default parameters
                self.emission_params[state] = {
                    'price_mu': 0.0,
                    'price_sigma': 0.01,
                    'atr_mu': 100,
                    'atr_sigma': 50
                }
        
        # Learn duration parameters (geometric distribution)
        self.duration_params = {}
        for state in self.states:
            state_mask = (labeled_states == state).values
            durations = self._extract_durations(state_mask)
            
            if durations:
                avg_duration = np.mean(durations)
                p = 1.0 / max(avg_duration, 2.0)  # Geometric param
                self.duration_params[state] = {
                    'p': p,
                    'mean': avg_duration
                }
            else:
                self.duration_params[state] = {
                    'p': 0.2,
                    'mean': 5.0
                }
    
    def _label_states_heuristic(self, df: pd.DataFrame) -> pd.Series:
        """
        Simple heuristic state labeling for initialization
        
        Args:
            df: DataFrame with price data and moving averages
        
        Returns:
            Series with state labels
            
        Raises:
            ValueError: If required features (sma_20, sma_50) are missing
        """
        
        # CRITICAL: Check for required features
        required_features = ['sma_20', 'sma_50']
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            raise ValueError(
                f"HSMM heuristic labeling requires {required_features}. "
                f"Missing: {missing_features}. "
                f"Ensure RegimeAgent._prepare_data() provides these features. "
                f"Without them, initialization defaults to Range state, "
                f"causing structural bias in regime detection."
            )
        
        labels = []
        
        for idx, row in df.iterrows():
            # Check for NaN values (still possible during warm-up period)
            if pd.isna(row.get('sma_20')) or pd.isna(row.get('sma_50')):
                labels.append('Range')
                continue
            
            # Trend detection
            trend_up = (row['sma_20'] > row['sma_50'] and 
                       row['close'] > row['sma_20'])
            trend_down = (row['sma_20'] < row['sma_50'] and 
                         row['close'] < row['sma_20'])
            
            if trend_up:
                labels.append('Trend+')
            elif trend_down:
                labels.append('Trend-')
            else:
                labels.append('Range')
        
        return pd.Series(labels, index=df.index)
    
    def _extract_durations(self, state_mask: np.ndarray) -> List[int]:
        """
        Extract state durations from binary mask
        
        Args:
            state_mask: Boolean array indicating state presence
        
        Returns:
            List of duration lengths
        """
        durations = []
        current_duration = 0
        
        for is_state in state_mask:
            if is_state:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return durations
    
    def emission_probability(self, observation: Dict, state: str) -> float:
        """
        Compute log-probability of observation given state
        
        Args:
            observation: Dict with 'price' (return), 'atr', etc.
            state: State name
        
        Returns:
            Log-probability
        """
        params = self.emission_params[state]
        log_prob = 0.0
        
        # Price return (Normal distribution)
        if 'price' in observation and not np.isnan(observation['price']):
            log_prob += stats.norm.logpdf(
                observation['price'],
                loc=params['price_mu'],
                scale=params['price_sigma']
            )
        
        # ATR (Normal distribution)
        if 'atr' in observation and not np.isnan(observation['atr']):
            log_prob += stats.norm.logpdf(
                observation['atr'],
                loc=params['atr_mu'],
                scale=params['atr_sigma']
            )
        
        return log_prob
    
    def forward_backward(self, observations: List[Dict]) -> np.ndarray:
        """
        Forward-Backward algorithm for state probability inference
        
        Args:
            observations: List of observation dicts with keys 'price' and 'atr'
        
        Returns:
            Array of shape (T, n_states) with smoothed probabilities
        """
        # Defensive assertion
        if not isinstance(observations, list):
            raise TypeError(
                f"HSMM expects List[Dict] observations, got {type(observations)}. "
                f"Each observation must be a dict with 'price' and 'atr' keys."
            )
        
        if len(observations) > 0 and not isinstance(observations[0], dict):
            raise TypeError(
                f"HSMM expects observations as List[Dict], got list of {type(observations[0])}. "
                f"Each observation must be a dict with 'price' and 'atr' keys."
            )
        
        T = len(observations)
        
        # Handle empty observations
        if T == 0:
            return np.array([])
        
        # Forward pass
        log_alpha = np.full((T, self.n_states), -np.inf)
        
        # Initialize
        for s in range(self.n_states):
            log_alpha[0, s] = (
                np.log(self.initial_probs[s] + 1e-10) +
                self.emission_probability(observations[0], self.states[s])
            )
        
        # Forward recursion
        for t in range(1, T):
            for s in range(self.n_states):
                log_probs = []
                for s_prev in range(self.n_states):
                    log_prob = (
                        log_alpha[t-1, s_prev] +
                        np.log(self.transition_matrix[s_prev, s] + 1e-10)
                    )
                    log_probs.append(log_prob)
                
                log_alpha[t, s] = (
                    logsumexp(log_probs) +
                    self.emission_probability(observations[t], self.states[s])
                )
        
        # Backward pass
        log_beta = np.full((T, self.n_states), -np.inf)
        log_beta[-1, :] = 0  # Terminal
        
        # Backward recursion
        for t in range(T-2, -1, -1):
            for s in range(self.n_states):
                log_probs = []
                for s_next in range(self.n_states):
                    log_prob = (
                        log_beta[t+1, s_next] +
                        np.log(self.transition_matrix[s, s_next] + 1e-10) +
                        self.emission_probability(observations[t+1], self.states[s_next])
                    )
                    log_probs.append(log_prob)
                
                log_beta[t, s] = logsumexp(log_probs)
        
        # Combine forward-backward
        log_gamma = log_alpha + log_beta
        
        # Normalize to probabilities
        gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))
        
        return gamma
    
    def viterbi(self, observations: List[Dict]) -> List[str]:
        """
        Viterbi algorithm for most likely state sequence
        
        Args:
            observations: List of observation dicts
        
        Returns:
            List of most likely states
        """
        T = len(observations)
        
        # Initialize
        log_delta = np.full((T, self.n_states), -np.inf)
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # t=0
        for s in range(self.n_states):
            log_delta[0, s] = (
                np.log(self.initial_probs[s] + 1e-10) +
                self.emission_probability(observations[0], self.states[s])
            )
        
        # Forward
        for t in range(1, T):
            for s in range(self.n_states):
                log_probs = []
                for s_prev in range(self.n_states):
                    log_prob = (
                        log_delta[t-1, s_prev] +
                        np.log(self.transition_matrix[s_prev, s] + 1e-10)
                    )
                    log_probs.append(log_prob)
                
                psi[t, s] = np.argmax(log_probs)
                log_delta[t, s] = (
                    log_probs[psi[t, s]] +
                    self.emission_probability(observations[t], self.states[s])
                )
        
        # Backtrack
        states = []
        states.append(np.argmax(log_delta[-1]))
        
        for t in range(T-1, 0, -1):
            states.append(psi[t, states[-1]])
        
        states.reverse()
        
        return [self.states[s] for s in states]
    
    def get_confidence_score(self, state_probs: np.ndarray) -> float:
        """
        Convert state probabilities to confidence score (SdC)
        
        Args:
            state_probs: Array of state probabilities (last timestep)
        
        Returns:
            Confidence score 0-10
        """
        max_prob = np.max(state_probs)
        return max_prob * 10
    
    def get_dominant_state(self, state_probs: np.ndarray) -> Tuple[str, float]:
        """
        Get dominant state and its probability
        
        Args:
            state_probs: Array of state probabilities
        
        Returns:
            (state_name, probability)
        """
        idx = np.argmax(state_probs)
        return self.states[idx], state_probs[idx]


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("HSMM Module - Example Usage")
    print("="*80)
    
    # Create synthetic data
    dates = pd.date_range('2020-01-01', periods=1000, freq='1h')
    data = pd.DataFrame({
        'close': 10000 + np.cumsum(np.random.randn(1000) * 100),
        'returns': np.random.randn(1000) * 0.02,
        'atr_14': 100 + np.random.randn(1000) * 20,
        'sma_20': 10000 + np.cumsum(np.random.randn(1000) * 50),
        'sma_50': 10000 + np.cumsum(np.random.randn(1000) * 30)
    }, index=dates)
    
    # Initialize HSMM
    hsmm = SemiMarkovHMM()
    print(f"\n✓ Initialized HSMM with states: {hsmm.states}")
    
    # Learn parameters
    hsmm.initialize_parameters(data)
    print(f"✓ Learned parameters from {len(data)} observations")
    
    # Prepare observations
    observations = []
    for idx, row in data.tail(100).iterrows():
        obs = {
            'price': row['returns'],
            'atr': row['atr_14']
        }
        observations.append(obs)
    
    # Run Forward-Backward
    state_probs = hsmm.forward_backward(observations)
    print(f"\n✓ Computed state probabilities: shape {state_probs.shape}")
    
    # Get latest confidence
    latest_probs = state_probs[-1]
    confidence = hsmm.get_confidence_score(latest_probs)
    dominant_state, prob = hsmm.get_dominant_state(latest_probs)
    
    print(f"\nLatest State Analysis:")
    print(f"  Trend+: {latest_probs[0]*100:.1f}%")
    print(f"  Range:  {latest_probs[1]*100:.1f}%")
    print(f"  Trend-: {latest_probs[2]*100:.1f}%")
    print(f"  Dominant: {dominant_state} ({prob*100:.1f}%)")
    print(f"  Confidence (SdC): {confidence:.2f}/10")
    
    print("\n" + "="*80)
    print("✅ HSMM Module Working")
    print("="*80)
