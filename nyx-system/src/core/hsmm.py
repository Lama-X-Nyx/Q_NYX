"""
Semi-Markov Hidden Markov Model (HSMM)
Advanced state detection with duration modeling

P4a: 5-state support — Trend+, Range, Trend-, Squeeze, Distribution
P4b: Liquidation as a nearly-absorbing 6th state
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from typing import List, Dict, Optional, Tuple


# State sets recognised by the smart-prior logic
_STATES_3 = ['Trend+', 'Range', 'Trend-']
_STATES_5 = ['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution']
_STATES_6 = ['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation']


class SemiMarkovHMM:
    """
    Semi-Markov Hidden Markov Model for market regime detection.

    Supports 3, 5, or 6 states:
      3-state: Trend+, Range, Trend-
      5-state: + Squeeze, Distribution          (P4a)
      6-state: + Liquidation (nearly absorbing) (P4b)

    Usage:
        hsmm = SemiMarkovHMM(states=['Trend+','Range','Trend-','Squeeze','Distribution'])
        hsmm.initialize_parameters(data)
        probs = hsmm.forward_backward(observations)
    """

    def __init__(self, states: List[str] = None):
        self.states = states or ['Trend+', 'Range', 'Trend-']
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}

        self.initial_probs = None
        self.transition_matrix = None
        self.emission_params = None
        self.duration_params = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_parameters(self, data: pd.DataFrame):
        """
        Initialise / learn model parameters from OHLCV data.

        Builds:
        - initial_probs (uniform)
        - transition_matrix (domain-informed prior, per known state sets)
        - emission_params (learned from heuristic labels)
        - duration_params (geometric, learned from run-lengths)
        """
        self.initial_probs = np.ones(self.n_states) / self.n_states
        self.transition_matrix = self._build_transition_prior()

        labeled_states = self._label_states_heuristic(data)

        # Emission parameters per state
        self.emission_params = {}
        for state in self.states:
            state_data = data[labeled_states == state]
            if len(state_data) > 10 and 'returns' in state_data.columns:
                returns = state_data['returns'].dropna()
                self.emission_params[state] = {
                    'price_mu':    float(returns.mean()),
                    'price_sigma': float(max(returns.std(), 0.001)),
                    'atr_mu':    float(state_data['atr_14'].mean()) if 'atr_14' in state_data else 100.0,
                    'atr_sigma': float(max(state_data['atr_14'].std(), 0.001)) if 'atr_14' in state_data else 50.0,
                }
            else:
                self.emission_params[state] = self._default_emission(state)

        # Duration parameters (geometric)
        self.duration_params = {}
        for state in self.states:
            mask = (labeled_states == state).values
            durations = self._extract_durations(mask)
            if durations:
                avg = float(np.mean(durations))
                self.duration_params[state] = {'p': 1.0 / max(avg, 2.0), 'mean': avg}
            else:
                self.duration_params[state] = {'p': 0.2, 'mean': 5.0}

    def _build_transition_prior(self) -> np.ndarray:
        """
        Domain-informed transition prior.

        For known state sets (3, 5, 6) we use hand-crafted priors that
        encode market knowledge.  Unknown configurations fall back to the
        uniform prior (diagonal 0.75, off-diagonal equal).
        """
        n = self.n_states
        states = self.states

        # ---- 3-state (original) ----
        if states == _STATES_3:
            # [Trend+, Range, Trend-]
            A = np.array([
                [0.75, 0.15, 0.10],
                [0.15, 0.70, 0.15],
                [0.10, 0.15, 0.75],
            ])
            return A

        # ---- 5-state (P4a) ----
        if states == _STATES_5:
            # rows: Trend+, Range, Trend-, Squeeze, Distribution
            # Squeeze exits to Trend+/Trend- (breakout); Distribution leads to Trend-
            A = np.array([
                # T+     Rng    T-     Sqz    Dist
                [0.75,  0.06,  0.05,  0.04,  0.10],  # Trend+  → often goes to Distribution
                [0.10,  0.68,  0.10,  0.06,  0.06],  # Range
                [0.05,  0.08,  0.75,  0.06,  0.06],  # Trend-
                [0.22,  0.10,  0.22,  0.42,  0.04],  # Squeeze → breaks out to Trend+ or Trend-
                [0.05,  0.08,  0.65,  0.04,  0.18],  # Distribution → leads to Trend-
            ])
            # Normalise rows (safety)
            return A / A.sum(axis=1, keepdims=True)

        # ---- 6-state (P4b: + Liquidation) ----
        if states == _STATES_6:
            # rows: Trend+, Range, Trend-, Squeeze, Distribution, Liquidation
            A = np.array([
                # T+     Rng    T-     Sqz    Dist   Liq
                [0.74,  0.06,  0.05,  0.04,  0.09,  0.02],  # Trend+
                [0.10,  0.66,  0.10,  0.06,  0.06,  0.02],  # Range
                [0.04,  0.07,  0.73,  0.06,  0.06,  0.04],  # Trend-  (higher Liq risk)
                [0.21,  0.09,  0.21,  0.41,  0.04,  0.04],  # Squeeze
                [0.04,  0.07,  0.62,  0.04,  0.17,  0.06],  # Distribution (highest Liq)
                [0.01,  0.01,  0.01,  0.01,  0.01,  0.95],  # Liquidation — nearly absorbing
            ])
            return A / A.sum(axis=1, keepdims=True)

        # ---- Generic fallback ----
        A = np.full((n, n), 0.25 / max(n - 1, 1))
        np.fill_diagonal(A, 0.75)
        return A / A.sum(axis=1, keepdims=True)

    def _default_emission(self, state: str) -> Dict:
        """Default emission parameters when insufficient data for a state."""
        defaults = {
            'Trend+':       {'price_mu':  0.0015, 'price_sigma': 0.010, 'atr_mu': 90,  'atr_sigma': 30},
            'Trend-':       {'price_mu': -0.0015, 'price_sigma': 0.010, 'atr_mu': 110, 'atr_sigma': 35},
            'Range':        {'price_mu':  0.0000, 'price_sigma': 0.005, 'atr_mu': 80,  'atr_sigma': 25},
            'Squeeze':      {'price_mu':  0.0000, 'price_sigma': 0.003, 'atr_mu': 50,  'atr_sigma': 15},
            'Distribution': {'price_mu': -0.0005, 'price_sigma': 0.008, 'atr_mu': 100, 'atr_sigma': 30},
            'Liquidation':  {'price_mu': -0.0100, 'price_sigma': 0.030, 'atr_mu': 300, 'atr_sigma': 100},
        }
        return defaults.get(state, {'price_mu': 0.0, 'price_sigma': 0.01, 'atr_mu': 100, 'atr_sigma': 50})

    # ------------------------------------------------------------------
    # Heuristic labeling (supports 3, 5, 6 states)
    # ------------------------------------------------------------------

    def _label_states_heuristic(self, df: pd.DataFrame) -> pd.Series:
        """
        Label each bar with the most likely regime state.

        Required: sma_20, sma_50
        Optional (for new states):
          - atr_14, atr_50          → Squeeze detection (ATR compression)
          - volume, volume_ma20     → Distribution detection (high vol + range)
          - drawdown_20             → Liquidation detection (rolling -10% drawdown)

        Priority (highest first):
          Liquidation > Squeeze > Distribution > Trend+ / Trend- > Range
        """
        required = ['sma_20', 'sma_50']
        missing = [f for f in required if f not in df.columns]
        if missing:
            raise ValueError(
                f"HSMM heuristic labeling requires {required}. "
                f"Missing: {missing}. "
                f"Ensure _prepare_data() provides these features."
            )

        has_squeeze      = 'Squeeze' in self.states
        has_distribution = 'Distribution' in self.states
        has_liquidation  = 'Liquidation' in self.states

        # Pre-compute optional features (vectorised for speed)
        squeeze_mask      = self._compute_squeeze_mask(df) if has_squeeze else None
        distribution_mask = self._compute_distribution_mask(df) if has_distribution else None
        liquidation_mask  = self._compute_liquidation_mask(df) if has_liquidation else None

        labels = []
        for i, (idx, row) in enumerate(df.iterrows()):
            # NaN guard for SMA warm-up
            if pd.isna(row.get('sma_20')) or pd.isna(row.get('sma_50')):
                labels.append('Range')
                continue

            # 1. Liquidation (highest priority)
            if has_liquidation and liquidation_mask is not None and liquidation_mask.iloc[i]:
                labels.append('Liquidation')
                continue

            # 2. Squeeze
            if has_squeeze and squeeze_mask is not None and squeeze_mask.iloc[i]:
                labels.append('Squeeze')
                continue

            # 3. Distribution
            if has_distribution and distribution_mask is not None and distribution_mask.iloc[i]:
                labels.append('Distribution')
                continue

            # 4-6. Standard trend detection
            trend_up   = row['sma_20'] > row['sma_50'] and row['close'] > row['sma_20']
            trend_down = row['sma_20'] < row['sma_50'] and row['close'] < row['sma_20']

            if trend_up:
                labels.append('Trend+')
            elif trend_down:
                labels.append('Trend-')
            else:
                labels.append('Range')

        return pd.Series(labels, index=df.index)

    def _compute_squeeze_mask(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Squeeze: ATR(14) < 0.8 × ATR(50) — volatility compression.
        Returns boolean Series or None if features unavailable.
        """
        if 'atr_14' not in df.columns or 'atr_50' not in df.columns:
            return None
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = df['atr_14'] / df['atr_50'].replace(0, np.nan)
        return ratio < 0.8

    def _compute_distribution_mask(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Distribution: volume > 1.2 × MA20(volume) AND price range-bound
        (|sma_20 − sma_50| / sma_50 < 2%).
        Returns boolean Series or None if features unavailable.
        """
        if 'volume' not in df.columns or 'volume_ma20' not in df.columns:
            return None
        if 'sma_20' not in df.columns or 'sma_50' not in df.columns:
            return None
        high_vol = df['volume'] > df['volume_ma20'].replace(0, np.nan) * 1.2
        with np.errstate(invalid='ignore', divide='ignore'):
            sma_diff_pct = (df['sma_20'] - df['sma_50']).abs() / df['sma_50'].replace(0, np.nan)
        in_range = sma_diff_pct < 0.02
        return high_vol & in_range

    def _compute_liquidation_mask(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Liquidation: rolling 20-bar drawdown from peak ≤ −10%.
        """
        if 'close' not in df.columns:
            return None
        rolling_peak = df['close'].rolling(20, min_periods=1).max()
        with np.errstate(invalid='ignore', divide='ignore'):
            drawdown = (df['close'] - rolling_peak) / rolling_peak.replace(0, np.nan)
        return drawdown <= -0.10

    # ------------------------------------------------------------------
    # Duration extraction
    # ------------------------------------------------------------------

    def _extract_durations(self, state_mask: np.ndarray) -> List[int]:
        """Extract consecutive run-lengths from a boolean mask."""
        durations, cur = [], 0
        for is_state in state_mask:
            if is_state:
                cur += 1
            else:
                if cur > 0:
                    durations.append(cur)
                cur = 0
        if cur > 0:
            durations.append(cur)
        return durations

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def emission_probability(self, observation: Dict, state: str) -> float:
        """Log-probability of observation given state."""
        params = self.emission_params[state]
        log_prob = 0.0

        if 'price' in observation and not np.isnan(observation['price']):
            log_prob += stats.norm.logpdf(
                observation['price'],
                loc=params['price_mu'],
                scale=params['price_sigma']
            )

        if 'atr' in observation and not np.isnan(observation['atr']):
            log_prob += stats.norm.logpdf(
                observation['atr'],
                loc=params['atr_mu'],
                scale=params['atr_sigma']
            )

        return log_prob

    def forward_backward(self, observations: List[Dict]) -> np.ndarray:
        """
        Forward-Backward algorithm — works for any number of states.

        Returns array (T, n_states) with smoothed posteriors.
        """
        if not isinstance(observations, list):
            raise TypeError(
                f"HSMM expects List[Dict], got {type(observations)}."
            )
        if len(observations) > 0 and not isinstance(observations[0], dict):
            raise TypeError(
                f"HSMM expects List[Dict], got list of {type(observations[0])}."
            )

        T = len(observations)
        if T == 0:
            return np.array([])

        # Forward pass
        log_alpha = np.full((T, self.n_states), -np.inf)
        for s in range(self.n_states):
            log_alpha[0, s] = (
                np.log(self.initial_probs[s] + 1e-10)
                + self.emission_probability(observations[0], self.states[s])
            )

        for t in range(1, T):
            for s in range(self.n_states):
                lp = [
                    log_alpha[t-1, sp] + np.log(self.transition_matrix[sp, s] + 1e-10)
                    for sp in range(self.n_states)
                ]
                log_alpha[t, s] = logsumexp(lp) + self.emission_probability(observations[t], self.states[s])

        # Backward pass
        log_beta = np.full((T, self.n_states), -np.inf)
        log_beta[-1, :] = 0.0

        for t in range(T - 2, -1, -1):
            for s in range(self.n_states):
                lp = [
                    log_beta[t+1, sn]
                    + np.log(self.transition_matrix[s, sn] + 1e-10)
                    + self.emission_probability(observations[t+1], self.states[sn])
                    for sn in range(self.n_states)
                ]
                log_beta[t, s] = logsumexp(lp)

        log_gamma = log_alpha + log_beta
        gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))
        return gamma

    def viterbi(self, observations: List[Dict]) -> List[str]:
        """Viterbi algorithm — works for any number of states."""
        T = len(observations)
        log_delta = np.full((T, self.n_states), -np.inf)
        psi = np.zeros((T, self.n_states), dtype=int)

        for s in range(self.n_states):
            log_delta[0, s] = (
                np.log(self.initial_probs[s] + 1e-10)
                + self.emission_probability(observations[0], self.states[s])
            )

        for t in range(1, T):
            for s in range(self.n_states):
                lp = [
                    log_delta[t-1, sp] + np.log(self.transition_matrix[sp, s] + 1e-10)
                    for sp in range(self.n_states)
                ]
                psi[t, s] = int(np.argmax(lp))
                log_delta[t, s] = lp[psi[t, s]] + self.emission_probability(observations[t], self.states[s])

        path = [int(np.argmax(log_delta[-1]))]
        for t in range(T - 1, 0, -1):
            path.append(psi[t, path[-1]])
        path.reverse()
        return [self.states[s] for s in path]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_confidence_score(self, state_probs: np.ndarray) -> float:
        """SdC = 10 × max P(s | O)."""
        return float(np.max(state_probs)) * 10

    def get_dominant_state(self, state_probs: np.ndarray) -> Tuple[str, float]:
        """Return (state_name, probability) for the dominant state."""
        idx = int(np.argmax(state_probs))
        return self.states[idx], float(state_probs[idx])


if __name__ == "__main__":
    print("=" * 70)
    print("HSMM — P4 5-state + Liquidation demo")
    print("=" * 70)

    np.random.seed(42)
    n = 500
    dates = pd.date_range('2021-01-01', periods=n, freq='1h')
    close = 40000 + np.cumsum(np.random.randn(n) * 200)
    tr = np.abs(np.random.randn(n) * 150) + 50

    data = pd.DataFrame({
        'close':      close,
        'high':       close + np.abs(np.random.randn(n) * 100),
        'low':        close - np.abs(np.random.randn(n) * 100),
        'volume':     np.abs(np.random.randn(n) * 500) + 100,
        'returns':    np.concatenate([[0], np.diff(close) / close[:-1]]),
        'atr_14':     pd.Series(tr).rolling(14).mean().values,
        'atr_50':     pd.Series(tr).rolling(50).mean().values,
        'sma_20':     pd.Series(close).rolling(20).mean().values,
        'sma_50':     pd.Series(close).rolling(50).mean().values,
        'volume_ma20': pd.Series(np.abs(np.random.randn(n) * 500) + 100).rolling(20).mean().values,
    }, index=dates)

    for state_set, label in [
        (['Trend+', 'Range', 'Trend-'], '3-state'),
        (['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution'], '5-state'),
        (['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation'], '6-state'),
    ]:
        hsmm = SemiMarkovHMM(states=state_set)
        hsmm.initialize_parameters(data)

        obs = [{'price': data['returns'].iloc[i], 'atr': data['atr_14'].iloc[i]}
               for i in range(50)]
        probs = hsmm.forward_backward(obs)
        dom, p = hsmm.get_dominant_state(probs[-1])
        print(f"\n{label}: dominant={dom} ({p:.2%})  A shape={hsmm.transition_matrix.shape}")
        print(f"  Liquidation row: {hsmm.transition_matrix[-1] if 'Liquidation' in state_set else 'N/A'}")

    print("\n✅ P4 HSMM Working")
