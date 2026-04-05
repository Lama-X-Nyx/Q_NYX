"""
Semi-Markov Hidden Markov Model (HSMM)
Advanced state detection with duration modeling

P4a: 5-state support — Trend+, Range, Trend-, Squeeze, Distribution
P4b: Liquidation as a nearly-absorbing 6th state
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Fast logsumexp — drop-in for scipy.special.logsumexp but without the
# array-API dispatch overhead that dominates cost for small (T×n) matrices.
# ---------------------------------------------------------------------------
def _logsumexp(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    # Replace -inf peaks with 0 to avoid nan in exp
    a_max_safe = np.where(np.isneginf(a_max), 0.0, a_max)
    out = np.log(np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=keepdims))
    peak = a_max_safe if keepdims else a_max_safe.squeeze(axis=axis)
    return out + peak


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

        # Speed caches (backtest hot path — zero impact on trade logic)
        self._init_fingerprint: Optional[int] = None
        self._fb_fingerprint:   Optional[int] = None
        self._fb_cache:         Optional[np.ndarray] = None

        # Locked after Baum-Welch EM: initialize_parameters() becomes a no-op
        # so pre-trained params are not overwritten by heuristic re-init each bar.
        self._locked: bool = False

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

        Speed note: if called with identical data (same tail fingerprint as the
        previous call) the method returns immediately — parameters are unchanged.
        This is transparent to callers and does not affect trade logic.

        Lock note: if fit() has been called (Baum-Welch EM), this method is a
        no-op — the learned parameters are preserved for OOS inference.
        Call unlock() to re-enable heuristic re-init (e.g. for online retraining).
        """
        if self._locked:
            return  # EM pre-trained — keep learned parameters

        # Fingerprint: hash of the last 10 close prices (cheap, stable proxy)
        fp = hash(data['close'].iloc[-10:].values.tobytes()) if len(data) >= 10 else None
        if fp is not None and fp == self._init_fingerprint and self.emission_params is not None:
            return  # Data unchanged — reuse existing parameters
        self._init_fingerprint = fp

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

        # ---- Fully vectorised labeling (no iterrows) ----
        sma20 = df['sma_20'].values
        sma50 = df['sma_50'].values
        close = df['close'].values
        nan_mask = np.isnan(sma20) | np.isnan(sma50)

        # Base: Range everywhere (default)
        labels = np.full(len(df), 'Range', dtype=object)

        # Trend detection (lowest priority among valid bars)
        valid = ~nan_mask
        trend_up   = valid & (sma20 > sma50) & (close > sma20)
        trend_down = valid & (sma20 < sma50) & (close < sma20)
        labels[trend_up]   = 'Trend+'
        labels[trend_down] = 'Trend-'

        # Higher-priority layers applied on top (overwrite lower priority)
        if has_distribution:
            dm = self._compute_distribution_mask(df)
            if dm is not None:
                labels[dm.fillna(False).values] = 'Distribution'

        if has_squeeze:
            sm = self._compute_squeeze_mask(df)
            if sm is not None:
                labels[sm.fillna(False).values] = 'Squeeze'

        if has_liquidation:
            lm = self._compute_liquidation_mask(df)
            if lm is not None:
                labels[lm.fillna(False).values] = 'Liquidation'

        # NaN warm-up bars always → Range (applied last to guarantee correctness)
        labels[nan_mask] = 'Range'

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

    def _compute_log_B(self, observations: List[Dict]) -> np.ndarray:
        """
        Pre-compute full emission matrix log_B (T × n_states).
        Fully vectorised over both T and n_states — no Python loops.
        """
        T = len(observations)

        # Param arrays (n_states,)
        price_mu    = np.array([self.emission_params[s]['price_mu']    for s in self.states])
        price_sigma = np.array([self.emission_params[s]['price_sigma'] for s in self.states])
        atr_mu      = np.array([self.emission_params[s]['atr_mu']      for s in self.states])
        atr_sigma   = np.array([self.emission_params[s]['atr_sigma']   for s in self.states])

        # Observation arrays (T,)
        prices = np.array([obs.get('price', np.nan) for obs in observations])
        atrs   = np.array([obs.get('atr',   np.nan) for obs in observations])

        # Vectorised logpdf: broadcast (T,1) × (1,n) → (T,n)
        log_B = np.zeros((T, self.n_states))

        valid_p = ~np.isnan(prices)
        if valid_p.any():
            log_B[valid_p] += stats.norm.logpdf(
                prices[valid_p, np.newaxis],
                loc=price_mu[np.newaxis, :],
                scale=price_sigma[np.newaxis, :]
            )

        valid_a = ~np.isnan(atrs)
        if valid_a.any():
            log_B[valid_a] += stats.norm.logpdf(
                atrs[valid_a, np.newaxis],
                loc=atr_mu[np.newaxis, :],
                scale=atr_sigma[np.newaxis, :]
            )

        return log_B

    def forward_backward(self, observations: List[Dict]) -> np.ndarray:
        """
        Vectorised Forward-Backward algorithm — works for any number of states.

        Replaces the original O(T × n²) Python-loop version with pure numpy
        broadcasting: the inner per-state logsumexp becomes a single matrix op.

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

        # FB result cache: if the observations tail is identical to last call,
        # the gamma matrix is unchanged — return cached result immediately.
        tail = observations[-min(5, T):]
        fb_fp = hash(tuple(
            (o.get('price', 0.0), o.get('atr', 0.0)) for o in tail
        ))
        if fb_fp == self._fb_fingerprint and self._fb_cache is not None:
            # Cache hit — shape may differ if T changed; validate
            if self._fb_cache.shape[0] == T:
                return self._fb_cache
        self._fb_fingerprint = fb_fp

        # Pre-compute emission matrix and log-transition matrix once
        log_B = self._compute_log_B(observations)                  # (T, n)
        log_A = np.log(self.transition_matrix + 1e-10)             # (n, n)  A[sp, s]

        # Forward pass — vectorised over states
        log_alpha = np.full((T, self.n_states), -np.inf)
        log_alpha[0] = np.log(self.initial_probs + 1e-10) + log_B[0]

        for t in range(1, T):
            # log_alpha[t, s] = logsumexp_sp( log_alpha[t-1, sp] + log_A[sp, s] ) + log_B[t, s]
            # log_alpha[t-1, :, None] + log_A  has shape (n, n); logsumexp over axis=0
            log_alpha[t] = _logsumexp(
                log_alpha[t-1, :, np.newaxis] + log_A, axis=0
            ) + log_B[t]

        # Backward pass — vectorised over states
        log_beta = np.zeros((T, self.n_states))   # log(1) = 0 at T-1

        for t in range(T - 2, -1, -1):
            # log_beta[t, s] = logsumexp_sn( log_A[s, sn] + log_B[t+1, sn] + log_beta[t+1, sn] )
            # log_A + log_B[t+1] + log_beta[t+1]  has shape (n, n); logsumexp over axis=1
            log_beta[t] = _logsumexp(
                log_A + log_B[t+1] + log_beta[t+1], axis=1
            )

        log_gamma = log_alpha + log_beta
        gamma = np.exp(log_gamma - _logsumexp(log_gamma, axis=1, keepdims=True))
        self._fb_cache = gamma  # store for next call
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
    # Baum-Welch EM
    # ------------------------------------------------------------------

    def _df_to_observations(self, data: pd.DataFrame) -> List[Dict]:
        """Convert a prepared DataFrame to the List[Dict] format used by F-B."""
        obs = []
        returns_col = data['returns'].values if 'returns' in data.columns else np.full(len(data), np.nan)
        atr_col     = data['atr_14'].values  if 'atr_14'   in data.columns else np.full(len(data), np.nan)
        for r, a in zip(returns_col, atr_col):
            obs.append({
                'price': float(r) if not np.isnan(r) else np.nan,
                'atr':   float(a) if not np.isnan(a) else np.nan,
            })
        return obs

    def _compute_xi(
        self,
        log_alpha: np.ndarray,
        log_beta:  np.ndarray,
        log_B:     np.ndarray,
    ) -> np.ndarray:
        """
        ξ(t, i, j) = P(q_t=i, q_{t+1}=j | O)   shape (T-1, n, n)

        Fully vectorised — no Python loops.
        """
        T = log_alpha.shape[0]
        log_A = np.log(self.transition_matrix + 1e-10)

        # (T-1, n, 1) + (1, n, n) + (T-1, 1, n) + (T-1, 1, n)  → (T-1, n, n)
        log_xi = (
            log_alpha[:-1, :, np.newaxis]
            + log_A[np.newaxis, :, :]
            + log_B[1:, np.newaxis, :]
            + log_beta[1:, np.newaxis, :]
        )
        # Normalise each timestep slice
        log_Z = _logsumexp(log_xi.reshape(T - 1, -1), axis=1)          # (T-1,)
        xi = np.exp(log_xi - log_Z[:, np.newaxis, np.newaxis])          # (T-1, n, n)
        return xi

    def _m_step(
        self,
        gamma:    np.ndarray,
        xi:       np.ndarray,
        prices:   np.ndarray,
        atrs:     np.ndarray,
    ) -> None:
        """
        M-step: update π, A, and Gaussian emission params.

        gamma  : (T, n)     — smoothed state posteriors
        xi     : (T-1, n, n) — joint transition posteriors
        prices : (T,)        — return observations (may contain NaN)
        atrs   : (T,)        — ATR observations   (may contain NaN)
        """
        EPS = 1e-8

        # ---- Initial distribution ----
        self.initial_probs = gamma[0] / (gamma[0].sum() + EPS)

        # ---- Transition matrix ----
        A_num = xi.sum(axis=0)                                  # (n, n)
        A_den = A_num.sum(axis=1, keepdims=True) + EPS          # (n, 1)
        A_new = A_num / A_den
        # Floor at 1e-4 to prevent degenerate absorbing states
        A_new = np.clip(A_new, 1e-4, None)
        self.transition_matrix = A_new / A_new.sum(axis=1, keepdims=True)

        # ---- Emission params (Gaussian, per state) ----
        for i, state in enumerate(self.states):
            w = gamma[:, i]                                     # (T,)

            # Price
            vp = ~np.isnan(prices)
            w_p = w[vp]; p = prices[vp]
            if w_p.sum() > EPS:
                mu_p  = np.dot(w_p, p) / w_p.sum()
                sig_p = np.sqrt(np.dot(w_p, (p - mu_p) ** 2) / w_p.sum())
            else:
                mu_p  = self.emission_params[state]['price_mu']
                sig_p = self.emission_params[state]['price_sigma']

            # ATR
            va = ~np.isnan(atrs)
            w_a = w[va]; a = atrs[va]
            if w_a.sum() > EPS:
                mu_a  = np.dot(w_a, a) / w_a.sum()
                sig_a = np.sqrt(np.dot(w_a, (a - mu_a) ** 2) / w_a.sum())
            else:
                mu_a  = self.emission_params[state]['atr_mu']
                sig_a = self.emission_params[state]['atr_sigma']

            self.emission_params[state] = {
                'price_mu':    float(mu_p),
                'price_sigma': float(max(sig_p, 1e-4)),
                'atr_mu':      float(mu_a),
                'atr_sigma':   float(max(sig_a, 0.01)),
            }

    def fit(
        self,
        observations: List[Dict],
        n_iter: int = 30,
        tol:    float = 1e-4,
    ) -> List[float]:
        """
        Baum-Welch EM — learn A, B, π from observations.

        Requires initialize_parameters() to have been called first
        (heuristic warm-start avoids bad local optima).

        Returns list of log-likelihoods per iteration.
        """
        if self.emission_params is None:
            raise RuntimeError("Call initialize_parameters() before fit().")

        T = len(observations)
        if T < 2 * self.n_states:
            return []

        prices = np.array([o.get('price', np.nan) for o in observations])
        atrs   = np.array([o.get('atr',   np.nan) for o in observations])

        log_A  = np.log(self.transition_matrix + 1e-10)
        ll_history: List[float] = []
        prev_ll = -np.inf

        for iteration in range(n_iter):
            # ---- E-step ----
            log_B = self._compute_log_B(observations)

            # Forward
            log_alpha = np.full((T, self.n_states), -np.inf)
            log_alpha[0] = np.log(self.initial_probs + 1e-10) + log_B[0]
            for t in range(1, T):
                log_alpha[t] = _logsumexp(
                    log_alpha[t - 1, :, np.newaxis] + log_A, axis=0
                ) + log_B[t]

            # Backward
            log_beta = np.zeros((T, self.n_states))
            for t in range(T - 2, -1, -1):
                log_beta[t] = _logsumexp(
                    log_A + log_B[t + 1] + log_beta[t + 1], axis=1
                )

            # Log-likelihood
            ll = float(_logsumexp(log_alpha[-1]))
            ll_history.append(ll)

            # γ
            log_gamma = log_alpha + log_beta
            gamma = np.exp(
                log_gamma - _logsumexp(log_gamma, axis=1, keepdims=True)
            )

            # ξ
            xi = self._compute_xi(log_alpha, log_beta, log_B)

            # ---- M-step ----
            self._m_step(gamma, xi, prices, atrs)

            # Refresh log_A with updated matrix
            log_A = np.log(self.transition_matrix + 1e-10)

            # Convergence check
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        # Invalidate speed caches — parameters changed
        self._init_fingerprint = None
        self._fb_fingerprint   = None
        self._fb_cache         = None

        # Lock: prevent heuristic re-init from overwriting learned params
        self._locked = True

        return ll_history

    def unlock(self) -> None:
        """
        Re-enable heuristic re-init (e.g. before periodic re-training in live trading).
        After calling this, initialize_parameters() will run normally again.
        """
        self._locked = False

    def initialize_parameters_with_em(
        self,
        data:   pd.DataFrame,
        n_iter: int = 30,
        tol:    float = 1e-4,
    ) -> List[float]:
        """
        Full training: heuristic warm-start → Baum-Welch EM.

        Call once on historical training data (e.g., 6-12 months).
        Afterwards, use forward_backward() for online inference without
        calling initialize_parameters() again.

        Returns EM log-likelihood history.
        """
        # Warm-start: heuristic labels + prior transition
        self.initialize_parameters(data)

        # Build observations from prepared data
        observations = self._df_to_observations(data)
        if len(observations) < 2 * self.n_states:
            return []

        return self.fit(observations, n_iter=n_iter, tol=tol)

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
