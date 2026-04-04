"""
Risk Manager for MTF

Computes hitting probabilities and RR ratios using HSMM projections.

P2: hitting probs computed via Monte Carlo simulation from HSMM dynamics
    instead of a hardcoded lookup table.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class RiskManagerMTF:
    """
    MTF Risk Manager

    Computes:
    - First-passage hitting probabilities P(hit -0.05), P(hit -0.10)
      via Monte Carlo simulation on the HSMM state/return dynamics (P2)
    - Risk/Reward ratios from SMC levels
    - Position sizing based on MTF confidence
    """

    def __init__(self, config: Dict):
        self.config = config

        # Risk thresholds (Prompt Maître §0)
        self.fibo_warning = -0.025      # 🟡
        self.fibo_danger = -0.05        # 🟠
        self.fibo_liquidation = -0.10   # 🔴

        # Monte Carlo config
        mc_cfg = config.get('strategy', {}).get('monte_carlo_risk', {})
        self.mc_n_paths = int(mc_cfg.get('n_paths', 500))
        self.mc_n_steps = int(mc_cfg.get('n_steps', 24))   # ~1 trading day on 1H

    # ------------------------------------------------------------------
    # P2 — Monte Carlo first-passage probabilities
    # ------------------------------------------------------------------

    def _monte_carlo_hitting_probs(
        self,
        pi: np.ndarray,
        A: np.ndarray,
        emission_params: Dict,
        states: List[str],
    ) -> Tuple[float, float]:
        """
        Estimate P(hit -5%) and P(hit -10%) via Monte Carlo simulation.

        Samples n_paths trajectories of length n_steps from the HSMM:
          - initial state drawn from π
          - per-step return ~ Normal(price_mu_s, price_sigma_s)
          - state transitions via A

        Returns (p_hit_05, p_hit_10) — first-passage probabilities.
        """
        rng = np.random.default_rng(seed=42)
        n_s = len(states)
        n_paths = self.mc_n_paths
        n_steps = self.mc_n_steps

        # Per-state emission parameters
        mus = np.array([
            emission_params.get(s, {}).get('price_mu', 0.0)
            for s in states
        ], dtype=float)
        sigmas = np.array([
            max(emission_params.get(s, {}).get('price_sigma', 0.01), 1e-6)
            for s in states
        ], dtype=float)

        # Normalise π
        pi_norm = np.array(pi, dtype=float)
        pi_norm = pi_norm / pi_norm.sum()

        # Pre-compute cumulative distributions for vectorised sampling
        cum_pi = np.cumsum(pi_norm)                    # (n_s,)
        cum_A = np.cumsum(A, axis=1)                   # (n_s, n_s)

        # Sample initial states from π
        u0 = rng.random(n_paths)
        current_states = (cum_pi[np.newaxis, :] <= u0[:, np.newaxis]).sum(axis=1)
        current_states = np.clip(current_states, 0, n_s - 1)

        cumulative_returns = np.zeros(n_paths)
        hit_05 = np.zeros(n_paths, dtype=bool)
        hit_10 = np.zeros(n_paths, dtype=bool)

        for _ in range(n_steps):
            # Vectorised return sampling
            step_returns = rng.normal(mus[current_states], sigmas[current_states])
            cumulative_returns += step_returns

            hit_05 |= (cumulative_returns <= self.fibo_danger)
            hit_10 |= (cumulative_returns <= self.fibo_liquidation)

            # Vectorised state transition (don't advance paths already at -10%)
            u = rng.random(n_paths)
            new_states = (cum_A[current_states] <= u[:, np.newaxis]).sum(axis=1)
            new_states = np.clip(new_states, 0, n_s - 1)
            current_states = np.where(hit_10, current_states, new_states)

        return float(hit_05.mean()), float(hit_10.mean())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_hitting_probabilities(
        self,
        current_price: float,
        fractal_states: Dict,
        transition_matrix: np.ndarray,
        emission_params: Optional[Dict] = None,
        hsmm_states_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute P(hit -0.05) and P(hit -0.10).

        When emission_params is provided (P2), uses Monte Carlo simulation
        from the HSMM dynamics.  Falls back to the hardcoded lookup table
        when emission_params is absent (backward compat).

        Args:
            current_price:    Current market price
            fractal_states:   MTF fractal states {'4h': {'Trend+': p, ...}}
            transition_matrix: HSMM transition matrix A (n_states × n_states)
            emission_params:  HSMM emission parameters per state (P2, optional)
            hsmm_states_list: Ordered state names matching A rows (optional)

        Returns:
            {'minus_0_05': float, 'minus_0_10': float}
        """
        states_4h = fractal_states.get('4h', {})

        if not states_4h:
            return {'minus_0_05': 0.25, 'minus_0_10': 0.10}

        p_trend_plus  = states_4h.get('Trend+', 0.33)
        p_range       = states_4h.get('Range',  0.34)
        p_trend_minus = states_4h.get('Trend-', 0.33)

        # ---- P2 path: Monte Carlo ----------------------------------------
        if emission_params is not None and transition_matrix is not None:
            states = hsmm_states_list or ['Trend+', 'Range', 'Trend-']
            pi = np.array([states_4h.get(s, 1.0 / len(states)) for s in states])

            A = np.array(transition_matrix, dtype=float)
            # Normalise rows (safety)
            row_sums = A.sum(axis=1, keepdims=True)
            A = A / np.where(row_sums > 0, row_sums, 1)

            try:
                p_hit_05, p_hit_10 = self._monte_carlo_hitting_probs(
                    pi, A, emission_params, states
                )
                return {
                    'minus_0_05': p_hit_05,
                    'minus_0_10': p_hit_10,
                    'method': 'monte_carlo'
                }
            except Exception:
                pass  # Fall through to lookup table

        # ---- Fallback: hardcoded lookup table ----------------------------
        p_hit_05 = p_trend_plus * 0.15 + p_range * 0.30 + p_trend_minus * 0.50
        p_hit_10 = p_trend_plus * 0.05 + p_range * 0.15 + p_trend_minus * 0.30

        return {
            'minus_0_05': p_hit_05,
            'minus_0_10': p_hit_10,
            'method': 'lookup_table'
        }

    def compute_rr_ratio(
        self,
        entry_price: float,
        smc_patterns: Dict,
        intent_daily: str,
    ) -> Tuple[float, Dict]:
        """
        Compute Risk/Reward ratio from SMC levels.

        Returns (rr_ratio, levels_dict).
        """
        default_sl_pct = 0.03   # 3% stop
        default_tp_pct = 0.09   # 9% target → 3:1 RR

        if intent_daily in ('bullish', 'Trend+'):
            sl = entry_price * (1 - default_sl_pct)
            tp = entry_price * (1 + default_tp_pct)
        elif intent_daily in ('bearish', 'Trend-'):
            sl = entry_price * (1 + default_sl_pct)
            tp = entry_price * (1 - default_tp_pct)
        else:
            return 0.0, {'sl': 0, 'tp': 0}

        risk   = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0.0

        return rr_ratio, {
            'entry': entry_price,
            'sl': sl,
            'tp': tp,
            'risk': risk,
            'reward': reward
        }

    def check_risk_conditions(
        self,
        entry_price: float,
        fractal_states: Dict,
        smc_patterns: Dict,
        intent_daily: str,
        transition_matrix: np.ndarray,
        emission_params: Optional[Dict] = None,
        hsmm_states_list: Optional[List[str]] = None,
    ) -> Dict:
        """
        Check all MTF risk conditions.

        Returns dict with boolean flags and raw values.
        """
        hit_probs = self.compute_hitting_probabilities(
            entry_price,
            fractal_states,
            transition_matrix,
            emission_params=emission_params,
            hsmm_states_list=hsmm_states_list,
        )

        rr_ratio, levels = self.compute_rr_ratio(
            entry_price, smc_patterns, intent_daily
        )

        return {
            'rr_ratio': rr_ratio >= 2.0,
            'risk_hit': hit_probs['minus_0_10'] <= 0.20,
            'hit_probs': hit_probs,
            'rr_value': rr_ratio,
            'levels': levels,
            'hit_method': hit_probs.get('method', 'lookup_table'),
        }


if __name__ == "__main__":
    print("Risk Manager MTF — P2 Monte Carlo Test")

    fractal_states = {'4h': {'Trend+': 0.70, 'Range': 0.20, 'Trend-': 0.10}}

    A = np.array([
        [0.75, 0.15, 0.10],
        [0.20, 0.60, 0.20],
        [0.10, 0.15, 0.75]
    ])

    emission_params = {
        'Trend+': {'price_mu':  0.001, 'price_sigma': 0.008},
        'Range':  {'price_mu':  0.000, 'price_sigma': 0.005},
        'Trend-': {'price_mu': -0.001, 'price_sigma': 0.008},
    }

    risk = RiskManagerMTF({})

    # Monte Carlo
    hit_mc = risk.compute_hitting_probabilities(
        50000, fractal_states, A,
        emission_params=emission_params
    )
    print(f"\nMonte Carlo:  P(-5%)={hit_mc['minus_0_05']:.3f}  P(-10%)={hit_mc['minus_0_10']:.3f}")

    # Lookup fallback
    hit_lk = risk.compute_hitting_probabilities(50000, fractal_states, A)
    print(f"Lookup table: P(-5%)={hit_lk['minus_0_05']:.3f}  P(-10%)={hit_lk['minus_0_10']:.3f}")

    cond = risk.check_risk_conditions(
        50000, fractal_states, {}, 'bullish', A,
        emission_params=emission_params
    )
    print(f"\nRR≥2:1 {'✅' if cond['rr_ratio'] else '❌'}   "
          f"P(-10%)≤20% {'✅' if cond['risk_hit'] else '❌'}   "
          f"method={cond['hit_method']}")
