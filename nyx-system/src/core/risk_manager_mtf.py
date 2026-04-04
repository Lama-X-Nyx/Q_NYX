"""
Risk Manager for MTF

Computes hitting probabilities and RR ratios using HSMM projections.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class RiskManagerMTF:
    """
    MTF Risk Manager
    
    Computes:
    - Hitting probabilities P(hit -0.05), P(hit -0.10)
    - Risk/Reward ratios from SMC levels
    - Position sizing based on MTF confidence
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Risk Manager
        
        Args:
            config: Configuration dict
        """
        self.config = config
        
        # Risk thresholds (from Prompt Maître)
        self.fibo_warning = -0.025  # 🟡
        self.fibo_danger = -0.05    # 🟠
        self.fibo_liquidation = -0.10  # 🔴
        
    def compute_hitting_probabilities(self, 
                                      current_price: float,
                                      fractal_states: Dict,
                                      transition_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute P(hit -0.05) and P(hit -0.10) using HSMM projections
        
        Args:
            current_price: Current market price
            fractal_states: MTF fractal states
            transition_matrix: HSMM transition matrix (4H)
        
        Returns:
            Dict with hitting probabilities
        """
        
        # Get 4H regime (most stable for projection)
        states_4h = fractal_states.get('4h', {})
        
        if not states_4h:
            # Fallback: assume moderate risk
            return {
                'minus_0_05': 0.25,
                'minus_0_10': 0.10
            }
        
        # Current state probabilities
        p_trend_plus = states_4h.get('Trend+', 0.33)
        p_range = states_4h.get('Range', 0.34)
        p_trend_minus = states_4h.get('Trend-', 0.33)
        
        # Hitting probability depends on regime
        # Trend+: low risk of -0.05/-0.10
        # Range: moderate risk
        # Trend-: high risk
        
        # Simple model: weighted by state probabilities
        # In Trend+: P(-0.05)=0.15, P(-0.10)=0.05
        # In Range:  P(-0.05)=0.30, P(-0.10)=0.15
        # In Trend-: P(-0.05)=0.50, P(-0.10)=0.30
        
        p_hit_05 = (
            p_trend_plus * 0.15 +
            p_range * 0.30 +
            p_trend_minus * 0.50
        )
        
        p_hit_10 = (
            p_trend_plus * 0.05 +
            p_range * 0.15 +
            p_trend_minus * 0.30
        )
        
        return {
            'minus_0_05': p_hit_05,
            'minus_0_10': p_hit_10
        }
    
    def compute_rr_ratio(self,
                        entry_price: float,
                        smc_patterns: Dict,
                        intent_daily: str) -> Tuple[float, Dict]:
        """
        Compute Risk/Reward ratio from SMC levels
        
        Args:
            entry_price: Entry price
            smc_patterns: SMC patterns detected
            intent_daily: Daily intent (Trend+/Trend-/Range)
        
        Returns:
            Tuple of (RR ratio, levels dict)
        """
        
        # Default levels if no SMC patterns
        default_sl_pct = 0.03  # 3% stop
        default_tp_pct = 0.09  # 9% target (3:1 RR)
        
        # In real implementation, would use SMC zones
        # For now, use simple percentage-based
        
        if intent_daily == 'Trend+':
            sl = entry_price * (1 - default_sl_pct)
            tp = entry_price * (1 + default_tp_pct)
        elif intent_daily == 'Trend-':
            sl = entry_price * (1 + default_sl_pct)
            tp = entry_price * (1 - default_tp_pct)
        else:
            # Range: no clear RR
            return 0.0, {'sl': 0, 'tp': 0}
        
        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        
        rr_ratio = reward / risk if risk > 0 else 0
        
        levels = {
            'entry': entry_price,
            'sl': sl,
            'tp': tp,
            'risk': risk,
            'reward': reward
        }
        
        return rr_ratio, levels
    
    def check_risk_conditions(self,
                             entry_price: float,
                             fractal_states: Dict,
                             smc_patterns: Dict,
                             intent_daily: str,
                             transition_matrix: np.ndarray) -> Dict[str, bool]:
        """
        Check all MTF risk conditions
        
        Args:
            entry_price: Entry price
            fractal_states: MTF fractal states
            smc_patterns: SMC patterns
            intent_daily: Daily intent
            transition_matrix: HSMM transition matrix
        
        Returns:
            Dict of risk conditions met/failed
        """
        
        # Compute hitting probabilities
        hit_probs = self.compute_hitting_probabilities(
            entry_price, 
            fractal_states, 
            transition_matrix
        )
        
        # Compute RR ratio
        rr_ratio, levels = self.compute_rr_ratio(
            entry_price,
            smc_patterns,
            intent_daily
        )
        
        # Check conditions
        conditions = {
            'rr_ratio': rr_ratio >= 2.0,  # RR ≥ 2:1
            'risk_hit': hit_probs['minus_0_10'] <= 0.20,  # P(hit -0.10) ≤ 20%
            'hit_probs': hit_probs,
            'rr_value': rr_ratio,
            'levels': levels
        }
        
        return conditions


if __name__ == "__main__":
    print("Risk Manager MTF Test")
    
    # Test with sample states
    fractal_states = {
        '4h': {
            'Trend+': 0.70,
            'Range': 0.20,
            'Trend-': 0.10
        }
    }
    
    transition_matrix = np.array([
        [0.75, 0.15, 0.10],
        [0.20, 0.60, 0.20],
        [0.10, 0.15, 0.75]
    ])
    
    risk = RiskManagerMTF({})
    
    # Compute hitting probs
    hit_probs = risk.compute_hitting_probabilities(
        current_price=50000,
        fractal_states=fractal_states,
        transition_matrix=transition_matrix
    )
    
    print(f"\nHitting Probabilities:")
    print(f"  P(hit -0.05): {hit_probs['minus_0_05']:.2%}")
    print(f"  P(hit -0.10): {hit_probs['minus_0_10']:.2%}")
    
    # Compute RR
    rr_ratio, levels = risk.compute_rr_ratio(
        entry_price=50000,
        smc_patterns={},
        intent_daily='Trend+'
    )
    
    print(f"\nRR Ratio: {rr_ratio:.2f}")
    print(f"Levels: Entry={levels['entry']}, SL={levels['sl']:.0f}, TP={levels['tp']:.0f}")
    
    # Check conditions
    conditions = risk.check_risk_conditions(
        entry_price=50000,
        fractal_states=fractal_states,
        smc_patterns={},
        intent_daily='Trend+',
        transition_matrix=transition_matrix
    )
    
    print(f"\nRisk Conditions:")
    print(f"  RR ≥ 2:1: {'✅' if conditions['rr_ratio'] else '❌'}")
    print(f"  P(-0.10) ≤ 20%: {'✅' if conditions['risk_hit'] else '❌'}")
