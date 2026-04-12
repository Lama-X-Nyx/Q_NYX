"""
NYX Engine - Multi-Timeframe Edition

4-TF fractal swing agent following Prompt Maître v5.5M specification.

Timeframe Hierarchy:
- 1D: Strategic Intent & Bias (Intent_Daily)
- 4H: Operational Stability (HSMM primary, Stability check)
- 1H: Intermediate Bridge (flip detection)
- 15M: SMC Setup (OB, FVG, BoS/CHoCH, Alignment)
- 5M: Entry Timing (optional)

Entry Conditions (ALL required):
1. SdC > 5
2. Stability_4H ≥ 0.60
3. Alignment_15M ≥ 0.60 (in direction of Intent_Daily)
4. RR ≥ 2:1
5. macro_block = OFF
6. P(hit -0.10) ≤ 0.20
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class NYXEngineMTF:
    """
    NYX Multi-Timeframe Engine
    
    Implements 4-TF fractal architecture
    """
    
    def __init__(self, config: Dict):
        """
        Initialize MTF engine
        
        Args:
            config: Configuration dict with mtf section
        """
        self.config = config
        
        # Extract MTF config
        self.mtf_config = config.get('mtf', {})
        self.weights = self.mtf_config.get('weights', {})
        self.timeframes = self.mtf_config.get('timeframes', {})
        self.mtf_conditions = config.get('strategy', {}).get('mtf_conditions', {})
        
        # Import HSMM
        from src.core.hsmm import SemiMarkovHMM
        self.hsmm = SemiMarkovHMM(states=['Trend+', 'Range', 'Trend-'])
        
        # Import SMC
        from src.core.smc import SMCDetector
        self.smc = SMCDetector()
        
        # Import Risk Manager
        from src.core.risk_manager_mtf import RiskManagerMTF
        self.risk_manager = RiskManagerMTF(config)
        
        # Other modules (placeholders for now)
        self.macro_filter = None
    
    def generate_signal_mtf(self, pair: str, mtf_data: Dict[str, pd.DataFrame],
                           current_date: str) -> Dict:
        """
        Generate signal from multi-timeframe data
        
        Args:
            pair: Trading pair
            mtf_data: Dict mapping TF to DataFrame
            current_date: Current timestamp
        
        Returns:
            Signal dict with MTF analysis
        """
        
        # Initialize signal
        signal = {
            'pair': pair,
            'timestamp': current_date,
            'action': 'HOLD',
            'confidence': 0,
            'fractal_states': {},
            'intent_daily': None,
            'stability_4h': 0,
            'alignment_15m': 0,
            'score_fl': 0,
            'sdc': 0,
            'conditions_met': {},
            'reasons': []
        }
        
        # Check minimum data
        for tf, df in mtf_data.items():
            if len(df) < 50:
                signal['reasons'].append(f'Insufficient {tf} data ({len(df)} bars)')
                return signal
        
        # Step 1: Compute fractal states for each TF
        signal['fractal_states'] = self._compute_fractal_states(mtf_data)
        
        # Step 2: Compute Intent_Daily (from 1D HSMM states)
        signal['intent_daily'] = self._compute_intent_daily(
            mtf_data['1d'],
            signal['fractal_states'].get('1d', {})
        )
        
        # Step 3: Compute Stability_4H (from 4H HSMM)
        signal['stability_4h'] = self._compute_stability_4h(mtf_data['4h'])
        
        # Step 4: Compute Alignment_15M (15M trend alignment with Intent_Daily)
        signal['alignment_15m'], signal['smc_patterns'] = self._compute_alignment_15m(
            mtf_data['15m'], 
            signal['intent_daily']
        )
        
        # Step 5: Compute fractal score
        signal['score_fl'] = self._compute_fractal_score(signal['fractal_states'])
        
        # Step 6: Compute SdC (Score de Confiance)
        signal['sdc'] = self._compute_sdc(signal['intent_daily'], signal['fractal_states'])
        
        # Step 7: Check MTF entry conditions
        conditions = self._check_mtf_conditions(signal, mtf_data)
        signal['conditions_met'] = conditions
        
        # Step 8: Determine action
        if all(conditions.values()):
            # All conditions met - generate entry signal
            signal = self._generate_entry_signal(signal, mtf_data)
        else:
            # Some condition failed
            failed = [k for k, v in conditions.items() if not v]
            signal['action'] = 'HOLD'
            signal['reasons'].append(f"MTF conditions not met: {', '.join(failed)}")
        
        return signal
    
    def _compute_fractal_states(self, mtf_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Compute state probabilities for each timeframe using HSMM
        
        Args:
            mtf_data: Dict of TF DataFrames
        
        Returns:
            Dict mapping TF to state probabilities
        """
        
        fractal_states = {}
        
        for tf, df in mtf_data.items():
            if len(df) < 50:
                continue
            
            # Prepare data for HSMM
            df_prepared = self._prepare_hsmm_data(df)
            
            # Initialize HSMM with this TF's data
            self.hsmm.initialize_parameters(df_prepared)
            
            # Build observations (last 50 bars)
            window_size = min(50, len(df_prepared))
            window = df_prepared.tail(window_size)
            
            observations = []
            for idx in range(len(window)):
                obs = {
                    'price': window.iloc[idx]['returns'] if 'returns' in window.columns else 0.0,
                    'atr': window.iloc[idx]['atr_14'] if 'atr_14' in window.columns else window.iloc[idx]['close'] * 0.02
                }
                observations.append(obs)
            
            # Run forward-backward
            try:
                state_probs = self.hsmm.forward_backward(observations)
                
                if len(state_probs) > 0:
                    # Get current state probabilities
                    current_probs = state_probs[-1]
                    
                    states = {
                        'Trend+': current_probs[0],
                        'Range': current_probs[1],
                        'Trend-': current_probs[2]
                    }
                else:
                    states = {'Trend+': 0.33, 'Range': 0.34, 'Trend-': 0.33}
                
            except Exception as e:
                # Fallback to simplified
                close = df['close'].values
                sma_20 = pd.Series(close).rolling(20).mean().values
                sma_50 = pd.Series(close).rolling(50).mean().values
                
                if len(sma_20) > 0 and len(sma_50) > 0 and not np.isnan(sma_20[-1]) and not np.isnan(sma_50[-1]):
                    trend_up = sma_20[-1] > sma_50[-1]
                    trend_down = sma_20[-1] < sma_50[-1]
                else:
                    trend_up = False
                    trend_down = False
                
                if trend_up:
                    states = {'Trend+': 0.65, 'Range': 0.25, 'Trend-': 0.10}
                elif trend_down:
                    states = {'Trend+': 0.10, 'Range': 0.25, 'Trend-': 0.65}
                else:
                    states = {'Trend+': 0.30, 'Range': 0.40, 'Trend-': 0.30}
            
            fractal_states[tf] = states
        
        return fractal_states
    
    def _prepare_hsmm_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataframe for HSMM (add returns and ATR if missing)
        
        Args:
            df: Raw OHLCV DataFrame
        
        Returns:
            DataFrame with returns and atr_14
        """
        
        df_prepared = df.copy()
        
        # Add returns if missing
        if 'returns' not in df_prepared.columns:
            df_prepared['returns'] = df_prepared['close'].pct_change()
        
        # Add ATR if missing
        if 'atr_14' not in df_prepared.columns:
            high = df_prepared['high'].values
            low = df_prepared['low'].values
            close = df_prepared['close'].values
            
            tr = np.maximum(high - low, 
                           np.maximum(np.abs(high - np.roll(close, 1)),
                                     np.abs(low - np.roll(close, 1))))
            tr[0] = high[0] - low[0]  # First TR
            
            atr = pd.Series(tr).rolling(14).mean().values
            df_prepared['atr_14'] = atr
        
        return df_prepared
    
    def _compute_intent_daily(self, df_1d: pd.DataFrame, fractal_states_1d: Optional[Dict] = None) -> str:
        """
        Compute Intent_Daily from 1D HSMM states
        
        Args:
            df_1d: Daily DataFrame
            fractal_states_1d: 1D fractal states from HSMM
        
        Returns:
            Intent string: 'Trend+', 'Trend-', 'Range', etc.
        """
        
        # If fractal states provided (from HSMM), use them
        if fractal_states_1d:
            # Intent = state with highest probability
            intent = max(fractal_states_1d.items(), key=lambda x: x[1])[0]
            return intent
        
        # Fallback to SMA if HSMM not available
        if len(df_1d) < 50:
            return 'Range'
        
        close = df_1d['close'].values
        sma_20 = pd.Series(close).rolling(20).mean().values[-1]
        sma_50 = pd.Series(close).rolling(50).mean().values[-1]
        
        if not np.isnan(sma_20) and not np.isnan(sma_50):
            if sma_20 > sma_50 * 1.02:
                return 'Trend+'
            elif sma_20 < sma_50 * 0.98:
                return 'Trend-'
        
        return 'Range'
    
    def _compute_stability_4h(self, df_4h: pd.DataFrame) -> float:
        """
        Compute Stability_4H = P(s_t = s_t+1 | O) using HSMM
        
        Args:
            df_4h: 4H DataFrame
        
        Returns:
            Stability probability [0, 1]
        """
        
        if len(df_4h) < 100:
            return 0.0
        
        # Prepare data
        df_prepared = self._prepare_hsmm_data(df_4h)
        
        # Initialize HSMM
        self.hsmm.initialize_parameters(df_prepared)
        
        # Build observations (last 50 bars)
        window_size = min(50, len(df_prepared))
        window = df_prepared.tail(window_size)
        
        observations = []
        for idx in range(len(window)):
            obs = {
                'price': window.iloc[idx]['returns'] if 'returns' in window.columns else 0.0,
                'atr': window.iloc[idx]['atr_14'] if 'atr_14' in window.columns else window.iloc[idx]['close'] * 0.02
            }
            observations.append(obs)
        
        try:
            # Get state probabilities
            state_probs = self.hsmm.forward_backward(observations)
            
            if len(state_probs) > 0:
                # Current state probabilities
                current_probs = state_probs[-1]
                
                # Get most likely current state
                current_state_idx = np.argmax(current_probs)
                
                # Stability = transition probability of staying in same state
                tm = self.hsmm.transition_matrix
                if tm is None:
                    return 0.5
                stability = tm[current_state_idx, current_state_idx]
                
                return stability
            else:
                return 0.5
                
        except Exception as e:
            # Fallback to simple consistency measure
            close = df_4h['close'].values[-20:]
            changes = np.diff(close)
            up_count = np.sum(changes > 0)
            down_count = np.sum(changes < 0)
            stability = max(up_count, down_count) / len(changes)
            return stability
    
    def _compute_alignment_15m(self, df_15m: pd.DataFrame, intent_daily: str) -> Tuple[float, Dict]:
        """
        Compute Alignment_15M using real SMC patterns on 15M timeframe
        
        Alignment = max{P(Trend+), P(Trend-)}_{15M} in direction of Intent_Daily
        
        Args:
            df_15m: 15M DataFrame
            intent_daily: Intent from Daily
        
        Returns:
            Tuple of (alignment score [0, 1], smc_patterns dict)
        """
        
        if len(df_15m) < 50:
            return 0.0, {}
        
        # Detect SMC patterns on 15M
        try:
            smc_patterns = self.smc.detect_all(df_15m)
        except Exception as e:
            # Fallback if SMC fails
            smc_patterns = {
                'bullish_ob': False,
                'bearish_ob': False,
                'bullish_fvg': False,
                'bearish_fvg': False
            }
        
        # Check alignment with Intent_Daily
        has_bullish = smc_patterns.get('bullish_ob', False) or smc_patterns.get('bullish_fvg', False)
        has_bearish = smc_patterns.get('bearish_ob', False) or smc_patterns.get('bearish_fvg', False)
        
        if intent_daily == 'Trend+':
            # Need bullish patterns for long bias
            if has_bullish and not has_bearish:
                alignment = 0.75  # Strong alignment
            elif has_bullish and has_bearish:
                alignment = 0.55  # Mixed signals
            else:
                alignment = 0.30  # No bullish patterns
                
        elif intent_daily == 'Trend-':
            # Need bearish patterns for short bias
            if has_bearish and not has_bullish:
                alignment = 0.75  # Strong alignment
            elif has_bearish and has_bullish:
                alignment = 0.55  # Mixed signals
            else:
                alignment = 0.30  # No bearish patterns
        else:
            # Range or unknown - neutral
            alignment = 0.40
        
        return alignment, smc_patterns
    
    def _compute_fractal_score(self, fractal_states: Dict) -> float:
        """
        Compute Score_FL = Σ(w_TF × Score_TF)
        
        Args:
            fractal_states: Dict of TF states
        
        Returns:
            Fractal score
        """
        
        score = 0.0
        
        for tf, states in fractal_states.items():
            weight = self.weights.get(tf, 1.0)
            
            # Score_TF = max probability
            score_tf = max(states.values()) if states else 0
            
            score += weight * score_tf
        
        return score
    
    def _compute_sdc(self, intent_daily: str, fractal_states: Dict) -> float:
        """
        Compute SdC = 10 × P(Intent_Daily | O)
        
        Args:
            intent_daily: Daily intent
            fractal_states: Fractal states
        
        Returns:
            Score de Confiance [0, 10]
        """
        
        # Get 1D probability of intent
        states_1d = fractal_states.get('1d', {})
        
        prob = states_1d.get(intent_daily, 0.3)
        
        sdc = 10 * prob
        
        return sdc
    
    def _check_mtf_conditions(self, signal: Dict, mtf_data: Dict) -> Dict[str, bool]:
        """
        Check all MTF entry conditions
        
        Args:
            signal: Current signal
            mtf_data: MTF data
        
        Returns:
            Dict of condition name to pass/fail
        """
        
        conditions = {}
        
        # 1. SdC > 5
        conditions['sdc'] = signal['sdc'] > self.mtf_conditions.get('sdc_min', 5.0)
        
        # 2. Stability_4H ≥ 0.60
        conditions['stability_4h'] = signal['stability_4h'] >= self.mtf_conditions.get('stability_4h_min', 0.60)
        
        # 3. Alignment_15M ≥ 0.60
        conditions['alignment_15m'] = signal['alignment_15m'] >= self.mtf_conditions.get('alignment_15m_min', 0.60)
        
        # 4-6. Risk conditions (using Risk Manager)
        if '15m' in mtf_data and len(mtf_data['15m']) > 0:
            entry_price = mtf_data['15m'].iloc[-1]['close']
            
            tm = self.hsmm.transition_matrix
            if tm is None:
                tm = np.ones((self.hsmm.n_states, self.hsmm.n_states)) / self.hsmm.n_states
            risk_conditions = self.risk_manager.check_risk_conditions(
                entry_price=entry_price,
                fractal_states=signal['fractal_states'],
                smc_patterns=signal.get('smc_patterns', {}),
                intent_daily=signal['intent_daily'],
                transition_matrix=tm
            )
            
            conditions['rr'] = risk_conditions['rr_ratio']
            conditions['risk_hit'] = risk_conditions['risk_hit']
            
            # Store risk details in signal
            signal['risk_analysis'] = risk_conditions
        else:
            conditions['rr'] = False
            conditions['risk_hit'] = False
        
        # 5. macro_block = OFF (placeholder)
        conditions['macro_block'] = True  # Assume no macro block for now
        
        return conditions
    
    def _generate_entry_signal(self, signal: Dict, mtf_data: Dict) -> Dict:
        """
        Generate entry signal when all conditions met
        
        Args:
            signal: Current signal
            mtf_data: MTF data
        
        Returns:
            Updated signal with action
        """
        
        intent = signal['intent_daily']
        
        if intent == 'Trend+':
            signal['action'] = 'BUY'
            signal['confidence'] = signal['sdc']
        elif intent == 'Trend-':
            signal['action'] = 'SELL'
            signal['confidence'] = signal['sdc']
        else:
            signal['action'] = 'HOLD'
            signal['reasons'].append('Intent_Daily is Range - no directional bias')
        
        return signal


if __name__ == "__main__":
    print("NYX Engine MTF")
    
    import yaml
    from src.data.mtf_loader import load_mtf_sample
    
    # Load config
    with open('config/validation_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load MTF data
    mtf_data = load_mtf_sample('BTCUSDT', sample_bars=1000)
    
    print(f"\nLoaded MTF data:")
    for tf, df in mtf_data.items():
        print(f"  {tf}: {len(df)} bars")
    
    # Initialize engine
    engine = NYXEngineMTF(config)
    
    # Generate signal
    signal = engine.generate_signal_mtf(
        pair='BTCUSDT',
        mtf_data=mtf_data,
        current_date=mtf_data['15m'].index[-1].isoformat()
    )
    
    print(f"\nSignal:")
    print(f"  Action: {signal['action']}")
    print(f"  Intent Daily: {signal['intent_daily']}")
    print(f"  Stability 4H: {signal['stability_4h']:.3f}")
    print(f"  Alignment 15M: {signal['alignment_15m']:.3f}")
    print(f"  SdC: {signal['sdc']:.1f}")
    print(f"  Score FL: {signal['score_fl']:.2f}")
    print(f"\n  Conditions met:")
    for cond, met in signal['conditions_met'].items():
        print(f"    {cond}: {'✅' if met else '❌'}")
