"""
NYX Engine v0.8 — LEGACY (kept for its 9 active callers only).

This file was the central trading engine in v0.8: HSMM regime +
SMC patterns + macro event context. As of Ticket 04 it is segregated
to this module so that `src/core/nyx_engine.py::NYXEngine` can become
the CANONICAL runtime entrypoint without colliding with this legacy
stack.

Still imported by (frozen list — do NOT add new imports of this
module):
  - src/runner/run_paper.py           (legacy v0.8 paper runner)
  - src/validation/walk_forward.py
  - src/validation/oos_report.py
  - src/validation/benchmarks.py
  - src/validation/smc_diagnostics.py
  - src/validation/signal_funnel.py
  - src/validation/pattern_quality.py
  - scripts/run_backtest.py

When those legacy callers are retired (deletion or rewrite against
the canonical `NYXEngine`), this file can also be removed.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import sys
from pathlib import Path

# Import components
from src.core.hsmm import SemiMarkovHMM
from src.core.smc import SMCDetector
from src.macro.real_macro_engine import RealMacroEngine


class NYXEngine:
    """
    Central NYX Trading Engine
    
    Combines:
    - HSMM (regime detection)
    - SMC (pattern detection)
    - Macro (event-based context)
    
    Produces standardized signals for all consumers
    """
    
    def __init__(self, config: Dict):
        """
        Initialize NYX Engine
        
        Args:
            config: Configuration dict from config.yaml
        """
        self.config = config
        
        # Initialize components
        self.hsmm = SemiMarkovHMM()
        self.smc = SMCDetector()
        self.macro = RealMacroEngine('data/macro_events.json')
        
        # Config thresholds
        strategy_config = config.get('strategy', {})
        self.sdc_threshold = strategy_config.get('hsmm', {}).get('sdc_threshold', 3.5)
        self.prob_threshold = strategy_config.get('hsmm', {}).get('prob_threshold', 0.55)
        self.min_required_bars = strategy_config.get('min_required_bars', 100)
        
        # Risk config
        risk_config = config.get('risk', {})
        self.base_size = risk_config.get('base_size', 0.05)
        self.max_position = risk_config.get('max_position', 0.25)
        self.stop_loss_pct = risk_config.get('stop_loss', 0.05)
        self.take_profit_pct = risk_config.get('take_profit', 0.15)
        
        # Regime multipliers
        self.bull_multiplier = risk_config.get('bull_multiplier', 2.5)
        self.bear_multiplier = risk_config.get('bear_multiplier', 0.7)
        self.range_multiplier = risk_config.get('range_multiplier', 1.0)
        
        # HSMM initialized flag
        self.hsmm_initialized = False
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Precompute all features needed for signal generation
        
        This method calculates features once to avoid repeated computation
        in validation loops (OOS, walk-forward).
        
        Args:
            data: Raw OHLCV DataFrame
        
        Returns:
            DataFrame with precomputed features
        """
        prepared = data.copy()
        
        # Returns
        if 'returns' not in prepared.columns:
            prepared['returns'] = prepared['close'].pct_change().fillna(0)
        
        # ATR
        if 'atr_14' not in prepared.columns:
            prepared['atr_14'] = (prepared['high'] - prepared['low']).rolling(14).mean().fillna(0)
        
        # SMAs (for SMC patterns)
        if 'sma_20' not in prepared.columns:
            prepared['sma_20'] = prepared['close'].rolling(20).mean()
        
        if 'sma_50' not in prepared.columns:
            prepared['sma_50'] = prepared['close'].rolling(50).mean()
        
        return prepared
    
    def generate_signal_at_index(
        self,
        pair: str,
        prepared_data: pd.DataFrame,
        i: int,
        current_date: Optional[str] = None,
        lookback_bars: Optional[int] = None
    ) -> Dict:
        """
        Generate signal at a specific index using precomputed data
        
        Optimized for validation loops with BOUNDED LOOKBACK.
        Instead of using all history [0:i+1], uses only the last
        lookback_bars to keep computational cost constant.
        
        Args:
            pair: Trading pair
            prepared_data: DataFrame with precomputed features
            i: Current index
            current_date: Optional date override
            lookback_bars: Max historical bars to use (default: 1000)
        
        Returns:
            Same signal format as generate_signal()
        """
        # Default lookback
        if lookback_bars is None:
            lookback_bars = 1000
        
        # Calculate bounded window
        # Use only [max(0, i - lookback + 1) : i + 1]
        start_idx = max(0, i - lookback_bars + 1)
        data_slice = prepared_data.iloc[start_idx:i+1]
        
        if current_date is None:
            current_date = str(data_slice.index[-1].isoformat())

        # Call standard generate_signal with bounded window
        return self.generate_signal(pair, data_slice, current_date or "")
    
    def generate_signal(
        self, 
        pair: str, 
        data: pd.DataFrame, 
        current_date: str
    ) -> Dict:
        """
        Generate standardized trading signal
        
        THIS IS THE SINGLE SOURCE OF TRUTH for all trading decisions.
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            data: Historical OHLCV DataFrame
            current_date: Current date string 'YYYY-MM-DD'
        
        Returns:
            {
                'action': 'BUY' | 'SELL' | 'HOLD',
                'confidence': 0-10 (SdC score),
                'position_size': 0-1 (% of capital),
                'stop_loss': price level,
                'take_profit': price level,
                'regime': 'Bull' | 'Bear' | 'Range',
                'macro_signal': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'macro_strength': 0-1,
                'reasons': [list of reasons],
                'hsmm_state': raw HSMM state,
                'smc_patterns': dict of patterns
            }
        """
        
        if len(data) < self.min_required_bars:
            return self._hold_signal(f"Insufficient data (< {self.min_required_bars} candles)")
        
        # 1. HSMM State Detection
        hsmm_result = self._get_hsmm_signal(data)
        if hsmm_result is None:
            return self._hold_signal("HSMM initialization failed")
        
        hsmm_state = hsmm_result['state']
        confidence = hsmm_result['confidence']
        regime = hsmm_result['regime']
        
        # 2. SMC Pattern Detection
        smc_patterns = self.smc.detect_all(data)
        
        # 3. Macro Context
        asset = pair.replace('USDT', '')
        macro_signal = self.macro.get_macro_signal(asset, current_date)
        
        # 4. Combine into decision
        signal = self._combine_signals(
            pair=pair,
            data=data,
            hsmm_state=hsmm_state,
            confidence=confidence,
            regime=regime,
            smc_patterns=smc_patterns,
            macro_signal=macro_signal
        )
        
        return signal
    
    def _get_hsmm_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Get HSMM state and confidence
        
        Returns:
            {
                'state': 'Trend+' | 'Trend-' | 'Range',
                'confidence': 0-10,
                'regime': 'Bull' | 'Bear' | 'Range',
                'state_probs': array of probabilities
            }
        """
        try:
            # Initialize if needed (HSMM expects DataFrame)
            if not self.hsmm_initialized:
                # HSMM needs DataFrame with 'returns' and 'atr_14' columns
                if 'returns' not in data.columns:
                    data = data.copy()
                    data['returns'] = data['close'].pct_change().fillna(0)
                
                if 'atr_14' not in data.columns:
                    data['atr_14'] = (data['high'] - data['low']).rolling(14).mean().fillna(0)
                
                self.hsmm.initialize_parameters(data)
                self.hsmm_initialized = True
            
            # Prepare observations in CORRECT format (List[Dict])
            # CRITICAL: HSMM expects dicts with 'price' and 'atr' keys
            returns = data['close'].pct_change().fillna(0).values
            atr = (data['high'] - data['low']).rolling(14).mean().fillna(0).values
            
            # Build observations as List[Dict]
            observations = [
                {'price': float(returns[i]), 'atr': float(atr[i])}
                for i in range(len(returns))
            ]
            
            # Get state probabilities
            state_probs = self.hsmm.forward_backward(observations)
            
            if len(state_probs) == 0:
                return None
            
            # Current state
            current_probs = state_probs[-1]
            dominant_idx = np.argmax(current_probs)
            dominant_state = self.hsmm.states[dominant_idx]
            
            # Confidence (SdC score)
            confidence = np.max(current_probs) * 10
            
            # Map to regime
            regime = self._map_hsmm_to_regime(dominant_state)
            
            return {
                'state': dominant_state,
                'confidence': confidence,
                'regime': regime,
                'state_probs': current_probs
            }
        
        except Exception as e:
            print(f"⚠️ HSMM error: {e}")
            return None
    
    def _map_hsmm_to_regime(self, hsmm_state: str) -> str:
        """
        Map HSMM states to regime names
        
        Args:
            hsmm_state: 'Trend+', 'Range', or 'Trend-'
        
        Returns:
            'Bull', 'Range', or 'Bear'
        """
        mapping = {
            'Trend+': 'Bull',
            'Trend-': 'Bear',
            'Range': 'Range'
        }
        return mapping.get(hsmm_state, 'Range')
    
    def _combine_signals(
        self,
        pair: str,
        data: pd.DataFrame,
        hsmm_state: str,
        confidence: float,
        regime: str,
        smc_patterns: Dict,
        macro_signal: Dict
    ) -> Dict:
        """
        Combine all signals into final decision
        
        Logic:
        1. Check confidence threshold
        2. Check regime + SMC confluence
        3. Check macro reinforcement/conflict
        4. Calculate position size
        5. Set stop/TP levels
        """
        
        reasons = []
        current_price = data['close'].iloc[-1]
        
        # Check confidence threshold
        if confidence < self.sdc_threshold:
            return self._hold_signal(
                f"Low confidence: {confidence:.1f} < {self.sdc_threshold}",
                confidence=confidence,
                regime=regime,
                macro_signal=macro_signal
            )
        
        reasons.append(f"Confidence: {confidence:.1f}/10")
        
        # Bullish setup
        has_bullish_smc = (
            smc_patterns.get('bullish_ob', False) or 
            smc_patterns.get('bullish_fvg', False)
        )
        
        # Bearish setup
        has_bearish_smc = (
            smc_patterns.get('bearish_ob', False) or 
            smc_patterns.get('bearish_fvg', False)
        )
        
        # BUY Logic
        if (regime == 'Bull' and 
            has_bullish_smc and 
            macro_signal['signal'] in ['BULLISH', 'NEUTRAL']):
            
            reasons.append(f"Regime: {regime}")
            reasons.append("SMC: Bullish pattern")
            reasons.append(f"Macro: {macro_signal['signal']} ({macro_signal['strength']:.2f})")
            
            # Calculate size
            position_size = self._calculate_position_size(
                confidence, 
                regime, 
                macro_signal
            )
            
            # Set levels
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            
            return {
                'action': 'BUY',
                'confidence': confidence,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'regime': regime,
                'macro_signal': macro_signal['signal'],
                'macro_strength': macro_signal['strength'],
                'reasons': reasons,
                'hsmm_state': hsmm_state,
                'smc_patterns': smc_patterns
            }
        
        # SELL Logic (for closing positions)
        elif (regime == 'Bear' and 
              has_bearish_smc and 
              macro_signal['signal'] in ['BEARISH', 'NEUTRAL']):
            
            reasons.append(f"Regime: {regime}")
            reasons.append("SMC: Bearish pattern")
            reasons.append(f"Macro: {macro_signal['signal']}")
            
            return {
                'action': 'SELL',
                'confidence': confidence,
                'position_size': 0,
                'stop_loss': None,
                'take_profit': None,
                'regime': regime,
                'macro_signal': macro_signal['signal'],
                'macro_strength': macro_signal['strength'],
                'reasons': reasons,
                'hsmm_state': hsmm_state,
                'smc_patterns': smc_patterns
            }
        
        # HOLD (no setup)
        else:
            reasons.append("No confluence")
            if not has_bullish_smc and not has_bearish_smc:
                reasons.append("No SMC pattern")
            if regime not in ['Bull', 'Bear']:
                reasons.append(f"Regime: {regime}")
            
            return self._hold_signal(
                ' | '.join(reasons),
                confidence=confidence,
                regime=regime,
                macro_signal=macro_signal,
                hsmm_state=hsmm_state,
                smc_patterns=smc_patterns
            )
    
    def _calculate_position_size(
        self, 
        confidence: float, 
        regime: str, 
        macro_signal: Dict
    ) -> float:
        """
        Calculate position size based on:
        - Base size
        - Confidence multiplier
        - Regime multiplier
        - Macro reinforcement
        """
        
        # Base
        size = self.base_size
        
        # Confidence multiplier
        confidence_mult = confidence / 10.0
        size *= confidence_mult
        
        # Regime multiplier
        regime_mult = {
            'Bull': self.bull_multiplier,
            'Bear': self.bear_multiplier,
            'Range': self.range_multiplier
        }.get(regime, 1.0)
        size *= regime_mult
        
        # Macro reinforcement
        if macro_signal['signal'] == 'BULLISH':
            size *= 1.5
        elif macro_signal['signal'] == 'BEARISH':
            size *= 0.5
        
        # Cap at max
        size = min(size, self.max_position)
        
        return size
    
    def _hold_signal(
        self,
        reason: str,
        confidence: float = 0,
        regime: str = 'Unknown',
        macro_signal: Optional[Dict] = None,
        hsmm_state: Optional[str] = None,
        smc_patterns: Optional[Dict] = None
    ) -> Dict:
        """Generate HOLD signal"""
        
        return {
            'action': 'HOLD',
            'confidence': confidence,
            'position_size': 0,
            'stop_loss': None,
            'take_profit': None,
            'regime': regime,
            'macro_signal': macro_signal['signal'] if macro_signal else 'UNKNOWN',
            'macro_strength': macro_signal['strength'] if macro_signal else 0,
            'reasons': [reason],
            'hsmm_state': hsmm_state,
            'smc_patterns': smc_patterns or {}
        }


if __name__ == "__main__":
    # Example usage
    import yaml
    
    print("="*80)
    print("NYX ENGINE v0.8 - Test")
    print("="*80)
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize engine
    engine = NYXEngine(config)
    print("\n✓ Engine initialized")
    print(f"  SdC threshold: {engine.sdc_threshold}")
    print(f"  Base size: {engine.base_size*100}%")
    
    # Load sample data
    data = pd.read_csv('data/sample/BTCUSDT_1h_sample.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    print(f"\n✓ Sample data loaded: {len(data)} rows")
    
    # Generate signal
    signal = engine.generate_signal(
        pair='BTCUSDT',
        data=data,
        current_date='2024-01-15'
    )
    
    print("\n" + "="*80)
    print("SIGNAL GENERATED")
    print("="*80)
    print(f"Action:          {signal['action']}")
    print(f"Confidence:      {signal['confidence']:.1f}/10")
    print(f"Position Size:   {signal['position_size']*100:.1f}%")
    print(f"Regime:          {signal['regime']}")
    print(f"Macro:           {signal['macro_signal']} ({signal['macro_strength']:.2f})")
    print(f"\nReasons:")
    for reason in signal['reasons']:
        print(f"  • {reason}")
    print("="*80)
