"""
Smart Money Concepts (SMC) Detection
Order Blocks, Fair Value Gaps, Liquidity Zones
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class SMCDetector:
    """
    Smart Money Concepts pattern detection
    
    Patterns:
    - Order Blocks (OB): Institutional demand/supply zones
    - Fair Value Gaps (FVG): Imbalance zones
    - Liquidity Sweeps: Stop hunts
    - Break of Structure (BOS): Trend continuation
    - Change of Character (ChoCH): Trend reversal
    
    Usage:
        smc = SMCDetector()
        patterns = smc.detect_all(data)
        if patterns['bullish_ob']:
            # Entry signal
    """
    
    def __init__(self, 
                 ob_range_threshold: float = 0.015,
                 fvg_min_gap: float = 0.005,
                 liquidity_lookback: int = 20):
        """
        Initialize SMC Detector
        
        Args:
            ob_range_threshold: Minimum range for Order Block (1.5% default)
            fvg_min_gap: Minimum gap for FVG (0.5% default)
            liquidity_lookback: Bars to look back for liquidity (20 default)
        
        Raises:
            TypeError: If parameters are not numeric types
        """
        # Type validation - defensive programming
        if not isinstance(ob_range_threshold, (int, float)):
            raise TypeError(
                f"ob_range_threshold must be numeric (int or float), "
                f"got {type(ob_range_threshold).__name__}"
            )
        
        if not isinstance(fvg_min_gap, (int, float)):
            raise TypeError(
                f"fvg_min_gap must be numeric (int or float), "
                f"got {type(fvg_min_gap).__name__}"
            )
        
        if not isinstance(liquidity_lookback, int):
            raise TypeError(
                f"liquidity_lookback must be int, "
                f"got {type(liquidity_lookback).__name__}"
            )
        
        self.ob_range_threshold = float(ob_range_threshold)
        self.fvg_min_gap = float(fvg_min_gap)
        self.liquidity_lookback = int(liquidity_lookback)
        
        # INSTRUMENTATION: Track candidate flow (diagnostic only, no logic change)
        self.diagnostics = {
            'fvg_candidates_seen': 0,
            'fvg_rejected_gap': 0,
            'fvg_rejected_structure': 0,
            'fvg_kept': 0,
            'ob_candidates_seen': 0,
            'ob_rejected_range': 0,
            'ob_rejected_structure': 0,
            'ob_kept': 0
        }
    
    def reset_diagnostics(self):
        """Reset diagnostic counters (call before each detect_all)"""
        self.diagnostics = {
            'fvg_candidates_seen': 0,
            'fvg_rejected_gap': 0,
            'fvg_rejected_structure': 0,
            'fvg_kept': 0,
            'ob_candidates_seen': 0,
            'ob_rejected_range': 0,
            'ob_rejected_structure': 0,
            'ob_kept': 0
        }
    
    def detect_all(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect all SMC patterns
        
        Args:
            data: DataFrame with OHLCV
        
        Returns:
            Dict with pattern flags
        """
        # Reset diagnostics for new detection cycle
        self.reset_diagnostics()
        
        patterns = {
            'bullish_ob': self.detect_order_block(data, bullish=True),
            'bearish_ob': self.detect_order_block(data, bullish=False),
            'bullish_fvg': self.detect_fvg(data, bullish=True),
            'bearish_fvg': self.detect_fvg(data, bullish=False),
            'liquidity_sweep': self.detect_liquidity_sweep(data),
            'bos_bullish': self.detect_bos(data, bullish=True),
            'bos_bearish': self.detect_bos(data, bullish=False)
        }
        
        return patterns
    
    def detect_order_block(self, data: pd.DataFrame, bullish: bool = True) -> bool:
        """
        Detect Order Block pattern
        
        OB = Strong move after consolidation
        Bullish OB: Last down candle before strong rally
        Bearish OB: Last up candle before strong selloff
        
        Args:
            data: DataFrame with OHLC
            bullish: True for bullish OB, False for bearish
        
        Returns:
            True if OB detected
        """
        if len(data) < 5:
            return False
        
        # Get recent candles
        recent = data.tail(5)
        
        if bullish:
            # Look for down candle followed by strong rally
            for i in range(len(recent) - 2):
                candle = recent.iloc[i]
                next_candle = recent.iloc[i+1]
                
                # Down candle
                is_down = candle['close'] < candle['open']
                
                # Strong rally after
                rally = (next_candle['close'] - next_candle['open']) / next_candle['open']
                is_strong_rally = rally > self.ob_range_threshold
                
                if is_down and is_strong_rally:
                    return True
        
        else:
            # Look for up candle followed by strong selloff
            for i in range(len(recent) - 2):
                candle = recent.iloc[i]
                next_candle = recent.iloc[i+1]
                
                # Up candle
                is_up = candle['close'] > candle['open']
                
                # Strong selloff after
                selloff = (next_candle['open'] - next_candle['close']) / next_candle['open']
                is_strong_selloff = selloff > self.ob_range_threshold
                
                if is_up and is_strong_selloff:
                    return True
        
        return False
    
    def detect_fvg(self, data: pd.DataFrame, bullish: bool = True) -> bool:
        """
        Detect Fair Value Gap
        
        FVG = Gap between candles (imbalance)
        Bullish FVG: Gap up (high[i-1] < low[i+1])
        Bearish FVG: Gap down (low[i-1] > high[i+1])
        
        Args:
            data: DataFrame with OHLC
            bullish: True for bullish FVG
        
        Returns:
            True if FVG detected
        """
        if len(data) < 3:
            return False
        
        # Check last 3 candles
        recent = data.tail(3)
        
        if len(recent) < 3:
            return False
        
        prev = recent.iloc[0]
        mid = recent.iloc[1]
        curr = recent.iloc[2]
        
        if bullish:
            # Bullish FVG: high[i-1] < low[i+1]
            gap = (curr['low'] - prev['high']) / prev['high']
            if gap > self.fvg_min_gap:
                return True
        
        else:
            # Bearish FVG: low[i-1] > high[i+1]
            gap = (prev['low'] - curr['high']) / prev['low']
            if gap > self.fvg_min_gap:
                return True
        
        return False
    
    def detect_liquidity_sweep(self, data: pd.DataFrame) -> bool:
        """
        Detect Liquidity Sweep (stop hunt)
        
        Pattern:
        - Price breaks recent high/low
        - Immediately reverses
        - Indicates stop hunt before real move
        
        Args:
            data: DataFrame with OHLC
        
        Returns:
            True if sweep detected
        """
        if len(data) < self.liquidity_lookback + 2:
            return False
        
        recent = data.tail(self.liquidity_lookback + 2)
        
        # Find recent high/low
        recent_high = recent['high'].iloc[:-1].max()
        recent_low = recent['low'].iloc[:-1].min()
        
        # Last candle
        last = recent.iloc[-1]
        prev = recent.iloc[-2]
        
        # Check for sweep and reversal
        swept_high = last['high'] > recent_high and last['close'] < prev['close']
        swept_low = last['low'] < recent_low and last['close'] > prev['close']
        
        # Explicitly convert to Python bool
        result = swept_high or swept_low
        return bool(result)
    
    def detect_bos(self, data: pd.DataFrame, bullish: bool = True) -> bool:
        """
        Detect Break of Structure (BOS)
        
        BOS = Breaking previous swing high/low
        Indicates trend continuation
        
        Args:
            data: DataFrame with OHLC
            bullish: True for bullish BOS
        
        Returns:
            True if BOS detected
        """
        if len(data) < 10:
            return False
        
        recent = data.tail(10)
        
        if bullish:
            # Find previous swing high
            swing_high = recent['high'].iloc[:-2].max()
            
            # Check if recent close breaks it
            current_close = recent['close'].iloc[-1]
            
            if current_close > swing_high:
                return True
        
        else:
            # Find previous swing low
            swing_low = recent['low'].iloc[:-2].min()
            
            # Check if recent close breaks it
            current_close = recent['close'].iloc[-1]
            
            if current_close < swing_low:
                return True
        
        return False
    
    def get_ob_zones(self, data: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        """
        Get all Order Block zones
        
        Args:
            data: DataFrame with OHLC
            lookback: Bars to analyze
        
        Returns:
            List of OB zones with {type, high, low, strength}
        """
        zones = []
        
        recent = data.tail(lookback)
        
        for i in range(len(recent) - 2):
            candle = recent.iloc[i]
            next_candle = recent.iloc[i+1]
            
            # Bullish OB
            if candle['close'] < candle['open']:
                rally = (next_candle['close'] - next_candle['open']) / next_candle['open']
                
                if rally > self.ob_range_threshold:
                    zones.append({
                        'type': 'bullish',
                        'high': candle['high'],
                        'low': candle['low'],
                        'strength': rally,
                        'index': i
                    })
            
            # Bearish OB
            elif candle['close'] > candle['open']:
                selloff = (next_candle['open'] - next_candle['close']) / next_candle['open']
                
                if selloff > self.ob_range_threshold:
                    zones.append({
                        'type': 'bearish',
                        'high': candle['high'],
                        'low': candle['low'],
                        'strength': selloff,
                        'index': i
                    })
        
        return zones
    
    def get_fvg_zones(self, data: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        """
        Get all FVG zones
        
        Args:
            data: DataFrame with OHLC
            lookback: Bars to analyze
        
        Returns:
            List of FVG zones
        """
        zones = []
        
        recent = data.tail(lookback)
        
        for i in range(1, len(recent) - 1):
            prev = recent.iloc[i-1]
            curr = recent.iloc[i+1]
            
            # Bullish FVG
            gap_up = (curr['low'] - prev['high']) / prev['high']
            if gap_up > self.fvg_min_gap:
                zones.append({
                    'type': 'bullish',
                    'high': curr['low'],
                    'low': prev['high'],
                    'gap': gap_up,
                    'index': i
                })
            
            # Bearish FVG
            gap_down = (prev['low'] - curr['high']) / prev['low']
            if gap_down > self.fvg_min_gap:
                zones.append({
                    'type': 'bearish',
                    'high': prev['low'],
                    'low': curr['high'],
                    'gap': gap_down,
                    'index': i
                })
        
        return zones


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("SMC Detector - Example Usage")
    print("="*80)
    
    # Create synthetic price data
    np.random.seed(42)
    n = 100
    
    data = pd.DataFrame({
        'open': 10000 + np.cumsum(np.random.randn(n) * 50),
        'high': 10000 + np.cumsum(np.random.randn(n) * 50) + np.random.rand(n) * 100,
        'low': 10000 + np.cumsum(np.random.randn(n) * 50) - np.random.rand(n) * 100,
        'close': 10000 + np.cumsum(np.random.randn(n) * 50),
        'volume': 1000000 + np.random.rand(n) * 500000
    })
    
    # Ensure high/low logic
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    print(f"\n✓ Created {len(data)} candles of synthetic data")
    
    # Initialize SMC detector
    smc = SMCDetector()
    print(f"✓ Initialized SMC Detector")
    
    # Detect patterns
    patterns = smc.detect_all(data)
    
    print(f"\nPattern Detection Results:")
    for pattern, detected in patterns.items():
        status = "✓ DETECTED" if detected else "✗ Not found"
        print(f"  {pattern:20s}: {status}")
    
    # Get OB zones
    ob_zones = smc.get_ob_zones(data, lookback=50)
    print(f"\nOrder Block Zones: {len(ob_zones)} found")
    if ob_zones:
        for zone in ob_zones[:3]:
            print(f"  {zone['type']:8s} OB: {zone['low']:.0f}-{zone['high']:.0f} (strength: {zone['strength']:.2%})")
    
    # Get FVG zones
    fvg_zones = smc.get_fvg_zones(data, lookback=50)
    print(f"\nFair Value Gaps: {len(fvg_zones)} found")
    if fvg_zones:
        for zone in fvg_zones[:3]:
            print(f"  {zone['type']:8s} FVG: {zone['low']:.0f}-{zone['high']:.0f} (gap: {zone['gap']:.2%})")
    
    print("\n" + "="*80)
    print("✅ SMC Module Working")
    print("="*80)
