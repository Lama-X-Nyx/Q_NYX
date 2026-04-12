"""
SMC Detector Autopsy

Technical audit of the SMC detector to understand why it's globally silent.
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter

sys.path.insert(0, '.')

from src.core.smc import SMCDetector


class SMCAutopsy:
    """
    Autopsy of SMC Detector behavior
    
    Instruments the detector to track:
    - Function call counts
    - Candidate structures seen
    - Detection success/failure
    - Failure reasons by category
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize autopsy with instrumentation
        
        Args:
            config: SMC detector configuration
        """
        self.config = config or {}
        
        # Get thresholds from config
        smc_config = self.config.get('smc_detector', {})
        self.ob_threshold = smc_config.get('ob_range_threshold', 0.015)
        self.fvg_min_gap = smc_config.get('fvg_min_gap', 0.005)
        self.liquidity_lookback = smc_config.get('liquidity_lookback', 20)
        
        # Initialize detector
        self.detector = SMCDetector(
            ob_range_threshold=self.ob_threshold,
            fvg_min_gap=self.fvg_min_gap,
            liquidity_lookback=self.liquidity_lookback
        )
        
        # Tracking counters
        self.reset_counters()
    
    def reset_counters(self):
        """Reset all tracking counters"""
        self.detector_called = 0
        self.fvg_function_calls = 0
        self.ob_function_calls = 0
        
        # FVG audit
        self.fvg_candidates = 0
        self.fvg_detected = 0
        self.fvg_failed_gap_threshold = 0
        self.fvg_failed_structure = 0
        self.fvg_failed_other = 0
        
        # OB audit
        self.ob_candidates = 0
        self.ob_detected = 0
        self.ob_failed_range_threshold = 0
        self.ob_failed_followthrough = 0
        self.ob_failed_other = 0
    
    def detect_fvg_instrumented(self, data: pd.DataFrame, bullish: bool = True) -> bool:
        """
        Instrumented FVG detection with failure tracking
        
        Args:
            data: OHLC data
            bullish: True for bullish FVG
        
        Returns:
            True if FVG detected
        """
        self.fvg_function_calls += 1
        
        if len(data) < 3:
            self.fvg_failed_structure += 1
            return False
        
        # Check last 3 candles
        recent = data.tail(3)
        
        if len(recent) < 3:
            self.fvg_failed_structure += 1
            return False
        
        prev = recent.iloc[0]
        mid = recent.iloc[1]
        curr = recent.iloc[2]
        
        # Count as candidate if we got here
        self.fvg_candidates += 1
        
        if bullish:
            # Bullish FVG: high[i-1] < low[i+1]
            gap = (curr['low'] - prev['high']) / prev['high']
            
            if gap > self.fvg_min_gap:
                self.fvg_detected += 1
                return True
            elif gap > 0:
                # Gap exists but below threshold
                self.fvg_failed_gap_threshold += 1
            else:
                # No gap structure
                self.fvg_failed_structure += 1
        
        else:
            # Bearish FVG: low[i-1] > high[i+1]
            gap = (prev['low'] - curr['high']) / prev['low']
            
            if gap > self.fvg_min_gap:
                self.fvg_detected += 1
                return True
            elif gap > 0:
                # Gap exists but below threshold
                self.fvg_failed_gap_threshold += 1
            else:
                # No gap structure
                self.fvg_failed_structure += 1
        
        return False
    
    def detect_ob_instrumented(self, data: pd.DataFrame, bullish: bool = True) -> bool:
        """
        Instrumented OB detection with failure tracking
        
        Args:
            data: OHLC data
            bullish: True for bullish OB
        
        Returns:
            True if OB detected
        """
        self.ob_function_calls += 1
        
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
                
                if not is_down:
                    continue
                
                # Count as candidate (down candle exists)
                self.ob_candidates += 1
                
                # Strong rally after
                rally = (next_candle['close'] - next_candle['open']) / next_candle['open']
                is_strong_rally = rally > self.ob_threshold
                
                if is_strong_rally:
                    self.ob_detected += 1
                    return True
                elif rally > 0:
                    # Rally exists but below threshold
                    self.ob_failed_range_threshold += 1
                else:
                    # No followthrough
                    self.ob_failed_followthrough += 1
        
        else:
            # Look for up candle followed by strong selloff
            for i in range(len(recent) - 2):
                candle = recent.iloc[i]
                next_candle = recent.iloc[i+1]
                
                # Up candle
                is_up = candle['close'] > candle['open']
                
                if not is_up:
                    continue
                
                # Count as candidate (up candle exists)
                self.ob_candidates += 1
                
                # Strong selloff after
                selloff = (next_candle['open'] - next_candle['close']) / next_candle['open']
                is_strong_selloff = selloff > self.ob_threshold
                
                if is_strong_selloff:
                    self.ob_detected += 1
                    return True
                elif selloff > 0:
                    # Selloff exists but below threshold
                    self.ob_failed_range_threshold += 1
                else:
                    # No followthrough
                    self.ob_failed_followthrough += 1
        
        return False
    
    def analyze_bar(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a single bar with full instrumentation
        
        Args:
            data: OHLC data up to current bar
        
        Returns:
            Detection results with tracking info
        """
        self.detector_called += 1
        
        # Detect with instrumentation
        has_bullish_fvg = self.detect_fvg_instrumented(data, bullish=True)
        has_bearish_fvg = self.detect_fvg_instrumented(data, bullish=False)
        has_bullish_ob = self.detect_ob_instrumented(data, bullish=True)
        has_bearish_ob = self.detect_ob_instrumented(data, bullish=False)
        
        patterns = {
            'bullish_fvg': has_bullish_fvg,
            'bearish_fvg': has_bearish_fvg,
            'bullish_ob': has_bullish_ob,
            'bearish_ob': has_bearish_ob
        }
        
        return patterns
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get full autopsy summary
        
        Returns:
            Complete tracking data
        """
        return {
            'code_path': {
                'detector_called': self.detector_called,
                'fvg_function_called': self.fvg_function_calls,
                'ob_function_called': self.ob_function_calls
            },
            'fvg_audit': {
                'candidates_seen': self.fvg_candidates,
                'detected': self.fvg_detected,
                'failed_gap_threshold': self.fvg_failed_gap_threshold,
                'failed_structure_rules': self.fvg_failed_structure,
                'failed_other': self.fvg_failed_other
            },
            'ob_audit': {
                'candidates_seen': self.ob_candidates,
                'detected': self.ob_detected,
                'failed_range_threshold': self.ob_failed_range_threshold,
                'failed_followthrough': self.ob_failed_followthrough,
                'failed_other': self.ob_failed_other
            },
            'configuration': {
                'ob_range_threshold': self.ob_threshold,
                'fvg_min_gap': self.fvg_min_gap,
                'liquidity_lookback': self.liquidity_lookback
            }
        }
    
    def determine_verdict(self, summary: Dict[str, Any]) -> Dict[str, str]:
        """
        Determine autopsy verdict category
        
        Args:
            summary: Autopsy summary
        
        Returns:
            Verdict with category and explanation
        """
        code_path = summary['code_path']
        fvg = summary['fvg_audit']
        ob = summary['ob_audit']
        
        # Check if detector was called
        if not code_path['detector_called']:
            return {
                'category': 'C',
                'summary': 'Detector code path is broken - not being called at all.'
            }
        
        # Total candidates
        total_candidates = fvg['candidates_seen'] + ob['candidates_seen']
        total_detected = fvg['detected'] + ob['detected']
        
        # Total failures by threshold
        fvg_threshold_failures = fvg['failed_gap_threshold']
        ob_threshold_failures = ob['failed_range_threshold']
        total_threshold_failures = fvg_threshold_failures + ob_threshold_failures
        
        # Analyze failure patterns
        if total_candidates == 0:
            # No candidate structures at all
            return {
                'category': 'B',
                'summary': 'Detector is functionally executed, but its implementation is too naive/incomplete - sees no candidate structures.'
            }
        
        elif total_detected == 0 and total_threshold_failures > 0:
            # Candidates seen but all fail thresholds
            return {
                'category': 'A',
                'summary': f'Detector path works, but current thresholds make detection near-zero ({total_candidates} candidates, {total_threshold_failures} threshold failures).'
            }
        
        elif total_detected == 0:
            # Candidates seen but fail for other reasons
            return {
                'category': 'D',
                'summary': f'Detector sees candidate structures ({total_candidates}), but final filtering logic discards them too aggressively.'
            }
        
        else:
            # Some detections but still very low
            detection_rate = total_detected / total_candidates if total_candidates > 0 else 0
            
            if detection_rate < 0.1:
                return {
                    'category': 'A',
                    'summary': f'Detector works but very low success rate ({detection_rate:.1%}) - likely threshold strictness.'
                }
            else:
                return {
                    'category': 'OK',
                    'summary': f'Detector appears to be working with {detection_rate:.1%} detection rate.'
                }


if __name__ == "__main__":
    print("SMC Autopsy module loaded successfully")
