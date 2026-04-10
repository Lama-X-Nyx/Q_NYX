"""
Test SMCDetector Initialization

Validates type checking and parameter handling in SMCDetector.
"""

import sys
sys.path.insert(0, '.')

import pytest
from src.core.smc import SMCDetector


class TestSMCDetectorInit:
    """Test suite for SMCDetector initialization"""
    
    def test_init_with_valid_floats(self):
        """Test initialization with valid float parameters"""
        detector = SMCDetector(
            ob_range_threshold=0.015,
            fvg_min_gap=0.005,
            liquidity_lookback=20
        )
        
        assert detector.ob_range_threshold == 0.015
        assert detector.fvg_min_gap == 0.005
        assert detector.liquidity_lookback == 20
    
    def test_init_with_valid_ints(self):
        """Test initialization accepts ints for numeric params"""
        detector = SMCDetector(
            ob_range_threshold=1,  # Will be converted to float
            fvg_min_gap=1,
            liquidity_lookback=20
        )
        
        assert detector.ob_range_threshold == 1.0
        assert detector.fvg_min_gap == 1.0
        assert detector.liquidity_lookback == 20
    
    def test_init_rejects_dict_for_ob_threshold(self):
        """Test that passing dict for ob_range_threshold raises TypeError"""
        with pytest.raises(TypeError) as exc_info:
            SMCDetector(
                ob_range_threshold={'value': 0.015},  # Invalid
                fvg_min_gap=0.005,
                liquidity_lookback=20
            )
        
        assert 'ob_range_threshold must be numeric' in str(exc_info.value)
        assert 'dict' in str(exc_info.value)
    
    def test_init_rejects_dict_for_fvg_gap(self):
        """Test that passing dict for fvg_min_gap raises TypeError"""
        with pytest.raises(TypeError) as exc_info:
            SMCDetector(
                ob_range_threshold=0.015,
                fvg_min_gap={'value': 0.005},  # Invalid
                liquidity_lookback=20
            )
        
        assert 'fvg_min_gap must be numeric' in str(exc_info.value)
        assert 'dict' in str(exc_info.value)
    
    def test_init_rejects_string_for_ob_threshold(self):
        """Test that passing string raises TypeError"""
        with pytest.raises(TypeError) as exc_info:
            SMCDetector(
                ob_range_threshold="0.015",  # Invalid
                fvg_min_gap=0.005,
                liquidity_lookback=20
            )
        
        assert 'ob_range_threshold must be numeric' in str(exc_info.value)
    
    def test_init_rejects_string_for_liquidity_lookback(self):
        """Test that passing string for int param raises TypeError"""
        with pytest.raises(TypeError) as exc_info:
            SMCDetector(
                ob_range_threshold=0.015,
                fvg_min_gap=0.005,
                liquidity_lookback="20"  # Invalid
            )
        
        assert 'liquidity_lookback must be int' in str(exc_info.value)
    
    def test_init_with_defaults(self):
        """Test initialization with default values"""
        detector = SMCDetector()
        
        assert detector.ob_range_threshold == 0.015
        assert detector.fvg_min_gap == 0.005
        assert detector.liquidity_lookback == 20
    
    def test_init_converts_numeric_to_correct_types(self):
        """Test that numeric values are converted to correct types"""
        detector = SMCDetector(
            ob_range_threshold=15,  # int -> should become float
            fvg_min_gap=5,          # int -> should become float
            liquidity_lookback=20.0 # float -> should become int
        )
        
        assert isinstance(detector.ob_range_threshold, float)
        assert isinstance(detector.fvg_min_gap, float)
        assert isinstance(detector.liquidity_lookback, int)
