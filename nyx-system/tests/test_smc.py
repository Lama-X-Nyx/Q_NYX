"""
Unit Tests for Smart Money Concepts Module

Run:
    pytest tests/test_smc.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.smc import SMCDetector


@pytest.fixture
def sample_data():
    """Create sample OHLCV data"""
    np.random.seed(42)
    n = 100
    
    base_price = 10000
    data = pd.DataFrame({
        'open': base_price + np.cumsum(np.random.randn(n) * 50),
        'high': base_price + np.cumsum(np.random.randn(n) * 50) + np.random.rand(n) * 100,
        'low': base_price + np.cumsum(np.random.randn(n) * 50) - np.random.rand(n) * 100,
        'close': base_price + np.cumsum(np.random.randn(n) * 50),
        'volume': 1000000 + np.random.rand(n) * 500000
    })
    
    # Ensure OHLC logic
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data


class TestSMCInitialization:
    """Test SMC detector initialization"""
    
    def test_default_params(self):
        """Test default parameters"""
        smc = SMCDetector()
        
        assert smc.ob_range_threshold == 0.015
        assert smc.fvg_min_gap == 0.005
        assert smc.liquidity_lookback == 20
    
    def test_custom_params(self):
        """Test custom parameters"""
        smc = SMCDetector(
            ob_range_threshold=0.02,
            fvg_min_gap=0.01,
            liquidity_lookback=30
        )
        
        assert smc.ob_range_threshold == 0.02
        assert smc.fvg_min_gap == 0.01
        assert smc.liquidity_lookback == 30


class TestSMCPatternDetection:
    """Test pattern detection methods"""
    
    def test_detect_all_returns_dict(self, sample_data):
        """Test that detect_all returns dict with all patterns"""
        smc = SMCDetector()
        patterns = smc.detect_all(sample_data)
        
        assert isinstance(patterns, dict)
        assert 'bullish_ob' in patterns
        assert 'bearish_ob' in patterns
        assert 'bullish_fvg' in patterns
        assert 'bearish_fvg' in patterns
        assert 'liquidity_sweep' in patterns
        assert 'bos_bullish' in patterns
        assert 'bos_bearish' in patterns
    
    def test_order_block_detection(self, sample_data):
        """Test Order Block detection"""
        smc = SMCDetector()
        
        bullish_ob = smc.detect_order_block(sample_data, bullish=True)
        bearish_ob = smc.detect_order_block(sample_data, bullish=False)
        
        assert isinstance(bullish_ob, bool)
        assert isinstance(bearish_ob, bool)
    
    def test_fvg_detection(self, sample_data):
        """Test Fair Value Gap detection"""
        smc = SMCDetector()
        
        bullish_fvg = smc.detect_fvg(sample_data, bullish=True)
        bearish_fvg = smc.detect_fvg(sample_data, bullish=False)
        
        assert isinstance(bullish_fvg, bool)
        assert isinstance(bearish_fvg, bool)
    
    def test_liquidity_sweep_detection(self, sample_data):
        """Test liquidity sweep detection"""
        smc = SMCDetector()
        
        sweep = smc.detect_liquidity_sweep(sample_data)
        
        assert isinstance(sweep, bool)
    
    def test_bos_detection(self, sample_data):
        """Test Break of Structure detection"""
        smc = SMCDetector()
        
        bos_bull = smc.detect_bos(sample_data, bullish=True)
        bos_bear = smc.detect_bos(sample_data, bullish=False)
        
        assert isinstance(bos_bull, bool)
        assert isinstance(bos_bear, bool)


class TestSMCZones:
    """Test zone extraction"""
    
    def test_get_ob_zones(self, sample_data):
        """Test Order Block zone extraction"""
        smc = SMCDetector()
        
        zones = smc.get_ob_zones(sample_data, lookback=50)
        
        assert isinstance(zones, list)
        
        for zone in zones:
            assert 'type' in zone
            assert 'high' in zone
            assert 'low' in zone
            assert 'strength' in zone
            assert zone['type'] in ['bullish', 'bearish']
            assert zone['high'] >= zone['low']
    
    def test_get_fvg_zones(self, sample_data):
        """Test FVG zone extraction"""
        smc = SMCDetector()
        
        zones = smc.get_fvg_zones(sample_data, lookback=50)
        
        assert isinstance(zones, list)
        
        for zone in zones:
            assert 'type' in zone
            assert 'high' in zone
            assert 'low' in zone
            assert 'gap' in zone
            assert zone['type'] in ['bullish', 'bearish']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
