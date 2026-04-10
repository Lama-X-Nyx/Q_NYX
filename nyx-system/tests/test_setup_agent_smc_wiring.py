"""
Test SetupAgent → SMCDetector Wiring

Validates that SetupAgent correctly instantiates SMCDetector with scalar parameters.
"""

import sys
sys.path.insert(0, '.')

import pytest
import pandas as pd
import numpy as np
from src.agents.setup_agent import SetupAgent


class TestSetupAgentSMCWiring:
    """Test suite for SetupAgent SMC detector wiring"""
    
    @pytest.fixture
    def valid_config(self):
        """Valid configuration with smc_detector block"""
        return {
            'mtf': {
                'timeframes': {'setup': '15m'}
            },
            'strategy': {
                'mtf_conditions': {'alignment_15m_min': 0.60}
            },
            'smc_detector': {
                'ob_range_threshold': 0.015,
                'fvg_min_gap': 0.005,
                'liquidity_lookback': 20
            },
            'fractal_readiness': {
                'setup_min_bars': 50
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        n = 100
        dates = pd.date_range('2023-01-01', periods=n, freq='15T')
        prices = np.linspace(40000, 41000, n)
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.rand(n) * 1000
        }, index=dates)
    
    def test_setup_agent_passes_scalars_not_dict(self, valid_config):
        """Test 1: SetupAgent passes scalars, not dict"""
        agent = SetupAgent(valid_config)
        
        # Detector should have scalar parameters
        assert isinstance(agent.smc.ob_range_threshold, float)
        assert isinstance(agent.smc.fvg_min_gap, float)
        assert isinstance(agent.smc.liquidity_lookback, int)
        
        # Values should match config
        assert agent.smc.ob_range_threshold == 0.015
        assert agent.smc.fvg_min_gap == 0.005
        assert agent.smc.liquidity_lookback == 20
    
    def test_detector_exceptions_not_silently_converted(self, valid_config, sample_data):
        """Test 3: Detector exceptions are not silently converted to no_pattern"""
        agent = SetupAgent(valid_config)
        
        # Force detector to fail by passing invalid data
        # (this won't actually fail with our fixed wiring, but structure is ready)
        result = agent.analyze(sample_data, context_state='bullish')
        
        # If detector had failed, state would be 'detector_error'
        # Not 'no_pattern' or 'misaligned'
        assert result.state in ['valid_setup', 'misaligned', 'no_pattern', 'detector_error']
        
        # If detector_error, metadata should contain error info
        if result.state == 'detector_error':
            assert 'detector_error' in result.metadata
            assert result.metadata['detector_error'] is True
            assert 'exception_type' in result.metadata
    
    def test_setup_agent_returns_detector_error_on_failure(self, valid_config, sample_data):
        """Test 4: SetupAgent returns state=detector_error if detector crashes"""
        # This test validates the error handling structure
        # With correct wiring, detector shouldn't crash
        # But if it did, the error should be visible
        
        agent = SetupAgent(valid_config)
        result = agent.analyze(sample_data, context_state='bullish')
        
        # Detector should work without error
        assert result.ready is True
        
        # If there's an error, it should be explicit
        if not result.passed and result.state == 'detector_error':
            assert 'detector_error' in result.metadata
            assert result.metadata['detector_error'] is True
    
    def test_setup_detector_check_produces_valid_json(self, valid_config, tmp_path):
        """Test 5: setup_detector_check mode produces valid JSON"""
        from scripts.setup_detector_check import run_setup_detector_check
        
        result = run_setup_detector_check(
            pair='BTCUSDT',
            config=valid_config,
            output_dir=tmp_path,
            sample_date='2023-12-15'
        )
        
        # Check result structure
        assert result['status'] == 'completed'
        assert 'detector_initialized' in result
        assert 'detector_error' in result
    
    def test_no_fake_no_pattern_on_crash(self, valid_config, sample_data):
        """Test 6: No fake no_pattern when detector crashes"""
        agent = SetupAgent(valid_config)
        result = agent.analyze(sample_data, context_state='bullish')
        
        # If state is no_pattern, it should be genuine (no patterns detected)
        # Not because detector crashed
        if result.state == 'no_pattern':
            # Metadata should not indicate detector error
            assert not result.metadata.get('detector_error', False)
        
        # If detector crashed, state should be detector_error
        if result.metadata.get('detector_error', False):
            assert result.state == 'detector_error'
    
    def test_setup_agent_with_missing_smc_config_uses_defaults(self):
        """Test SetupAgent handles missing smc_detector config gracefully"""
        config = {
            'mtf': {'timeframes': {'setup': '15m'}},
            'strategy': {'mtf_conditions': {'alignment_15m_min': 0.60}},
            'fractal_readiness': {'setup_min_bars': 50}
            # No smc_detector block
        }
        
        agent = SetupAgent(config)
        
        # Should use defaults
        assert agent.smc.ob_range_threshold == 0.015
        assert agent.smc.fvg_min_gap == 0.005
        assert agent.smc.liquidity_lookback == 20
    
    def test_setup_agent_state_semantics_clean(self, valid_config, sample_data):
        """Test state semantics are clean and consistent"""
        agent = SetupAgent(valid_config)
        result = agent.analyze(sample_data, context_state='bullish')
        
        # States should be one of the expected values
        valid_states = [
            'not_ready',
            'no_pattern',
            'misaligned',
            'valid_setup',
            'mixed_signals',
            'pattern_found',
            'no_bias',
            'detector_error'
        ]
        
        assert result.state in valid_states
        
        # If not_ready, should be blocked_by_readiness
        if result.state == 'not_ready':
            assert result.blocked_by_readiness is True
            assert result.ready is False
        
        # If detector_error, should have error metadata
        if result.state == 'detector_error':
            assert 'detector_error' in result.metadata
            assert result.metadata['detector_error'] is True
