"""
Tests for HSMM Input Contract - P1-06c

Validates that NYXEngine and HSMM use the same agreed format.

Run:
    pytest tests/test_hsmm_input_contract.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.hsmm import SemiMarkovHMM
from src.core.nyx_engine import NYXEngine


class TestHSMMInputContract:
    """Test HSMM input format contract"""
    
    def test_hsmm_expects_list_of_dicts(self):
        """HSMM forward_backward should expect List[Dict]"""
        hsmm = SemiMarkovHMM()
        
        # Create sample data for initialization
        data = pd.DataFrame({
            'close': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'returns': np.random.randn(100) * 0.01,
            'atr_14': np.random.randn(100) * 2 + 5
        })
        
        hsmm.initialize_parameters(data)
        
        # Correct format: List[Dict]
        observations = [
            {'price': 0.01, 'atr': 5.0},
            {'price': -0.02, 'atr': 5.5},
            {'price': 0.015, 'atr': 4.8}
        ]
        
        # Should work without error
        probs = hsmm.forward_backward(observations)
        
        assert probs.shape == (3, 3)  # 3 observations, 3 states
    
    def test_hsmm_rejects_wrong_type_ndarray(self):
        """HSMM should reject np.ndarray input"""
        hsmm = SemiMarkovHMM()
        
        # Initialize
        data = pd.DataFrame({
            'close': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'returns': np.random.randn(100) * 0.01,
            'atr_14': np.random.randn(100) * 2 + 5
        })
        
        hsmm.initialize_parameters(data)
        
        # Wrong format: np.ndarray
        observations = np.random.randn(10, 2)
        
        # Should raise TypeError
        with pytest.raises(TypeError, match="HSMM expects List\\[Dict\\]"):
            hsmm.forward_backward(observations)  # type: ignore[arg-type]

    def test_hsmm_rejects_list_of_non_dicts(self):
        """HSMM should reject list of non-dict types"""
        hsmm = SemiMarkovHMM()
        
        # Initialize
        data = pd.DataFrame({
            'close': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'returns': np.random.randn(100) * 0.01,
            'atr_14': np.random.randn(100) * 2 + 5
        })
        
        hsmm.initialize_parameters(data)
        
        # Wrong format: list of tuples
        observations = [(0.01, 5.0), (-0.02, 5.5)]

        # Should raise TypeError
        with pytest.raises(TypeError, match="HSMM expects observations as List\\[Dict\\]"):
            hsmm.forward_backward(observations)  # type: ignore[arg-type]


class TestNYXEngineHSMMIntegration:
    """Test NYXEngine produces correct HSMM input"""
    
    def test_nyx_engine_produces_correct_format(self):
        """NYXEngine should send List[Dict] to HSMM"""
        
        # Create config
        config = {
            'strategy': {
                'hsmm': {'sdc_threshold': 3.5, 'prob_threshold': 0.55}
            },
            'risk': {
                'base_size': 0.05,
                'max_position': 0.25,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'bull_multiplier': 2.5,
                'bear_multiplier': 0.7,
                'range_multiplier': 1.0
            }
        }
        
        engine = NYXEngine(config)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        data = pd.DataFrame({
            'open': np.random.randn(200) * 100 + 50000,
            'high': np.random.randn(200) * 100 + 50100,
            'low': np.random.randn(200) * 100 + 49900,
            'close': np.random.randn(200) * 100 + 50000,
            'volume': np.random.lognormal(10, 1, 200)
        }, index=dates)
        
        # This should work without TypeError
        signal = engine.generate_signal(
            pair='BTCUSDT',
            data=data,
            current_date='2024-01-08'
        )
        
        # Signal should be generated
        assert signal is not None
        assert 'action' in signal
        assert 'confidence' in signal


class TestEmissionProbability:
    """Test emission probability uses correct observation format"""
    
    def test_emission_uses_price_and_atr_keys(self):
        """Emission probability should access 'price' and 'atr' keys"""
        hsmm = SemiMarkovHMM()
        
        # Initialize
        data = pd.DataFrame({
            'close': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'returns': np.random.randn(100) * 0.01,
            'atr_14': np.random.randn(100) * 2 + 5
        })
        
        hsmm.initialize_parameters(data)
        
        # Observation with correct keys
        obs = {'price': 0.01, 'atr': 5.0}
        
        # Should return non-zero log probability
        log_prob = hsmm.emission_probability(obs, 'Trend+')
        
        assert log_prob != 0.0  # Not uniform
        assert not np.isnan(log_prob)
    
    def test_emission_handles_missing_keys(self):
        """Emission should handle observations with missing keys"""
        hsmm = SemiMarkovHMM()
        
        # Initialize
        data = pd.DataFrame({
            'close': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'returns': np.random.randn(100) * 0.01,
            'atr_14': np.random.randn(100) * 2 + 5
        })
        
        hsmm.initialize_parameters(data)
        
        # Observation with only price
        obs = {'price': 0.01}
        
        # Should still work (partial observation)
        log_prob = hsmm.emission_probability(obs, 'Trend+')
        
        assert not np.isnan(log_prob)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
