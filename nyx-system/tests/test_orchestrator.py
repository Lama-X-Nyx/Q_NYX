"""
Test Orchestrator
"""

import sys
sys.path.insert(0, '.')

import pytest
import pandas as pd
import numpy as np
from src.agents.orchestrator import Orchestrator
from src.agents.contracts import OrchestratorDecision


class TestOrchestrator:
    """Test suite for Orchestrator"""
    
    @pytest.fixture
    def config(self):
        return {
            'fractal': {
                'enabled': True,
                'use_entry_agent': False,
                'context_tf': '1d',
                'regime_tf': '1h',
                'setup_tf': '15m'
            },
            'strategy': {
                'mtf_conditions': {
                    'sdc_min': 5.0,
                    'stability_4h_min': 0.60,
                    'alignment_15m_min': 0.60
                },
                'smc': {
                    'ob_range_threshold': 0.015,
                    'fvg_threshold': 0.01
                }
            }
        }
    
    @pytest.fixture
    def orchestrator(self, config):
        return Orchestrator(config)
    
    @pytest.fixture
    def mtf_data(self):
        """Create multi-timeframe data"""
        dates_1d = pd.date_range('2023-01-01', periods=100, freq='1D')
        dates_1h = pd.date_range('2023-01-01', periods=400, freq='1H')
        dates_15m = pd.date_range('2023-01-01', periods=1000, freq='15T')
        
        prices_1d = np.linspace(40000, 50000, 100)
        prices_1h = np.linspace(40000, 50000, 400)
        prices_15m = np.linspace(49000, 50000, 1000)
        
        return {
            '1d': pd.DataFrame({
                'open': prices_1d,
                'high': prices_1d * 1.01,
                'low': prices_1d * 0.99,
                'close': prices_1d,
                'volume': np.random.rand(100) * 1000
            }, index=dates_1d),
            
            '1h': pd.DataFrame({
                'open': prices_1h,
                'high': prices_1h * 1.01,
                'low': prices_1h * 0.99,
                'close': prices_1h,
                'volume': np.random.rand(400) * 1000
            }, index=dates_1h),
            
            '15m': pd.DataFrame({
                'open': prices_15m,
                'high': prices_15m * 1.005,
                'low': prices_15m * 0.995,
                'close': prices_15m,
                'volume': np.random.rand(1000) * 1000
            }, index=dates_15m)
        }
    
    def test_returns_valid_decision(self, orchestrator, mtf_data):
        """Test that orchestrator returns valid OrchestratorDecision"""
        decision = orchestrator.decide(mtf_data)
        
        assert isinstance(decision, OrchestratorDecision)
        assert decision.action in ['BUY', 'SELL', 'WAIT']
        assert isinstance(decision.score, float)
        assert isinstance(decision.reason, str)
        assert isinstance(decision.blocked_by, list)
        assert isinstance(decision.components, dict)
    
    def test_has_3_agents(self, orchestrator, mtf_data):
        """Test that orchestrator uses 3 agents (no entry)"""
        decision = orchestrator.decide(mtf_data)
        
        # Should have exactly 3 components
        assert len(decision.components) == 3
        assert 'context' in decision.components
        assert 'regime' in decision.components
        assert 'setup' in decision.components
        assert 'entry' not in decision.components
    
    def test_blocked_by_tracking(self, orchestrator, mtf_data):
        """Test that orchestrator tracks which agents block"""
        decision = orchestrator.decide(mtf_data)
        
        # If action is WAIT, should have blocked_by
        if decision.action == 'WAIT':
            # Check that blocked_by is consistent with components
            for agent_name in decision.blocked_by:
                if agent_name != 'risk':  # Risk is external
                    assert agent_name in decision.components
                    assert decision.components[agent_name].passed == False
    
    def test_all_agents_called(self, orchestrator, mtf_data):
        """Test that all 3 agents are called"""
        decision = orchestrator.decide(mtf_data)
        
        # All agents should have results
        assert 'context' in decision.components
        assert 'regime' in decision.components
        assert 'setup' in decision.components
        
        # All results should be AgentResult objects
        from src.agents.contracts import AgentResult
        for result in decision.components.values():
            assert isinstance(result, AgentResult)
    
    def test_score_bounded(self, orchestrator, mtf_data):
        """Test that aggregate score is in [0, 1]"""
        decision = orchestrator.decide(mtf_data)
        
        assert 0 <= decision.score <= 1
    
    def test_no_5m_fallback(self, orchestrator, mtf_data):
        """Test that there's no hidden 5m fallback"""
        # mtf_data doesn't have 5m
        assert '5m' not in mtf_data
        
        # Should still work fine
        decision = orchestrator.decide(mtf_data)
        assert isinstance(decision, OrchestratorDecision)
        
        # No entry agent in components
        assert 'entry' not in decision.components


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
