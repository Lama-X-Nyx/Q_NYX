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
        dates_1h = pd.date_range('2023-01-01', periods=400, freq='1h')
        dates_15m = pd.date_range('2023-01-01', periods=1000, freq='15min')
        
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
        assert '5m' not in mtf_data
        decision = orchestrator.decide(mtf_data)
        assert isinstance(decision, OrchestratorDecision)
        assert 'entry' not in decision.components

    def test_blocked_by_allows_external_blockers(self, orchestrator, mtf_data):
        """blocked_by may contain non-agent names (risk, macro_block, readiness)"""
        decision = orchestrator.decide(mtf_data)
        EXTERNAL = {'risk', 'macro_block', 'readiness'}
        for agent_name in decision.blocked_by:
            if agent_name not in EXTERNAL:
                assert agent_name in decision.components, \
                    f"Unknown blocker '{agent_name}' not in components"

    # ------------------------------------------------------------------
    # P2 — hitting probs from Monte Carlo
    # ------------------------------------------------------------------

    def test_risk_analysis_has_hit_method(self, orchestrator, mtf_data):
        """When risk_analysis is present, hit_method key should exist"""
        decision = orchestrator.decide(mtf_data)
        if decision.risk_analysis:
            if 'hit_probs' in decision.risk_analysis:
                assert 'method' in decision.risk_analysis['hit_probs'], \
                    "hit_probs must include 'method' key (monte_carlo or lookup_table)"

    def test_monte_carlo_produces_valid_probs(self):
        """Monte Carlo hitting probs must be in [0, 1] and P(-10%) ≤ P(-5%)"""
        from src.core.risk_manager_mtf import RiskManagerMTF
        import numpy as np

        rm = RiskManagerMTF({})
        A = np.array([[0.75, 0.15, 0.10],
                      [0.20, 0.60, 0.20],
                      [0.10, 0.15, 0.75]])
        emission_params = {
            'Trend+': {'price_mu':  0.001, 'price_sigma': 0.008},
            'Range':  {'price_mu':  0.000, 'price_sigma': 0.005},
            'Trend-': {'price_mu': -0.001, 'price_sigma': 0.008},
        }
        fractal_states = {'4h': {'Trend+': 0.60, 'Range': 0.25, 'Trend-': 0.15}}

        result = rm.compute_hitting_probabilities(
            50000, fractal_states, A, emission_params=emission_params
        )

        assert 0.0 <= result['minus_0_05'] <= 1.0
        assert 0.0 <= result['minus_0_10'] <= 1.0
        assert result['minus_0_10'] <= result['minus_0_05'] + 1e-9, \
            "P(-10%) must be ≤ P(-5%)"
        assert result.get('method') == 'monte_carlo'

    def test_lookup_fallback_when_no_emission_params(self):
        """Without emission_params, falls back to lookup table"""
        from src.core.risk_manager_mtf import RiskManagerMTF
        import numpy as np

        rm = RiskManagerMTF({})
        fractal_states = {'4h': {'Trend+': 0.70, 'Range': 0.20, 'Trend-': 0.10}}
        A = np.eye(3)

        result = rm.compute_hitting_probabilities(50000, fractal_states, A)
        assert result.get('method') == 'lookup_table'

    # ------------------------------------------------------------------
    # P3 — macro block
    # ------------------------------------------------------------------

    def test_macro_disabled_by_default(self, orchestrator):
        """Macro engine must be disabled when config has no macro.enabled=True"""
        assert orchestrator._macro_enabled is False
        assert orchestrator.macro_engine is None

    def test_macro_enabled_with_config(self):
        """Macro engine is instantiated when macro.enabled=True"""
        config = {
            'fractal': {'enabled': True, 'use_entry_agent': False,
                        'context_tf': '1d', 'regime_tf': '1h', 'setup_tf': '15m'},
            'strategy': {'mtf_conditions': {'sdc_min': 5.0,
                                             'stability_4h_min': 0.60,
                                             'alignment_15m_min': 0.60}},
            'macro': {'enabled': True, 'events_file': 'data/macro_events.json'}
        }
        orch = Orchestrator(config)
        assert orch._macro_enabled is True
        assert orch.macro_engine is not None

    def test_macro_block_returns_wait(self):
        """When macro block fires, action must be WAIT with 'macro_block' in blocked_by"""
        from unittest.mock import patch, MagicMock

        config = {
            'fractal': {'enabled': True, 'use_entry_agent': False,
                        'context_tf': '1d', 'regime_tf': '1h', 'setup_tf': '15m'},
            'strategy': {'mtf_conditions': {'sdc_min': 5.0,
                                             'stability_4h_min': 0.60,
                                             'alignment_15m_min': 0.60}},
            'macro': {'enabled': True, 'block_threshold': 0.3,
                      'events_file': 'data/macro_events.json'}
        }

        dates_1d  = pd.date_range('2023-01-01', periods=100, freq='1D')
        dates_1h  = pd.date_range('2023-01-01', periods=400, freq='1h')
        dates_15m = pd.date_range('2023-01-01', periods=1000, freq='15min')
        prices    = np.linspace(40000, 50000, 100)

        mtf = {
            '1d':  pd.DataFrame({'open': prices, 'high': prices*1.01,
                                  'low': prices*0.99, 'close': prices,
                                  'volume': np.ones(100)*1000}, index=dates_1d),
            '1h':  pd.DataFrame({'open': np.linspace(40000,50000,400),
                                  'high': np.linspace(40000,50000,400)*1.01,
                                  'low':  np.linspace(40000,50000,400)*0.99,
                                  'close':np.linspace(40000,50000,400),
                                  'volume': np.ones(400)*1000}, index=dates_1h),
            '15m': pd.DataFrame({'open': np.linspace(49000,50000,1000),
                                  'high': np.linspace(49000,50000,1000)*1.005,
                                  'low':  np.linspace(49000,50000,1000)*0.995,
                                  'close':np.linspace(49000,50000,1000),
                                  'volume': np.ones(1000)*1000}, index=dates_15m),
        }

        orch = Orchestrator(config)

        # Force macro engine to return a strong BEARISH signal
        orch.macro_engine.get_macro_signal = MagicMock(return_value={
            'signal': 'BEARISH', 'strength': 0.80,
            'active_events': [], 'cumulative_impact': -0.80
        })

        decision = orch.decide(mtf)
        assert decision.action == 'WAIT'
        assert 'macro_block' in decision.blocked_by


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
