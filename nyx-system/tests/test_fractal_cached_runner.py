"""
Test Fractal Cached Runner

Validates cached runner behavior and parity with uncached pipeline.
"""

import sys
sys.path.insert(0, '.')

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

from src.core.fractal_cached_runner import FractalCachedRunner
from src.agents.orchestrator import Orchestrator


class TestFractalCachedRunner:
    """Test suite for fractal cached runner"""
    
    @pytest.fixture
    def config(self):
        """Load real config"""
        with open('config/validation_baseline.yaml') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def sample_mtf_data(self):
        """Create sample MTF data"""
        n_1d = 200
        n_1h = 500
        n_15m = 1000
        
        dates_1d = pd.date_range('2023-01-01', periods=n_1d, freq='1D')
        dates_1h = pd.date_range('2023-01-01', periods=n_1h, freq='1H')
        dates_15m = pd.date_range('2023-01-01', periods=n_15m, freq='15T')
        
        prices_1d = np.linspace(40000, 45000, n_1d)
        prices_1h = np.linspace(40000, 45000, n_1h)
        prices_15m = np.linspace(40000, 45000, n_15m)
        
        return {
            '1d': pd.DataFrame({
                'open': prices_1d,
                'high': prices_1d * 1.01,
                'low': prices_1d * 0.99,
                'close': prices_1d,
                'volume': np.random.rand(n_1d) * 1000
            }, index=dates_1d),
            '1h': pd.DataFrame({
                'open': prices_1h,
                'high': prices_1h * 1.01,
                'low': prices_1h * 0.99,
                'close': prices_1h,
                'volume': np.random.rand(n_1h) * 1000
            }, index=dates_1h),
            '15m': pd.DataFrame({
                'open': prices_15m,
                'high': prices_15m * 1.01,
                'low': prices_15m * 0.99,
                'close': prices_15m,
                'volume': np.random.rand(n_15m) * 1000
            }, index=dates_15m)
        }
    
    def test_runner_initializes(self, config):
        """Test runner initializes without errors"""
        runner = FractalCachedRunner(config)
        
        assert runner.cache is not None
        assert runner.context_agent is not None
        assert runner.regime_agent is not None
        assert runner.setup_agent is not None
    
    def test_first_call_computes_all_agents(self, config, sample_mtf_data):
        """Test first call recomputes all agents"""
        runner = FractalCachedRunner(config)
        
        # First decision
        decision = runner.decide(sample_mtf_data)
        
        # Check stats - all should have 1 recompute
        stats = runner.get_cache_stats()
        
        assert stats['context']['stats']['recompute_count'] == 1
        assert stats['regime']['stats']['recompute_count'] == 1
        assert stats['setup']['stats']['recompute_count'] == 1
    
    def test_second_call_reuses_htf_states(self, config, sample_mtf_data):
        """Test second call with same data reuses HTF states"""
        runner = FractalCachedRunner(config)
        
        # First decision
        runner.decide(sample_mtf_data)
        
        # Second decision with same data
        runner.decide(sample_mtf_data)
        
        # Check stats
        stats = runner.get_cache_stats()
        
        # Context and Regime should have cache hits
        # (Setup always recomputes but still gets cached for tracking)
        assert stats['context']['stats']['cache_hit_count'] >= 1
        assert stats['regime']['stats']['cache_hit_count'] >= 1
        
        # Context and Regime should not recompute twice
        assert stats['context']['stats']['recompute_count'] == 1
        assert stats['regime']['stats']['recompute_count'] == 1
        
        # Setup should recompute twice (always fresh)
        assert stats['setup']['stats']['recompute_count'] == 2
    
    def test_cached_produces_valid_decision(self, config, sample_mtf_data):
        """Test cached runner produces valid decision"""
        runner = FractalCachedRunner(config)
        
        decision = runner.decide(sample_mtf_data)
        
        assert decision is not None
        assert decision.action in ['BUY', 'SELL', 'WAIT']
        assert 0.0 <= decision.score <= 1.0
        assert decision.reason is not None
    
    def test_reset_cache_clears_state(self, config, sample_mtf_data):
        """Test reset_cache clears all cached state"""
        runner = FractalCachedRunner(config)
        
        # Make decision to populate cache
        runner.decide(sample_mtf_data)
        
        # Reset
        runner.reset_cache()
        
        # Check stats are reset
        stats = runner.get_cache_stats()
        assert stats['context']['stats']['recompute_count'] == 0
        assert stats['regime']['stats']['recompute_count'] == 0
        assert stats['setup']['stats']['recompute_count'] == 0


class TestFractalCacheParity:
    """Test parity between cached and uncached pipelines"""
    
    @pytest.fixture
    def config(self):
        """Load real config"""
        with open('config/validation_baseline.yaml') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def sample_mtf_data(self):
        """Create sample MTF data"""
        n_1d = 200
        n_1h = 500
        n_15m = 1000
        
        dates_1d = pd.date_range('2023-01-01', periods=n_1d, freq='1D')
        dates_1h = pd.date_range('2023-01-01', periods=n_1h, freq='1H')
        dates_15m = pd.date_range('2023-01-01', periods=n_15m, freq='15T')
        
        prices_1d = np.linspace(40000, 45000, n_1d)
        prices_1h = np.linspace(40000, 45000, n_1h)
        prices_15m = np.linspace(40000, 45000, n_15m)
        
        return {
            '1d': pd.DataFrame({
                'open': prices_1d,
                'high': prices_1d * 1.01,
                'low': prices_1d * 0.99,
                'close': prices_1d,
                'volume': np.random.rand(n_1d) * 1000
            }, index=dates_1d),
            '1h': pd.DataFrame({
                'open': prices_1h,
                'high': prices_1h * 1.01,
                'low': prices_1h * 0.99,
                'close': prices_1h,
                'volume': np.random.rand(n_1h) * 1000
            }, index=dates_1h),
            '15m': pd.DataFrame({
                'open': prices_15m,
                'high': prices_15m * 1.01,
                'low': prices_15m * 0.99,
                'close': prices_15m,
                'volume': np.random.rand(n_15m) * 1000
            }, index=dates_15m)
        }
    
    def test_parity_identical_timestamp(self, config, sample_mtf_data):
        """Test: cached and uncached produce same decision at identical timestamp"""
        # Uncached
        orchestrator = Orchestrator(config)
        uncached_decision = orchestrator.decide(sample_mtf_data)
        
        # Cached
        cached_runner = FractalCachedRunner(config)
        cached_decision = cached_runner.decide(sample_mtf_data)
        
        # Compare actions
        assert cached_decision.action == uncached_decision.action
        
        # Scores should be close (may differ slightly due to floating point)
        assert abs(cached_decision.score - uncached_decision.score) < 0.01
    
    def test_multiple_calls_maintain_parity(self, config, sample_mtf_data):
        """Test parity maintained across multiple calls"""
        orchestrator = Orchestrator(config)
        cached_runner = FractalCachedRunner(config)
        
        # Make 3 calls with same data
        for i in range(3):
            uncached = orchestrator.decide(sample_mtf_data)
            cached = cached_runner.decide(sample_mtf_data)
            
            assert cached.action == uncached.action
            assert abs(cached.score - uncached.score) < 0.01
