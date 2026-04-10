"""
Test Fractal State Cache

Validates timestamp-based caching of agent states.
"""

import sys
sys.path.insert(0, '.')

import pytest
from datetime import datetime
import pandas as pd

from src.cache.fractal_state_cache import FractalStateCache, get_closed_bar_timestamp
from src.cache.contracts import CacheEntry, CacheStats
from src.agents.contracts import AgentResult


class TestFractalStateCache:
    """Test suite for fractal state cache"""
    
    @pytest.fixture
    def cache(self):
        """Create fresh cache"""
        return FractalStateCache()
    
    @pytest.fixture
    def sample_result(self):
        """Create sample agent result"""
        return AgentResult(
            agent='context',
            state='bullish',
            score=0.75,
            passed=True,
            ready=True,
            blocked_by_readiness=False,
            reason='Test result',
            metadata={'test': True}
        )
    
    def test_cache_hit_when_timestamp_unchanged(self, cache, sample_result):
        """Test 1: Cache hit if closed bar timestamp hasn't changed"""
        timestamp = datetime(2023, 12, 15, 0, 0, 0)
        
        # Store result
        cache.set('context', '1d', timestamp, sample_result, {'bars': 100})
        
        # Retrieve with same timestamp
        retrieved = cache.get_if_valid('context', '1d', timestamp)
        
        assert retrieved is not None
        assert retrieved.state == 'bullish'
        assert retrieved.score == 0.75
        
        # Check stats
        stats = cache.stats()['context']
        assert stats.cache_hit_count == 1
        assert stats.cache_miss_count == 0
    
    def test_cache_miss_when_timestamp_changed(self, cache, sample_result):
        """Test 2: Cache miss if closed bar timestamp has changed"""
        old_timestamp = datetime(2023, 12, 15, 0, 0, 0)
        new_timestamp = datetime(2023, 12, 15, 1, 0, 0)
        
        # Store with old timestamp
        cache.set('context', '1d', old_timestamp, sample_result, {'bars': 100})
        
        # Try to retrieve with new timestamp
        retrieved = cache.get_if_valid('context', '1d', new_timestamp)
        
        assert retrieved is None
        
        # Check stats
        stats = cache.stats()['context']
        assert stats.cache_hit_count == 0
        assert stats.cache_miss_count == 1
    
    def test_invalidation_works(self, cache, sample_result):
        """Test 3: Invalidation marks cache entry invalid"""
        timestamp = datetime(2023, 12, 15, 0, 0, 0)
        
        # Store result
        cache.set('context', '1d', timestamp, sample_result, {'bars': 100})
        
        # Invalidate
        cache.invalidate('context')
        
        # Try to retrieve - should be None
        retrieved = cache.get_if_valid('context', '1d', timestamp)
        
        assert retrieved is None
        
        # Should record as cache miss
        stats = cache.stats()['context']
        assert stats.cache_miss_count == 1
    
    def test_no_cross_timeframe_contamination(self, cache, sample_result):
        """Test 4: No cross-timeframe cache contamination"""
        timestamp = datetime(2023, 12, 15, 0, 0, 0)
        
        # Store Context result
        cache.set('context', '1d', timestamp, sample_result, {'bars': 100})
        
        # Try to retrieve as Regime - should be None
        retrieved = cache.get_if_valid('regime', '1h', timestamp)
        
        assert retrieved is None
    
    def test_separate_agent_caches(self, cache, sample_result):
        """Test agents have separate caches"""
        timestamp = datetime(2023, 12, 15, 0, 0, 0)
        
        # Store results for different agents
        context_result = sample_result
        regime_result = AgentResult(
            agent='regime',
            state='trend_plus',
            score=0.80,
            passed=True,
            ready=True,
            blocked_by_readiness=False,
            reason='Test',
            metadata={}
        )
        
        cache.set('context', '1d', timestamp, context_result, {})
        cache.set('regime', '1h', timestamp, regime_result, {})
        
        # Retrieve both
        context_retrieved = cache.get_if_valid('context', '1d', timestamp)
        regime_retrieved = cache.get_if_valid('regime', '1h', timestamp)
        
        assert context_retrieved.state == 'bullish'
        assert regime_retrieved.state == 'trend_plus'
    
    def test_no_crash_when_cache_empty(self, cache):
        """Test 6: No crash if cache is empty"""
        timestamp = datetime(2023, 12, 15, 0, 0, 0)
        
        # Try to retrieve from empty cache
        retrieved = cache.get_if_valid('context', '1d', timestamp)
        
        assert retrieved is None
        
        # Should record as cache miss
        stats = cache.stats()['context']
        assert stats.cache_miss_count == 1
    
    def test_stats_tracking_consistent(self, cache, sample_result):
        """Test 7: Stats tracking is consistent"""
        timestamp = datetime(2023, 12, 15, 0, 0, 0)
        
        # Cache miss (empty)
        cache.get_if_valid('context', '1d', timestamp)
        
        # Store
        cache.set('context', '1d', timestamp, sample_result, {})
        cache.record_recompute('context')
        
        # Cache hit
        cache.get_if_valid('context', '1d', timestamp)
        
        # Cache hit again
        cache.get_if_valid('context', '1d', timestamp)
        
        # Check stats
        stats = cache.stats()['context']
        assert stats.cache_hit_count == 2
        assert stats.cache_miss_count == 1
        assert stats.recompute_count == 1
        
        # Check hit rate
        assert stats.hit_rate() == pytest.approx(66.67, abs=0.1)
    
    def test_get_closed_bar_timestamp(self):
        """Test get_closed_bar_timestamp helper"""
        # Create sample dataframe
        dates = pd.date_range('2023-12-15', periods=5, freq='1H')
        df = pd.DataFrame({'close': [100, 101, 102, 103, 104]}, index=dates)
        
        timestamp = get_closed_bar_timestamp(df)
        
        assert timestamp == dates[-1]
    
    def test_cache_summary(self, cache, sample_result):
        """Test cache summary provides complete state"""
        timestamp = datetime(2023, 12, 15, 0, 0, 0)
        
        # Store some data
        cache.set('context', '1d', timestamp, sample_result, {'bars': 100})
        cache.get_if_valid('context', '1d', timestamp)
        
        # Get summary
        summary = cache.summary()
        
        assert 'context' in summary
        assert 'regime' in summary
        assert 'setup' in summary
        assert summary['context']['cached'] is True
        assert summary['context']['stats']['cache_hit_count'] == 1
