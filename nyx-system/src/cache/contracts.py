"""
Fractal State Cache Contracts

Defines data structures for cached agent states.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CacheEntry:
    """
    Single cache entry for an agent state
    
    Attributes:
        timeframe: Timeframe of the cached data (e.g., '1d', '1h', '15m')
        last_closed_bar_timestamp: Timestamp of the last closed bar
        result: The cached agent result (AgentResult object)
        metadata: Additional metadata about the cached computation
        is_valid: Whether this cache entry is valid
    """
    
    timeframe: str
    last_closed_bar_timestamp: datetime
    result: Any  # AgentResult
    metadata: Dict[str, Any]
    is_valid: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'timeframe': self.timeframe,
            'last_closed_bar_timestamp': self.last_closed_bar_timestamp.isoformat(),
            'result': self._serialize_result(self.result),
            'metadata': self.metadata,
            'is_valid': self.is_valid
        }
    
    def _serialize_result(self, result) -> dict:
        """Serialize agent result"""
        if hasattr(result, '__dict__'):
            return {
                'agent': result.agent,
                'state': result.state,
                'score': result.score,
                'passed': result.passed,
                'ready': result.ready,
                'blocked_by_readiness': result.blocked_by_readiness,
                'reason': result.reason,
                'metadata': result.metadata if hasattr(result, 'metadata') else {}
            }
        return result


@dataclass
class CacheStats:
    """
    Cache statistics for monitoring
    
    Attributes:
        cache_hit_count: Number of successful cache hits
        cache_miss_count: Number of cache misses (recompute needed)
        recompute_count: Total recompute operations
    """
    
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    recompute_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'cache_hit_count': self.cache_hit_count,
            'cache_miss_count': self.cache_miss_count,
            'recompute_count': self.recompute_count,
            'hit_rate': self.hit_rate()
        }
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hit_count + self.cache_miss_count
        if total == 0:
            return 0.0
        return self.cache_hit_count / total * 100


@dataclass
class FractalCacheState:
    """
    Complete cache state for all fractal agents
    
    Attributes:
        context_cache: Cache entry for Context agent
        regime_cache: Cache entry for Regime agent
        setup_cache: Cache entry for Setup agent
        context_stats: Statistics for Context cache
        regime_stats: Statistics for Regime cache
        setup_stats: Statistics for Setup cache
    """
    
    context_cache: Optional[CacheEntry] = None
    regime_cache: Optional[CacheEntry] = None
    setup_cache: Optional[CacheEntry] = None
    
    context_stats: CacheStats = None
    regime_stats: CacheStats = None
    setup_stats: CacheStats = None
    
    def __post_init__(self):
        """Initialize stats if not provided"""
        if self.context_stats is None:
            self.context_stats = CacheStats()
        if self.regime_stats is None:
            self.regime_stats = CacheStats()
        if self.setup_stats is None:
            self.setup_stats = CacheStats()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for reporting"""
        return {
            'context': {
                'cached': self.context_cache is not None,
                'last_timestamp': self.context_cache.last_closed_bar_timestamp.isoformat() 
                    if self.context_cache else None,
                'stats': self.context_stats.to_dict()
            },
            'regime': {
                'cached': self.regime_cache is not None,
                'last_timestamp': self.regime_cache.last_closed_bar_timestamp.isoformat()
                    if self.regime_cache else None,
                'stats': self.regime_stats.to_dict()
            },
            'setup': {
                'cached': self.setup_cache is not None,
                'last_timestamp': self.setup_cache.last_closed_bar_timestamp.isoformat()
                    if self.setup_cache else None,
                'stats': self.setup_stats.to_dict()
            }
        }
