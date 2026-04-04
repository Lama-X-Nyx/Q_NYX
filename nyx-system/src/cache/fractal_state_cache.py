"""
Fractal State Cache

Manages caching of agent states to avoid unnecessary recomputation.

V1 Safe Implementation:
- Timestamp-based invalidation only
- No complex incremental feature updates
- Simple state reuse when closed bar hasn't changed
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd

from src.cache.contracts import CacheEntry, CacheStats, FractalCacheState

logger = logging.getLogger(__name__)


class FractalStateCache:
    """
    Fractal State Cache
    
    Caches agent states by timeframe to avoid recomputation when
    the closed bar timestamp hasn't changed.
    
    V1 Safe: timestamp-based reuse only, no look-ahead, no logic changes.
    """
    
    def __init__(self):
        """Initialize cache"""
        self.state = FractalCacheState()
        self.logger = logging.getLogger(__name__)
    
    def get_if_valid(
        self, 
        agent: str, 
        timeframe: str, 
        current_closed_bar_timestamp: datetime
    ) -> Optional[Any]:
        """
        Get cached result if valid for current timestamp
        
        Args:
            agent: Agent name ('context', 'regime', 'setup')
            timeframe: Timeframe string
            current_closed_bar_timestamp: Current closed bar timestamp
        
        Returns:
            Cached result if valid, None if cache miss
        """
        
        # Get cache entry for agent
        cache_entry = self._get_cache_entry(agent)
        
        if cache_entry is None:
            self.logger.debug(f"{agent}: cache_miss (empty)")
            self._record_miss(agent)
            return None
        
        # Check if timestamps match
        if cache_entry.last_closed_bar_timestamp != current_closed_bar_timestamp:
            self.logger.debug(
                f"{agent}: cache_miss (timestamp mismatch: "
                f"cached={cache_entry.last_closed_bar_timestamp}, "
                f"current={current_closed_bar_timestamp})"
            )
            self._record_miss(agent)
            return None
        
        # Check if valid
        if not cache_entry.is_valid:
            self.logger.debug(f"{agent}: cache_miss (invalidated)")
            self._record_miss(agent)
            return None
        
        # Cache hit!
        self.logger.debug(f"{agent}: cache_hit")
        self._record_hit(agent)
        return cache_entry.result
    
    def set(
        self,
        agent: str,
        timeframe: str,
        closed_bar_timestamp: datetime,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store agent result in cache
        
        Args:
            agent: Agent name ('context', 'regime', 'setup')
            timeframe: Timeframe string
            closed_bar_timestamp: Closed bar timestamp
            result: Agent result to cache
            metadata: Optional metadata
        """
        
        if metadata is None:
            metadata = {}
        
        # Create cache entry
        entry = CacheEntry(
            timeframe=timeframe,
            last_closed_bar_timestamp=closed_bar_timestamp,
            result=result,
            metadata=metadata,
            is_valid=True
        )
        
        # Store in appropriate cache slot
        self._set_cache_entry(agent, entry)
        
        self.logger.debug(
            f"{agent}: cached (timestamp={closed_bar_timestamp})"
        )
    
    def invalidate(self, agent: str):
        """
        Invalidate cache for specific agent
        
        Args:
            agent: Agent name to invalidate
        """
        cache_entry = self._get_cache_entry(agent)
        if cache_entry:
            cache_entry.is_valid = False
            self.logger.debug(f"{agent}: cache invalidated")
    
    def invalidate_all(self):
        """Invalidate all caches"""
        self.invalidate('context')
        self.invalidate('regime')
        self.invalidate('setup')
    
    def stats(self) -> Dict[str, CacheStats]:
        """
        Get cache statistics
        
        Returns:
            Dictionary of agent -> CacheStats
        """
        return {
            'context': self.state.context_stats,
            'regime': self.state.regime_stats,
            'setup': self.state.setup_stats
        }
    
    def summary(self) -> dict:
        """Get summary statistics"""
        return self.state.to_dict()
    
    def _get_cache_entry(self, agent: str) -> Optional[CacheEntry]:
        """Get cache entry for agent"""
        if agent == 'context':
            return self.state.context_cache
        elif agent == 'regime':
            return self.state.regime_cache
        elif agent == 'setup':
            return self.state.setup_cache
        else:
            raise ValueError(f"Unknown agent: {agent}")
    
    def _set_cache_entry(self, agent: str, entry: CacheEntry):
        """Set cache entry for agent"""
        if agent == 'context':
            self.state.context_cache = entry
        elif agent == 'regime':
            self.state.regime_cache = entry
        elif agent == 'setup':
            self.state.setup_cache = entry
        else:
            raise ValueError(f"Unknown agent: {agent}")
    
    def _record_hit(self, agent: str):
        """Record cache hit"""
        if agent == 'context':
            self.state.context_stats.cache_hit_count += 1
        elif agent == 'regime':
            self.state.regime_stats.cache_hit_count += 1
        elif agent == 'setup':
            self.state.setup_stats.cache_hit_count += 1
    
    def _record_miss(self, agent: str):
        """Record cache miss"""
        if agent == 'context':
            self.state.context_stats.cache_miss_count += 1
        elif agent == 'regime':
            self.state.regime_stats.cache_miss_count += 1
        elif agent == 'setup':
            self.state.setup_stats.cache_miss_count += 1
    
    def record_recompute(self, agent: str):
        """Record a recompute operation"""
        if agent == 'context':
            self.state.context_stats.recompute_count += 1
        elif agent == 'regime':
            self.state.regime_stats.recompute_count += 1
        elif agent == 'setup':
            self.state.setup_stats.recompute_count += 1


def get_closed_bar_timestamp(df: pd.DataFrame) -> datetime:
    """
    Get timestamp of last closed bar from dataframe
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        Timestamp of last closed bar
    """
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    return df.index[-1]
