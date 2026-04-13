"""
Fractal Cached Runner

Orchestrates fractal agents with intelligent state caching.

V1 Safe:
- Reuses higher-timeframe states when closed bar hasn't changed
- Always recomputes lower timeframe (setup on 15m)
- No trading logic changes
- No look-ahead
"""

import logging
from typing import Dict, Optional
import pandas as pd
from datetime import datetime

from src.cache.fractal_state_cache import FractalStateCache, get_closed_bar_timestamp
from src.agents.context_agent import ContextAgent
from src.agents.regime_agent import RegimeAgent
from src.agents.setup_agent import SetupAgent
from src.agents.orchestrator import Orchestrator
from src.agents.contracts import OrchestratorDecision

logger = logging.getLogger(__name__)


class FractalCachedRunner:
    """
    Fractal Cached Runner
    
    Orchestrates fractal agents with state caching to avoid
    unnecessary recomputation of slow higher-timeframe layers.
    
    Recompute rules:
    - Context (1d): recompute only when 1d closed bar changes
    - Regime (1h): recompute only when 1h closed bar changes  
    - Setup (15m): always recompute (decision timeframe)
    
    Final decision is always fresh, composed from latest states.
    """
    
    def __init__(self, config: dict):
        """
        Initialize cached runner
        
        Args:
            config: System configuration
        """
        self.config = config
        self.cache = FractalStateCache()
        
        # Initialize agents
        self.context_agent = ContextAgent(config)
        self.regime_agent = RegimeAgent(config)
        self.setup_agent = SetupAgent(config)
        
        # Initialize orchestrator (for final decision composition)
        self.orchestrator = Orchestrator(config)
        
        # Get timeframes from config
        fractal_config = config.get('fractal', {})
        self.context_tf = fractal_config.get('context_tf', '1d')
        self.regime_tf = fractal_config.get('regime_tf', '1h')
        self.setup_tf = fractal_config.get('setup_tf', '15m')
        
        self.logger = logging.getLogger(__name__)
    
    def decide(
        self, 
        mtf_data: Dict[str, pd.DataFrame], 
        current_price: Optional[float] = None
    ) -> OrchestratorDecision:
        """
        Make trading decision with intelligent caching
        
        Args:
            mtf_data: Multi-timeframe data
            current_price: Current market price
        
        Returns:
            OrchestratorDecision
        """
        
        # Step 1: Get or compute Context state
        context_result = self._get_or_compute_context(mtf_data)
        
        # Step 2: Get or compute Regime state
        regime_result = self._get_or_compute_regime(mtf_data, context_result.state)
        
        # Step 3: Always compute Setup (decision timeframe)
        setup_result = self._compute_setup(mtf_data, context_result.state)
        
        # Step 4: Use orchestrator to compose final decision
        # The orchestrator's decide() method expects mtf_data and will
        # recompute agents, but we can override by directly composing
        # the decision from our cached/computed results
        
        # For V1, we'll use a simplified approach: call orchestrator
        # but since we've already computed the agents, their internal
        # state should be fresh
        
        # Alternative: manually compose decision (cleaner for V1)
        decision = self._compose_decision(
            context_result,
            regime_result,
            setup_result,
            current_price or mtf_data[self.setup_tf].iloc[-1]['close']
        )
        
        return decision
    
    def _get_or_compute_context(self, mtf_data: Dict[str, pd.DataFrame]):
        """Get context state from cache or recompute"""
        
        context_data = mtf_data.get(self.context_tf)
        if context_data is None or len(context_data) == 0:
            # No data - compute without cache
            self.logger.warning("No context data available")
            return self.context_agent.analyze(pd.DataFrame())
        
        # Get current closed bar timestamp
        current_timestamp = get_closed_bar_timestamp(context_data)
        
        # Try to get from cache
        cached_result = self.cache.get_if_valid(
            'context', 
            self.context_tf, 
            current_timestamp
        )
        
        if cached_result is not None:
            # Cache hit - reuse
            return cached_result
        
        # Cache miss - recompute
        self.logger.info(f"context: recompute (new {self.context_tf} close)")
        result = self.context_agent.analyze(context_data)
        
        # Store in cache
        self.cache.set(
            'context',
            self.context_tf,
            current_timestamp,
            result,
            metadata={'bars': len(context_data)}
        )
        self.cache.record_recompute('context')
        
        return result
    
    def _get_or_compute_regime(self, mtf_data: Dict[str, pd.DataFrame], context_state: str):
        """Get regime state from cache or recompute"""
        
        regime_data = mtf_data.get(self.regime_tf)
        if regime_data is None or len(regime_data) == 0:
            # No data - compute without cache
            self.logger.warning("No regime data available")
            return self.regime_agent.analyze(pd.DataFrame(), context_state=context_state)
        
        # Get current closed bar timestamp
        current_timestamp = get_closed_bar_timestamp(regime_data)
        
        # Try to get from cache
        cached_result = self.cache.get_if_valid(
            'regime',
            self.regime_tf,
            current_timestamp
        )
        
        if cached_result is not None:
            # Cache hit - reuse
            return cached_result
        
        # Cache miss - recompute
        self.logger.info(f"regime: recompute (new {self.regime_tf} close)")
        result = self.regime_agent.analyze(regime_data, context_state=context_state)
        
        # Store in cache
        self.cache.set(
            'regime',
            self.regime_tf,
            current_timestamp,
            result,
            metadata={'bars': len(regime_data)}
        )
        self.cache.record_recompute('regime')
        
        return result
    
    def _compute_setup(self, mtf_data: Dict[str, pd.DataFrame], context_state: str):
        """Always recompute setup (decision timeframe)"""
        
        setup_data = mtf_data.get(self.setup_tf)
        if setup_data is None or len(setup_data) == 0:
            # No data
            self.logger.warning("No setup data available")
            return self.setup_agent.analyze(pd.DataFrame(), context_state=context_state)
        
        # Get current closed bar timestamp
        current_timestamp = get_closed_bar_timestamp(setup_data)
        
        # Always recompute setup (decision timeframe)
        self.logger.debug(f"setup: recompute (new {self.setup_tf} close)")
        result = self.setup_agent.analyze(setup_data, context_state=context_state)
        
        # Store in cache for stats tracking
        self.cache.set(
            'setup',
            self.setup_tf,
            current_timestamp,
            result,
            metadata={'bars': len(setup_data)}
        )
        self.cache.record_recompute('setup')
        
        return result
    
    def _compose_decision(
        self,
        context_result,
        regime_result,
        setup_result,
        current_price: float
    ) -> OrchestratorDecision:
        """
        Compose final decision from agent results
        
        Delegates to Orchestrator.compose_from_components to ensure
        identical logic and guarantee parity.
        """
        
        components = {
            'context': context_result,
            'regime': regime_result,
            'setup': setup_result
        }
        
        # Use orchestrator's composition logic for perfect parity
        return self.orchestrator.compose_from_components(components, current_price)
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return self.cache.summary()
    
    def reset_cache(self):
        """Reset cache (useful for testing)"""
        self.cache = FractalStateCache()
