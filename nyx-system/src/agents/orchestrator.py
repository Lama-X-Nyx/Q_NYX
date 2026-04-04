"""
Orchestrator

Coordinates the 4 fractal agents and makes final decision.
"""

from typing import Dict, List, Any
import pandas as pd
from src.agents.contracts import AgentResult, OrchestratorDecision
from src.agents.context_agent import ContextAgent
from src.agents.regime_agent import RegimeAgent
from src.agents.setup_agent import SetupAgent
from src.agents.entry_agent import EntryAgent
from src.core.risk_manager_mtf import RiskManagerMTF


class Orchestrator:
    """
    Orchestrator - Central Decision Maker
    
    Responsibilities:
    1. Call 4 agents in sequence
    2. Aggregate results
    3. Apply AND logic
    4. Check centralized risk
    5. Produce final decision
    6. Log which agent blocked/passed
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Orchestrator
        
        Args:
            config: Configuration dict
        """
        self.config = config
        
        # Initialize agents
        self.context_agent = ContextAgent(config)
        self.regime_agent = RegimeAgent(config)
        self.setup_agent = SetupAgent(config)
        self.entry_agent = EntryAgent(config)
        
        # Initialize risk manager
        self.risk_manager = RiskManagerMTF(config)
    
    def decide(self, mtf_data: Dict[str, pd.DataFrame], current_price: float = None) -> OrchestratorDecision:
        """
        Make final trading decision using active agents
        
        CURRENT: 3-agent baseline (Context, Regime, Setup)
        Entry agent deferred until 5M data available
        
        Args:
            mtf_data: Dict mapping timeframe to DataFrame
            current_price: Current market price (optional, defaults to last close of lowest TF)
        
        Returns:
            OrchestratorDecision with final action
        """
        
        # Get current price if not provided
        if current_price is None:
            lowest_tf = min(mtf_data.keys(), key=lambda x: self._tf_to_minutes(x))
            current_price = mtf_data[lowest_tf].iloc[-1]['close']
        
        # Get fractal config
        fractal_config = self.config.get('fractal', {})
        use_entry_agent = fractal_config.get('use_entry_agent', False)
        
        context_tf = fractal_config.get('context_tf', '1d')
        regime_tf = fractal_config.get('regime_tf', '1h')
        setup_tf = fractal_config.get('setup_tf', '15m')
        
        # Step 1a: Run HSMM on 4H data to compute Intent_1D via π_4H · A^k (P1)
        # This is a projection-only pass — result used solely by ContextAgent.
        # It does NOT participate in the AND-gate logic.
        regime_4h_for_context = None
        structure_tf = fractal_config.get('structure_tf', '4h')
        if structure_tf in mtf_data and not mtf_data[structure_tf].empty:
            regime_4h_for_context = self.regime_agent.analyze(
                mtf_data[structure_tf],
                context_state=None
            )

        # Step 1b: Call Context Agent with 4H HSMM posterior (P1)
        context_result = self.context_agent.analyze(
            mtf_data.get(context_tf, pd.DataFrame()),
            regime_4h_result=regime_4h_for_context
        )

        # Step 2: Call Regime Agent on its operational TF (1H) with context
        regime_result = self.regime_agent.analyze(
            mtf_data.get(regime_tf, pd.DataFrame()),
            context_state=context_result.state
        )
        
        # Step 3: Call Setup Agent (with context)
        setup_result = self.setup_agent.analyze(
            mtf_data.get(setup_tf, pd.DataFrame()),
            context_state=context_result.state
        )
        
        # Aggregate results (3 agents only)
        components = {
            'context': context_result,
            'regime': regime_result,
            'setup': setup_result
        }
        
        # Step 4: Entry Agent (only if enabled)
        if use_entry_agent:
            # Future: when 5M data ready
            entry_tf = fractal_config.get('entry_tf', '5m')
            if entry_tf in mtf_data:
                entry_result = self.entry_agent.analyze(
                    mtf_data[entry_tf],
                    setup_state=setup_result.state
                )
                components['entry'] = entry_result
        
        # Step 5: Check readiness FIRST (before logic)
        all_ready = all([result.ready for result in components.values()])
        
        if not all_ready:
            # Agents not ready - track which ones
            not_ready_agents = [name for name, result in components.items() if not result.ready]
            
            return OrchestratorDecision(
                action='WAIT',
                score=0.0,
                reason=f'Pipeline not ready: {", ".join(not_ready_agents)} need more data',
                blocked_by=['readiness'],  # Special marker for readiness blocks
                components=components
            )
        
        # Step 6: All agents READY - now check logic (all must pass)
        all_passed = all([result.passed for result in components.values()])
        
        blocked_by = [name for name, result in components.items() if not result.passed]
        
        # If any agent blocked by logic
        if not all_passed:
            return OrchestratorDecision(
                action='WAIT',
                score=self._calculate_aggregate_score(components),
                reason=f'Blocked by: {", ".join(blocked_by)}',
                blocked_by=blocked_by,  # Logic blocks (agents were ready but said no)
                components=components
            )
        
        # Step 7: All agents passed - check risk
        risk_conditions = self.risk_manager.check_risk_conditions(
            entry_price=current_price,
            fractal_states=self._extract_fractal_states(regime_result),
            smc_patterns=setup_result.metadata.get('patterns', {}),
            intent_daily=context_result.state,
            transition_matrix=self.regime_agent.hsmm.transition_matrix
        )
        
        risk_passed = risk_conditions['rr_ratio'] and risk_conditions['risk_hit']
        
        if not risk_passed:
            blocked_by.append('risk')
            return OrchestratorDecision(
                action='WAIT',
                score=self._calculate_aggregate_score(components),
                reason='Blocked by risk conditions',
                blocked_by=blocked_by,
                components=components,
                risk_analysis=risk_conditions
            )
        
        # Step 7: All passed - determine action from Context
        if context_result.state == 'bullish':
            action = 'BUY'
        elif context_result.state == 'bearish':
            action = 'SELL'
        else:
            action = 'WAIT'
        
        # Calculate aggregate score
        aggregate_score = self._calculate_aggregate_score(components)
        
        return OrchestratorDecision(
            action=action,
            score=aggregate_score,
            reason=f'All agents approved: Context={context_result.state}, Regime={regime_result.state}, Setup={setup_result.state}',
            blocked_by=[],
            components=components,
            risk_analysis=risk_conditions
        )
    
    def compose_from_components(
        self,
        components: Dict[str, Any],
        current_price: float,
        regime_4h_for_context: Any = None  # P1: unused here, kept for API compat
    ) -> OrchestratorDecision:
        """
        Compose final decision from provided agent components
        
        This method can be used by cached runners to compose decisions
        from pre-computed agent results, ensuring identical logic.
        
        Args:
            components: Dict of agent results (context, regime, setup, entry)
            current_price: Current market price
        
        Returns:
            OrchestratorDecision
        """
        
        # Extract results
        context_result = components.get('context')
        regime_result = components.get('regime')
        setup_result = components.get('setup')
        
        # Step 1: Check readiness FIRST
        all_ready = all([result.ready for result in components.values()])
        
        if not all_ready:
            not_ready_agents = [name for name, result in components.items() if not result.ready]
            
            return OrchestratorDecision(
                action='WAIT',
                score=0.0,
                reason=f'Pipeline not ready: {", ".join(not_ready_agents)} need more data',
                blocked_by=['readiness'],
                components=components
            )
        
        # Step 2: All agents READY - check logic (all must pass)
        all_passed = all([result.passed for result in components.values()])
        
        blocked_by = [name for name, result in components.items() if not result.passed]
        
        if not all_passed:
            return OrchestratorDecision(
                action='WAIT',
                score=self._calculate_aggregate_score(components),
                reason=f'Blocked by: {", ".join(blocked_by)}',
                blocked_by=blocked_by,
                components=components
            )
        
        # Step 3: Check risk
        risk_conditions = self.risk_manager.check_risk_conditions(
            entry_price=current_price,
            fractal_states=self._extract_fractal_states(regime_result),
            smc_patterns=setup_result.metadata.get('patterns', {}),
            intent_daily=context_result.state,
            transition_matrix=self.regime_agent.hsmm.transition_matrix
        )
        
        risk_passed = risk_conditions['rr_ratio'] and risk_conditions['risk_hit']
        
        if not risk_passed:
            blocked_by.append('risk')
            return OrchestratorDecision(
                action='WAIT',
                score=self._calculate_aggregate_score(components),
                reason='Blocked by risk conditions',
                blocked_by=blocked_by,
                components=components,
                risk_analysis=risk_conditions
            )
        
        # Step 4: All passed - determine action
        if context_result.state == 'bullish':
            action = 'BUY'
        elif context_result.state == 'bearish':
            action = 'SELL'
        else:
            action = 'WAIT'
        
        aggregate_score = self._calculate_aggregate_score(components)
        
        return OrchestratorDecision(
            action=action,
            score=aggregate_score,
            reason=f'All agents approved: Context={context_result.state}, Regime={regime_result.state}, Setup={setup_result.state}',
            blocked_by=[],
            components=components,
            risk_analysis=risk_conditions
        )
    
    def _calculate_aggregate_score(self, components: Dict[str, AgentResult]) -> float:
        """
        Calculate weighted aggregate score from components
        
        Weights adapt to number of active agents:
        - 3 agents: Context 35%, Regime 35%, Setup 30%
        - 4 agents: Context 30%, Regime 30%, Setup 25%, Entry 15%
        """
        
        if len(components) == 3:
            # 3-agent baseline
            weights = {
                'context': 0.35,
                'regime': 0.35,
                'setup': 0.30
            }
        else:
            # 4-agent (when entry active)
            weights = {
                'context': 0.30,
                'regime': 0.30,
                'setup': 0.25,
                'entry': 0.15
            }
        
        total_score = sum(
            components[name].score * weights.get(name, 0)
            for name in components.keys()
        )
        
        return total_score
    
    def _extract_fractal_states(self, regime_result: AgentResult) -> Dict:
        """Extract fractal states from regime result for risk calculation"""
        hsmm_states = regime_result.metadata.get('hsmm_states', {})
        
        # Return in format expected by risk manager
        return {
            '4h': hsmm_states
        }
    
    def _tf_to_minutes(self, tf: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return mapping.get(tf, 1440)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    import numpy as np
    
    print("Orchestrator Test")
    
    # Create sample MTF data
    dates_1d = pd.date_range('2023-01-01', periods=100, freq='1D')
    dates_4h = pd.date_range('2023-01-01', periods=400, freq='4h')
    dates_15m = pd.date_range('2023-01-01', periods=1000, freq='15min')
    
    prices_1d = np.linspace(40000, 50000, 100)  # Uptrend
    prices_4h = np.linspace(40000, 50000, 400)
    prices_15m = np.linspace(49000, 50000, 1000)
    
    mtf_data = {
        '1d': pd.DataFrame({
            'open': prices_1d,
            'high': prices_1d * 1.01,
            'low': prices_1d * 0.99,
            'close': prices_1d,
            'volume': np.random.rand(100) * 1000
        }, index=dates_1d),
        
        '4h': pd.DataFrame({
            'open': prices_4h,
            'high': prices_4h * 1.01,
            'low': prices_4h * 0.99,
            'close': prices_4h,
            'volume': np.random.rand(400) * 1000
        }, index=dates_4h),
        
        '15m': pd.DataFrame({
            'open': prices_15m,
            'high': prices_15m * 1.005,
            'low': prices_15m * 0.995,
            'close': prices_15m,
            'volume': np.random.rand(1000) * 1000
        }, index=dates_15m)
    }
    
    # Config
    config = {
        'mtf': {
            'timeframes': {
                'context': '1d',
                'regime': '4h',
                'setup': '15m',
                'entry': '15m'  # Use 15m as entry (no 5m data)
            }
        },
        'strategy': {
            'mtf_conditions': {
                'sdc_min': 5.0,
                'stability_4h_min': 0.60,
                'alignment_15m_min': 0.60
            }
        }
    }
    
    # Test
    orchestrator = Orchestrator(config)
    decision = orchestrator.decide(mtf_data)
    
    print(f"\n✅ Decision:")
    print(f"  Action: {decision.action}")
    print(f"  Score: {decision.score:.2f}")
    print(f"  Reason: {decision.reason}")
    print(f"  Blocked by: {decision.blocked_by}")
    print(f"\n  Components:")
    for name, result in decision.components.items():
        print(f"    {name}: {result.state} (passed={result.passed}, score={result.score:.2f})")
