"""
Agent Contracts

Standard output contract for all NYX fractal agents.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class AgentResult:
    """
    Standard output contract for all agents
    
    All agents (Context, Regime, Setup, Entry) must return this structure.
    """
    
    agent: str
    """Agent name: 'context', 'regime', 'setup', 'entry'"""
    
    state: str
    """Agent-specific state (e.g., 'bullish', 'trend_plus', 'valid_setup', 'ready', 'not_ready')"""
    
    score: float
    """Confidence/quality score in [0, 1]"""
    
    passed: bool
    """Decision gate - True if agent approves, False if agent blocks"""
    
    ready: bool = True
    """Readiness flag - True if agent has sufficient data, False if warming up"""
    
    blocked_by_readiness: bool = False
    """True if agent blocked due to insufficient data (not ready), False if logic block"""
    
    reason: str = ""
    """Human-readable explanation of the decision"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Agent-specific extra data"""
    
    timestamp: Optional[datetime] = None
    """When this result was generated"""
    
    def __post_init__(self):
        """Validate contract"""
        
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Validation
        assert self.agent in ['context', 'regime', 'setup', 'entry'], \
            f"Agent must be one of: context, regime, setup, entry. Got: {self.agent}"
        
        assert isinstance(self.state, str) and len(self.state) > 0, \
            "State must be non-empty string"
        
        assert 0 <= self.score <= 1, \
            f"Score must be in [0, 1]. Got: {self.score}"
        
        assert isinstance(self.passed, bool), \
            f"Passed must be boolean. Got: {type(self.passed)}"
        
        assert isinstance(self.ready, bool), \
            f"Ready must be boolean. Got: {type(self.ready)}"
        
        assert isinstance(self.blocked_by_readiness, bool), \
            f"Blocked_by_readiness must be boolean. Got: {type(self.blocked_by_readiness)}"
        
        # Logic check: if not ready, must be blocked by readiness
        if not self.ready:
            assert self.blocked_by_readiness, \
                "If agent is not ready, blocked_by_readiness must be True"
            assert not self.passed, \
                "If agent is not ready, passed must be False"
            assert self.state == "not_ready", \
                f"If agent is not ready, state must be 'not_ready'. Got: {self.state}"
        
        assert isinstance(self.reason, str), \
            "Reason must be string"
        
        assert isinstance(self.metadata, dict), \
            "Metadata must be dict"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'agent': self.agent,
            'state': self.state,
            'score': self.score,
            'passed': self.passed,
            'ready': self.ready,
            'blocked_by_readiness': self.blocked_by_readiness,
            'reason': self.reason,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class OrchestratorDecision:
    """
    Orchestrator final decision
    
    Output of the orchestrator after aggregating all agents.
    """
    
    action: str
    """Final action: 'BUY', 'SELL', 'WAIT'"""
    
    score: float
    """Aggregate confidence score [0, 1]"""
    
    reason: str
    """Human-readable explanation"""
    
    blocked_by: list = field(default_factory=list)
    """List of agent names that blocked (empty if all passed)"""
    
    components: Dict[str, AgentResult] = field(default_factory=dict)
    """Individual agent results"""
    
    risk_analysis: Optional[Dict] = None
    """Risk manager output (if computed)"""
    
    timestamp: Optional[datetime] = None
    """When decision was made"""
    
    def __post_init__(self):
        """Validate decision"""
        
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        assert self.action in ['BUY', 'SELL', 'WAIT'], \
            f"Action must be BUY, SELL, or WAIT. Got: {self.action}"
        
        assert 0 <= self.score <= 1, \
            f"Score must be in [0, 1]. Got: {self.score}"
        
        assert isinstance(self.blocked_by, list), \
            "Blocked_by must be list"
        
        # If action is WAIT and no agent blocked, should have explanation
        if self.action == 'WAIT' and not self.blocked_by:
            # Could be risk blocking, macro, or other reason
            pass  # Allow for now
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'action': self.action,
            'score': self.score,
            'reason': self.reason,
            'blocked_by': self.blocked_by,
            'components': {
                name: result.to_dict() 
                for name, result in self.components.items()
            },
            'risk_analysis': self.risk_analysis,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


if __name__ == "__main__":
    # Test contracts
    
    # Valid AgentResult
    result = AgentResult(
        agent='context',
        state='bullish',
        score=0.72,
        passed=True,
        reason='Higher timeframe structure intact',
        metadata={'timeframe': '1d', 'trend_strength': 0.72}
    )
    
    print("✅ AgentResult created:")
    print(f"  Agent: {result.agent}")
    print(f"  State: {result.state}")
    print(f"  Passed: {result.passed}")
    print(f"  Score: {result.score}")
    
    # Test validation
    try:
        bad_result = AgentResult(
            agent='invalid',  # Should fail
            state='test',
            score=0.5,
            passed=True,
            reason='test'
        )
    except AssertionError as e:
        print(f"\n✅ Validation works: {e}")
    
    # Valid OrchestratorDecision
    decision = OrchestratorDecision(
        action='BUY',
        score=0.71,
        reason='All agents approved',
        blocked_by=[],
        components={'context': result}
    )
    
    print(f"\n✅ OrchestratorDecision created:")
    print(f"  Action: {decision.action}")
    print(f"  Blocked by: {decision.blocked_by}")
    
    # Test serialization
    decision_dict = decision.to_dict()
    print(f"\n✅ Serialization works:")
    print(f"  Keys: {list(decision_dict.keys())}")
