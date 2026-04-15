"""
Agent Contracts — unified cross-layer vocabulary.

Historical `AgentResult` / `OrchestratorDecision` live here (used by
the 5 Jesse agents — see `docs/JESSE_AGENTS_STATUS.md`).

Ticket 03 introduces 4 additional canonical types so every layer
(fractal reporters, Meta-GBM, risk, execution) speaks the same
language :

  FractalReport         — per-timeframe, per-agent report
  MetaDecision          — strategy-brain output for one bar
  TradePlan             — risk-sized + stop/TP resolved
  ExecutionInstruction  — broker-ready order payload

Chain :

  FractalReport(×N)  →  MetaDecision  →  TradePlan  →  ExecutionInstruction

See `docs/ARCHITECTURE_CANONIQUE.md` for the runtime path these
contracts serialise.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime


# Canonical constants — use these instead of free-form strings.
CANONICAL_TIMEFRAMES = ('15m', '1h', '4h', '1d')
CANONICAL_AGENTS = ('context', 'regime', 'setup', 'entry')
CANONICAL_DIRECTIONS = (-1, 0, 1)
CANONICAL_ORDER_TYPES = ('post_only_limit', 'market')
CANONICAL_SIDES = ('buy', 'sell')


if TYPE_CHECKING:
    # Forward references for adapters (avoid circular imports at runtime).
    pass


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

        # Coerce numpy scalar types → Python native (agents use numpy ops internally)
        self.score                = float(self.score)
        self.passed               = bool(self.passed)
        self.ready                = bool(self.ready)
        self.blocked_by_readiness = bool(self.blocked_by_readiness)

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

    def to_fractal_report(
        self,
        asset: str,
        timeframe: str,
        timestamp: Optional[str] = None,
    ) -> 'FractalReport':
        """Adapter — convert legacy `AgentResult` to canonical
        `FractalReport` for cross-layer vocabulary."""
        ts = timestamp
        if ts is None and self.timestamp is not None:
            ts = self.timestamp.isoformat()
        if ts is None:
            ts = datetime.now().isoformat()
        reasons: List[str] = []
        if not self.passed:
            reasons.append(self.reason or 'agent blocked')
        return FractalReport(
            asset=asset,
            agent=self.agent,
            timeframe=timeframe,
            state=self.state,
            score=self.score,
            passed=self.passed,
            block_reasons=reasons,
            timestamp=ts,
            metadata=dict(self.metadata),
        )


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


# ==========================================================================
# TICKET 03 — canonical cross-layer contracts
# ==========================================================================


@dataclass
class FractalReport:
    """Canonical per-timeframe, per-agent report.

    Unified vocabulary replacing `AgentResult` for cross-layer use.
    Every fractal reporter (Context/Regime/Setup/Entry on its TF)
    emits exactly this shape so Meta-GBM and downstream consumers
    speak one language.
    """

    asset: str
    agent: str                       # in CANONICAL_AGENTS
    timeframe: str                   # in CANONICAL_TIMEFRAMES
    state: str                       # agent-specific free-form state
    score: float                     # [0, 1] — confidence / quality
    passed: bool                     # gate approval
    block_reasons: List[str]         # why blocked; empty if passed
    timestamp: str                   # ISO8601
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = float(self.score)
        self.passed = bool(self.passed)
        if self.agent not in CANONICAL_AGENTS:
            raise ValueError(
                f"agent={self.agent!r} must be one of {CANONICAL_AGENTS}"
            )
        if self.timeframe not in CANONICAL_TIMEFRAMES:
            raise ValueError(
                f"timeframe={self.timeframe!r} must be one of "
                f"{CANONICAL_TIMEFRAMES}"
            )
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score={self.score} out of [0,1]")
        if not isinstance(self.state, str) or not self.state:
            raise ValueError("state must be non-empty string")
        if not isinstance(self.block_reasons, list):
            raise ValueError("block_reasons must be list[str]")
        if not self.passed and not self.block_reasons:
            raise ValueError(
                "passed=False requires non-empty block_reasons"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset':         self.asset,
            'agent':         self.agent,
            'timeframe':     self.timeframe,
            'state':         self.state,
            'score':         self.score,
            'passed':        self.passed,
            'block_reasons': list(self.block_reasons),
            'timestamp':     self.timestamp,
            'metadata':      dict(self.metadata),
        }


@dataclass
class MetaDecision:
    """Canonical strategy-brain output for one bar.

    The Meta-GBM (or any future brain) emits exactly this shape per
    bar per asset. `direction != 0` ⇒ actionable. Risk / execution
    layers consume `MetaDecision` — never a free-form candidate dict.
    """

    asset: str
    timestamp: str                   # ISO8601
    timeframe: str                   # in CANONICAL_TIMEFRAMES
    direction: int                   # in CANONICAL_DIRECTIONS
    probability: float               # P(positive class) in [0,1]
    threshold_used: float            # threshold applied in [0,1]
    passed: bool                     # direction != 0 AND all gates OK
    block_reasons: List[str]         # empty iff passed=True
    expected_edge_net: float         # bps proxy, may be negative
    candidate_quality: float         # [0,1]
    fractal_reports: Dict[str, 'FractalReport'] = field(default_factory=dict)
    features_snapshot: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.direction = int(self.direction)
        self.probability = float(self.probability)
        self.threshold_used = float(self.threshold_used)
        self.passed = bool(self.passed)
        self.expected_edge_net = float(self.expected_edge_net)
        self.candidate_quality = float(self.candidate_quality)
        if self.direction not in CANONICAL_DIRECTIONS:
            raise ValueError(
                f"direction={self.direction} must be in "
                f"{CANONICAL_DIRECTIONS}"
            )
        if self.timeframe not in CANONICAL_TIMEFRAMES:
            raise ValueError(
                f"timeframe={self.timeframe!r} must be one of "
                f"{CANONICAL_TIMEFRAMES}"
            )
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability={self.probability} out of [0,1]")
        if not 0.0 <= self.threshold_used <= 1.0:
            raise ValueError(
                f"threshold_used={self.threshold_used} out of [0,1]"
            )
        if not 0.0 <= self.candidate_quality <= 1.0:
            raise ValueError(
                f"candidate_quality={self.candidate_quality} out of [0,1]"
            )
        if self.passed and self.direction == 0:
            raise ValueError(
                "passed=True requires direction != 0"
            )
        if not self.passed and not self.block_reasons:
            raise ValueError(
                "passed=False requires non-empty block_reasons"
            )
        for k, v in self.fractal_reports.items():
            if not isinstance(v, FractalReport):
                raise ValueError(
                    f"fractal_reports[{k!r}] must be FractalReport, "
                    f"got {type(v).__name__}"
                )

    def is_actionable(self) -> bool:
        return self.passed and self.direction != 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset':             self.asset,
            'timestamp':         self.timestamp,
            'timeframe':         self.timeframe,
            'direction':         self.direction,
            'probability':       self.probability,
            'threshold_used':    self.threshold_used,
            'passed':            self.passed,
            'block_reasons':     list(self.block_reasons),
            'expected_edge_net': self.expected_edge_net,
            'candidate_quality': self.candidate_quality,
            'fractal_reports':   {k: r.to_dict()
                                   for k, r in self.fractal_reports.items()},
            'features_snapshot': dict(self.features_snapshot),
        }


@dataclass
class TradePlan:
    """Canonical risk-sized + stop/TP resolved plan.

    Produced by the risk / sizing layer after it accepts a
    `MetaDecision`. Carries everything execution needs (prices,
    size, hold window, traceability to source decision).
    """

    asset: str
    timestamp: str                   # ISO8601
    direction: int                   # -1 or +1 (no FLAT plans)
    size_fraction: float             # fraction of per-asset risk cap in [0,1]
    entry_price: float
    stop_loss: float
    take_profit: float
    expected_hold_bars: int
    source: 'MetaDecision'           # traceability

    def __post_init__(self) -> None:
        self.direction = int(self.direction)
        self.size_fraction = float(self.size_fraction)
        self.entry_price = float(self.entry_price)
        self.stop_loss = float(self.stop_loss)
        self.take_profit = float(self.take_profit)
        self.expected_hold_bars = int(self.expected_hold_bars)
        if self.direction not in (-1, 1):
            raise ValueError(
                f"TradePlan direction must be -1 or +1, got {self.direction} "
                "(FLAT decisions have no plan)"
            )
        if not 0.0 <= self.size_fraction <= 1.0:
            raise ValueError(
                f"size_fraction={self.size_fraction} out of [0,1]"
            )
        if self.entry_price <= 0 or self.stop_loss <= 0 or self.take_profit <= 0:
            raise ValueError("prices must be positive")
        if self.expected_hold_bars < 0:
            raise ValueError(
                f"expected_hold_bars={self.expected_hold_bars} < 0"
            )
        # Geometry checks
        if self.direction == 1:
            if not (self.stop_loss < self.entry_price < self.take_profit):
                raise ValueError(
                    f"long plan requires stop_loss < entry < take_profit, "
                    f"got SL={self.stop_loss} E={self.entry_price} "
                    f"TP={self.take_profit}"
                )
        else:  # short
            if not (self.stop_loss > self.entry_price > self.take_profit):
                raise ValueError(
                    f"short plan requires stop_loss > entry > take_profit, "
                    f"got SL={self.stop_loss} E={self.entry_price} "
                    f"TP={self.take_profit}"
                )
        if int(self.source.direction) != self.direction:
            raise ValueError(
                f"TradePlan.direction ({self.direction}) must match "
                f"source.direction ({self.source.direction})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset':              self.asset,
            'timestamp':          self.timestamp,
            'direction':          self.direction,
            'size_fraction':      self.size_fraction,
            'entry_price':        self.entry_price,
            'stop_loss':          self.stop_loss,
            'take_profit':        self.take_profit,
            'expected_hold_bars': self.expected_hold_bars,
            'source':             self.source.to_dict(),
        }


@dataclass
class ExecutionInstruction:
    """Canonical broker-ready order payload.

    Translated from a `TradePlan` by the execution layer. Consumed by
    `PostOnlyPaperBroker` (or a future live broker adapter). No further
    strategic logic lives here — it is a pure order instruction.
    """

    asset: str
    timestamp: str                   # ISO8601
    side: str                        # 'buy' or 'sell'
    quantity: float                  # absolute quantity > 0
    order_type: str                  # in CANONICAL_ORDER_TYPES
    limit_price: Optional[float]     # None for market orders
    max_wait_bars: int               # post-only timeout
    source: 'TradePlan'              # traceability

    def __post_init__(self) -> None:
        self.quantity = float(self.quantity)
        self.max_wait_bars = int(self.max_wait_bars)
        if self.side not in CANONICAL_SIDES:
            raise ValueError(
                f"side={self.side!r} must be one of {CANONICAL_SIDES}"
            )
        if self.order_type not in CANONICAL_ORDER_TYPES:
            raise ValueError(
                f"order_type={self.order_type!r} must be one of "
                f"{CANONICAL_ORDER_TYPES}"
            )
        if self.quantity <= 0:
            raise ValueError(f"quantity={self.quantity} must be > 0")
        if self.max_wait_bars < 0:
            raise ValueError(
                f"max_wait_bars={self.max_wait_bars} < 0"
            )
        if self.order_type == 'post_only_limit' and self.limit_price is None:
            raise ValueError("post_only_limit order requires limit_price")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError(f"limit_price={self.limit_price} must be > 0")
        # Side ↔ source.direction consistency
        expected_side = 'buy' if int(self.source.direction) == 1 else 'sell'
        if self.side != expected_side:
            raise ValueError(
                f"side={self.side!r} inconsistent with source.direction="
                f"{self.source.direction} (expected {expected_side!r})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset':         self.asset,
            'timestamp':     self.timestamp,
            'side':          self.side,
            'quantity':      self.quantity,
            'order_type':    self.order_type,
            'limit_price':   self.limit_price,
            'max_wait_bars': self.max_wait_bars,
            'source':        self.source.to_dict(),
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
