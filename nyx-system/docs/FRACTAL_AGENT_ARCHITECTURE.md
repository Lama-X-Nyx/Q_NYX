# NYX Fractal Agent Architecture

**Version:** 1.0  
**Date:** 2025-03-29  
**Status:** Implementation

---

## 📐 **ARCHITECTURE OVERVIEW**

NYX is a **fractal multi-timeframe trading system** where each timeframe serves a distinct role in the decision process.

Instead of a monolithic engine, NYX uses **3 specialized agents** coordinated by **1 orchestrator**.

**CURRENT BASELINE: 3-LAYER FRACTAL (Active)**
- Context Agent (1D)
- Regime Agent (1H) 
- Setup Agent (15M)
- Orchestrator

**DEFERRED: Entry Agent (5M)**
- Status: Not yet implemented
- Reason: No 5M data currently available
- No fallback to 15M - cleanly disabled
- Will be added when 5M data pipeline is ready

---

## 🎯 **DESIGN PRINCIPLES**

1. **One Agent = One Role = One Timeframe**
2. **No Duplicate Logic** - Reuse existing components (HSMM, SMC, Risk)
3. **Single Responsibility** - Each agent answers ONE question
4. **Centralized Risk** - Only orchestrator accesses risk/execution
5. **Clear Diagnostics** - Know exactly which agent blocked a decision

---

## 🏗️ **ARCHITECTURE DIAGRAM**

**CURRENT ACTIVE BASELINE (3-LAYER FRACTAL):**

```
┌─────────────────────────────────────────────────────────────┐
│                      MTF Data Loader                        │
│         (Loads & Aligns: 1D, 1H, 15M - Closed Candles)     │
└────────┬────────────────────────────────────────────────────┘
         │
         ├─────► Context Agent (1D)  → "Bullish bias?"
         │
         ├─────► Regime Agent (1H)   → "Trend+ regime?"
         │
         ├─────► Setup Agent (15M)   → "Valid SMC pattern?"
         │
         └─────► Orchestrator
                    │
                    ├─→ Aggregate Results (3 agents)
                    ├─→ Apply AND Logic
                    ├─→ Check Risk (centralized)
                    │
                    └─→ Decision: BUY / SELL / WAIT
                           │
                           └─→ Execution (if approved)
```

**DEFERRED (Not Yet Active):**
- Entry Agent (5M) - awaiting 5M data pipeline
- No fallback to 15M - cleanly disabled

---

## 🤖 **AGENT SPECIFICATIONS**

**ACTIVE AGENTS (Current Baseline):**

### **1. Context Agent** ✅ ACTIVE

**Timeframe:** 1D  
**Role:** Determine macro bias and structural direction

**Inputs:**
- DataFrame (1D or 4H OHLCV)
- Historical bars (≥ 100)

**Responsibilities:**
- Identify higher timeframe trend
- Determine favored/unfavored direction
- Provide strategic context score

**Output:**
```python
{
    "agent": "context",
    "state": "bullish" | "bearish" | "neutral",
    "score": 0.72,  # [0, 1]
    "passed": True,
    "reason": "Higher timeframe structure intact",
    "metadata": {
        "timeframe": "1d",
        "trend_strength": 0.72,
        "structure": "uptrend"
    }
}
```

**Decision Gate:**
- `passed = True` if state aligns with intended direction
- For long: requires state = "bullish"
- For short: requires state = "bearish"

**Implementation:**
- Uses simple SMA/EMA analysis (no HSMM needed here)
- Looks for higher timeframe structure
- Returns bias score

---

### **2. Regime Agent** ✅ ACTIVE

**Timeframe:** 1H  
**Role:** Detect current market regime with confidence

**Inputs:**
- DataFrame (4H or 1H OHLCV)
- Historical bars (≥ 200 for HSMM)

**Responsibilities:**
- Run HSMM to detect regime
- Calculate Stability (persistence probability)
- Provide regime confidence (SdC)

**Output:**
```python
{
    "agent": "regime",
    "state": "trend_plus" | "range" | "trend_minus",
    "score": 0.81,  # SdC / 10
    "passed": True,
    "reason": "HSMM Trend+ with 81% confidence, Stability 0.75",
    "metadata": {
        "timeframe": "4h",
        "sdc": 8.1,
        "stability": 0.75,
        "hsmm_states": {
            "Trend+": 0.81,
            "Range": 0.12,
            "Trend-": 0.07
        }
    }
}
```

**Decision Gate:**
- `passed = True` if:
  - SdC > 5.0
  - Stability ≥ 0.60
  - State matches Context direction

**Implementation:**
- Wraps existing `SemiMarkovHMM`
- Computes stability from transition matrix
- Returns Intent_Daily equivalent

---

### **3. Setup Agent** ✅ ACTIVE

**Timeframe:** 15M  
**Role:** Detect and validate SMC patterns

**Inputs:**
- DataFrame (15M OHLCV)
- Context bias (from Context Agent)
- Historical bars (≥ 100)

**Responsibilities:**
- Run SMC detector (OB, FVG, BoS/CHoCH)
- Check pattern alignment with Context
- Validate pattern quality

**Output:**
```python
{
    "agent": "setup",
    "state": "valid_setup" | "no_pattern" | "misaligned",
    "score": 0.66,  # alignment score
    "passed": True,
    "reason": "Bullish FVG aligned with Context",
    "metadata": {
        "timeframe": "15m",
        "patterns": {
            "bullish_ob": False,
            "bearish_ob": False,
            "bullish_fvg": True,
            "bearish_fvg": False
        },
        "alignment": 0.66,
        "pattern_quality": "good"
    }
}
```

**Decision Gate:**
- `passed = True` if:
  - Pattern exists (OB or FVG)
  - Pattern aligns with Context direction
  - Alignment score ≥ 0.60

**Implementation:**
- Wraps existing `SMCDetector`
- Compares patterns with Context state
- Returns alignment score

---

### **4. Entry Agent** ⏸️ DEFERRED

**Timeframe:** 5M  
**Status:** **NOT YET ACTIVE**  
**Role:** Validate entry timing (when implemented)

**Why Deferred:**
- 5M data pipeline not yet available
- Priority: Validate 3-layer baseline first
- No fake 15M fallback - cleanly disabled

**When Ready:**
- Add 5M data to MTF loader
- Enable `use_entry_agent: true` in config
- Integrate into Orchestrator decision logic

**Current Behavior:**
- Orchestrator operates with 3 agents only
- No entry timing validation
- Direct decision from Context + Regime + Setup

---

**Inputs:**
- DataFrame (5M OHLCV)
- Setup confirmation (from Setup Agent)
- Historical bars (≥ 50)

**Responsibilities:**
- Check entry timing conditions
- Validate execution readiness
- Confirm micro-structure alignment

**Output:**
```python
{
    "agent": "entry",
    "state": "ready" | "early" | "late",
    "score": 0.58,
    "passed": True,
    "reason": "Timing conditions met on entry TF",
    "metadata": {
        "timeframe": "5m",
        "micro_structure": "aligned",
        "timing_quality": "good"
    }
}
```

**Decision Gate:**
- `passed = True` if:
  - Timing is "ready"
  - No conflicting signals on 5M
  - Micro-structure supports entry

**Implementation:**
- Simple momentum/structure check
- Can be minimal initially
- Confirms setup hasn't invalidated

---

## 🎭 **ORCHESTRATOR**

**Role:** Central decision maker - aggregates agents and applies final logic

**Inputs:**
- Results from 4 agents (AgentResult objects)
- Current price
- MTF data (for risk calculation)

**Responsibilities:**
1. Collect agent results
2. Apply decision logic (AND gate)
3. Check centralized risk
4. Produce final decision
5. Log which agent blocked/passed

**Decision Logic (AND Gate):**
```python
def decide(agent_results):
    context = agent_results['context']
    regime = agent_results['regime']
    setup = agent_results['setup']
    entry = agent_results['entry']
    
    # All gates must pass
    all_passed = all([
        context.passed,
        regime.passed,
        setup.passed,
        entry.passed
    ])
    
    if not all_passed:
        blocked_by = [name for name, result in agent_results.items() 
                      if not result.passed]
        return {
            'action': 'WAIT',
            'blocked_by': blocked_by,
            'reason': f'Blocked by: {", ".join(blocked_by)}'
        }
    
    # All passed - check risk
    risk_ok = check_risk(...)
    if not risk_ok:
        return {
            'action': 'WAIT',
            'blocked_by': ['risk'],
            'reason': 'Risk conditions not met'
        }
    
    # Determine direction from Context
    if context.state == 'bullish':
        action = 'BUY'
    elif context.state == 'bearish':
        action = 'SELL'
    else:
        action = 'WAIT'
    
    return {
        'action': action,
        'score': calculate_aggregate_score(agent_results),
        'reason': 'All agents approved',
        'components': agent_results
    }
```

**Output:**
```python
{
    "action": "BUY" | "SELL" | "WAIT",
    "score": 0.71,  # Aggregate score
    "reason": "Context bullish + regime trend_plus + valid setup + entry ready",
    "blocked_by": [],  # Empty if all passed
    "components": {
        "context": {...},
        "regime": {...},
        "setup": {...},
        "entry": {...}
    },
    "risk_analysis": {...},  # From centralized risk manager
    "timestamp": "2024-01-01T00:00:00"
}
```

---

## 🔒 **CENTRALIZED RISK & EXECUTION**

### **Why Centralized?**

1. **Single Source of Truth** - No conflicting risk calculations
2. **Simpler Testing** - Test risk logic once, not 4 times
3. **Compliance** - Easier to audit/control
4. **Performance** - No duplicate computation

### **Risk Manager Responsibilities:**

- Compute P(hit -0.05), P(hit -0.10)
- Calculate RR ratio from SMC levels
- Position sizing
- Risk gates (final check before execution)

**Called by:** Orchestrator only (after agents approve)

### **Execution Layer:**

- Receives final decision from Orchestrator
- Places orders
- Manages stops/targets
- Logs trades

**Called by:** Orchestrator only (if risk approved)

---

## 📊 **DATA FLOW**

```
1. MTF Loader
   ↓
   Provides aligned data per timeframe
   
2. Agent Execution (parallel possible)
   Context Agent ← df_1d
   Regime Agent  ← df_4h
   Setup Agent   ← df_15m (+ Context state)
   Entry Agent   ← df_5m (+ Setup state)
   
3. Orchestrator
   ↓
   Aggregates results → Apply AND logic
   ↓
   Check Risk (centralized)
   ↓
   Final Decision: BUY/SELL/WAIT
   
4. Execution (if approved)
   ↓
   Order placement
```

---

## 🔍 **DIAGNOSTICS**

### **Question: "Why did NYX not trade?"**

**Before (Monolithic):**
```
Signal: HOLD
Reasons: ['Some condition failed']
```
→ Unclear which component blocked

**After (Fractal):**
```json
{
  "action": "WAIT",
  "blocked_by": ["setup", "entry"],
  "components": {
    "context": {"passed": true, "state": "bullish"},
    "regime": {"passed": true, "state": "trend_plus"},
    "setup": {"passed": false, "state": "no_pattern"},
    "entry": {"passed": false, "state": "early"}
  }
}
```
→ Clear: Setup had no pattern, Entry was early

### **Diagnostic Metrics:**

```python
# Over N signals
blocked_by_context: 12
blocked_by_regime: 45
blocked_by_setup: 187  # ← Main bottleneck
blocked_by_entry: 34
blocked_by_risk: 8
approved_buy: 23
approved_sell: 5
wait_neutral: 686
```

→ Immediately see: **Setup is the bottleneck** (187 blocks)

---

## 🧪 **TESTING STRATEGY**

### **Unit Tests (Per Agent):**

```python
def test_context_agent():
    agent = ContextAgent(config)
    result = agent.analyze(df_1d)
    
    assert isinstance(result, AgentResult)
    assert result.agent == 'context'
    assert result.state in ['bullish', 'bearish', 'neutral']
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)

def test_regime_agent():
    agent = RegimeAgent(config)
    result = agent.analyze(df_4h)
    
    assert result.agent == 'regime'
    assert 'sdc' in result.metadata
    assert 'stability' in result.metadata
```

### **Integration Test (Orchestrator):**

```python
def test_orchestrator_all_passed():
    orch = Orchestrator(config, agents)
    
    # Mock all agents passing
    agent_results = {
        'context': AgentResult(agent='context', state='bullish', score=0.7, passed=True),
        'regime': AgentResult(agent='regime', state='trend_plus', score=0.8, passed=True),
        'setup': AgentResult(agent='setup', state='valid_setup', score=0.6, passed=True),
        'entry': AgentResult(agent='entry', state='ready', score=0.5, passed=True)
    }
    
    decision = orch.decide(agent_results, mtf_data, current_price)
    
    assert decision['action'] == 'BUY'
    assert decision['blocked_by'] == []

def test_orchestrator_setup_blocked():
    # Mock setup blocking
    agent_results = {
        'context': AgentResult(..., passed=True),
        'regime': AgentResult(..., passed=True),
        'setup': AgentResult(..., passed=False),  # ← Blocks
        'entry': AgentResult(..., passed=True)
    }
    
    decision = orch.decide(agent_results, mtf_data, current_price)
    
    assert decision['action'] == 'WAIT'
    assert 'setup' in decision['blocked_by']
```

### **End-to-End Test:**

```python
def test_fractal_pipeline():
    from src.agents.fractal_pipeline import FractalPipeline
    
    pipeline = FractalPipeline(config)
    mtf_data = load_mtf_sample('BTCUSDT', 1000)
    
    signal = pipeline.generate_signal(mtf_data, current_date)
    
    assert 'action' in signal
    assert 'components' in signal
    assert all(agent in signal['components'] for agent in 
               ['context', 'regime', 'setup', 'entry'])
```

---

## 🔄 **MIGRATION STRATEGY**

### **Phase 1: Parallel Implementation**

Keep existing `nyx_engine_mtf.py` working while building fractal architecture.

```python
# Old (still works)
signal = engine.generate_signal_mtf(pair, mtf_data, date)

# New (fractal)
signal = engine.generate_fractal_signal(pair, mtf_data, date)
```

### **Phase 2: Validation**

Run both in parallel on same data, compare results.

### **Phase 3: Deprecation**

Once validated, deprecate old method with warning.

### **Phase 4: Removal**

Remove old MTF engine after full transition.

---

## 📋 **IMPLEMENTATION CHECKLIST**

- [ ] Contracts defined (`contracts.py`)
- [ ] Context Agent implemented
- [ ] Regime Agent implemented (wraps HSMM)
- [ ] Setup Agent implemented (wraps SMC)
- [ ] Entry Agent implemented
- [ ] Orchestrator implemented
- [ ] Risk remains centralized
- [ ] Execution remains centralized
- [ ] CLI mode `fractal_check` added
- [ ] Tests for each agent
- [ ] Test for orchestrator
- [ ] Test for pipeline
- [ ] Documentation complete
- [ ] Migration plan documented

---

## 🎯 **SUCCESS CRITERIA**

After implementation, we should be able to:

1. ✅ **Clearly see which agent blocked a decision**
2. ✅ **Test each agent in isolation**
3. ✅ **Reuse existing components (HSMM, SMC, Risk)**
4. ✅ **Run diagnostics per agent** (blocked_by_setup count, etc.)
5. ✅ **Understand data flow at a glance**
6. ✅ **No duplicate risk/execution logic**

---

## 📝 **APPENDIX: Standard Contract**

```python
from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime

@dataclass
class AgentResult:
    """Standard output contract for all agents"""
    
    agent: str              # 'context', 'regime', 'setup', 'entry'
    state: str              # Agent-specific state
    score: float            # Confidence/quality score [0, 1]
    passed: bool            # Decision gate (True = approved)
    reason: str             # Human-readable explanation
    metadata: Dict[str, Any]  # Agent-specific extra data
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Validation
        assert 0 <= self.score <= 1, "Score must be in [0, 1]"
        assert isinstance(self.passed, bool), "Passed must be boolean"
```

---

**This architecture transforms NYX from a monolithic MTF engine into a clear, diagnosable fractal system.**
