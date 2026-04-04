# Fractal Readiness Rules

**Version:** 1.0  
**Date:** 2025-03-29  
**Purpose:** Define when fractal agents are "ready" vs "blocked by logic"

---

## 🎯 **CORE PRINCIPLE**

**Two types of blocks:**

1. **NOT READY** - Agent lacks sufficient data to make any decision
   - Example: Context has only 10 bars (needs 50)
   - State: `not_ready`
   - Reason: `Insufficient data`

2. **LOGIC BLOCKED** - Agent has sufficient data but rejects the trade
   - Example: Context has 100 bars, detects neutral (no bias)
   - State: `neutral`, `range`, `no_pattern`, etc.
   - Reason: `Neutral bias blocks directional trades`

**These must NEVER be mixed.**

---

## 📊 **MINIMUM BARS PER AGENT**

### **Context Agent (1D)**

**Timeframe:** 1D  
**Minimum Required:** 50 bars  
**Why:** Need sufficient history for SMA 20/50 comparison

**Readiness Check:**
```python
if len(df_1d) < 50:
    ready = False
    state = "not_ready"
    reason = f"Insufficient data ({len(df_1d)} bars, need 50)"
```

**Logic Check (when ready):**
```python
if len(df_1d) >= 50:
    ready = True
    if sma_diff < -2%:
        state = "bearish"
        passed = True
    elif sma_diff > 2%:
        state = "bullish"
        passed = True
    else:
        state = "neutral"
        passed = False  # Logic block
```

---

### **Regime Agent (1H)**

**Timeframe:** 1H  
**Minimum Required:** 100 bars  
**Why:** HSMM needs sufficient window for forward-backward algorithm

**Readiness Check:**
```python
if len(df_1h) < 100:
    ready = False
    state = "not_ready"
    reason = f"Insufficient data for HSMM ({len(df_1h)} bars, need 100)"
```

**Logic Check (when ready):**
```python
if len(df_1h) >= 100:
    ready = True
    # Run HSMM
    if sdc < 5.0:
        state = "low_confidence"
        passed = False  # Logic block (SdC too low)
    elif stability < 0.60:
        state = "unstable"
        passed = False  # Logic block (stability too low)
    else:
        state = "trend_plus" / "range" / "trend_minus"
        passed = True (if aligned) or False (if misaligned)
```

---

### **Setup Agent (15M)**

**Timeframe:** 15M  
**Minimum Required:** 50 bars  
**Why:** SMC detector needs lookback window for patterns

**Readiness Check:**
```python
if len(df_15m) < 50:
    ready = False
    state = "not_ready"
    reason = f"Insufficient data ({len(df_15m)} bars, need 50)"
```

**Logic Check (when ready):**
```python
if len(df_15m) >= 50:
    ready = True
    # Detect SMC patterns
    if no_context_provided:
        state = "no_bias"
        passed = False  # Logic block (no context to align)
    elif patterns_exist and aligned:
        state = "valid_setup"
        passed = True
    else:
        state = "no_pattern" / "misaligned"
        passed = False  # Logic block
```

---

## 🔄 **FRACTAL DECISION ELIGIBILITY**

### **When is a decision "ready to count"?**

**ALL of the following must be true:**

1. ✅ Context Agent: `ready = True`
2. ✅ Regime Agent: `ready = True`
3. ✅ Setup Agent: `ready = True`

**If ANY agent is `not_ready`:**
- Decision is marked as `WAIT`
- `blocked_by = ["readiness"]`
- Does NOT count in logic statistics
- Counts only in `bars_skipped_due_to_readiness`

---

## 📈 **TYPICAL WARMUP TIMELINE**

**Example: 10,000 bars 15M sample**

| Timeframe | Bars Loaded | Min Required | Ready After |
|-----------|-------------|--------------|-------------|
| 1D        | 105         | 50           | ✅ Immediately (105 > 50) |
| 1H        | 2,500       | 100          | ✅ Immediately (2,500 > 100) |
| 15M       | 10,000      | 50           | ✅ Immediately (10,000 > 50) |

**In this case:** Fractal ready from bar 0 (all TFs have sufficient history)

**Example: 1,000 bars 15M sample**

| Timeframe | Bars Loaded | Min Required | Ready After |
|-----------|-------------|--------------|-------------|
| 1D        | 11          | 50           | ❌ NEVER (only 11 bars) |
| 1H        | 250         | 100          | ✅ Immediately (250 > 100) |
| 15M       | 1,000       | 50           | ✅ Immediately (1,000 > 50) |

**In this case:** Fractal NEVER ready (Context lacks data)  
**Result:** All 1,000 signals = `blocked_by_readiness`

---

## 🔍 **DIAGNOSTIC INTERPRETATION**

### **Scenario 1: High Readiness Blocks**
```
bars_analyzed:                1000
bars_ready_for_decision:        0
bars_skipped_due_to_readiness: 1000

blocked_by_context_readiness:  1000
blocked_by_regime_readiness:      0
blocked_by_setup_readiness:       0
```

**Interpretation:** Sample too small - Context needs more 1D data  
**Action:** Increase `--sample-bars` or use different period

---

### **Scenario 2: Logic Blocks Dominate**
```
bars_analyzed:                1000
bars_ready_for_decision:       742
bars_skipped_due_to_readiness: 258

blocked_by_context_logic:       85
blocked_by_regime_logic:       210
blocked_by_setup_logic:        317
blocked_by_risk:                 8

approved_buy:                    8
approved_sell:                   2
wait_neutral:                  120
```

**Interpretation:** Pipeline ready, but Setup is main bottleneck  
**Action:** This is REAL diagnostic - Setup rejects most setups

---

### **Scenario 3: Mixed Blocks**
```
bars_analyzed:                1000
bars_ready_for_decision:       400
bars_skipped_due_to_readiness: 600

blocked_by_regime_readiness:   600
blocked_by_setup_logic:        280
```

**Interpretation:** First 600 bars = warmup, then Setup blocks  
**Action:** Valid diagnostic on ready bars only

---

## ⚠️ **CRITICAL RULES**

### **1. Never Mix Readiness and Logic**

❌ **WRONG:**
```python
if len(df) < 50:
    return AgentResult(
        state="neutral",  # ← WRONG! This is not a logic decision
        passed=False
    )
```

✅ **CORRECT:**
```python
if len(df) < 50:
    return AgentResult(
        ready=False,
        blocked_by_readiness=True,
        state="not_ready",  # ← Clear distinction
        passed=False,
        reason=f"Insufficient data ({len(df)} bars)"
    )
```

---

### **2. Always Set `ready` Flag**

Every AgentResult MUST include:
- `ready: bool`
- `blocked_by_readiness: bool`

---

### **3. Orchestrator Must Check Readiness First**

```python
# BEFORE checking agent.passed
if not all(agent.ready for agent in agents):
    return OrchestratorDecision(
        action='WAIT',
        blocked_by=['readiness'],
        reason='Pipeline not ready'
    )

# THEN check logic
if not all(agent.passed for agent in agents):
    return OrchestratorDecision(
        action='WAIT',
        blocked_by=[name for name, agent in agents if not agent.passed],
        reason='Logic blocks'
    )
```

---

### **4. fractal_check Must Count Separately**

```python
for signal in signals:
    if any(not agent.ready for agent in signal.agents):
        bars_skipped_due_to_readiness += 1
        # Count which agents not ready
    else:
        bars_ready_for_decision += 1
        # Count logic blocks
```

---

## 📋 **CONFIGURATION**

**File:** `config/validation_baseline.yaml`

```yaml
fractal_readiness:
  context_min_bars: 50    # Context Agent (1D)
  regime_min_bars: 100    # Regime Agent (1H)
  setup_min_bars: 50      # Setup Agent (15M)
```

**These values are:**
- Configurable (not hardcoded)
- Based on agent requirements (SMA windows, HSMM needs, SMC lookback)
- Tested and validated

---

## 🎯 **SUCCESS CRITERIA**

After implementing readiness rules, fractal_check must be able to answer:

1. ✅ "How many bars were skipped due to insufficient data?"
2. ✅ "Which agent needs more data?"
3. ✅ "Of the bars with sufficient data, which agent blocks most?"
4. ✅ "Is the bottleneck warmup or logic?"

**Before:** Mixed confusion  
**After:** Clear separation

---

## 📊 **EXAMPLE OUTPUT**

**Good Diagnostic (Enough Data):**
```
FRACTAL CHECK - BTCUSDT
Bars analyzed:               10000
Bars ready for decision:      9742
Bars skipped (readiness):      258

Blocked by readiness:         258
  Context:                       0
  Regime:                      258  ← Early warmup
  Setup:                         0

Blocked by logic:            9130
  Context:                     850
  Regime:                     2100
  Setup:                      6180  ← Main bottleneck

Approved:
  BUY:                         487
  SELL:                        125
```

**Bad Diagnostic (Insufficient Data):**
```
FRACTAL CHECK - BTCUSDT
Bars analyzed:                1000
Bars ready for decision:         0
Bars skipped (readiness):     1000

Blocked by readiness:        1000
  Context:                    1000  ← Sample too small!
  Regime:                        0
  Setup:                         0

⚠️ WARNING: No bars ready for decision.
   Increase --sample-bars or use longer period.
```

---

**This document defines the contract for fractal readiness vs logic blocking.**
