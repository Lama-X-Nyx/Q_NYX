# Fractal Bottleneck Discovery

**Version:** 1.0  
**Date:** 2025-03-29  
**Purpose:** Identify the real business bottleneck in the fractal pipeline after warmup

---

## 🎯 **OBJECTIVE**

**Question to answer:**
> "Once the fractal pipeline is ready, which agent is the primary blocker of trading decisions?"

**NOT the question:**
> "Which agent needs more data?" ← That's readiness (already solved)

**The difference:**
- **Readiness bottleneck:** Agent doesn't have enough bars yet (warmup phase)
- **Logic bottleneck:** Agent has enough bars but rejects trades (business logic)

---

## 📊 **WHAT WE MEASURE**

### **Primary Metrics (Ready Bars Only)**

**Pass Rates:**
- `context_pass_rate`: % of ready bars where Context approves
- `regime_pass_rate`: % of ready bars where Regime approves  
- `setup_pass_rate`: % of ready bars where Setup approves
- `final_trade_rate`: % of ready bars that result in BUY/SELL

**Logic Blocks:**
- `context_blocks`: Count of Context rejections (on ready bars)
- `regime_blocks`: Count of Regime rejections (on ready bars)
- `setup_blocks`: Count of Setup rejections (on ready bars)
- `risk_blocks`: Count of Risk rejections (on ready bars)

**Bottleneck Ranking:**
- Agents ordered by number of blocks (descending)
- Primary bottleneck = agent with most blocks

---

## 🔍 **HOW TO INTERPRET RESULTS**

### **Scenario 1: Setup is the Bottleneck**

**Example:**
```
Context pass rate:  85%
Regime pass rate:   70%
Setup pass rate:    25%  ← Bottleneck
Final trade rate:   0.8%

Logic blocks:
  Setup: 5,500  ← Primary blocker
  Regime: 2,200
  Context: 1,100
```

**Interpretation:**
- Context and Regime are relatively permissive
- Setup (SMC patterns) blocks most trades
- Only 25% of ready bars have valid SMC alignment

**Potential causes:**
- SMC patterns too rare (OB/FVG detection)
- Alignment threshold too strict (`alignment_15m_min`)
- Pattern quality filter too tight

**Next steps:**
- Review SMC detector parameters
- Check if OB/FVG thresholds are too strict
- Consider pattern quality vs quantity tradeoff
- Analyze sample blocked decisions manually

---

### **Scenario 2: Regime is the Bottleneck**

**Example:**
```
Context pass rate:  80%
Regime pass rate:   40%  ← Bottleneck
Setup pass rate:    75%
Final trade rate:   1.2%

Logic blocks:
  Regime: 4,400  ← Primary blocker
  Setup: 1,800
  Context: 1,500
```

**Interpretation:**
- HSMM regime detection is the main filter
- Context and Setup are more permissive
- Only 40% of ready bars have confident regime

**Potential causes:**
- `sdc_min` (5.0) too high
- `stability_4h_min` (0.60) too strict
- Range regime detected too often
- HSMM may be too conservative

**Next steps:**
- Review HSMM confidence thresholds
- Check if SdC calculation is too strict
- Analyze regime state distribution
- Consider if Range detection is too sensitive

---

### **Scenario 3: Context is the Bottleneck**

**Example:**
```
Context pass rate:  35%  ← Bottleneck
Regime pass rate:   85%
Setup pass rate:    80%
Final trade rate:   0.9%

Logic blocks:
  Context: 4,800  ← Primary blocker
  Regime: 1,100
  Setup: 1,500
```

**Interpretation:**
- Macro bias detection is the main filter
- Lower timeframes are more permissive
- Only 35% of ready bars have clear directional bias

**Potential causes:**
- SMA threshold (2%) too strict
- Market in prolonged sideways/neutral period
- Bias detection logic too conservative

**Next steps:**
- Review SMA 20/50 threshold
- Check if neutral bias is too frequent
- Consider if 1D timeframe is appropriate
- Analyze market conditions during test period

---

### **Scenario 4: Risk is the Bottleneck**

**Example:**
```
Context pass rate:  80%
Regime pass rate:   75%
Setup pass rate:    70%
Final trade rate:   2.5%

Logic blocks:
  Risk: 5,200  ← Primary blocker
  Setup: 2,200
  Regime: 1,800
```

**Interpretation:**
- All agents approve frequently
- Risk management blocks most trades
- Agents are permissive, risk is strict

**Potential causes:**
- `rr_min` (2.0) too high
- Stop loss / take profit levels too tight
- Risk calculation formula too conservative
- `hit_minus_010_max` (0.20) too strict

**Next steps:**
- Review risk management parameters
- Check if RR ratio is realistic
- Analyze stop/target placement
- Consider risk/reward tradeoff

---

## 📋 **METRICS CALCULATION**

### **Pass Rate Formula**

```python
# Only on ready bars (not warmup)
ready_bars = [bar for bar in all_bars if all_agents_ready(bar)]

context_pass_rate = (
    count(context.passed for bar in ready_bars) / len(ready_bars)
) * 100

regime_pass_rate = (
    count(regime.passed for bar in ready_bars) / len(ready_bars)
) * 100

setup_pass_rate = (
    count(setup.passed for bar in ready_bars) / len(ready_bars)
) * 100

final_trade_rate = (
    count(action in ['BUY', 'SELL'] for bar in ready_bars) / len(ready_bars)
) * 100
```

### **Logic Block Count**

```python
# Only on ready bars
for bar in ready_bars:
    if not context.passed:
        context_blocks += 1
    if not regime.passed:
        regime_blocks += 1
    if not setup.passed:
        setup_blocks += 1
    if 'risk' in decision.blocked_by:
        risk_blocks += 1
```

---

## ⚠️ **CRITICAL RULES**

### **1. Only Count Ready Bars**

❌ **WRONG:**
```python
pass_rate = context_passed / total_bars  # Includes warmup!
```

✅ **CORRECT:**
```python
pass_rate = context_passed / ready_bars  # Only evaluable bars
```

### **2. Separate Readiness from Logic**

**In the report:**
```json
{
  "bars_ready_for_decision": 7420,
  "bars_skipped_due_to_readiness": 2580,
  "diagnostics": {
    "readiness_blocks": {...},  ← Separate
    "logic_blocks": {...}       ← Separate
  }
}
```

### **3. Primary Bottleneck = Most Blocks**

```python
bottleneck_ranking = sorted(
    logic_blocks.items(),
    key=lambda x: x[1],
    reverse=True
)
primary_bottleneck = bottleneck_ranking[0][0]
```

---

## 📊 **EXAMPLE OUTPUT**

### **CLI Summary**
```
FRACTAL BOTTLENECK - BTCUSDT

Bars analyzed:               10000
Bars ready for decision:      7420
Bars skipped (readiness):     2580

READY-ONLY PASS RATES
Context pass rate:            88.9%
Regime pass rate:             62.4%
Setup pass rate:              31.8%  ← Lowest
Final trade rate:              0.7%

LOGIC BLOCKS (Ready Bars Only)
Context blocked:               820
Regime blocked:              1,900
Setup blocked:               3,050  ← Most blocks
Risk blocked:                  110

ACTIONS
BUY approved:                   42
SELL approved:                  11
WAIT neutral:                1,487

PRIMARY BOTTLENECK: setup
```

### **JSON Report**
```json
{
  "pair": "BTCUSDT",
  "bars_analyzed": 10000,
  "bars_ready_for_decision": 7420,
  "bars_skipped_due_to_readiness": 2580,
  "ready_only_summary": {
    "context_pass_rate": 88.9,
    "regime_pass_rate": 62.4,
    "setup_pass_rate": 31.8,
    "final_trade_rate": 0.7,
    "primary_bottleneck": "setup"
  },
  "bottleneck_ranking": [
    {"agent": "setup", "blocks": 3050},
    {"agent": "regime", "blocks": 1900},
    {"agent": "context", "blocks": 820},
    {"agent": "risk", "blocks": 110}
  ]
}
```

---

## 🎯 **WHAT THIS TELLS US**

### **If Setup is the bottleneck:**
→ SMC patterns are too strict or rare  
→ Next: Review SMC detector settings

### **If Regime is the bottleneck:**
→ HSMM is too conservative  
→ Next: Review confidence/stability thresholds

### **If Context is the bottleneck:**
→ Bias detection is too strict  
→ Next: Review SMA threshold or timeframe

### **If Risk is the bottleneck:**
→ Risk management is too strict  
→ Next: Review RR ratio and hit probabilities

---

## 📝 **IMPORTANT NOTES**

### **This is NOT optimization yet**

**Purpose:** Identify where the pipeline blocks  
**NOT:** Change thresholds to force more trades

**Workflow:**
1. Measure bottleneck (this ticket)
2. Understand why it blocks (manual analysis)
3. Decide if it's a bug or feature (judgment call)
4. THEN optimize if needed (separate ticket)

### **Trade rate is not the goal**

**Low trade rate (0.7%) might be correct:**
- If market conditions don't meet criteria
- If agents are correctly conservative
- If the baseline is designed to be selective

**High trade rate might be wrong:**
- If agents are too permissive
- If logic is broken
- If quality trades are diluted

**The goal is to know WHERE decisions get blocked, not to increase the rate artificially.**

---

## ✅ **SUCCESS CRITERIA**

After running fractal_bottleneck, you should be able to say:

1. ✅ "X% of bars were ready for decision"
2. ✅ "Of ready bars, Context passed Y%"
3. ✅ "Of ready bars, Regime passed Z%"
4. ✅ "The primary bottleneck is [agent]"
5. ✅ "This is because [agent] blocked N times"
6. ✅ "Sample blocked decisions show [pattern]"

**If you can't answer these clearly, the diagnostic failed.**

---

**This document defines how to identify and interpret the fractal business bottleneck.**
