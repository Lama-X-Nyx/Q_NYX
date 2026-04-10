# Bullish Context/Regime Attribution Audit

## Purpose

**Objective:** Understand why NYX doesn't express a usable bullish bias on BTC during periods that should be bullish.

**Core Question:**
> Is the bullish under-expression due to Context being too neutral, or Regime being too conservative?

This diagnostic helps attribute the root cause of bullish signal failure to the correct layer of the fractal architecture.

---

## Why This Matters

### **The Problem**

NYX is supposed to detect and trade bullish BTC periods. But previous diagnostics showed:
- Setup finds 0 patterns (addressed in other tickets)
- But even if Setup worked, would Context/Regime provide bullish bias?

**We need to know:**
1. Does Context Agent correctly identify bullish market structure?
2. Does Regime Agent correctly identify trend_plus during bull markets?
3. Or do they both collapse into neutral/range?

### **Impact**

If Context/Regime don't express bullish correctly:
- Even perfect Setup detection won't help
- System structurally can't capture bull moves
- Need to address higher-level agents first

---

## Usage

### **Basic Usage (default periods)**

```bash
python scripts/run_validation.py --mode bullish_audit \
    --pair BTCUSDT
```

**Default test periods:**
- 2023-01-15 (early bull)
- 2023-03-15 (mid bull)
- 2023-10-15 (late bull)
- 2023-12-15 (range comparison)

### **Custom Periods**

```bash
python scripts/run_validation.py --mode bullish_audit \
    --pair BTCUSDT --dates 2023-01-15,2023-03-15,2023-10-15
```

**Output:**
- CLI summary comparing all periods
- JSON file: `reports/validation/fractal/BTCUSDT_bullish_audit.json`

---

## How to Read Results

### **CLI Output Example**

```
2023-01-15 (early_bull)
  Context: bullish    (0.71) ✅
  Regime:  range      (0.98) ✅
  Verdict: regime_too_conservative

2023-03-15 (mid_bull)
  Context: neutral    (0.40) ❌
  Regime:  range      (1.00) ✅
  Verdict: bullish_not_expressed

2023-10-15 (late_bull)
  Context: bullish    (0.71) ✅
  Regime:  range      (0.59) ✅
  Verdict: regime_too_conservative

CROSS-PERIOD SUMMARY
Context bullish rate:     75.0%
Regime trend_plus rate:   0.0%

Main issue: regime_too_conservative

VERDICT:
Context is often bullish, but regime collapses too often into range.
```

---

## Understanding the Metrics

### **Per-Period Fields**

**Context:**
- **state:** bullish / neutral / bearish
- **score:** Agent confidence (0.0-1.0)
- **passed:** Did agent pass its threshold?
- **reason:** Why this state was chosen

**Regime:**
- **state:** trend_plus / range / trend_minus
- **score:** Agent confidence
- **sdc:** State Duration Certainty (HSMM metric)
- **stability:** Regime stability score
- **passed:** Did agent pass its threshold?

**Joint Interpretation:**
- `bullish_coherent`: Both Context and Regime bullish ✅
- `regime_too_conservative`: Context bullish, Regime range ⚠️
- `context_too_neutral`: Context neutral, Regime variable ⚠️
- `bullish_not_expressed`: Both neutral/range ❌
- `bearish_detected`: Wrong direction ❌

---

### **Cross-Period Summary**

**Rates:**
- **context_bullish_rate:** % of periods where Context is bullish
- **context_neutral_rate:** % of periods where Context is neutral
- **regime_trend_plus_rate:** % of periods where Regime is trend_plus
- **regime_range_rate:** % of periods where Regime is range

**Main Issue:**
- `regime_too_conservative`: Context often bullish, Regime often range
- `context_too_neutral`: Context often neutral, regardless of Regime
- `both_too_neutral`: Both agents too neutral
- `sample_dependent`: Inconsistent across samples

---

## Verdict Categories

The audit automatically produces one of 4 verdicts:

### **Verdict A: Both Express Correctly**

```
"Context and regime both express bullish BTC correctly."
```

**Meaning:**
- Context bullish rate ≥ 75%
- Regime trend_plus rate ≥ 75%
- Both agents working as expected

**Action:** No Context/Regime fixes needed, focus on Setup

---

### **Verdict B: Regime Too Conservative** ← **CURRENT FINDING**

```
"Context is often bullish, but regime collapses too often into range."
```

**Meaning:**
- Context bullish rate ≥ 60%
- Regime trend_plus rate < 40%
- Context sees bullish structure
- Regime refuses to confirm trend

**Root Cause:** Regime Agent threshold too strict or HSMM parameters too conservative

**Action:** 
1. Review Regime Agent threshold (currently requires what score?)
2. Review HSMM parameters (stability too demanding?)
3. Consider if "range" detection is too sensitive

**Example from audit:**
```
Context bullish: 75% (3/4 periods)
Regime trend_plus: 0% (0/4 periods)
Regime range: 100% (4/4 periods)

→ Regime ALWAYS sees range, even when Context sees bullish
```

---

### **Verdict C: Context Too Neutral**

```
"Context itself is too neutral during bullish BTC periods."
```

**Meaning:**
- Context bullish rate < 40%
- Context fails to identify bullish structure

**Root Cause:** Context Agent threshold too strict or structure detection too conservative

**Action:**
1. Review Context Agent logic (how does it detect bullish?)
2. Review threshold for "bullish" state
3. Check if Context timeframe (1D) is appropriate

---

### **Verdict D: Sample Dependent**

```
"Bullish expression is inconsistent and depends strongly on the chosen sample."
```

**Meaning:**
- No clear pattern across periods
- Sometimes Context bullish, sometimes not
- Sometimes Regime trend, sometimes not

**Root Cause:** Sample selection issue or genuine market ambiguity

**Action:**
1. Test more periods to confirm pattern
2. Review if "bullish" sample selection is accurate
3. Consider if markets were genuinely ambiguous

---

## Real-World Example (Current Audit)

### **Finding: Verdict B - Regime Too Conservative**

**Tested periods:**
- 2023-01-15 (early BTC bull)
- 2023-03-15 (mid BTC bull)
- 2023-10-15 (late BTC bull)
- 2023-12-15 (range comparison)

**Results:**

| Period | Context | Regime | Issue |
|--------|---------|--------|-------|
| 2023-01-15 | bullish (0.71) | **range (0.98)** | regime_too_conservative |
| 2023-03-15 | neutral (0.40) | **range (1.00)** | bullish_not_expressed |
| 2023-10-15 | bullish (0.71) | **range (0.59)** | regime_too_conservative |
| 2023-12-15 | bullish (1.00) | **range (1.00)** | regime_too_conservative |

**Cross-period:**
- Context bullish: 75% (good!)
- Regime trend_plus: **0%** (problem!)
- Regime range: **100%** (always!)

**Interpretation:**

Context Agent correctly identifies bullish structure in 3/4 periods. **Context works.**

Regime Agent sees "range" in ALL 4 periods, even confirmed bull markets. **Regime is broken or too conservative.**

**Root Cause:** Regime Agent (HSMM-based) is too strict about declaring "trend_plus". It defaults to "range" safety.

---

## What This Means

### **Before This Audit**

**Uncertainty:**
> "NYX doesn't trade bullish. Is it Context, Regime, Setup, or all three?"

### **After This Audit**

**Clarity:**
> "Context works (75% bullish on bull periods). Regime is the blocker (0% trend_plus, 100% range)."

**Actionable:**
- Don't waste time fixing Context (already working)
- Focus on Regime Agent parameters
- Review HSMM threshold for trend_plus
- Check if "range" detection is too sensitive

---

## Next Steps Based on Verdict

### **If Verdict B (Regime Conservative) ← Current**

**Immediate actions:**

1. **Inspect Regime Agent threshold:**
   - What score required for trend_plus?
   - Is it too strict?

2. **Inspect HSMM parameters:**
   - Stability threshold
   - SDC (State Duration Certainty) requirements
   - Are they too demanding?

3. **Visual validation:**
   - Chart 2023-01-15, 2023-10-15 (bull periods)
   - Manually assess: should these be "trend_plus"?
   - If yes: Regime is too conservative (fix params)
   - If no: Samples were actually ambiguous (OK)

4. **Consider Regime sensitivity tuning ticket:**
   - Test alternative HSMM parameters
   - Backtest with relaxed trend_plus threshold
   - Measure impact on false positives

---

### **If Verdict C (Context Neutral)**

**Actions:**
1. Review Context Agent bullish detection logic
2. Check Context threshold requirements
3. Validate on charts manually

---

### **If Verdict D (Sample Dependent)**

**Actions:**
1. Test more diverse periods
2. Review sample selection criteria
3. Accept if genuinely ambiguous markets

---

## JSON Structure

```json
{
  "pair": "BTCUSDT",
  "periods": {
    "2023-01-15": {
      "period_label": "early_bull",
      "context": {
        "state": "bullish",
        "score": 0.71,
        "passed": true,
        "reason": "Higher timeframe structure bullish"
      },
      "regime": {
        "state": "range",
        "score": 0.98,
        "passed": true,
        "reason": "HSMM range",
        "sdc": 8.2,
        "stability": 0.76
      },
      "joint_interpretation": "regime_too_conservative"
    },
    ...
  },
  "cross_period_summary": {
    "context_bullish_rate": 75.0,
    "context_neutral_rate": 25.0,
    "regime_trend_plus_rate": 0.0,
    "regime_range_rate": 100.0,
    "main_issue": "regime_too_conservative",
    "verdict": "Context is often bullish, but regime collapses too often into range.",
    "interpretation_distribution": {
      "regime_too_conservative": 3,
      "bullish_not_expressed": 1
    }
  }
}
```

---

## Limitations

### **What This CAN Tell You**

- Whether Context identifies bullish correctly
- Whether Regime identifies trend_plus correctly
- Which layer (Context or Regime) is the blocker
- If bullish expression is sample-dependent

### **What This CAN'T Tell You**

- Why Regime is conservative (need deeper HSMM inspection)
- Optimal threshold values (need separate tuning ticket)
- If Setup would work even with good Context/Regime (separate issue)
- Whether samples are "truly" bullish (manual validation needed)

---

## Interpretation Guide

### **Good Patterns**

✅ **Context bullish ≥ 75%, Regime trend_plus ≥ 75%**
- Both agents working correctly
- Bullish expression is strong
- Focus can shift to Setup

✅ **Context bullish ≥ 60%, Regime trend_plus ≥ 50%**
- Decent bullish expression
- Some conservatism but acceptable
- Minor tuning may help

### **Problem Patterns**

⚠️ **Context bullish ≥ 60%, Regime trend_plus < 40%** ← **Current**
- Context works, Regime blocks
- Fix Regime Agent

⚠️ **Context bullish < 40%, Regime trend_plus ≥ 50%**
- Regime works, Context blocks
- Fix Context Agent

❌ **Context bullish < 40%, Regime trend_plus < 40%**
- Both agents too neutral
- Systemic issue, both need review

❌ **Highly variable across samples**
- Sample selection issue
- Or genuinely ambiguous markets

---

## Related Documentation

- **FRACTAL_BOTTLENECK.md** - Agent-level pass rates
- **SETUP_BOTTLENECK.md** - Why Setup blocks
- **SMC_AUTOPSY.md** - SMC detector analysis

---

## Summary

**Question:** 
> "Why doesn't NYX express bullish bias on bull BTC periods?"

**Answer (from audit):**
> "Context identifies bullish correctly (75%). Regime refuses to confirm trend_plus (0%). Regime is the blocker."

**Action:**
> "Fix Regime Agent conservatism. Review HSMM parameters for trend_plus threshold."

**Impact:**
> "Don't waste time on Context (working). Focus Regime tuning. Then revisit Setup/SMC."

---

**This audit identifies WHICH layer blocks bullish expression, enabling targeted fixes.**
