# SMC Detector Autopsy

## Purpose

This diagnostic mode provides a **technical autopsy** of the SMC (Smart Money Concepts) detector to understand why it's globally silent across all market regimes (bull/bear/range).

**Question answered:**
> Is the detector silent because it's too strict, mal-wired, too naive, or filtering too aggressively?

**Approach:**
- Instrument the detector to track what actually happens
- Count function calls, candidates seen, detections made
- Identify failure reasons (threshold, structure, followthrough)
- Test sensitivity with alternative thresholds
- Produce a structured verdict (A/B/C/D)

---

## Usage

### **Basic Autopsy**
```bash
python scripts/run_validation.py --mode smc_autopsy \
    --pair BTCUSDT --sample-date 2023-12-15
```

### **With Sensitivity Analysis**
```bash
python scripts/run_validation.py --mode smc_autopsy \
    --pair BTCUSDT --sample-date 2023-12-15 --sensitivity
```

**Output:**
- CLI summary with verdict
- JSON file: `reports/validation/fractal/BTCUSDT_smc_autopsy.json`
- Optional: Sensitivity analysis with multiple threshold variants

---

## How to Read Results

### **CLI Output Example**

```
SMC AUTOPSY - BTCUSDT
Reference date: 2023-12-15

CODE PATH
Detector called:           YES
FVG function calls:         94
OB function calls:          94

FVG AUDIT
Candidates seen:            94
Detected:                    2
Main failure:              structure rules (10 gap, 82 structure)

OB AUDIT
Candidates seen:           141
Detected:                    0
Main failure:              range threshold (81 range, 60 followthrough)

FINAL OUTPUT
Patterns detected:           2
Setup passed:                0

VERDICT
Category: A
Detector works but very low success rate (0.9%) - likely threshold strictness.
```

---

## Understanding the Metrics

### **Code Path**

**detector_called:**
- Number of times the detector was invoked
- If 0 → Detector not being called at all (Category C)
- If > 0 → Detector is in the execution path ✅

**fvg_function_called / ob_function_called:**
- Number of times each detection function was called
- Usually 2x detector_called (bullish + bearish)

---

### **FVG Audit**

**Fair Value Gap** = Imbalance where price gaps between candles

**candidates_seen:**
- How many potential FVG structures were examined
- If 0 → No 3-bar patterns checked (too naive)
- If > 0 → Detector is looking for FVG structures ✅

**detected:**
- How many FVG patterns passed all criteria
- If 0 → All candidates failed
- If > 0 → Some FVGs detected ✅

**failed_gap_threshold:**
- Candidates with gap < `fvg_min_gap` (default 0.5%)
- High count → Threshold too strict

**failed_structure_rules:**
- Candidates with no gap at all (price didn't gap)
- High count → Market structure doesn't produce FVGs

---

### **OB Audit**

**Order Block** = Strong move after consolidation candle

**candidates_seen:**
- How many down/up candles were examined
- If 0 → No consolidation candles found
- If > 0 → Detector is looking for OB structures ✅

**detected:**
- How many OB patterns passed all criteria
- If 0 → All candidates failed
- If > 0 → Some OBs detected ✅

**failed_range_threshold:**
- Candidates with followthrough < `ob_range_threshold` (default 1.5%)
- High count → Threshold too strict

**failed_followthrough:**
- Candidates with no move after consolidation candle
- High count → Market doesn't show strong followthrough

---

### **Sensitivity Analysis**

Tests multiple threshold configurations in memory:

```
Testing: baseline
  OB threshold: 0.015 (1.5%)
  FVG min gap:  0.005 (0.5%)
  → Patterns detected: 2 (4.3%)

Testing: very_relaxed
  OB threshold: 0.005 (0.5%)
  FVG min gap:  0.001 (0.1%)
  → Patterns detected: 7 (14.9%)
```

**Interpretation:**
- **No change:** Thresholds not the issue
- **Moderate improvement:** Thresholds contribute but not sole cause
- **Large improvement:** Thresholds are the main blocker

---

## Verdict Categories

The autopsy produces one of these verdicts:

### **Category A: Threshold Strictness**
```
"Detector path works, but current thresholds make detection near-zero."
```

**Meaning:**
- Detector is called ✅
- Candidates are seen ✅
- Most fail threshold checks ❌

**Indicators:**
- `failed_gap_threshold` high (FVG)
- `failed_range_threshold` high (OB)
- Sensitivity analysis shows improvement

**Action:**
- Review `fvg_min_gap` (currently 0.5%)
- Review `ob_range_threshold` (currently 1.5%)
- Consider if current strictness is intentional

---

### **Category B: Naive Implementation**
```
"Detector is functionally executed, but its implementation is too naive/incomplete."
```

**Meaning:**
- Detector is called ✅
- No candidates seen at all ❌
- Implementation doesn't match market structure

**Indicators:**
- `candidates_seen` = 0 (both FVG and OB)
- Detector logic too simple
- Missing key market patterns

**Action:**
- Review detection algorithms
- Compare with SMC reference implementations
- Add more pattern types (liquidity, breaker blocks)

---

### **Category C: Broken Code Path**
```
"Detector code path is broken - not being called at all."
```

**Meaning:**
- Detector not called ❌
- Integration issue

**Indicators:**
- `detector_called` = 0

**Action:**
- Check Setup Agent integration
- Verify detector instantiation
- Review orchestrator logic

---

### **Category D: Aggressive Filtering**
```
"Detector sees candidates, but final filtering logic discards them too aggressively."
```

**Meaning:**
- Detector is called ✅
- Candidates seen ✅
- Low detection not due to thresholds ❌

**Indicators:**
- `candidates_seen` > 0
- `detected` = 0
- Fails NOT due to threshold/followthrough
- Sensitivity analysis shows no improvement

**Action:**
- Review filtering logic beyond simple thresholds
- Check alignment scoring
- Review pattern validation rules

---

## Example Autopsy Scenarios

### **Scenario 1: Strict Thresholds (Category A)**

```
FVG AUDIT:
  Candidates: 94
  Detected: 2
  Failed gap threshold: 80
  Failed structure: 12

OB AUDIT:
  Candidates: 141
  Detected: 0
  Failed range threshold: 120
  Failed followthrough: 21

Sensitivity: baseline 4% → very_relaxed 15%

Verdict: A
```

**Interpretation:** Thresholds are too strict. Many candidates but most fail the threshold check.

---

### **Scenario 2: Naive Detector (Category B)**

```
FVG AUDIT:
  Candidates: 0
  Detected: 0

OB AUDIT:
  Candidates: 0
  Detected: 0

Sensitivity: No improvement

Verdict: B
```

**Interpretation:** Detector doesn't even see candidate structures. Implementation too simple.

---

### **Scenario 3: Broken Integration (Category C)**

```
CODE PATH:
  Detector called: 0

Verdict: C
```

**Interpretation:** Detector never called. Integration problem.

---

## JSON Structure

```json
{
  "pair": "BTCUSDT",
  "reference_date": "2023-12-15T00:00:00",
  "bars_analyzed": 47,
  "code_path": {
    "detector_called": 47,
    "fvg_function_called": 94,
    "ob_function_called": 94
  },
  "fvg_audit": {
    "candidates_seen": 94,
    "detected": 2,
    "failed_gap_threshold": 10,
    "failed_structure_rules": 82,
    "failed_other": 0
  },
  "ob_audit": {
    "candidates_seen": 141,
    "detected": 0,
    "failed_range_threshold": 81,
    "failed_followthrough": 60,
    "failed_other": 0
  },
  "configuration": {
    "ob_range_threshold": 0.015,
    "fvg_min_gap": 0.005,
    "liquidity_lookback": 20
  },
  "final_output": {
    "patterns_detected": 2,
    "setup_passed": 0
  },
  "verdict": {
    "category": "A",
    "summary": "Detector works but very low success rate..."
  },
  "sensitivity_analysis": {
    "test_configurations": [...],
    "summary": "Relaxing thresholds to 'very_relaxed' improves..."
  }
}
```

---

## Current Configuration (Baseline)

**From `config/validation_baseline.yaml`:**

```yaml
smc_detector:
  ob_range_threshold: 0.015  # 1.5% minimum move for OB
  fvg_min_gap: 0.005         # 0.5% minimum gap for FVG
  liquidity_lookback: 20     # bars to look back
```

**Interpretation:**
- **1.5% OB threshold:** Very strict (especially for crypto)
- **0.5% FVG gap:** Moderately strict
- These values may be appropriate for highly selective strategy
- Or may be too conservative and miss valid setups

---

## Sensitivity Variants Tested

When `--sensitivity` flag is used:

1. **baseline:** Current config (1.5% OB, 0.5% FVG)
2. **relaxed_25pct:** 25% more permissive (1.1% OB, 0.375% FVG)
3. **relaxed_50pct:** 50% more permissive (0.75% OB, 0.25% FVG)
4. **very_relaxed:** Very permissive (0.5% OB, 0.1% FVG)

**Purpose:** Understand if moderate threshold adjustments meaningfully improve detection.

---

## Related Documentation

- **SETUP_BOTTLENECK.md** - Why Setup blocks (no patterns)
- **SETUP_COVERAGE_MULTI.md** - Silence across regimes
- **SMC_VISUAL_TRUTH_AUDIT.md** - Manual validation protocol
- **FRACTAL_BOTTLENECK.md** - Agent-level bottleneck analysis

---

## Limitations

### **What This CAN Tell You**

- Whether detector is called
- How many candidates are examined
- Why candidates fail (threshold vs structure)
- If relaxing thresholds helps
- Verdict category (A/B/C/D)

### **What This CAN'T Tell You**

- If detected patterns are "good quality"
- If patterns exist visually but detector misses them
- Optimal threshold values (requires backtesting)
- Whether selectivity is appropriate for strategy

---

## Next Steps Based on Verdict

### **If Category A (Threshold Strictness)**
1. Review if current strictness is intentional
2. Run sensitivity to find reasonable relaxation
3. Backtest with alternative thresholds
4. Consider if selective strategy is desired

### **If Category B (Naive Implementation)**
1. Compare with SMC reference implementations
2. Review detection algorithms
3. Consider additional pattern types
4. Visual audit to see what detector misses

### **If Category C (Broken Path)**
1. Debug Setup Agent integration
2. Verify detector instantiation
3. Check orchestrator calls

### **If Category D (Aggressive Filtering)**
1. Review filtering logic beyond thresholds
2. Check alignment scoring
3. Audit pattern validation rules

---

## Summary

**Question:** Why is the SMC detector globally silent?

**Answer:** This autopsy tells you:
- **Category A:** Thresholds too strict (most common)
- **Category B:** Implementation too naive
- **Category C:** Code path broken
- **Category D:** Filtering too aggressive

**Then:** You can take targeted action based on root cause rather than guessing.

**But:** This is still diagnostics - changes require separate optimization ticket with backtesting.
