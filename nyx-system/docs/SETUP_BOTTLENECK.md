# Setup Bottleneck Diagnosis

## Purpose

Now that fractal geometry and readiness are correct, this diagnostic mode answers the critical question:

> **Why does the Setup Agent block 100% of ready decisions?**

This is a **diagnostic-only** mode. It does not change trading logic, thresholds, or SMC rules.

---

## Problem Context

After fixing fractal geometry and separating readiness from logic blocks, we observed:

- **Context Agent:** 100% pass rate ✅
- **Regime Agent:** 100% pass rate ✅
- **Setup Agent:** 0% pass rate ❌

Setup is the clear bottleneck. But we need to understand **why** before making any adjustments.

---

## What It Diagnoses

For each ready-for-decision bar, the diagnostic categorizes Setup rejection into one of these reasons:

### **1. no_pattern**
- **Meaning:** No SMC patterns (FVG/OB) detected at all
- **Implication:** SMC detector finds nothing to work with
- **Example:** Quiet, choppy price action with no clear imbalances

### **2. misaligned_pattern**
- **Meaning:** Pattern exists but alignment score is far below threshold
- **Implication:** Pattern direction conflicts with higher timeframes
- **Example:** Bearish FVG when Context is bullish, score 0.2 vs threshold 0.6

### **3. alignment_score_too_low**
- **Meaning:** Pattern exists and somewhat aligned, but score just below threshold
- **Implication:** Close call - pattern almost passed
- **Example:** Bullish FVG aligned with bullish Context, score 0.58 vs threshold 0.6

### **4. direction_conflict**
- **Meaning:** Pattern has good alignment score but direction conflicts with Context
- **Implication:** Rare edge case - pattern score high but still rejected
- **Example:** Setup passes alignment but filtered by Context directional bias

### **5. setup_not_ready**
- **Meaning:** Setup component itself wasn't ready (should be 0 after geometry fix)
- **Implication:** If this appears, readiness separation failed
- **Example:** N/A - shouldn't happen with corrected geometry

### **6. unknown**
- **Meaning:** Rejection reason doesn't fit above categories
- **Implication:** Need deeper investigation
- **Example:** Edge cases not yet classified

---

## How to Read Results

### **Block Reasons Distribution**

```
BLOCK REASONS
No pattern:                   47   ← Most common
Misaligned pattern:            0
Alignment too low:             0
Direction conflict:            0
Setup not ready:               0
Unknown:                       0
```

**Interpretation:**
- If `no_pattern` dominates → SMC detector is too strict or patterns are genuinely rare
- If `misaligned_pattern` dominates → Patterns exist but fight higher TF context
- If `alignment_score_too_low` dominates → Threshold may be too strict
- If `direction_conflict` dominates → Context filter is too aggressive

### **Pattern Observations**

```
PATTERNS OBSERVED
Bullish FVG:                   0
Bearish FVG:                   0
Bullish OB:                    0
Bearish OB:                    0
No pattern:                   47
```

**Interpretation:**
- Zero patterns → Detector finds nothing
- Many patterns → Patterns exist but get filtered

### **Alignment Statistics**

```
ALIGNMENT
Mean alignment score:       0.30
Median alignment score:     0.30
Min alignment score:        0.30
Max alignment score:        0.30
Threshold required:         0.60
```

**Interpretation:**
- If mean << threshold → Patterns are far from passing
- If mean close to threshold (e.g., 0.55) → Threshold adjustment might help
- If all scores are identical (e.g., all 0.30) → Likely default score for "no pattern"

---

## Verdict Interpretation

The mode produces a verdict with two fields:

```json
"verdict": {
  "primary_setup_issue": "no_pattern",
  "interpretation": "Setup is blocked mostly by total absence of patterns"
}
```

### **Four Possible Verdicts**

#### **Verdict A: No patterns**
```
"primary_setup_issue": "no_pattern"
"interpretation": "Setup is blocked mostly by total absence of patterns"
```
**Meaning:** SMC detector finds no valid FVG or OB patterns  
**Action:** Investigate SMC detector parameters or accept that this period has no patterns

#### **Verdict B: Misaligned patterns**
```
"primary_setup_issue": "misaligned_pattern"
"interpretation": "Setup is blocked mostly by misalignment, not by pattern absence"
```
**Meaning:** Patterns exist but fight higher timeframe context  
**Action:** Investigate why patterns conflict with Context/Regime

#### **Verdict C: Near-threshold scores**
```
"primary_setup_issue": "alignment_score_too_low"
"interpretation": "Setup is blocked mostly by scores close but below threshold"
```
**Meaning:** Many patterns almost pass  
**Action:** Consider if threshold 0.6 is too strict

#### **Verdict D: Mixed reasons**
```
"primary_setup_issue": "unknown"
"interpretation": "Setup blocks for mixed reasons and needs deeper decomposition"
```
**Meaning:** No single dominant reason  
**Action:** Deeper per-pattern investigation needed

---

## Why This Doesn't Change Strategy

This diagnostic mode:
- ✅ Reads Setup metadata
- ✅ Categorizes rejection reasons
- ✅ Computes statistics
- ✅ Produces verdict

But does **NOT**:
- ❌ Change alignment thresholds
- ❌ Modify SMC detection rules
- ❌ Relax pattern filters
- ❌ Adjust Context/Regime logic

**Purpose:** Understand the bottleneck before deciding if/how to fix it.

---

## Usage

```bash
python scripts/run_validation.py --mode setup_bottleneck \
    --pair BTCUSDT --sample-date 2023-12-15
```

**Output:**
- CLI summary showing block reasons and patterns
- JSON file: `reports/validation/fractal/BTCUSDT_setup_bottleneck.json`

---

## Example Analysis

### **Case Study: 2023-12-15 (Range Market)**

**Results:**
```
No pattern: 47 (100%)
Bullish FVG: 0
Bearish FVG: 0
Mean alignment: 0.30
Threshold: 0.60

Verdict: Setup is blocked mostly by total absence of patterns
```

**Interpretation:**
- Context detects bullish bias
- Regime detects range market
- Setup finds **zero** SMC patterns
- This is **Verdict A** - no patterns exist

**Business Question:**
- Is this expected? (Range markets may lack clear SMC setups)
- Is the detector too strict? (Missing valid patterns?)
- Is this period just unfavorable? (Test other dates)

**Next Steps (separate ticket):**
1. Test Setup on Bull market (2023-03-15)
2. Test Setup on Bear market (2022-06-15)
3. Compare pattern rates across regimes
4. Decide if detector needs tuning

---

## Technical Details

### **Data Source**

Uses corrected fractal geometry:
- 1D: 360 days context
- 4H: 7 days structure
- 1H: 5 days regime
- 15M: 1 day setup

### **Analyzed Bars**

Only bars where `ready_for_decision == True`:
- Context ready: ✅
- Regime ready: ✅
- Setup ready: ✅

Skips warmup bars (first 50).

### **Pattern Detection**

Reads Setup component metadata:
- `has_bullish_fvg`
- `has_bearish_fvg`
- `has_bullish_ob`
- `has_bearish_ob`

### **Alignment Scoring**

Uses Setup component score:
- Default score: 0.3 (when no pattern)
- Actual score: Based on pattern alignment with higher TFs
- Threshold: Configured in `smc_detector.alignment_15m_min`

---

## Limitations

### **What It Can't Tell You**

1. **Why patterns are missing**
   - Need to inspect raw price action
   - May require visual chart review

2. **Whether threshold is "correct"**
   - This is a business decision
   - Requires backtesting multiple thresholds

3. **Pattern quality**
   - Just counts presence/absence
   - Doesn't assess FVG size or OB strength

### **What You Need Next**

- **Pattern quality audit:** Inspect raw patterns on charts
- **Multi-period analysis:** Test Bull/Bear/Range separately
- **Threshold sensitivity:** Test if 0.5 vs 0.6 matters
- **Detector tuning:** Adjust FVG/OB detection rules if needed

---

## Related Documentation

- **FRACTAL_BOTTLENECK.md** - How Setup became the bottleneck
- **FRACTAL_CONTEXT_WINDOWS.md** - Why geometry matters
- **SMC_DETECTOR.md** - How patterns are detected

---

## Summary

**Question:** Why does Setup block 100% of ready decisions?

**Answer:** This diagnostic tells you:
- Is it absence of patterns?
- Is it misalignment?
- Is it threshold strictness?
- Is it directional conflict?

**Then:** You can make informed decisions about whether to:
- Accept the bottleneck (selective strategy)
- Tune the detector (find more patterns)
- Adjust thresholds (pass more patterns)
- Or investigate further

**But:** This mode itself makes no changes - it just reveals the truth.
