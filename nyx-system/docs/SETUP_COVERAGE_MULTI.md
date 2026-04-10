# Multi-Period Setup Coverage Audit

## Purpose

This diagnostic mode answers a critical question:

> **Is Setup silence specific to range markets... or a global detector problem?**

After discovering that Setup blocks 100% of decisions on 2023-12-15 (range market) due to "no patterns", we need to determine if this behavior:

**A.** Is specific to quiet/range periods (expected behavior)  
**B.** Reflects a globally over-silent SMC detector (problem)  
**C.** Is due to alignment issues (patterns exist but get filtered)  
**D.** Varies unpredictably across regimes (needs deeper analysis)

---

## What It Does

Runs `setup_bottleneck` diagnostic on multiple market periods and compares results:

**Default test periods:**
- **2022-06-15:** Bear/stress market
- **2023-03-15:** Bull market
- **2023-12-15:** Range/quiet market

For each period, it measures:
- Setup pass rate
- Pattern detection rate (FVG/OB counts)
- Dominant block reason (no_pattern vs misaligned_pattern)
- Alignment score distribution

Then produces a **cross-period verdict** to determine if silence is regime-dependent or global.

---

## Usage

### **Basic Usage (default periods)**
```bash
python scripts/run_validation.py --mode setup_coverage_multi \
    --pair BTCUSDT
```

### **Custom Periods**
```bash
python scripts/run_validation.py --mode setup_coverage_multi \
    --pair BTCUSDT --dates 2022-06-15,2023-03-15,2023-12-15
```

**Output:**
- CLI summary comparing all periods
- JSON file: `reports/validation/fractal/BTCUSDT_setup_coverage_multi.json`

---

## How to Read Results

### **CLI Output**

```
MULTI-PERIOD SETUP COVERAGE - BTCUSDT

2022-06-15 (BEAR)
  Ready bars:              47
  Setup pass rate:       12.8%
  Primary issue:     no_pattern
  Patterns: BFVG 3 / BeFVG 11 / BOB 1 / BeOB 4

2023-03-15 (BULL)
  Ready bars:              47
  Setup pass rate:       27.7%
  Primary issue: misaligned_pattern
  Patterns: BFVG 15 / BeFVG 2 / BOB 3 / BeOB 0

2023-12-15 (RANGE)
  Ready bars:              47
  Setup pass rate:        0.0%
  Primary issue:     no_pattern
  Patterns: 0 / 0 / 0 / 0

Cross-period conclusion:
Detector silence is regime-dependent and strongest in range markets.
```

**Interpretation:**
- Bull market: Higher pass rate, more bullish patterns
- Bear market: Lower pass rate, more bearish patterns
- Range market: Zero patterns, complete silence
- **Verdict:** Silence is regime-dependent (expected behavior)

---

### **Metrics Explained**

#### **setup_pass_rate**
Percentage of ready bars where Setup approves a trade.

- **0-5%:** Very low coverage (detector finds almost nothing)
- **5-20%:** Moderate coverage (some patterns found but most rejected)
- **20%+:** Good coverage (patterns frequently detected and aligned)

#### **Primary Issue**

**no_pattern:**
- Detector finds no FVG or OB patterns
- Could be: patterns don't exist, or detector too strict

**misaligned_pattern:**
- Patterns detected but conflict with higher timeframes
- Could be: alignment threshold too strict, or valid filtering

**alignment_score_too_low:**
- Patterns close to threshold but don't pass
- Could be: threshold needs tuning

#### **Pattern Observed**

- **BFVG:** Bullish Fair Value Gap
- **BeFVG:** Bearish Fair Value Gap
- **BOB:** Bullish Order Block
- **BeOB:** Bearish Order Block

High counts → Detector is working  
Zero counts → Detector finds nothing

---

## Cross-Period Verdict

The system automatically produces one of 4 verdicts:

### **Verdict A: Regime-Dependent Silence**
```
"Detector silence is regime-dependent and strongest in range markets"
```

**Meaning:**
- Bull/bear markets: Some patterns detected
- Range market: Very low or zero patterns
- **Interpretation:** Expected behavior - SMC thrives in trending markets

**Action:** Accept this as feature, not bug

---

### **Verdict B: Globally Too Silent**
```
"Detector is globally too silent across all tested regimes"
```

**Meaning:**
- ALL periods: Very low pattern coverage
- Bull/bear/range: Consistently near zero
- **Interpretation:** Detector is broken or too strict

**Action:** Investigate SMC detector parameters urgently

---

### **Verdict C: Alignment is the Blocker**
```
"Patterns exist across periods, but alignment is the dominant blocker"
```

**Meaning:**
- Patterns detected across regimes
- But most rejected due to alignment
- **Interpretation:** Threshold may be too strict

**Action:** Review alignment_15m_min threshold (currently 0.6)

---

### **Verdict D: Mixed Behavior**
```
"Setup behavior is mixed and needs deeper decomposition"
```

**Meaning:**
- No clear pattern across regimes
- Different issues in different periods
- **Interpretation:** Needs per-period investigation

**Action:** Inspect each period individually

---

## Example Comparison

### **Scenario 1: Healthy Detector (Verdict A)**

```
Bear:  15% pass, 8 bearish patterns, primary: misaligned
Bull:  30% pass, 18 bullish patterns, primary: alignment_too_low
Range:  0% pass, 0 patterns, primary: no_pattern

Verdict: Detector silence is regime-dependent
```

**Interpretation:** Detector works well in trending markets, silent in range. This is **expected**.

---

### **Scenario 2: Broken Detector (Verdict B)**

```
Bear:  0% pass, 0 patterns, primary: no_pattern
Bull:  0% pass, 0 patterns, primary: no_pattern
Range: 0% pass, 0 patterns, primary: no_pattern

Verdict: Detector is globally too silent
```

**Interpretation:** Detector finds nothing anywhere. This is **broken**.

---

### **Scenario 3: Strict Alignment (Verdict C)**

```
Bear:  8% pass, 25 patterns, primary: misaligned_pattern
Bull: 12% pass, 30 patterns, primary: alignment_score_too_low
Range: 5% pass, 10 patterns, primary: misaligned_pattern

Verdict: Patterns exist but alignment is the blocker
```

**Interpretation:** Patterns detected but most rejected. **Threshold may be too strict**.

---

## JSON Structure

```json
{
  "pair": "BTCUSDT",
  "periods": {
    "2022-06-15": {
      "regime_label": "bear",
      "bars_ready_for_decision": 47,
      "setup_pass_rate": 12.8,
      "block_reasons": {...},
      "pattern_observed": {...},
      "primary_setup_issue": "no_pattern",
      "sample_blocked_setups": [...]
    },
    ...
  },
  "cross_period_summary": {
    "average_setup_pass_rate": 13.5,
    "bull_behavior": "moderate pattern presence",
    "bear_behavior": "some bearish pattern presence",
    "range_behavior": "very low pattern coverage",
    "global_interpretation": "Detector silence is regime-dependent..."
  }
}
```

---

## Why This Matters

**Before multi-period test:**
- "Setup blocks 100% on 2023-12-15"
- Unknown if this is normal or broken

**After multi-period test:**
- **If Verdict A:** Silence is expected in range → Accept behavior
- **If Verdict B:** Silence is global → Fix detector urgently
- **If Verdict C:** Patterns exist but filtered → Consider threshold tuning
- **If Verdict D:** Behavior unclear → Investigate per-period

---

## Limitations

### **What This Can't Tell You**

1. **Why patterns are missing**
   - Visual chart inspection needed
   - May require manual pattern marking

2. **If detected patterns are "good"**
   - Just counts presence/absence
   - Doesn't assess pattern quality

3. **Optimal threshold value**
   - Would require sensitivity analysis
   - Separate optimization ticket

### **What You Need Next**

**If Verdict B (globally silent):**
- Inspect SMC detector code
- Check FVG/OB detection parameters
- Visual chart review to see if patterns exist

**If Verdict A (regime-dependent):**
- Accept as expected behavior
- Document that SMC is selective
- Consider alternative patterns for range

**If Verdict C (alignment blocker):**
- Run sensitivity analysis on alignment threshold
- Test if 0.5 vs 0.6 makes meaningful difference

---

## Related Documentation

- **SETUP_BOTTLENECK.md** - Single-period diagnosis
- **FRACTAL_BOTTLENECK.md** - Why Setup is the bottleneck
- **FRACTAL_CONTEXT_WINDOWS.md** - Geometry foundation

---

## Technical Details

### **Data Source**
Uses corrected fractal geometry for each period:
- 1D: 360 days context
- 4H: 7 days structure
- 1H: 5 days regime
- 15M: 1 day setup

### **Analyzed Bars**
Each period analyzes ~47 ready bars (after warmup).

### **Pattern Detection**
Same SMC detector used for all periods:
- FVG detection: Gap between candles
- OB detection: Strong rejection zones
- Alignment scoring: Pattern vs higher TF context

---

## Example Run

```bash
$ python scripts/run_validation.py --mode setup_coverage_multi --pair BTCUSDT

MULTI-PERIOD SETUP COVERAGE - BTCUSDT

Testing 3 periods:
  - 2022-06-15 (bear)
  - 2023-03-15 (bull)
  - 2023-12-15 (range)

[Runs setup_bottleneck for each period...]

CROSS-PERIOD CONCLUSION
Average setup pass rate: 0.0%

Bull behavior:  very low pattern coverage
Bear behavior:  very low pattern coverage
Range behavior: very low pattern coverage

Global interpretation:
Detector is globally too silent across all tested regimes

✓ Multi-period report saved
```

**Interpretation:** **Verdict B** - Detector is broken globally. Urgent investigation needed.

---

## Summary

**Question:** Is Setup silence specific to range or a global problem?

**Answer:** This diagnostic tells you one of:
- **Cas A:** Normal - silence is regime-dependent
- **Cas B:** Problem - detector globally silent
- **Cas C:** Tuning - patterns exist but alignment filters
- **Cas D:** Unclear - needs deeper analysis

**Then:** You can make informed decisions about:
- Accepting selective behavior
- Fixing broken detector
- Tuning alignment threshold
- Or investigating further

**But:** This mode itself makes no changes - it just reveals the truth across regimes.
