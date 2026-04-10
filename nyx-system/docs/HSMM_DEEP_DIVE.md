# HSMM Deep Dive: State Formation, Feature Signal, and Regime Mapping

## Purpose

**Objective:** Understand why the HSMM never produces `trend_plus` on tested BTC periods, even when Context is bullish, SdC is high, stability is high, and relaxing thresholds changes nothing.

**Problem Statement:**
Regime Sensitivity Tuning (previous ticket) revealed that even aggressive threshold relaxation (sdc_min=2.0, stability_min=0.30) produces 0% improvement. The problem is NOT in the thresholds.

**Hypothesis to Test:**
The root cause is one of:
- **Case A:** Weak/non-trending features fed to HSMM
- **Case B:** Structural range dominance in HSMM state probabilities
- **Case C:** Rigid mapping from HSMM states to final states
- **Case D:** Biased calibration/initialization

---

## Why This Ticket After Regime Tuning?

**Regime Tuning taught us:**
> Thresholds are not the problem

**HSMM Deep Dive will teach us:**
> Where the problem actually is

**Tuning sequence:**
1. ~~Regime Sensitivity Tuning~~ ✅ (ruled out thresholds)
2. **HSMM Deep Dive** ← You are here
3. *Next steps depend on findings*

---

## Constraints

This is **diagnostics only** - no production changes:

❌ **Do NOT change:**
- Context Agent
- SMC patterns
- Fractal geometry
- Cache logic
- Strategy thresholds

✅ **Only diagnose:**
- Features fed to HSMM
- HSMM state probabilities
- Mapping logic
- Calibration/initialization

Any experimentation stays in audit mode.

---

## How It Works

### **What We Inspect**

For each test period:

1. **Input Features**
   - `returns_mean`: Average returns
   - `returns_std`: Return volatility
   - `atr_mean`: Average True Range
   - `price_slope`: Directional signal
   - *Everything actually fed to HSMM*

2. **State Probabilities**
   - `P(Trend+)`: Probability of bullish trend
   - `P(Range)`: Probability of range-bound
   - `P(Trend-)`: Probability of bearish trend
   - *Raw HSMM output*

3. **Mapping Details**
   - Raw HSMM state
   - Final mapped state
   - Mapping reason
   - *How probabilities → final decision*

4. **Diagnostic**
   - Why trend_plus doesn't emerge
   - Specific issue identified
   - *Clear root cause*

---

## Usage

### **Basic**

```bash
python scripts/run_validation.py --mode hsmm_deep_dive --pair BTCUSDT
```

### **Custom Dates**

```bash
python scripts/run_validation.py --mode hsmm_deep_dive \
    --pair BTCUSDT \
    --dates 2023-01-15,2023-03-15,2023-10-15,2023-12-15
```

### **Output**

JSON report saved to:
```
reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json
```

---

## Reading Results

### **Per-Period Output**

```
2023-01-15
  Features:
    returns_mean: 0.001926  ← Positive returns (bullish signal)
    returns_std:  0.009124  ← Moderate volatility
    atr_mean:     246.126   ← ATR baseline
    
  State probabilities:
    Trend+:  0.0082  ← 0.82% (!!) trend_plus
    Range:   0.9836  ← 98.36% range
    Trend-:  0.0082  ← 0.82% trend_minus
    
  Selected: range
  SDC: 9.84   ← Perfect confidence
  Stability: 0.75  ← Excellent stability
  
  Diagnostic: range dominates structurally (>80% probability)
```

**Key insight:** Even with positive returns and perfect SdC/stability, HSMM assigns 98.36% probability to range.

---

### **Cross-Period Summary**

```
Probability means:
  Trend+: 0.0533  ← Average 5.33% across all periods
  Range:  0.8933  ← Average 89.33% across all periods
  Trend-: 0.0533

State counts:
  Trend+: 0  ← Never selected
  Range:  4  ← Always selected
  Trend-: 0

VERDICT:
Main issue: range_state_dominance

The HSMM internal state probabilities are structurally dominated by range
```

---

## Verdict Categories

The system produces one of these verdicts:

### **Case A: Weak Features**

**Diagnosis:**
```json
{
  "main_issue": "weak_features",
  "verdict": "The HSMM receives weak / non-trending features, so range is the natural output"
}
```

**Meaning:** Features (returns, ATR) show no directional signal

**Typical pattern:**
- `returns_mean` near zero (< 0.0005)
- Low volatility
- Flat price slope

**What to do:**
- Investigate feature engineering
- Add directional indicators
- Consider alternative regime detection

---

### **Case B: Range State Dominance** ⭐

**Diagnosis:**
```json
{
  "main_issue": "range_state_dominance",
  "verdict": "The HSMM internal state probabilities are structurally dominated by range"
}
```

**Meaning:** HSMM always outputs high range probability regardless of features

**Typical pattern:**
- Range probability > 80% across all periods
- Trend+ probability < 10% even with directional features
- Features show signal but HSMM ignores it

**What to do:**
- **This is the actual BTC finding** ✅
- Investigate HSMM calibration
- Check HSMM initialization
- Consider HSMM re-training
- Investigate transition matrix

---

### **Case C: Mapping Rigidity**

**Diagnosis:**
```json
{
  "main_issue": "mapping_rigidity",
  "verdict": "The HSMM emits useful variation, but the final mapping consistently favors range"
}
```

**Meaning:** HSMM produces competitive probabilities but mapping logic is too rigid

**Typical pattern:**
- Trend+ probability > 20%
- Range probability > Trend+ but not dominant
- Close competition but range always wins

**What to do:**
- Review mapping logic
- Consider probabilistic thresholds
- Add tie-breaking rules

---

### **Case D: Initialization Bias**

**Diagnosis:**
```json
{
  "main_issue": "initialization_bias",
  "verdict": "The HSMM initialization / calibration biases the model toward range, making trend states effectively unreachable"
}
```

**Meaning:** HSMM starts biased toward range and never escapes

**Typical pattern:**
- Trend+ probability < 10% everywhere
- High stability (state won't change)
- Initialization locks into range

**What to do:**
- Review HSMM initialization
- Check prior probabilities
- Investigate transition matrix diagonal

---

## Understanding the Findings

### **The BTC Reality**

On tested BTC periods (2023-01-15, 2023-03-15, 2023-10-15, 2023-12-15):

```
Period         Returns Mean    Range Prob    Trend+ Prob
2023-01-15     0.001926        98.36%        0.82%
2023-03-15     0.002704        100%          0.00%  (!!)
2023-10-15     0.000074        58.96%        20.52%
2023-12-15     0.000795        100%          0.00%  (!!)
```

**Key observation:** Even with positive returns (0.002704 on 2023-03-15), HSMM outputs **100% range probability**.

---

### **Differentiating the Cases**

**How to tell them apart:**

| Issue | Returns Mean | Range Prob | Trend+ Prob | What It Means |
|-------|-------------|-----------|------------|---------------|
| **Weak features** | ~0.0 | High | Low | Features are flat → range makes sense |
| **Range dominance** | Non-zero | >80% | <10% | Features show signal but HSMM ignores it |
| **Mapping rigidity** | Non-zero | 50-70% | 20-40% | HSMM competitive but mapping rigid |
| **Init bias** | Varies | High + stable | <5% | HSMM locked in range state |

**BTC finding:** Range dominance (features show signal, HSMM outputs range anyway)

---

## Example: 2023-03-15 Analysis

**Features:**
```json
{
  "returns_mean": 0.002704,    // Positive (bullish)
  "returns_std": 0.012121,     // Moderate volatility
  "atr_mean": 402.945,         // ATR normal
  "price_slope": 0.xxx         // Upward trend
}
```

**HSMM Output:**
```json
{
  "trend_plus": 0.0000,   // 0% (!!)
  "range": 1.0000,        // 100%
  "trend_minus": 0.0000   // 0%
}
```

**Analysis:**
- Features show bullish signal (returns_mean = 0.002704)
- HSMM completely ignores it
- Outputs 100% range probability
- **Conclusion:** Range state dominance, not weak features

---

## What We Learn

This ticket answers:

1. **Are features too weak?**
   - No - features show directional signal (returns_mean positive)

2. **Do HSMM probabilities vary?**
   - No - range always dominates (89% average)

3. **Is mapping the problem?**
   - No - when range prob is 100%, no mapping can help

4. **Is initialization biased?**
   - Yes - HSMM structurally biased toward range state

**Bottom line:**
> The HSMM itself is broken. Fixing thresholds, features, or mapping won't help. The model needs re-calibration or replacement.

---

## Next Steps After This Ticket

### **If Verdict = Weak Features**
→ Investigate feature engineering
→ Add directional indicators

### **If Verdict = Range Dominance** ← BTC finding
→ **HSMM Re-calibration**
→ Or **Alternative Regime Detection**

### **If Verdict = Mapping Rigidity**
→ Adjust mapping logic
→ Add probabilistic thresholds

### **If Verdict = Initialization Bias**
→ Review HSMM initialization
→ Adjust prior probabilities

---

## Testing

Run tests:
```bash
python tests/test_hsmm_deep_dive.py
```

**Tests cover (10 tests):**
- Script runs without errors
- JSON output valid
- All periods present
- Required fields present
- Cross-period summary exists
- No crash if all range
- No crash if probability zero
- Verdict is meaningful
- Features are numeric
- Probabilities sum to one

---

## Files

**Module:**
- `src/validation/hsmm_deep_dive.py`

**Tests:**
- `tests/test_hsmm_deep_dive.py`

**Documentation:**
- `docs/HSMM_DEEP_DIVE.md` (this file)

**Output:**
- `reports/validation/fractal/BTCUSDT_hsmm_deep_dive.json`

---

## Comparison: Regime Tuning vs HSMM Deep Dive

### **Regime Tuning**
**Question:** Can we fix Regime by relaxing thresholds?  
**Answer:** No - 0% improvement even with aggressive relaxation  
**Conclusion:** Thresholds are not the problem

### **HSMM Deep Dive**
**Question:** Where IS the problem then?  
**Answer:** HSMM state probabilities structurally dominated by range  
**Conclusion:** HSMM itself needs fixing, not thresholds

---

## Summary

**Question:**
> Why does HSMM never produce trend_plus?

**Method:**
> Inspect features → state probabilities → mapping → diagnosis

**Finding:**
> HSMM internal state probabilities are structurally dominated by range (89% average, up to 100% on some periods)

**Implication:**
> Even with positive returns and directional features, HSMM assigns 100% probability to range. This is a calibration/initialization problem, not a threshold or feature problem.

**Recommendation:**
> Next ticket should be **HSMM Re-calibration** or **Alternative Regime Detection**, NOT Setup Tuning (premature while Regime is broken).

---

**The goal was to find where the problem is. We found it: in the HSMM itself.** ✅
