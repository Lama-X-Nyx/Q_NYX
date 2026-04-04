# Regime Sensitivity Tuning

## Purpose

**Objective:** Tune the Regime layer only to determine if bullish expression can be unlocked by relaxing HSMM thresholds, without touching Context, fractal geometry, cache logic, or SMC patterns.

**Problem Statement:**
Diagnostics (bullish_audit) showed that Regime collapses to 'range' on 100% of BTC periods tested, even during known bullish periods. This prevents the fractal pipeline from ever expressing bullish intent, regardless of Context state.

**Hypothesis:**
The Regime layer's thresholds (sdc_min, stability_min) are too conservative, causing legitimate bullish trends to be classified as range-bound markets.

---

## Why Regime First?

The fractal pipeline follows a strict hierarchy:

```
Context (1D) → Regime (1H) → Setup (15M)
```

**Current state:**
- ✅ Context works reasonably well (detects bullish/bearish/neutral)
- ❌ Regime blocks all bullish expression (collapses to range)
- ⚠️ Setup cannot be properly evaluated until Regime is fixed

**Tuning priority:**
1. **Regime first** ← This ticket
2. **Setup second** (after Regime unlocked)
3. **SMC threshold tuning** (final polish)

Tuning Setup before Regime would be premature optimization - you cannot validate Setup's quality when Regime never lets bullish signals through.

---

## Constraints

This ticket is **strictly limited** to Regime tuning:

❌ **Do NOT change:**
- Context Agent logic
- Fractal geometry
- Cache logic
- SMC patterns
- Risk management

✅ **Only tune:**
- `sdc_min` (Score de Confiance minimum)
- `stability_min` (Stability threshold)

---

## How It Works

### **Tuning Grid**

The script tests 4 scenarios:

1. **Baseline (current)**
   - `sdc_min: 5.0`
   - `stability_min: 0.60`
   - Purpose: Establish current behavior

2. **Slightly Relaxed**
   - `sdc_min: 4.0`
   - `stability_min: 0.50`
   - Purpose: Test mild relaxation

3. **Moderately Relaxed**
   - `sdc_min: 3.0`
   - `stability_min: 0.40`
   - Purpose: Test moderate relaxation

4. **Aggressive**
   - `sdc_min: 2.0`
   - `stability_min: 0.30`
   - Purpose: Test if only aggressive relaxation helps

### **Test Periods**

Default test periods (can be customized with `--dates`):
- 2023-01-15 (Q1 - early year)
- 2023-03-15 (Q1 - mid quarter)
- 2023-10-15 (Q4 - autumn)
- 2023-12-15 (Q4 - end year)

These periods are chosen to represent different market conditions across the year.

### **Metrics Tracked**

For each scenario and period:
- `state`: Regime state (trend_plus, range, trend_minus)
- `score`: Normalized confidence (0-1)
- `sdc`: Score de Confiance (HSMM state probability × 10)
- `stability`: State persistence probability
- `trend_plus`: Boolean flag for bullish detection

**Aggregate metrics:**
- `trend_plus_rate`: % of periods classified as trend_plus
- `range_rate`: % of periods classified as range

---

## Usage

### **Basic**

```bash
python scripts/run_validation.py --mode regime_tuning --pair BTCUSDT
```

### **Custom Dates**

```bash
python scripts/run_validation.py --mode regime_tuning \
    --pair BTCUSDT \
    --dates 2023-01-15,2023-03-15,2023-06-15,2023-09-15,2023-12-15
```

### **Output**

JSON report saved to:
```
reports/validation/fractal/BTCUSDT_regime_tuning.json
```

---

## Reading Results

### **Scenario Summary**

```
Baseline (current)        (current)
  Trend+ rate:   0.0%
  Range rate:  100.0%

Slightly Relaxed          
  Trend+ rate:  25.0%
  Range rate:   75.0%

Moderately Relaxed        ⭐ BEST
  Trend+ rate:  50.0%
  Range rate:   50.0%

Aggressive                
  Trend+ rate: 100.0%
  Range rate:    0.0%
```

### **Interpreting Results**

**Too Conservative:**
```
Trend+ rate:  0-10%
Range rate:  90-100%
```
→ Regime blocks all bullish expression
→ Thresholds are too strict

**Best Compromise:**
```
Trend+ rate:  30-60%
Range rate:   40-70%
```
→ Regime expresses trends without over-detecting
→ Balanced sensitivity

**Over-Relaxed:**
```
Trend+ rate:  90-100%
Range rate:    0-10%
```
→ Regime detects trends everywhere
→ Thresholds too loose, unreliable

---

## Verdict Categories

The script produces one of four verdicts:

### **1. Regime remains too conservative**

**Meaning:** Best scenario improves trend_plus_rate by < 10%

**Implication:** Tuning sdc_min/stability_min alone is insufficient

**Next steps:**
- Investigate HSMM implementation
- Check Context-Regime alignment
- Consider deeper structural changes

---

### **2. Regime becomes usable with mild relaxation**

**Meaning:** Best scenario improves trend_plus_rate by 10-30%

**Implication:** Modest relaxation unlocks bullish expression

**Next steps:**
- Deploy recommended scenario
- Monitor for over-detection
- Proceed to Setup tuning

---

### **3. Regime only improves with aggressive relaxation**

**Meaning:** Best scenario improves trend_plus_rate by 30-60%

**Implication:** Significant relaxation needed, but risky

**Next steps:**
- Validate on broader dataset
- Check for false positives
- Consider if risk is acceptable

---

### **4. Regime tuning alone is not enough**

**Meaning:** All scenarios produce over-aggressive trend detection (>95%)

**Implication:** Problem is deeper than threshold tuning

**Next steps:**
- Investigate HSMM training/configuration
- Check feature engineering
- Consider alternative regime detection methods

---

## Example Output

### **CLI Summary**

```
REGIME SENSITIVITY TUNING - BTCUSDT

Test periods: 4
  - 2023-01-15
  - 2023-03-15
  - 2023-10-15
  - 2023-12-15

Scenarios to test: 4
  - Baseline (current)
  - Slightly Relaxed
  - Moderately Relaxed
  - Aggressive

RUNNING SCENARIOS

Testing Baseline (current)...
  Trend+: 0.0%
  Range:  100.0%

Testing Slightly Relaxed...
  Trend+: 0.0%
  Range:  100.0%

Testing Moderately Relaxed...
  Trend+: 0.0%
  Range:  100.0%

Testing Aggressive...
  Trend+: 0.0%
  Range:  100.0%

VERDICT

Conclusion: Regime remains too conservative

Best scenario improves trend_plus_rate by only 0.0%

Recommendation:
  Regime tuning alone insufficient - investigate Context alignment or deeper HSMM issues
```

### **JSON Structure**

```json
{
  "pair": "BTCUSDT",
  "test_dates": ["2023-01-15T00:00:00", ...],
  "scenarios": [
    {
      "scenario": "baseline",
      "label": "Baseline (current)",
      "description": "Current production settings - sdc_min=5.0, stability_min=0.6",
      "params": {
        "sdc_min": 5.0,
        "stability_min": 0.6
      },
      "results": [
        {
          "date": "2023-01-15T00:00:00",
          "state": "range",
          "score": 0.9836,
          "passed": true,
          "ready": true,
          "metadata": {
            "sdc": 9.8363,
            "stability": 0.75,
            "trend_plus": false
          }
        }
      ],
      "summary": {
        "total_periods": 4,
        "trend_plus_count": 0,
        "range_count": 4,
        "trend_plus_rate": 0.0,
        "range_rate": 100.0
      }
    }
  ],
  "verdict": {
    "conclusion": "Regime remains too conservative",
    "message": "Best scenario improves trend_plus_rate by only 0.0%",
    "best_scenario": "baseline",
    "recommendation": "Regime tuning alone insufficient - investigate Context alignment or deeper HSMM issues"
  }
}
```

---

## What We Learn

This ticket answers:

1. **Can Regime be fixed by threshold tuning?**
   - Yes → Deploy recommended scenario
   - No → Investigate deeper issues

2. **Is the problem in sdc_min or stability_min?**
   - Compare scenarios to isolate which parameter matters

3. **How aggressive must relaxation be?**
   - Mild, moderate, or aggressive?
   - Helps assess risk/reward

4. **Is Regime the real bottleneck?**
   - If tuning doesn't help → problem is elsewhere
   - If tuning helps → proceed to Setup tuning

---

## Important Notes

### **This Does NOT Change:**

- Context Agent logic
- Fractal hierarchy
- Cache behavior
- SMC pattern detection
- Risk management

### **This ONLY Tunes:**

- HSMM threshold sensitivity
- State classification boundaries

### **After This Ticket:**

If Regime tuning succeeds:
→ Proceed to **Setup Tuning**

If Regime tuning fails:
→ Investigate **HSMM implementation** or **Context-Regime interaction**

---

## Testing

Run tests:
```bash
python tests/test_regime_tuning.py
```

**Tests cover:**
- Script runs without errors
- JSON output is valid
- Multiple scenarios are tested
- Verdict is produced
- No crash on edge cases

---

## Files

**Script:**
- `scripts/regime_tuning.py`

**Tests:**
- `tests/test_regime_tuning.py`

**Documentation:**
- `docs/REGIME_TUNING.md` (this file)

**Output:**
- `reports/validation/fractal/BTCUSDT_regime_tuning.json`

---

## Summary

**Question:**
> Can Regime's conservatism be fixed by tuning sdc_min and stability_min?

**Method:**
> Test 4 scenarios (baseline → aggressive) on multiple periods

**Answer:**
> One of four verdicts:
> 1. Remains too conservative (tuning insufficient)
> 2. Usable with mild relaxation (success)
> 3. Requires aggressive relaxation (risky)
> 4. Tuning alone not enough (deeper issues)

**Next Ticket:**
> If successful → **Setup Tuning**
> If unsuccessful → **HSMM Deep Dive**

---

**The goal is NOT to force a solution, but to learn where the real problem lies.**
