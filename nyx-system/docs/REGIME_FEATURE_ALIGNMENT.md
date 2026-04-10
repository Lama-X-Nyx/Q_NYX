# Regime Feature Alignment Fix

## Summary

**Problem:** RegimeAgent → HSMM initialization feature mismatch  
**Cause:** RegimeAgent._prepare_data() did not provide sma_20 and sma_50  
**Impact:** HSMM initialization defaulted everything to Range state  
**Fix:** Added sma_20 and sma_50 to RegimeAgent, made HSMM explicit about requirements  
**Result:** Trend+ probability increased from 5.33% to 71.47% (13.4x improvement)

---

## The Bug

### What Happened

The HSMM Deep Dive (previous ticket) revealed that HSMM was producing 89% Range probability on average, with some periods showing 100% Range despite bullish features (returns_mean = 0.002704).

Code inspection revealed a **feature mismatch**:

**RegimeAgent._prepare_data()** provided:
- ✅ returns
- ✅ atr_14
- ❌ sma_20 (MISSING)
- ❌ sma_50 (MISSING)

**HSMM._label_states_heuristic()** expected:
- returns
- atr_14
- **sma_20** ← Required for trend detection
- **sma_50** ← Required for trend detection

### The Silent Fallback

```python
# src/core/hsmm.py, line 124-126 (BEFORE FIX):
if pd.isna(row.get('sma_20')) or pd.isna(row.get('sma_50')):
    labels.append('Range')  # ← Silent fallback!
    continue
```

**Without sma_20 and sma_50:**
- Every row was labeled 'Range' during initialization
- HSMM learned that Range is the dominant state
- Transition probabilities locked HSMM into Range
- trend_plus became effectively unreachable

**This explained:**
- Why Range dominated (89-100% probability)
- Why threshold tuning didn't help (0% improvement)
- Why features showed directional signal but HSMM ignored it

---

## The Fix

### 1. RegimeAgent._prepare_data()

**Before:**
```python
def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for HSMM (add returns and ATR)"""
    df_prepared = df.copy()
    
    # Add returns
    if 'returns' not in df_prepared.columns:
        df_prepared['returns'] = df_prepared['close'].pct_change()
    
    # Add ATR
    if 'atr_14' not in df_prepared.columns:
        # ... calculate ATR
    
    return df_prepared
```

**After:**
```python
def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for HSMM (add returns, ATR, and moving averages)
    
    CRITICAL: HSMM._label_states_heuristic() requires sma_20 and sma_50
    for proper initialization. Without them, HSMM defaults to Range state,
    causing structural bias in regime detection.
    """
    df_prepared = df.copy()
    
    # Add returns
    if 'returns' not in df_prepared.columns:
        df_prepared['returns'] = df_prepared['close'].pct_change()
    
    # Add ATR
    if 'atr_14' not in df_prepared.columns:
        # ... calculate ATR
    
    # Add SMA_20 (required by HSMM heuristic labeling)
    if 'sma_20' not in df_prepared.columns:
        df_prepared['sma_20'] = df_prepared['close'].rolling(window=20).mean()
    
    # Add SMA_50 (required by HSMM heuristic labeling)
    if 'sma_50' not in df_prepared.columns:
        df_prepared['sma_50'] = df_prepared['close'].rolling(window=50).mean()
    
    return df_prepared
```

### 2. HSMM._label_states_heuristic()

**Made requirements explicit:**

```python
def _label_states_heuristic(self, df: pd.DataFrame) -> pd.Series:
    """
    Simple heuristic state labeling for initialization
    
    Raises:
        ValueError: If required features (sma_20, sma_50) are missing
    """
    
    # CRITICAL: Check for required features
    required_features = ['sma_20', 'sma_50']
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        raise ValueError(
            f"HSMM heuristic labeling requires {required_features}. "
            f"Missing: {missing_features}. "
            f"Ensure RegimeAgent._prepare_data() provides these features. "
            f"Without them, initialization defaults to Range state, "
            f"causing structural bias in regime detection."
        )
    
    # ... rest of labeling logic
```

**Benefits:**
- No more silent fallback
- Clear error message if features missing
- Forces proper feature pipeline alignment

---

## Results

### Before/After Comparison

```
BEFORE FIX:
  Trend+ probability mean: 5.33%
  Range probability mean:  89.33%
  Selected trend+:         0/4 periods
  Verdict:                 "Range state dominance"

AFTER FIX:
  Trend+ probability mean: 71.47%  ← 13.4x increase!
  Range probability mean:  27.24%  ← 3.3x decrease!
  Selected trend+:         3/4 periods
  Verdict:                 "Trend states appear normally"

IMPROVEMENT:
  Trend+ delta:     +66.14 percentage points
  Increase factor:  13.4x
  Conclusion:       Initialization mismatch was THE MAJOR BLOCKER
```

### Period-by-Period

| Period     | Before (Trend+) | After (Trend+) | Improvement |
|------------|-----------------|----------------|-------------|
| 2023-01-15 | 0.82%          | 99.28%         | 121x        |
| 2023-03-15 | 0.00%          | 80.59%         | ∞ (from 0%) |
| 2023-10-15 | 20.52%         | 43.15%         | 2.1x        |
| 2023-12-15 | 0.00%          | 62.84%         | ∞ (from 0%) |

**Most dramatic:**
- 2023-03-15: 0% → 80.59% despite returns_mean = 0.002704 (bullish)
- 2023-01-15: 0.82% → 99.28% (121x improvement)

---

## Why This Matters

### Differentiating Root Causes

**During investigation, we ruled out:**

1. **Bad thresholds** (Regime Sensitivity Tuning)
   - Tested sdc_min: 5.0 → 2.0
   - Tested stability_min: 0.60 → 0.30
   - Result: 0% improvement
   - Conclusion: Thresholds NOT the problem

2. **Bad mapping** (HSMM Deep Dive)
   - HSMM output probabilities were 89-100% Range
   - No mapping logic can fix that
   - Conclusion: Mapping NOT the problem

3. **Bad features** (HSMM Deep Dive)
   - Features showed directional signal (returns_mean positive)
   - But HSMM ignored them
   - Conclusion: Features exist, but...

**The real problem:**
4. **Missing feature pipeline** ← THIS TICKET
   - Features existed in data
   - But weren't passed to HSMM
   - HSMM couldn't see them during initialization
   - Silent fallback contaminated everything

---

## What Changed vs What Didn't

### ✅ What Changed

- RegimeAgent now provides sma_20 and sma_50
- HSMM initialization sees proper trend signals
- trend_plus is now accessible (71% vs 5%)
- Regime layer can express bullish intent

### ❌ What Didn't Change

- Context Agent (untouched)
- SMC patterns (untouched)
- Fractal geometry (untouched)
- Threshold values (sdc_min, stability_min unchanged)
- Strategy logic (no optimization yet)

**This was a pure bug fix**, not an optimization.

---

## Validation

### New Tools Added

**1. regime_feature_check**

```bash
python scripts/run_validation.py --mode regime_feature_check \
    --pair BTCUSDT --sample-date 2023-12-15
```

**Output:**
```
Prepared columns:
  open, high, low, close, volume, returns, atr_14, sma_20, sma_50

Required by HSMM heuristic:
  ✓ sma_20
  ✓ sma_50

Verdict:
  ✅ All required HSMM features are present
```

**Purpose:**
- Validate feature pipeline alignment
- Prevent regression
- Make requirements explicit

**2. Before/After Comparison**

JSON report: `reports/validation/fractal/BTCUSDT_hsmm_before_after.json`

Contains:
- Before metrics (from original HSMM Deep Dive)
- After metrics (after fix)
- Delta and improvement factors
- Period-by-period details

---

## Tests

**New tests (9 total):**

**test_regime_feature_alignment.py (5 tests):**
1. RegimeAgent._prepare_data() adds sma_20 and sma_50
2. No future leak in SMA calculation
3. HSMM detects missing features (raises ValueError)
4. HSMM works with complete features
5. Existing tests remain compatible

**test_regime_feature_check.py (4 tests):**
1. regime_feature_check runs
2. JSON output valid
3. Correctly detects alignment
4. HSMM initialization succeeds

**All tests pass:** 19/19 ✅
- 5 new alignment tests
- 4 new check tests
- 10 existing HSMM deep dive tests

---

## Files Modified

**Modified:**
```
src/agents/regime_agent.py
  - _prepare_data(): Added sma_20 and sma_50 calculation

src/core/hsmm.py
  - _label_states_heuristic(): Added explicit feature checks

scripts/run_validation.py
  - Added regime_feature_check mode
```

**Created:**
```
scripts/regime_feature_check.py
  - New feature alignment validator

tests/test_regime_feature_alignment.py
  - Feature pipeline tests

tests/test_regime_feature_check.py
  - Alignment check tests

docs/REGIME_FEATURE_ALIGNMENT.md
  - This documentation

reports/validation/fractal/BTCUSDT_hsmm_before_after.json
  - Before/after comparison
```

---

## Lessons Learned

### 1. Silent Fallbacks Are Dangerous

The HSMM had a silent fallback to Range when features were missing. This masked the real problem for a long time.

**Lesson:** Make requirements explicit. Fail loudly when critical inputs are missing.

### 2. End-to-End Feature Pipelines Matter

Even when features exist in the data, if they're not passed through the pipeline, downstream components can't see them.

**Lesson:** Validate feature alignment at each pipeline stage.

### 3. Bug vs Tuning vs Calibration

We spent time on:
- Threshold tuning (ruled out)
- Deep diagnostics (identified root cause)
- Bug fix (this ticket)

The real issue was a **bug** (missing features), not a **tuning** problem.

**Lesson:** Fix bugs before optimizing parameters.

### 4. Metrics Validate Hypotheses

The 13.4x improvement proves the hypothesis:
- Before: 5.33% trend+ → structural bias
- After: 71.47% trend+ → normal distribution

The delta is so large it can't be noise.

---

## Next Steps

Now that Regime is fixed:

### ✅ Completed Investigation Path

1. ✅ Regime Sensitivity Tuning → Ruled out thresholds
2. ✅ HSMM Deep Dive → Identified range dominance
3. ✅ Feature Alignment Fix → **Solved the root cause**

### 🎯 What's Unlocked

With Regime working properly:

**Now possible:**
- Setup tuning (can finally validate Setup quality)
- SMC threshold optimization
- Full fractal pipeline validation
- Bullish setup detection

**Previously blocked:**
- Setup couldn't be evaluated (Regime blocked everything)
- SMC patterns couldn't trigger (no bullish Regime)
- Fractal pipeline couldn't express bullish intent

### 📋 Recommended Next Tickets

**Option A: Validate the full pipeline**
Run complete fractal validation to ensure:
- Context → Regime → Setup flow works
- Bullish setups can now be detected
- No other bottlenecks exist

**Option B: Setup optimization**
Now that Regime works:
- Tune Setup thresholds
- Validate Setup quality
- Optimize SMC pattern detection

**Option C: Backtest with fix**
Rerun historical validation:
- See if bullish trades now appear
- Measure impact on P&L
- Validate in production conditions

---

## Conclusion

**Question:**
> Was the sma_20/sma_50 mismatch the real blocker?

**Answer:**
> **YES.** Unequivocally.

**Evidence:**
- Trend+ probability: 5.33% → 71.47% (13.4x)
- Selected trend+: 0/4 → 3/4 periods
- Some periods went from 0% → 80%+

**This was not incremental improvement. This was unlocking a locked system.**

The feature mismatch caused HSMM initialization to label everything as Range, which locked the model into Range-dominated behavior. Once the features were provided, trend states became accessible.

**One sentence summary:**

> *Adding sma_20 and sma_50 to RegimeAgent._prepare_data() fixed a feature pipeline mismatch that was causing HSMM to initialize with 100% Range labels, unlocking trend detection and increasing trend_plus probability from 5.33% to 71.47% (13.4x improvement).*

---

**The blocker is removed. Regime is fixed. The pipeline is open.** ✅
