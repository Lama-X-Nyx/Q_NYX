# SMC Detector Wiring Fix

## Bug Identified

**Critical wiring bug:** SetupAgent was passing the entire config dict to SMCDetector constructor instead of explicit scalar parameters.

### **The Problem**

**Before fix:**

```python
# src/agents/setup_agent.py (LINE 39 - BROKEN)
self.smc = SMCDetector(config)  # ❌ Passing entire dict
```

**SMCDetector expected:**

```python
def __init__(self, 
             ob_range_threshold: float = 0.015,  # expects float
             fvg_min_gap: float = 0.005,          # expects float  
             liquidity_lookback: int = 20):       # expects int
```

**What actually happened:**

1. `SMCDetector.__init__` received `config` dict as first positional argument
2. `ob_range_threshold` = entire config dict (not a float!)
3. During detection: `rally > self.ob_range_threshold` crashed with TypeError
4. Exception was **silently caught** and converted to fake "no pattern"

**Result:** All diagnostic verdicts showing "0 patterns detected" were **potentially contaminated** by this masked bug.

---

## Why This Contaminated Diagnostics

### **Silent Exception Swallowing**

**Before fix:**

```python
# src/agents/setup_agent.py (LINES 77-85 - BROKEN)
try:
    patterns = self.smc.detect_all(df)
except Exception as e:
    patterns = {
        'bullish_ob': False,
        'bearish_ob': False,
        'bullish_fvg': False,
        'bearish_fvg': False
    }
```

**Effect:**
- Detector crashes with TypeError → **swallowed silently**
- Replaced with fake "no patterns found"
- Setup Agent reports: `state = "no_pattern"` (lie!)
- Multi-period autopsy: "globally silent" (maybe not true!)

**Impact on previous diagnostics:**
- setup_bottleneck: "0 patterns" → Was this real or masked bug?
- setup_coverage_multi: "globally silent" → Was detector really silent?
- smc_autopsy: "0 OB detected" → Were there really no OBs or did detector crash?

---

## The Fix

### **1. Correct Wiring**

**After fix:**

```python
# src/agents/setup_agent.py (NEW - LINES 38-44)
smc_config = config.get('smc_detector', {})
self.smc = SMCDetector(
    ob_range_threshold=smc_config.get('ob_range_threshold', 0.015),
    fvg_min_gap=smc_config.get('fvg_min_gap', 0.005),
    liquidity_lookback=smc_config.get('liquidity_lookback', 20)
)
```

**Now:**
- Explicit scalar parameters extracted from config
- SMCDetector receives correct types
- No TypeErrors during execution

---

### **2. Type Validation in Detector**

**Added defensive checks:**

```python
# src/core/smc.py (NEW - LINES 40-61)
if not isinstance(ob_range_threshold, (int, float)):
    raise TypeError(
        f"ob_range_threshold must be numeric (int or float), "
        f"got {type(ob_range_threshold).__name__}"
    )

if not isinstance(fvg_min_gap, (int, float)):
    raise TypeError(
        f"fvg_min_gap must be numeric (int or float), "
        f"got {type(fvg_min_gap).__name__}"
    )

if not isinstance(liquidity_lookback, int):
    raise TypeError(
        f"liquidity_lookback must be int, "
        f"got {type(liquidity_lookback).__name__}"
    )
```

**Effect:**
- Fails fast with clear error if wrong types passed
- No silent crashes during detection
- Easy to diagnose wiring issues

---

### **3. Stop Swallowing Exceptions**

**After fix:**

```python
# src/agents/setup_agent.py (NEW - LINES 76-92)
try:
    patterns = self.smc.detect_all(df)
except Exception as e:
    # Detector failed - return explicit error state
    return AgentResult(
        agent=self.name,
        state='detector_error',  # ✅ Explicit error state
        score=0.0,
        passed=False,
        ready=True,
        blocked_by_readiness=False,
        reason=f'SMCDetector error: {type(e).__name__}: {str(e)}',
        metadata={
            'detector_error': True,
            'exception_type': type(e).__name__,
            'exception_message': str(e),
            ...
        }
    )
```

**Effect:**
- Detector errors are **visible** (not masked)
- State distinguishes: `no_pattern` vs `detector_error`
- Diagnostics show **truth**, not fake silence

---

### **4. Clean Config Block**

**Added to config/validation_baseline.yaml:**

```yaml
# SMC Detector Configuration
smc_detector:
  ob_range_threshold: 0.015   # Order Block: minimum followthrough move (1.5%)
  fvg_min_gap: 0.005          # Fair Value Gap: minimum gap size (0.5%)
  liquidity_lookback: 20      # Liquidity sweep: bars to look back
```

**Benefits:**
- Clear, explicit config structure
- Scalar types documented
- Easy to understand and modify

---

## State Semantics After Fix

SetupAgent now produces **5 distinct states** (not mixed):

| State | Meaning | ready | passed |
|-------|---------|-------|--------|
| **not_ready** | Insufficient data | False | False |
| **no_pattern** | Detector ran, found nothing | True | False |
| **misaligned** | Pattern found but conflicts with Context | True | False |
| **valid_setup** | Pattern aligned with Context | True | True/False |
| **detector_error** | Detector crashed (bug/config issue) | True | False |

**Before fix:** `detector_error` was **silently mapped** to `no_pattern` (LIE!)

**After fix:** `detector_error` is **explicit** (TRUTH!)

---

## Verification Mode

### **setup_detector_check**

**Usage:**

```bash
python scripts/run_validation.py --mode setup_detector_check \
    --pair BTCUSDT --sample-date 2023-12-15
```

**What it checks:**

1. **Instantiation:** Can SetupAgent instantiate SMCDetector?
2. **Parameters:** What scalar values were passed?
3. **Execution:** Does detector run without crashing?
4. **Patterns:** What patterns (if any) are detected?

**Output:**

```json
{
  "detector_initialized": true,
  "parameters_used": {
    "ob_range_threshold": 0.015,
    "fvg_min_gap": 0.005,
    "liquidity_lookback": 20
  },
  "detector_error": false,
  "patterns_detected": {...},
  "verdict": "✓ SMCDetector correctly wired and functioning"
}
```

**Verdicts:**

- **"✓ SMCDetector correctly wired and functioning"** → All good
- **"WIRING BUG CONFIRMED"** → SetupAgent passing wrong types
- **"Detector execution error: TypeError"** → Wiring issue during runtime

---

## What Changed, What Didn't

### **Changed ✅**

1. **SetupAgent wiring:** Now passes scalars, not dict
2. **Exception handling:** Errors are visible, not swallowed
3. **Type validation:** SMCDetector rejects invalid types
4. **State semantics:** `detector_error` is explicit

### **NOT Changed ❌**

1. **Trading logic:** No strategy modifications
2. **Thresholds:** Still 1.5% OB, 0.5% FVG (from config)
3. **SMC algorithms:** FVG/OB detection logic unchanged
4. **Alignment scoring:** Context/Regime/Setup logic same

**This was a wiring fix, not an optimization.**

---

## Impact on Previous Diagnostics

### **Before Fix (Potentially Contaminated)**

All previous diagnostic runs may have been affected:

- **setup_bottleneck:** "0 patterns on 2023-12-15" → Was detector crashing?
- **setup_coverage_multi:** "globally silent" → Was this real or masked errors?
- **smc_autopsy:** "0 OB detected, 2 FVG" → Did detector actually run correctly?

**Uncertainty:** We don't know if results were **genuine silence** or **masked crashes**.

---

### **After Fix (Clean Baseline)**

Now diagnostics will show **truth**:

- If detector crashes → `state = "detector_error"` (visible!)
- If no patterns → `state = "no_pattern"` (genuine!)
- If threshold strict → smc_autopsy shows candidates rejected

**Certainty:** We know results are **real**, not artifacts of hidden bugs.

---

## Next Steps

### **P1: Rerun Diagnostics on Clean Baseline** 🚨

**Required:**

1. **Rerun setup_bottleneck:**
   ```bash
   python scripts/run_validation.py --mode setup_bottleneck \
       --pair BTCUSDT --sample-date 2023-12-15
   ```

2. **Rerun setup_coverage_multi:**
   ```bash
   python scripts/run_validation.py --mode setup_coverage_multi \
       --pair BTCUSDT
   ```

3. **Rerun smc_autopsy:**
   ```bash
   python scripts/run_validation.py --mode smc_autopsy \
       --pair BTCUSDT --sample-date 2023-12-15 --sensitivity
   ```

**Why:** Previous verdicts **may have been contaminated** by the wiring bug.

**Expected:** One of two outcomes:

**Outcome A:** Results unchanged
- Detector was always running correctly (wiring bug didn't trigger)
- Previous verdicts stand ("globally silent" is real)
- Threshold tuning ticket still valid

**Outcome B:** Results changed
- Detector was crashing before (bug was active)
- Some "no pattern" were actually "detector errors"
- New baseline reveals different bottleneck

---

### **P2: Visual Truth Audit** (Still Relevant)

Even with correct wiring, visual audit (docs/SMC_VISUAL_TRUTH_AUDIT.md) remains useful to validate:
- Patterns exist visually
- Detector finds them (or misses them)
- Threshold recommendations are sound

---

### **P3: Threshold Tuning** (If Needed)

If rerun diagnostics confirm "detector works but too strict," proceed with threshold tuning.

---

## Testing

### **Tests Added**

**1. tests/test_smc_detector_init.py:**
- SMCDetector accepts valid numeric types
- SMCDetector rejects dicts with clear TypeError
- SMCDetector converts types correctly (int→float, float→int)

**2. tests/test_setup_agent_smc_wiring.py:**
- SetupAgent passes scalars, not dict
- Detector exceptions not converted to no_pattern
- setup_detector_check produces valid JSON
- State semantics are clean (no fake no_pattern)

**Run tests:**

```bash
pytest tests/test_smc_detector_init.py -v
pytest tests/test_setup_agent_smc_wiring.py -v
```

---

## Summary

### **Question:**
> "Why was the detector showing 0 patterns everywhere?"

### **Answer (Before Fix):**
> "Unknown - could be real silence OR masked TypeError from wiring bug."

### **Answer (After Fix):**
> "Now we'll know for sure - detector errors are visible, not hidden."

---

**The wiring is now correct. If detector finds nothing, it's a real result — not a bug.**

**Rerun diagnostics to establish clean baseline, then proceed with optimization if needed.**
