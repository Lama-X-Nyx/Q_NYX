# Fractal Geometry Migration Guide

**Version:** 1.0  
**Date:** 2025-03-29  
**Purpose:** Guide for migrating from uniform sampling to fractal context windows

---

## 🎯 **WHAT CHANGED**

### **The Problem (Before)**

**Old approach:**
```python
# Single lowest-TF sample drives everything
mtf_data = load_mtf_sample(pair='BTCUSDT', sample_bars=10000)
```

**Result:**
- 15M: 10,000 bars (loaded)
- 1H: 2,500 bars (derived by resampling)
- 4H: 416 bars (derived by resampling)
- **1D: 104 bars** ← **ONLY 3.5 MONTHS!**

**Problem:**
- 1D context insufficient for macro bias
- Context Agent appears "not ready" even with data
- Fake readiness blocks masquerading as geometry issues

---

### **The Solution (After)**

**New approach:**
```python
# Each TF gets its own proper window
mtf_data = load_fractal_context(
    pair='BTCUSDT',
    current_date=datetime(2023, 12, 15),
    config=config
)
```

**Result:**
- **1D: 361 bars (1 year)** ← Proper macro context
- 4H: 43 bars (1 week) ← Weekly structure
- 1H: 121 bars (5 days) ← Operational regime
- 15M: 97 bars (1 day) ← Tactical setup

**Benefit:**
- Each TF sees correct temporal depth for its role
- No readiness blocks
- Real bottleneck discovery possible

---

## 📋 **MIGRATION STEPS**

### **Step 1: Update Config**

**Add to `config/validation_baseline.yaml`:**

```yaml
fractal:
  enabled: true
  use_entry_agent: false
  
  # Explicit timeframe assignment
  timeframes:
    context_tf: "1d"
    structure_tf: "4h"
    regime_tf: "1h"
    setup_tf: "15m"
  
  # Context windows per timeframe (in days)
  context_windows:
    context_1d_days: 360    # 1 year macro context
    structure_4h_days: 7    # 1 week structure
    regime_1h_days: 5       # 5 days regime (for 100+ bars)
    setup_15m_days: 1       # 1 day setup
```

---

### **Step 2: Update Data Loading**

**Replace this:**
```python
from src.data.mtf_loader import load_mtf_sample

# Old
mtf_data = load_mtf_sample(pair, sample_bars=10000)
```

**With this:**
```python
from src.data.mtf_loader import load_fractal_context
from datetime import datetime

# New
mtf_data = load_fractal_context(
    pair='BTCUSDT',
    current_date=datetime(2023, 12, 15),
    config=config
)
```

---

### **Step 3: Update Function Signatures**

**Change from sample_bars to current_date:**

**Before:**
```python
def run_fractal_check(pair: str, config: dict, output_dir: Path, sample_bars: int):
    mtf_data = load_mtf_sample(pair, sample_bars=sample_bars)
```

**After:**
```python
def run_fractal_check(pair: str, config: dict, output_dir: Path, sample_date: str):
    current_date = datetime.strptime(sample_date, '%Y-%m-%d')
    mtf_data = load_fractal_context(pair, current_date, config)
```

---

### **Step 4: Update Iteration Logic**

**The warmup needs adjustment:**

**Before:**
```python
warmup = 100  # Fixed warmup
n_signals = min(1000, len(mtf_data[lowest_tf]) - warmup)
```

**After:**
```python
warmup = 50  # Reduced to work with 1-day 15M window (96 bars)
available_bars = len(mtf_data[lowest_tf])
max_signals = min(100, available_bars - warmup)
```

**Rationale:**
- 1-day 15M window = ~96 bars
- Old warmup (100) exceeds available data
- New warmup (50) allows iteration

---

### **Step 5: Update CLI Arguments**

**Replace `--sample-bars` with `--sample-date`:**

**Before:**
```python
parser.add_argument(
    '--sample-bars',
    type=int,
    default=10000,
    help='Number of bars to sample'
)

# Usage:
python script.py --sample-bars 10000
```

**After:**
```python
parser.add_argument(
    '--sample-date',
    type=str,
    default='2024-01-01',
    help='Reference date (YYYY-MM-DD)'
)

# Usage:
python script.py --sample-date 2023-12-15
```

---

## ✅ **VALIDATION**

### **Step 6: Verify Geometry**

**Run geometry check:**
```bash
python scripts/run_validation.py --mode fractal_geometry \
    --pair BTCUSDT --sample-date 2023-12-15
```

**Expected output:**
```
1D Context:   361 bars (360 days) ✅
4H Structure:  43 bars (7 days)   ✅
1H Regime:    121 bars (5 days)   ✅
15M Setup:     97 bars (1 day)    ✅

Status: valid ✅
```

---

### **Step 7: Verify Readiness**

**Run fractal check:**
```bash
python scripts/run_validation.py --mode fractal_check \
    --pair BTCUSDT --sample-date 2023-12-15
```

**Expected changes:**

| Metric | Before | After |
|--------|--------|-------|
| bars_ready_for_decision | 0 | 47 ✅ |
| bars_skipped_due_to_readiness | 1000 | 0 ✅ |
| Context readiness_blocks | 1000 | 0 ✅ |
| Regime readiness_blocks | 800 | 0 ✅ |

---

## 🔍 **WHAT TO CHECK**

### **Readiness Blocks Should Drop**

**Before migration:**
```json
{
  "bars_skipped_due_to_readiness": 1000,
  "diagnostics": {
    "readiness_blocks": {
      "context": 1000,
      "regime": 800,
      "setup": 0
    }
  }
}
```

**After migration:**
```json
{
  "bars_skipped_due_to_readiness": 0,
  "diagnostics": {
    "readiness_blocks": {
      "context": 0,
      "regime": 0,
      "setup": 0
    }
  }
}
```

---

### **Real Bottleneck Visible**

**Before migration:**
```
Bottleneck: Unknown
Reason: All bars blocked by readiness
```

**After migration:**
```
Bottleneck: SETUP (47 blocks)
Reason: SMC patterns misaligned
```

**This is real diagnostic data!**

---

## ⚠️ **IMPORTANT NOTES**

### **1. Window Tuning**

**Initial windows (from docs):**
```yaml
regime_1h_days: 2  # 2 days
```

**Problem:** 2 days 1H = 48 bars, but Regime needs 100 minimum.

**Tuned windows (final):**
```yaml
regime_1h_days: 5  # 5 days → 120 bars ✅
```

**Lesson:** Window sizes must satisfy readiness requirements.

---

### **2. No Changes to Trading Logic**

**What did NOT change:**
- ✅ Agent algorithms (Context, Regime, Setup)
- ✅ Threshold parameters
- ✅ Strategy logic
- ✅ Risk management

**What changed:**
- ✅ Data loading (geometry only)
- ✅ How much history each TF sees

---

### **3. Backward Compatibility**

**Old function still exists:**
```python
# Still works for non-fractal use cases
mtf_data = load_mtf_sample(pair, sample_bars=10000)
```

**New function for fractal:**
```python
# Use for fractal architecture
mtf_data = load_fractal_context(pair, current_date, config)
```

---

## 🎯 **EXPECTED OUTCOMES**

### **1. Readiness Issues Resolved**

**Before:**
- Context appears "not ready" even with data
- Geometry confusion

**After:**
- Context is ready (361 bars)
- Clear diagnostics

---

### **2. True Bottleneck Discovery**

**Before:**
```
Cannot identify bottleneck
All bars blocked by readiness
```

**After:**
```
Setup is the bottleneck (47 blocks)
Context: passed (bullish)
Regime: passed (range)
Setup: blocked (misaligned)
```

---

### **3. Correct Fractal Geometry**

**Fractal principle:**
> "Memory decreases down the stack"

**Before (wrong):**
- 1D: 3 months
- 1H: 3 months
- 15M: 3 months
→ All same memory (not fractal!)

**After (correct):**
- 1D: 1 year
- 4H: 1 week
- 1H: 5 days
- 15M: 1 day
→ Memory decreases (fractal!)

---

## 📊 **FILES MODIFIED**

**Core changes:**
1. `src/data/mtf_loader.py` - Added `load_fractal_context()`
2. `config/validation_baseline.yaml` - Added `context_windows`
3. `scripts/fractal_check.py` - Updated to use new loader
4. `scripts/fractal_geometry.py` - New validation mode
5. `scripts/run_validation.py` - CLI integration

**No changes to:**
- Agent implementations
- Strategy logic
- Threshold parameters
- Risk management

---

## ✅ **CHECKLIST**

**After migration, verify:**

- [ ] Config has `context_windows` block
- [ ] `load_fractal_context()` replaces `load_mtf_sample()`
- [ ] CLI uses `--sample-date` instead of `--sample-bars`
- [ ] Warmup reduced (50 instead of 100)
- [ ] fractal_geometry mode runs successfully
- [ ] fractal_check shows 0 readiness blocks
- [ ] Real bottleneck is identified
- [ ] JSON output valid

---

## 🎯 **SUMMARY**

**What changed:**
- Data loading geometry (how much history each TF sees)

**What didn't change:**
- Trading logic, thresholds, strategy

**Benefit:**
- Proper fractal geometry
- No fake readiness blocks
- Real bottleneck discovery

**Migration effort:**
- Core changes: 2-3 hours
- Validation: 30 minutes
- Total: Half day

---

**This migration fundamentally fixes the fractal data geometry.**
