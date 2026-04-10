# OOS Bounded Lookback - P1-06d

## 📋 **OVERVIEW**

**Date:** 2025-03-28  
**Ticket:** P1-06d  
**Status:** Implemented

---

## 🎯 **PURPOSE**

Make Out-of-Sample and Walk-Forward validation practically runnable on the full dataset (43,848 rows) without changing NYX trading logic.

---

## ❓ **THE PROBLEM**

### **Before P1-06d:**

OOS and Walk-Forward used **unbounded historical lookback**:

```python
# At bar i, NYX sees all history from start
for i in range(len(data)):
    signal = engine.generate_signal(pair, data.iloc[:i+1], ...)
```

**Cost:**
- Bar 100: process 100 rows
- Bar 1000: process 1000 rows
- Bar 10000: process 10000 rows
- **Complexity: O(n²)**

**Result:** OOS timeout on real dataset

---

## ✅ **THE SOLUTION**

### **Bounded Lookback Window:**

Instead of unbounded history, NYX sees only the **last N bars** (default: 1000):

```python
# At bar i, NYX sees only [i-999 : i+1]
lookback_bars = 1000
for i in range(warmup, len(data)):
    start = max(0, i - lookback_bars + 1)
    window = prepared_data.iloc[start:i+1]
    signal = engine.generate_signal_at_index(pair, prepared_data, i, lookback_bars=lookback_bars)
```

**Cost:**
- Bar 100: process 100 rows (warmup)
- Bar 1000: process 1000 rows
- Bar 10000: process 1000 rows (bounded)
- **Complexity: O(n)**

**Result:** OOS completes in practical time

---

## ⚙️ **CONFIGURATION**

### **In `config/validation_baseline.yaml`:**

```yaml
validation:
  warmup_bars: 100           # Skip first 100 bars (NYX needs history)
  signal_lookback_bars: 1000 # Max historical window
```

### **Defaults:**
- `warmup_bars`: 100 (minimum history for HSMM initialization)
- `signal_lookback_bars`: 1000 (balanced performance vs memory)

---

## 📊 **BOUNDED MEMORY ASSUMPTION**

### **Key Assumption:**

> NYX decision-making is based on bounded memory of the last N bars, not infinite history.

### **Why This Is Acceptable for Phase 1:**

1. **HSMM State Detection:**
   - HSMM learns regime patterns from local price behavior
   - Recent 1000 bars capture current regime context
   - Older data has diminishing predictive value

2. **SMC Pattern Detection:**
   - Order blocks, FVGs, liquidity zones are local structures
   - Patterns older than 1000 bars (41 days @ 1h) unlikely to be relevant

3. **Regime Transitions:**
   - Market regimes change on shorter timescales
   - 1000-bar window captures recent regime shifts

4. **Computational Practicality:**
   - Unbounded lookback = impractical validation
   - Bounded lookback = runnable validation pipeline

### **What Changes vs Unbounded:**

**Behavior:**
- NYX sees rolling 1000-bar window instead of all history
- Decisions based on recent context, not ancient data

**Doesn't Change:**
- Signal logic (HSMM + SMC + Macro)
- Decision thresholds
- Risk parameters
- Entry/exit rules

---

## 🔬 **IMPLEMENTATION DETAILS**

### **1. NYXEngine.generate_signal_at_index():**

```python
def generate_signal_at_index(self, pair, prepared_data, i, lookback_bars=1000):
    # Calculate bounded window
    start_idx = max(0, i - lookback_bars + 1)
    data_slice = prepared_data.iloc[start_idx:i+1]
    
    # Standard signal generation on bounded window
    return self.generate_signal(pair, data_slice, current_date)
```

### **2. OOSReport._run_backtest():**

```python
# Prepare features once
prepared_data = engine.prepare_data(data)

# Get config
warmup = config['validation']['warmup_bars']
lookback_bars = config['validation']['signal_lookback_bars']

# Run with bounded lookback
for i in range(warmup, len(prepared_data)):
    signal = engine.generate_signal_at_index(
        pair, prepared_data, i, lookback_bars=lookback_bars
    )
```

### **3. WalkForward._run_window():**

Same bounded lookback logic applied to each walk-forward test window.

---

## 📈 **PERFORMANCE IMPACT**

### **Logged Metrics:**

Each OOS/WalkForward run now logs:
- `bars_processed`: Number of bars evaluated
- `warmup_bars`: Bars skipped at start
- `lookback_bars`: Window size used
- `execution_time_seconds`: Total runtime
- `avg_time_per_bar_ms`: Average cost per bar
- `signals_generated`: Total signals

### **Expected Performance:**

**Before P1-06d:**
```
OOS on 43k rows: timeout / impractical
Complexity: O(n²)
```

**After P1-06d:**
```
OOS on 43k rows: completes in reasonable time
Complexity: O(n)
Avg time per bar: ~constant
```

---

## ⚠️ **LIMITATIONS & CAVEATS**

### **Known Limitations:**

1. **Bounded Memory:**
   - NYX doesn't see pre-window history
   - Very long-term patterns (>1000 bars) not captured

2. **Warmup Period:**
   - First 100 bars skipped
   - No signals generated during warmup

3. **Window Edge Cases:**
   - If i < lookback_bars: uses [0:i+1] (smaller window)
   - This is handled automatically

### **Not Changed:**

- ✅ Trading logic unchanged
- ✅ Signal thresholds unchanged
- ✅ Risk parameters unchanged
- ✅ HSMM statistical model unchanged

---

## 🧪 **VALIDATION**

### **Tests Verify:**

1. Bounded window size respected
2. No crash when i < lookback_bars
3. Output schemas remain compatible
4. Performance logging functional

### **Run Tests:**

```bash
pytest tests/test_oos_bounded_lookback.py -v
pytest tests/test_walkforward_bounded_lookback.py -v
```

---

## 📝 **FUTURE CONSIDERATIONS**

### **Potential Enhancements (Post-Phase 1):**

1. **Adaptive Lookback:**
   - Vary window size based on volatility regime
   - Larger windows in stable regimes, smaller in volatile

2. **Incremental HSMM:**
   - True incremental state updates
   - No window needed - perfect memory efficiency

3. **Multi-Scale Windows:**
   - Different lookbacks for different components
   - HSMM: 2000 bars, SMC: 500 bars, etc.

### **Not Recommended for Phase 1:**

- ❌ Complex incremental state logic
- ❌ Multiple competing window sizes
- ❌ Dynamic window adaptation

**Reason:** Phase 1 is about validation, not optimization.

---

## ✅ **SUMMARY**

**Decision:** Bounded lookback (1000 bars) instead of unbounded history

**Why:** Makes validation practically runnable without changing trading logic

**Trade-off:** Bounded memory vs computational practicality

**Result:** OOS and Walk-Forward complete on full dataset

**Status:** ✅ Implemented and tested

---

**See also:**
- `docs/PHASE1_BASELINE.md` - Frozen parameters
- `docs/HSMM_INPUT_AUDIT.md` - HSMM contract fix
- `config/validation_baseline.yaml` - Configuration
