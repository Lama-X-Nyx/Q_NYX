# Fractal Context Windows

**Version:** 1.0  
**Date:** 2025-03-29  
**Purpose:** Define proper temporal context depth for each timeframe in the fractal architecture

---

## 🎯 **CORE PRINCIPLE**

**Each timeframe serves a different role and needs a different temporal horizon.**

**NOT this:**
```python
# ❌ WRONG: Derive everything from lowest TF sample
sample_bars_15m = 10000
bars_1d = sample_bars_15m / 96  # Only 104 bars 1D
bars_4h = sample_bars_15m / 4   # Only 2500 bars 4H
```

**THIS:**
```python
# ✅ CORRECT: Each TF gets its own context window
windows = {
    '1d': 360 days,   # Macro climate
    '4h': 7 days,     # Weekly structure
    '1h': 2 days,     # Operational regime
    '15m': 1 day      # Setup patterns
}
```

---

## 📊 **CONTEXT WINDOW DESIGN**

### **1D Context: 360 Days**

**Role:** Macro bias & strategic intent  
**Why 360 days:** Need a full year to see:
- Major market cycles
- Seasonal patterns
- Long-term trend direction
- Support/resistance at yearly scale

**Example:**
- SMA 20/50 on 1D needs reliable long-term data
- Bias detection requires multiple market phases
- Cannot determine "bullish year" from 30 days

**Minimum bars:** 360 (1 year)  
**Typical use:** Context Agent

---

### **4H Structure: 7 Days**

**Role:** Intermediate structure & weekly patterns  
**Why 7 days:** Captures:
- Full trading week
- Intraweek momentum shifts
- Weekly highs/lows
- Bridge between macro and operational

**Example:**
- Weekly trend structure
- Multi-day consolidations
- Swing high/low context

**Minimum bars:** 42 (7 days × 6 candles/day)  
**Typical use:** Structure analysis (future)

---

### **1H Regime: 2 Days**

**Role:** Operational regime detection (HSMM)  
**Why 2 days:** Sufficient for:
- Regime persistence detection
- Intraday trend vs range
- HSMM forward-backward algorithm
- Recent volatility context

**Example:**
- HSMM needs ~100 bars for reliable regimes
- 2 days = 48 bars (48h)
- Captures recent market behavior without noise

**Minimum bars:** 100 (configurable)  
**Typical use:** Regime Agent

---

### **15M Setup: 1 Day**

**Role:** Entry setup patterns (SMC)  
**Why 1 day:** Recent tactical patterns:
- Order blocks from last session
- Fair value gaps still relevant
- Break of structure signals
- Liquidity sweeps

**Example:**
- SMC patterns older than 1 day less relevant
- Recent price action for entry timing
- Tight tactical focus

**Minimum bars:** 96 (1 day × 4 candles/hour × 24h)  
**Typical use:** Setup Agent

---

## 🔍 **WHY NOT UNIFORM SAMPLING?**

### **Problem 1: Lowest-TF Dominance**

**Old approach:**
```python
# Load 10,000 bars at 15M
sample_bars = 10000

# Derive higher TFs
df_1d = resample_to_1d(df_15m[:10000])  # Only 104 bars!
df_4h = resample_to_4h(df_15m[:10000])  # Only 416 bars
```

**Result:**
- 1D gets 104 bars (3.5 months) ← **NOT ENOUGH**
- Cannot detect yearly trends
- Bias detection unreliable
- Readiness issues masquerading as logic issues

---

### **Problem 2: Mismatched Memory**

**Each TF role needs different memory:**

| Timeframe | Role | Needs Memory Of |
|-----------|------|-----------------|
| 1D | Climate | 1 year+ |
| 4H | Structure | 1 week |
| 1H | Regime | 2 days |
| 15M | Setup | 1 day |

**Uniform sampling gives:**
- Too little memory to higher TFs
- Too much memory to lower TFs
- Wrong geometry for every role

---

### **Problem 3: False Readiness Blocks**

**With old sampling (10k bars 15M):**
```
Context Agent: NOT READY (11 bars 1D, need 50)
Regime Agent: NOT READY (250 bars 1H, barely enough)
Setup Agent: READY (10k bars)
```

**Diagnosis:** "Context is the bottleneck!"  
**Reality:** Context was never given proper data

**With context windows:**
```
Context Agent: READY (360 bars 1D ✅)
Regime Agent: READY (48 bars 1H ✅)
Setup Agent: READY (96 bars 15M ✅)
```

**Diagnosis:** Now reflects real logic, not data geometry

---

## 📋 **WINDOW SPECIFICATION**

### **Configuration**

```yaml
fractal:
  enabled: true
  use_entry_agent: false

timeframes:
  context_tf: 1d
  structure_tf: 4h   # Reserved for future use
  regime_tf: 1h
  setup_tf: 15m

context_windows:
  context_1d_days: 360    # 1 year macro context
  structure_4h_days: 7    # 1 week structure
  regime_1h_days: 2       # 2 days regime
  setup_15m_days: 1       # 1 day setup
```

### **Loading Logic**

```python
def load_fractal_context(pair: str, current_date: datetime, config: dict):
    """
    Load each timeframe with its own proper context window
    
    Returns aligned MTF data centered on current_date
    """
    
    windows = config['fractal']['context_windows']
    
    # Each TF loaded independently
    mtf_data = {
        '1d': load_tf_window(pair, '1d', current_date, days=windows['context_1d_days']),
        '4h': load_tf_window(pair, '4h', current_date, days=windows['structure_4h_days']),
        '1h': load_tf_window(pair, '1h', current_date, days=windows['regime_1h_days']),
        '15m': load_tf_window(pair, '15m', current_date, days=windows['setup_15m_days'])
    }
    
    return mtf_data
```

### **Key Properties**

1. **Independent Loading:** Each TF loaded separately
2. **Temporal Alignment:** All windows end at same `current_date`
3. **Closed Candles Only:** No look-ahead
4. **Configurable:** Window sizes in config, not hardcoded

---

## 🔄 **FRACTAL GEOMETRY**

### **Visual Representation**

```
Reference Date: 2023-12-30
├── 1D Context    [2023-01-05 ─────────────────────► 2023-12-30]  360 days
├── 4H Structure  [2023-12-23 ───────────► 2023-12-30]            7 days
├── 1H Regime     [2023-12-28 ──► 2023-12-30]                     2 days
└── 15M Setup     [2023-12-29 ► 2023-12-30]                       1 day
```

**Properties:**
- All windows end at same point (2023-12-30)
- Higher TFs see longer history
- Lower TFs see recent tactical context
- True fractal: memory decreases down the stack

---

## ✅ **BENEFITS**

### **1. Proper Context Per Role**

**Before:**
- 1D sees 3 months → Cannot detect yearly trends
- 4H sees 2 weeks → Limited structural context
- All derived from single 15M sample

**After:**
- 1D sees 1 year → Proper macro context
- 4H sees 1 week → Full weekly structure
- Each loaded independently

### **2. Readiness Clarity**

**Before:**
```
bars_skipped_due_to_readiness: 1000
→ Context needs 50 bars (has 11)
```
**Interpretation:** Is Context too strict? Or just missing data?

**After:**
```
bars_skipped_due_to_readiness: 0
Context: 360 bars loaded ✅
```
**Interpretation:** Clear - if Context blocks now, it's logic, not data

### **3. True Bottleneck Discovery**

**With proper geometry:**
- Can finally measure real logic bottlenecks
- Not confused by data loading artifacts
- Diagnostics become trustworthy

---

## 📊 **EXPECTED BAR COUNTS**

### **For Reference Date: 2023-12-30**

| Timeframe | Window | Expected Bars | Actual Function |
|-----------|--------|---------------|-----------------|
| 1D | 360 days | ~360 | Macro bias |
| 4H | 7 days | ~42 | Structure |
| 1H | 2 days | ~48 | Regime |
| 15M | 1 day | ~96 | Setup |

**All aligned to closed candles ending 2023-12-30**

---

## ⚠️ **MIGRATION NOTES**

### **What Changes**

**Old:**
```python
# Single sample drives everything
mtf_data = load_mtf_sample(pair, sample_bars=10000)
```

**New:**
```python
# Each TF gets proper window
mtf_data = load_fractal_context(
    pair='BTCUSDT',
    current_date=datetime(2023, 12, 30),
    config=config
)
```

### **What Doesn't Change**

- ✅ Trading logic (agents, thresholds)
- ✅ Strategy parameters
- ✅ Agent roles
- ✅ Risk management

**Only changes:** How much history each TF sees

---

## 🎯 **VALIDATION**

### **Geometry Check**

```bash
python scripts/run_validation.py --mode fractal_geometry \
    --pair BTCUSDT --sample-date 2023-12-30
```

**Expected output:**
```
FRACTAL GEOMETRY - BTCUSDT
Reference date: 2023-12-30

1D context window:   360 days  → 360 bars loaded
4H structure window: 7 days    → 42 bars loaded
1H regime window:    2 days    → 48 bars loaded
15m setup window:    1 day     → 96 bars loaded

All windows aligned to closed candles.
No look-ahead detected.
```

### **Readiness Re-Check**

After implementing context windows, re-run:
```bash
python scripts/run_validation.py --mode fractal_check \
    --pair BTCUSDT
```

**Expected change:**
```
bars_skipped_due_to_readiness: 0  (was 1000)
bars_ready_for_decision: 9900     (was 0)
```

---

## 📋 **SUMMARY**

**Problem:** Uniform lowest-TF sampling gives wrong context to higher TFs

**Solution:** Each TF gets its own temporal window matching its role

**Result:** 
- Proper fractal geometry
- Clear readiness
- Trustworthy diagnostics
- Real bottleneck discovery

**Key Insight:** Memory should decrease down the fractal stack, not be uniform.

---

**This document defines the fractal context window architecture for NYX.**
