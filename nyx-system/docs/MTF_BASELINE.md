# NYX Multi-Timeframe Baseline Architecture

**Version:** 0.8.2-MTF  
**Date:** 2025-03-29  
**Status:** Restored from Prompt Maître v5.5M

---

## 📋 **OVERVIEW**

NYX is natively a **4-timeframe fractal swing agent** designed for disciplined execution on BTC/USDT.

This document defines the **exact mapping** between timeframes and system components, based on the original prompt specification.

---

## 🎯 **TIMEFRAME ARCHITECTURE**

### **Design Philosophy**

NYX operates on **5 timeframes** with distinct hierarchical roles:

```
1D (Daily)    → Strategic Intent & Bias
4H (Pivot)    → Operational Stability
1H (Bridge)   → Intermediate Confirmation
15M (Setup)   → Pattern Detection (SMC)
5M (Timing)   → Entry Execution
```

**Fractal Principle:** Higher timeframes constrain lower timeframes. Lower timeframes refine higher timeframes.

---

## 📊 **TIMEFRAME ROLES**

### **1D - Daily (Context/Strategic Intent)**

**Role:** Define the dominant bias and strategic direction.

**Usage:**
- Compute `Intent_Daily = argmax(π_4H · A^k)` (HSMM projection from 4H)
- Propagate hierarchical constraint to all lower TFs
- **No direct trading signals** - purely strategic context

**Weight in fractal score:** `w_1D = 3.0` (highest)

**Key Metrics:**
- State probabilities: `P(Trend+)`, `P(Trend-)`, `P(Range)`, etc.
- ATR(14) vs ATR(50) for volatility regime

**Read by:**
- `NYXEngine._get_intent_daily()`
- Fractal scoring logic

---

### **4H - Four-Hour (Pivot/Stability)**

**Role:** Primary operational regime with stability requirements.

**Usage:**
- Compute `Stability_4H = P(s_t = s_t+1 | O)`
- **Entry condition:** `Stability_4H ≥ 0.60` (≥0.70 if macro_block=ON)
- Detect major bias flips with 1H

**Weight in fractal score:** `w_4H = 2.0`

**Key Metrics:**
- HSMM state persistence probability
- Transition matrix A(4H)
- Volume impulse (>1.5× MA20)

**Read by:**
- `NYXEngine._check_stability()`
- HSMM module (primary TF for regime)

---

### **1H - One-Hour (Intermediate Bridge)**

**Role:** Bridge between strategic (4H) and tactical (15M) layers.

**Usage:**
- Detect flip probability: `P(flip 4H/1H at h ∈ {2,3}) > 0.35`
- Confirm alignment with Daily intent
- **Trailing mode:** Conservative (1H-based) if misaligned with Daily

**Weight in fractal score:** `w_1H = 1.5`

**Key Metrics:**
- Trend continuation probability
- BoS/CHoCH confirmations
- RSI regime (bull/bear/mean)

**Read by:**
- `NYXEngine._check_flip_risk()`
- Trailing stop logic

---

### **15M - Fifteen-Minute (Setup/SMC Confirmation)**

**Role:** SMC pattern detection and alignment verification.

**Usage:**
- Detect Order Blocks (OB), Fair Value Gaps (FVG), BoS/CHoCH
- **Entry condition:** `max{P(Trend+), P(Trend-)}_{15M} ≥ 0.60` **in direction of Intent_Daily**
- Primary TF for SMC emissions

**Weight in fractal score:** `w_15M = 1.0`

**Key Metrics:**
- SMC pattern validity (OB base 2-6 candles, FVG respect)
- Alignment score with Daily intent
- Liquidity sweep detection

**Read by:**
- `SMCDetector.detect_patterns()`
- `NYXEngine._check_alignment()`

---

### **5M - Five-Minute (Entry Timing)**

**Role:** Fine-grained entry timing and aggressive trailing.

**Usage:**
- Final confirmation before execution
- **Trailing mode:** Aggressive (5M-based) if aligned with Daily
- Monitor real-time liquidity conditions

**Weight in fractal score:** `w_5M = 0.5` (lowest)

**Key Metrics:**
- Spread ≤ budget
- Order book depth
- Expected slippage

**Read by:**
- Entry execution logic
- Trailing stop updates (if aggressive mode)

---

## 🔗 **SYNCHRONIZATION RULES**

### **Closed-Candle Requirement**

**Rule:** Only use **closed candles** from higher timeframes when making decisions on lower timeframes.

**Implementation:**
```python
# When evaluating at 15M index i:
# - Use 1D data up to most recent closed 1D candle
# - Use 4H data up to most recent closed 4H candle
# - Use 1H data up to most recent closed 1H candle
# - Current 15M candle at i is allowed (execution TF)
```

**Prevents:**
- Look-ahead bias
- Using incomplete higher TF candles
- Future information leakage

---

### **Temporal Alignment**

**Mapping from entry TF (15M/5M) to higher TFs:**

```python
# Given 15M timestamp t:
aligned_1H = floor(t to 1H boundary)
aligned_4H = floor(t to 4H boundary)
aligned_1D = floor(t to 1D boundary)

# Use only data up to these aligned timestamps
data_1D = df_1D[df_1D.index <= aligned_1D]
data_4H = df_4H[df_4H.index <= aligned_4H]
data_1H = df_1H[df_1H.index <= aligned_1H]
```

---

### **Macro-Block Constraint**

**Rule:** If `macro_block = ON` (FOMC, CPI, etc. within [-30min, +2h]):
- **Freeze all transitions**
- **Block all new entries**
- Only allow exits/stop management

**Implementation:**
```python
if macro_block:
    return {'action': 'HOLD', 'reason': 'macro_block_active'}
```

---

## 📐 **FRACTAL SCORING**

### **Formula**

```
Score_TF = max_{s ∈ S} P(s | O)

Score_FL = Σ(w_TF × Score_TF)
```

### **Dynamic Weight Adjustment**

**ATR-based:**
```
Δ_ATR = +0.2  if ATR(14) > 1.2 × ATR(50)
        -0.2  if ATR(14) < 0.8 × ATR(50)
         0    otherwise
```

**Opposition penalty (15M/5M against Intent_Daily):**
```
w_TF ← 0.5 × (w_TF^0 + Δ_ATR)  # Halve weight if opposing Daily
```

---

## 🎯 **ENTRY CONDITIONS (ALL REQUIRED)**

Based on multi-timeframe alignment:

1. **SdC > 5** (based on Intent_Daily from 1D)
2. **Stability_4H ≥ 0.60** (from 4H HSMM)
3. **Alignment_15M ≥ 0.60** (15M trend in direction of Intent_Daily)
4. **RR ≥ 2:1** (computed from 15M setup)
5. **macro_block = OFF**
6. **P(hit -0.10) ≤ 0.20** (HSMM projection)

**Any single failure → WAIT**

---

## 🔄 **MODULE MAPPING**

| Module | Primary TF | Usage |
|--------|-----------|-------|
| HSMM Regime | 4H | State estimation, stability |
| Intent Daily | 1D | Strategic bias, SdC calculation |
| SMC Detector | 15M | OB, FVG, BoS/CHoCH patterns |
| Macro Filter | 1D/4H | Trend alignment, regime detection |
| Risk Manager | All | Hitting probabilities, position sizing |
| Entry Timing | 5M | Final confirmation, execution |

---

## 📊 **DATA REQUIREMENTS**

### **Minimum History per TF**

```yaml
1D:  200 bars (for ATR, regime calibration)
4H:  500 bars (for HSMM training)
1H:  1000 bars (for bridge analysis)
15M: 2000 bars (for SMC pattern detection)
5M:  5000 bars (for execution timing)
```

### **Required Columns**

All timeframes must have:
- `datetime` (ISO-8601, timezone-aware)
- `open`, `high`, `low`, `close`, `volume`

---

## 🚫 **WHAT THIS BASELINE DOES NOT INCLUDE**

**Out of scope for restoration:**
- Strategy optimization
- Threshold tuning
- New alpha logic
- Additional filters
- Performance enhancement

**This is purely structural restoration** to enable proper diagnostics on the real NYX architecture.

---

## ✅ **VALIDATION CHECKLIST**

- [ ] All 5 TFs load correctly
- [ ] Temporal alignment verified
- [ ] Closed-candle rule enforced
- [ ] Fractal scoring computes across TFs
- [ ] Entry conditions check all TFs
- [ ] Macro-block freezes transitions
- [ ] No look-ahead bias
- [ ] Module-TF mapping respected

---

## 📝 **NEXT STEPS**

After baseline restoration:
1. Re-run diagnostics on 4-TF engine:
   - Funnel analysis
   - SMC diagnostics
   - Pattern quality
2. Compare results vs 1H-only version
3. Identify if multi-TF improves signal quality

---

**This baseline enables diagnostics on the real NYX system, not a simplified approximation.**
