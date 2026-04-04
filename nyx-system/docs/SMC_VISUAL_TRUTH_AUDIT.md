# SMC Visual Truth Audit Protocol

## Purpose

**Objective:** Manually verify if SMC patterns (FVG, OB) exist visually on charts, and compare against what the automated detector finds.

**Why this matters:**
- Autopsy shows detector is silent (Category A: too strict)
- But we don't know if patterns *actually exist* on charts
- Need manual ground truth to validate:
  - Are patterns absent (detector correct but strict)?
  - Are patterns present but missed (detector broken)?

---

## Test Periods

Perform visual audit on these 3 reference dates:

| Date | Regime | Why Test |
|------|--------|----------|
| **2022-06-15** | Bear/stress | High volatility, should have bearish patterns |
| **2023-03-15** | Bull | Strong trend, should have bullish patterns |
| **2023-12-15** | Range | Consolidation, patterns may be rare |

---

## Procedure

### **Step 1: Select Chart Window**

**Timeframe:** 15 minutes (Setup TF where detector runs)

**Window size:** 4 hours of data (~16 candles)

**Specific windows to check:**

```
2022-06-15: 12:00 - 16:00 UTC
2023-03-15: 12:00 - 16:00 UTC
2023-12-15: 12:00 - 16:00 UTC
```

**Why 4 hours:** Enough candles to see patterns, small enough to mark manually

---

### **Step 2: Visual Pattern Identification**

Open chart (TradingView, custom tool, or charting software) and mark patterns by eye.

#### **A. Fair Value Gap (FVG)**

**Definition:**
- 3-candle pattern
- Price gaps between candle[i-1] and candle[i+1]
- Leaves an imbalance zone

**Bullish FVG:**
```
Candle 0: Any
Candle 1: Usually bullish (but can be any)
Candle 2: High enough that low[2] > high[0]

Gap zone: From high[0] to low[2]
```

**Bearish FVG:**
```
Candle 0: Any
Candle 1: Usually bearish (but can be any)
Candle 2: Low enough that high[2] < low[0]

Gap zone: From high[2] to low[0]
```

**Visual markers:**
- Box/rectangle marking gap zone
- Color: Green (bullish) / Red (bearish)
- Note timestamp of middle candle

**Example:**
```
Time: 13:15
Pattern: Bullish FVG
Gap size: ~0.8% (visual estimate)
Notes: Clear gap, strong bullish move
```

---

#### **B. Order Block (OB)**

**Definition:**
- Consolidation candle followed by strong move
- Last supply/demand zone before breakout

**Bullish OB:**
```
Setup candle: Bearish (close < open)
Followthrough: Strong bullish move (>1.5% ideally)

OB zone: From low to high of setup candle
```

**Bearish OB:**
```
Setup candle: Bullish (close > open)
Followthrough: Strong bearish move (>1.5% ideally)

OB zone: From low to high of setup candle
```

**Visual markers:**
- Box/rectangle marking OB zone
- Color: Green (bullish) / Red (bearish)
- Note timestamp of setup candle
- Note strength of followthrough move

**Example:**
```
Time: 14:30
Pattern: Bearish OB
Setup: Bullish candle
Followthrough: -2.1% (strong selloff)
Notes: Clear institutional supply
```

---

### **Step 3: Record Findings**

Create a simple table for each period:

```
Period: 2023-12-15 (12:00-16:00)

| Time  | Pattern | Type      | Strength | Detected? | Notes |
|-------|---------|-----------|----------|-----------|-------|
| 12:15 | FVG     | Bullish   | 0.3%     | NO        | Gap too small |
| 13:00 | OB      | Bearish   | 0.8%     | NO        | Move too weak |
| 14:30 | FVG     | Bearish   | 0.6%     | YES       | Clean gap |
| 15:45 | OB      | Bullish   | 1.8%     | Likely    | Strong move |

Total patterns: 4
Detected by auto: 1
False negatives: 3
```

---

### **Step 4: Compare with Detector Output**

**Get detector's view:**

```bash
python scripts/run_validation.py --mode smc_autopsy \
    --pair BTCUSDT --sample-date 2023-12-15
```

**Check JSON:**
```json
{
  "fvg_audit": {
    "detected": 1  // Compare with manual count
  },
  "ob_audit": {
    "detected": 0  // Compare with manual count
  }
}
```

**Compare:**
- Manual FVG count vs `fvg_audit.detected`
- Manual OB count vs `ob_audit.detected`
- Note patterns you saw that detector missed

---

### **Step 5: Classify Mismatches**

For each pattern detector missed:

**Type 1: Below Threshold (Expected Miss)**
```
Manual: FVG gap 0.3%
Detector threshold: 0.5%
→ Expected miss (threshold working as designed)
```

**Type 2: Above Threshold (Unexpected Miss)**
```
Manual: FVG gap 0.7%
Detector threshold: 0.5%
→ Bug or false negative (should have detected)
```

**Type 3: Borderline (Ambiguous)**
```
Manual: FVG gap ~0.5% (hard to measure exactly)
Detector threshold: 0.5%
→ Measurement uncertainty
```

---

## Measurement Tips

### **How to Measure Gap Size (FVG)**

**Using TradingView or similar:**

1. Identify high of candle[0]
2. Identify low of candle[2]
3. Calculate: `(low[2] - high[0]) / high[0]`

**Example:**
```
Candle 0 high: 42,100
Candle 2 low:  42,310

Gap = (42310 - 42100) / 42100 = 0.00499 = 0.5%
```

**Quick visual estimate:**
- Small gap (~0.2-0.4%): Subtle, hard to see
- Medium gap (~0.5-1.0%): Clear gap
- Large gap (>1%): Obvious imbalance

---

### **How to Measure Followthrough (OB)**

**For bullish OB:**
```
Setup candle close: 42,000
Next candle close: 42,700

Move = (42700 - 42000) / 42000 = 0.0167 = 1.67%
```

**Quick visual estimate:**
- Weak (<1%): Small move
- Moderate (1-2%): Decent move
- Strong (>2%): Clear breakout

---

## Validation Checklist

For each test period, verify:

- [ ] Chart window selected (4 hours)
- [ ] All visible FVGs marked
- [ ] All visible OBs marked
- [ ] Timestamps recorded
- [ ] Strengths estimated
- [ ] Detector output obtained
- [ ] Comparison table filled
- [ ] Mismatches classified

---

## Expected Outcomes

### **Outcome 1: Patterns Exist, Detector Finds Some**
```
Manual patterns: 12
Detected: 3
Missed (below threshold): 7
Missed (above threshold): 2

Verdict: Thresholds too strict but detector works
Action: Consider relaxing thresholds
```

---

### **Outcome 2: Patterns Exist, Detector Finds None**
```
Manual patterns: 8
Detected: 0
Missed (below threshold): 2
Missed (above threshold): 6

Verdict: Detector broken or logic flawed
Action: Debug detection algorithm
```

---

### **Outcome 3: Few Patterns Exist, Detector Correct**
```
Manual patterns: 1
Detected: 1

Verdict: Market genuinely has few SMC setups
Action: Accept or add more pattern types
```

---

### **Outcome 4: Many Patterns, All Below Threshold**
```
Manual patterns: 15
Detected: 0
Missed (below threshold): 15
Missed (above threshold): 0

Verdict: Thresholds intentionally strict, working correctly
Action: Decide if selectivity is desired
```

---

## Recording Template

Use this template for each period:

```
===============================================
VISUAL AUDIT - [Date] ([Regime])
===============================================

Chart Window: [Start] - [End]
Timeframe: 15m
Pair: BTCUSDT

MANUAL FINDINGS
---------------
Total FVGs found: [N]
  Bullish: [N]
  Bearish: [N]

Total OBs found: [N]
  Bullish: [N]
  Bearish: [N]

DETECTOR FINDINGS
-----------------
FVG detected: [N]
OB detected: [N]

COMPARISON
----------
False negatives (patterns missed): [N]
  Below threshold: [N]
  Above threshold: [N]
  Borderline: [N]

False positives (detector wrong): [N]

PATTERN DETAILS
---------------
[Table with timestamps, types, strengths]

NOTES
-----
[Observations, anomalies, insights]
```

---

## Tools

### **Recommended Charting Tools**

1. **TradingView** (easiest)
   - Free tier sufficient
   - Drawing tools for boxes
   - Can measure distances
   - Export screenshots

2. **Python/Matplotlib** (for developers)
   - Plot OHLC from CSV
   - Mark patterns programmatically
   - Compare side-by-side with detector

3. **Custom Tool** (optional)
   - Load NYX data directly
   - Overlay detector output
   - Interactive marking

---

## Deliverables

After completing visual audit:

1. **Summary report** (markdown or doc)
2. **Comparison tables** for each period
3. **Annotated screenshots** (optional but helpful)
4. **Recommendations** based on findings

**Example summary:**
```
Visual Audit Summary - 3 Periods

2022-06-15 (Bear):
  Manual: 8 patterns (5 FVG, 3 OB)
  Detected: 2 FVG, 0 OB
  Issue: OB threshold too strict (1.5%)

2023-03-15 (Bull):
  Manual: 12 patterns (7 FVG, 5 OB)
  Detected: 3 FVG, 0 OB
  Issue: OB threshold too strict

2023-12-15 (Range):
  Manual: 3 patterns (2 FVG, 1 OB)
  Detected: 1 FVG, 0 OB
  Issue: Few patterns + strict threshold

Overall Conclusion:
Patterns DO exist but most are below current thresholds.
OB threshold (1.5%) is especially strict.
Recommend testing 0.8-1.0% for OB.
```

---

## Next Steps

**After visual audit, you can confidently say:**

- "Patterns exist but detector is too strict" (relax thresholds)
- "Patterns exist but detector misses them" (fix algorithm)
- "Few patterns exist, detector is correct" (accept or diversify)

**Then:** Proceed with appropriate fix/optimization ticket.

---

## Summary

**Purpose:** Ground truth validation of automated detector

**Method:** Manual chart marking + comparison with detector output

**Periods:** 3 test dates (bear/bull/range)

**Deliverable:** Evidence-based recommendation on detector performance

**Impact:** Transform "detector is silent" into actionable insight on why and what to fix.
