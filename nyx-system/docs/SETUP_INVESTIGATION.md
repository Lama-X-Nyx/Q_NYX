# Setup Investigation

## Overview

Setup Investigation transforms the diagnostic from Setup Deep Dive into **actionable root cause analysis**. While Setup Deep Dive tells you HOW Setup blocks, Investigation tells you WHY.

## Purpose

Answers the question: **"Why does Setup block? Pattern scarcity or filtering severity?"**

Setup Investigation identifies whether the dominant issue is:
1. **Pattern Scarcity** - Patterns genuinely don't exist
2. **Alignment Filtering** - Patterns exist but alignment threshold is too strict
3. **Mixed** - Both scarcity and filtering contribute
4. **Unclear** - Needs deeper decomposition

## Usage

```bash
# Run with default periods
python scripts/run_validation.py --mode setup_investigation --pair BTCUSDT

# Run with custom dates
python scripts/run_validation.py --mode setup_investigation \
    --pair BTCUSDT \
    --dates 2023-01-15,2023-03-15,2023-12-15
```

## How It Works

### 1. Run Setup Deep Dive First
```python
deep_dive_result = run_setup_deep_dive(pair, config, output_dir, dates)
```

Setup Investigation builds on Setup Deep Dive results rather than re-implementing analysis.

### 2. Load Deep Dive Results
```python
with open(f'{output_dir}/fractal/{pair}_setup_deep_dive.json') as f:
    deep_dive = json.load(f)
```

### 3. Aggregate Metrics
```python
total_no_pattern = sum(p['block_reasons']['no_pattern'] for p in periods.values())
total_alignment_low = sum(p['block_reasons']['alignment_score_too_low'] for p in periods.values())
total_blocks = sum(p['setup_blocked'] for p in periods.values())

no_pattern_rate = (total_no_pattern / total_blocks * 100)
alignment_rate = (total_alignment_low / total_blocks * 100)
```

### 4. Determine Root Cause

**Pattern Scarcity:**
```python
if no_pattern_rate > 80:
    return "pattern_scarcity"
```
→ Patterns genuinely don't exist. Need better pattern detector or more pattern types.

**Alignment Filtering:** ⚠️
```python
elif alignment_rate > 50:
    return "alignment_filtering"
```
→ **Patterns exist but are filtered by alignment threshold.** This is the most actionable finding - suggests relaxing alignment threshold.

**Mixed:**
```python
elif no_pattern_rate > 50 and alignment_rate > 20:
    return "mixed_scarcity_and_filtering"
```
→ Both issues contribute. Need to address both pattern detection and alignment.

**Unclear:**
```python
else:
    return "unclear"
```
→ Needs deeper decomposition (e.g., analyzing pattern candidate flow internally).

## Output

### JSON Report Structure

```json
{
  "pair": "BTCUSDT",
  "periods": {
    "2023-12-15T00:00:00": {
      "bars_ready_for_decision": 43,
      "setup_pass_rate": 2.3,
      "block_reasons": {
        "no_pattern": 20,
        "alignment_score_too_low": 18,
        ...
      },
      ...
    }
  },
  "cross_period_summary": {
    "no_pattern_rate": 46.5,
    "alignment_low_rate": 41.9,
    "main_root_cause": "alignment_filtering",
    "verdict": "Setup sees patterns, but alignment is the dominant blocker"
  }
}
```

## Example Results

### Typical Output (2023-12-15)

```
ROOT CAUSE ANALYSIS
===================

No pattern rate: 46.5%
Alignment too low rate: 41.9%

INVESTIGATION VERDICT
=====================

Main root cause: alignment_filtering

Setup sees patterns, but alignment is the dominant blocker
```

### Interpretation

**No pattern rate: 46.5%**
- About half the time, Setup genuinely doesn't find patterns
- This is expected - not every bar has valid SMC setups

**Alignment too low rate: 41.9%**
- About 40% of blocks happen because alignment_score < 0.6
- **This is the actionable finding**
- Patterns exist but are rejected by alignment filter

**Root cause: alignment_filtering**
- The dominant blocker is NOT pattern absence
- The dominant blocker IS alignment threshold being too strict
- **Action:** Relax alignment threshold from 0.6 to 0.4-0.5

## Causality Chain

Setup Investigation reveals the full causality chain:

```
Regime Feature Fix (sma_20/sma_50)
  ↓
Regime Unlocked (0% → 75% trend+)
  ↓
Setup Now Visible as Bottleneck (2.3% pass rate)
  ↓
Setup Deep Dive: alignment_dominant
  ↓
Setup Investigation: alignment_filtering
  ↓
ROOT CAUSE: Alignment threshold (0.6) too strict
  ↓
RECOMMENDATION: Relax alignment threshold to 0.4-0.5
```

## Root Cause Categories

### 1. Pattern Scarcity (>80% no_pattern)
**Diagnosis:** Patterns genuinely don't exist
**Actions:**
- Add more SMC pattern types (breaker blocks, mitigation zones)
- Lower pattern formation thresholds
- Expand pattern detection window

**Example:**
```
No pattern rate: 85%
Alignment rate: 10%
Root cause: pattern_scarcity
```

### 2. Alignment Filtering (>50% alignment_low) ⚠️
**Diagnosis:** Patterns exist but alignment threshold filters them
**Actions:**
- **Relax alignment threshold from 0.6 to 0.4-0.5** (RECOMMENDED)
- Re-evaluate alignment calculation
- Add alignment score distribution analysis

**Example:**
```
No pattern rate: 46.5%
Alignment rate: 41.9%
Root cause: alignment_filtering  ← Most common
```

### 3. Mixed Scarcity and Filtering
**Diagnosis:** Both pattern absence and alignment contribute
**Actions:**
- Address both issues in parallel
- Prioritize based on which rate is higher

**Example:**
```
No pattern rate: 55%
Alignment rate: 30%
Root cause: mixed_scarcity_and_filtering
```

### 4. Unclear
**Diagnosis:** Needs deeper decomposition
**Actions:**
- Analyze pattern candidate flow internally
- Add instrumentation to pattern formation
- Track rejection reasons inside detector

**Example:**
```
No pattern rate: 40%
Alignment rate: 35%
Root cause: unclear
```

## Actionable Recommendations

### Priority 1: Relax Alignment Threshold (RECOMMENDED)
**Current finding:** alignment_filtering (41.9% of blocks)

```yaml
# In config
setup:
  alignment_threshold: 0.4  # Was 0.6
```

**Expected impact:**
- Setup pass rate: 2.3% → 10-15%
- More signals reach Entry agent
- Some false positives, but currently too conservative

**Risk:** Low - currently filtering too aggressively

### Priority 2: Pattern Detection Enhancement
**If root cause is pattern_scarcity:**

Add more SMC pattern types:
- Breaker blocks
- Mitigation zones
- Liquidity voids
- Fair value gaps with extensions

**Risk:** Adds complexity without addressing root cause if alignment_filtering

### Priority 3: Pattern Filtering Audit
**If alignment filtering persists after threshold relaxation:**

Deep dive into pattern formation rules:
- Why are valid patterns getting low alignment scores?
- Is alignment calculation correct?
- Are there edge cases being missed?

**Risk:** Time-intensive investigation

## API

### `run_setup_investigation()`

```python
def run_setup_investigation(
    pair: str,
    config: Dict,
    output_dir: str,
    dates: List[datetime] = None
) -> Dict:
    """
    Setup Investigation - identify root cause
    
    Returns:
        {
            'status': 'completed',
            'pair': 'BTCUSDT',
            'main_root_cause': 'alignment_filtering',
            'verdict': 'Setup sees patterns, but alignment is the dominant blocker',
            'output': 'reports/.../BTCUSDT_setup_investigation.json'
        }
    """
```

## Testing

```bash
python tests/test_setup_investigation.py
```

Tests verify:
- Valid JSON output with root cause
- Root cause is one of expected values
- Verdict is actionable
- Metrics are aggregated correctly

## Files

**Created:**
- `scripts/setup_investigation.py` - Root cause analysis (~130 lines)
- `tests/test_setup_investigation.py` - Test suite
- `docs/SETUP_INVESTIGATION.md` - This documentation

**Output:**
- `reports/validation/fractal/{PAIR}_setup_investigation.json`

**Dependencies:**
- Runs `setup_deep_dive` first
- Loads and analyzes deep dive results

## Workflow

The complete diagnostic workflow:

```bash
# 1. Verify Regime is working
python scripts/run_validation.py --mode hsmm_deep_dive --pair BTCUSDT
# → Verdict: "Feature alignment fix successfully unlocked trend states"

# 2. Measure HOW Setup blocks
python scripts/run_validation.py --mode setup_deep_dive --pair BTCUSDT
# → Verdict: "Patterns exist, but alignment is the main blocker"

# 3. Identify WHY Setup blocks
python scripts/run_validation.py --mode setup_investigation --pair BTCUSDT
# → Root cause: "alignment_filtering"
# → Action: Relax alignment threshold to 0.4-0.5
```

## Key Insight

**Setup Investigation transforms vague "Setup blocks" into actionable "Relax alignment threshold from 0.6 to 0.4".**

This is the difference between knowing there's a problem and knowing how to fix it.
