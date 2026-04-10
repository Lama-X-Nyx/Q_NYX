# HSMM Verdict Refinement

## Overview

The HSMM Deep Dive verdict system was refined to correctly identify successful improvements after the Regime feature alignment fix (sma_20/sma_50), instead of returning vague "unclear" verdicts when metrics clearly show progress.

## Problem

**Before refinement:**
```
Metrics: 71.47% trend+, 27.24% range, 3/4 periods select trend+
Verdict: "unclear - Multiple factors may be contributing"
```

The original verdict logic couldn't detect when the HSMM had been successfully unlocked.

## Solution

Added new verdict cases that detect improvement by analyzing both:
- State probability distributions
- Selected state counts across periods

### New Verdict Cases

**Case A: Feature Alignment Fixed** ✅
```python
if avg_trend_plus > 0.60 and avg_range < 0.40:
    if trend_plus_rate >= 0.5:
        return "feature_alignment_fixed"
```
- Triggers when: >60% trend+ probability, <40% range, ≥50% periods select trend+
- Verdict: "HSMM feature alignment fix successfully unlocked trend states"

**Case B: Partial Improvement**
```python
if 0.30 < avg_trend_plus <= 0.60 and avg_range < 0.70:
    return "partial_improvement"
```
- Triggers when: 30-60% trend+ probability
- Verdict: "HSMM improved partially, but range still dominates some periods"

**Case C: Mixed Results** (instead of "unclear")
```python
return "mixed_results"  # Default case
```
- Better than vague "unclear"
- Indicates investigation needed but acknowledges mixed signals

## Usage

The verdict is automatically determined when running HSMM Deep Dive:

```bash
python scripts/run_validation.py --mode hsmm_deep_dive --pair BTCUSDT
```

## Results After Refinement

**After fix (71% trend+):**
```
Main issue: feature_alignment_fixed
Verdict: "HSMM feature alignment fix successfully unlocked trend states"
```

**Partial improvement (45% trend+):**
```
Main issue: partial_improvement
Verdict: "HSMM improved partially, but range still dominates some periods"
```

## API

### `_determine_verdict()`

```python
def _determine_verdict(
    trend_plus_probs: List[float],
    range_probs: List[float],
    returns_means: List[float],
    returns_stds: List[float],
    selected_states: List[str] = None
) -> tuple:
    """
    Determine main issue and verdict
    
    Returns:
        (main_issue, verdict) tuple
    """
```

**New parameter:**
- `selected_states`: List of selected states per period (e.g., ['trend_plus', 'trend_plus', 'range', 'trend_plus'])

**Returns:**
- `main_issue`: One of {feature_alignment_fixed, partial_improvement, range_state_dominance, weak_features, mapping_rigidity, initialization_bias, mixed_results}
- `verdict`: Human-readable explanation

## Testing

Run tests:
```bash
cd /home/claude/nyx-phase1/nyx-system
python tests/test_hsmm_verdict_refinement.py
```

Tests verify:
- Successful fix detection (71% trend+ → feature_alignment_fixed)
- Partial improvement detection (45% trend+ → partial_improvement)
- Strong metrics never return "unclear"

## Files Modified

- `src/validation/hsmm_deep_dive.py` - Added refined verdict logic

## Impact

**Before:** Confusing "unclear" verdicts despite clear improvement
**After:** Actionable verdicts that correctly identify successful fixes

This enables proper diagnosis of whether Regime improvements actually work.
