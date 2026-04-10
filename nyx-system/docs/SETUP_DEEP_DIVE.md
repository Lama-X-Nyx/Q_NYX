# Setup Deep Dive

## Overview

Setup Deep Dive measures precisely **HOW** Setup blocks the pipeline by analyzing orchestrator decisions across multiple periods. Now that Regime is no longer the main blocker (75% trend+), this diagnostic identifies where Setup failures occur.

## Purpose

Answers the question: **"How does Setup block 97.7% of signals?"**

Unlike simplified aggregations, Setup Deep Dive:
- Iterates through actual bars using orchestrator
- Analyzes `decision.components['setup']` for each decision
- Categorizes specific blocking reasons
- Counts patterns observed vs rejected
- Analyzes alignment score distributions

## Usage

```bash
# Run with default periods (2022-06-15, 2023-03-15, 2023-12-15)
python scripts/run_validation.py --mode setup_deep_dive --pair BTCUSDT

# Run with custom dates
python scripts/run_validation.py --mode setup_deep_dive \
    --pair BTCUSDT \
    --dates 2023-01-15,2023-03-15,2023-12-15
```

## How It Works

### 1. Load Fractal Context
```python
mtf_data = load_fractal_context(pair, reference_date, config)
```

### 2. Iterate Through Bars
```python
for i in range(warmup, warmup + n_signals):
    current_time = mtf_data[lowest_tf].index[i]
    aligned_data = loader.align_at_timestamp(mtf_data, current_time, lowest_tf)
    decision = orchestrator.decide(aligned_data)
```

### 3. Analyze Setup Component
```python
setup_comp = decision.components.get('setup')
setup_state = setup_comp.state
setup_passed = setup_comp.passed
setup_score = setup_comp.score
```

### 4. Categorize Blocking Reasons
- `no_pattern`: Setup score < 0.1, state is 'misaligned'
- `alignment_score_too_low`: Score < 0.6 (threshold)
- `misaligned_pattern`: Score ≥ 0.6 but still blocked
- `unknown`: Other reasons

### 5. Count Patterns
- `bullish_fvg`, `bearish_fvg`
- `bullish_ob`, `bearish_ob`
- `no_pattern`

## Output

### JSON Report Structure

```json
{
  "pair": "BTCUSDT",
  "periods": {
    "2023-12-15T00:00:00": {
      "bars_ready_for_decision": 43,
      "setup_passed": 1,
      "setup_blocked": 42,
      "setup_pass_rate": 2.3,
      "block_reasons": {
        "no_pattern": 20,
        "alignment_score_too_low": 18,
        "misaligned_pattern": 3,
        "direction_conflict": 0,
        "setup_not_ready": 0,
        "unknown": 1
      },
      "patterns_observed": {
        "bullish_fvg": 5,
        "bearish_fvg": 7,
        "bullish_ob": 3,
        "bearish_ob": 7,
        "no_pattern": 20
      },
      "alignment_summary": {
        "mean": 0.32,
        "median": 0.28,
        "min": 0.05,
        "max": 0.89,
        "threshold": 0.6
      },
      "primary_setup_issue": "alignment_score_too_low",
      "sample_blocked_setups": [...]
    }
  },
  "cross_period_summary": {
    "average_pass_rate": 2.3,
    "main_issue": "alignment_dominant",
    "verdict": "Patterns exist, but alignment is the main blocker"
  }
}
```

## Verdict Logic

Setup Deep Dive automatically determines the dominant blocking reason:

### Pattern Absence Dominant
```python
if no_pattern_rate > 0.80:
    return "Setup is dominated by total absence of patterns"
```

### Alignment Dominant ⚠️
```python
elif alignment_low_rate > 0.50:
    return "Patterns exist, but alignment is the main blocker"
```
**This is the most common verdict**, indicating alignment threshold (0.6) is too strict.

### Mixed Pattern and Alignment
```python
elif no_pattern_rate > 0.50 and alignment_low_rate > 0.20:
    return "Setup is mixed: both pattern scarcity and alignment matter"
```

### Severe Blocking
```python
elif avg_pass_rate < 5:
    return "Setup blocks almost everything (<5% pass rate)"
```

## Example Results

### 2023-12-15 Period
```
Bars ready: 43
Setup pass rate: 2.3%
Primary issue: alignment_score_too_low

Block reasons breakdown:
  No pattern: 20 (46.5%)
  Alignment too low: 18 (41.9%)
  Misaligned pattern: 3 (7.0%)
  Unknown: 1 (2.3%)

Alignment scores:
  Mean: 0.32
  Median: 0.28
  Threshold: 0.6

Verdict: "Patterns exist, but alignment is the main blocker"
```

## Key Insights

**Setup sees patterns (22 patterns observed) but rejects most due to alignment < 0.6.**

This indicates:
1. Pattern detector is working (finds FVG, OB)
2. Alignment filter is too conservative
3. **Action:** Consider relaxing alignment threshold to 0.4-0.5

## API

### `run_setup_deep_dive()`

```python
def run_setup_deep_dive(
    pair: str,
    config: Dict,
    output_dir: str,
    dates: List[datetime] = None
) -> Dict:
    """
    Run Setup Deep Dive analysis
    
    Returns:
        {
            'status': 'completed',
            'pair': 'BTCUSDT',
            'avg_pass_rate': 2.3,
            'main_issue': 'alignment_dominant',
            'output': 'reports/validation/fractal/BTCUSDT_setup_deep_dive.json'
        }
    """
```

### `_determine_setup_verdict()`

```python
def _determine_setup_verdict(period_results: List[Dict]) -> tuple:
    """
    Determine main Setup blocking reason across periods
    
    Returns:
        (main_issue, verdict) tuple
    """
```

## Testing

```bash
python tests/test_setup_deep_dive.py
```

Tests verify:
- Valid JSON output produced
- Blocking reasons categorized correctly
- Verdict logic works
- Handles zero patterns gracefully

## Files

**Created:**
- `scripts/setup_deep_dive.py` - Complete orchestrator analysis (~350 lines)
- `tests/test_setup_deep_dive.py` - Test suite
- `docs/SETUP_DEEP_DIVE.md` - This documentation

**Output:**
- `reports/validation/fractal/{PAIR}_setup_deep_dive.json`

## Next Steps

After identifying HOW Setup blocks (alignment dominant), use **Setup Investigation** to understand WHY this happens (root cause analysis).

```bash
python scripts/run_validation.py --mode setup_investigation --pair BTCUSDT
```
