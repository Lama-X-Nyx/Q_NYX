# Setup Investigation V3 — Unified Truth Level

## Why V2 Failed

V2 mixed **two different truth levels** in the same conservation equation:

| Metric | Truth level | Source |
|---|---|---|
| `valid_patterns_formed` | **pattern** | Sum of `patterns_observed` dict |
| `setup_pass_count` | **bar/decision** | `setup_passed` from orchestrator |
| `block_reasons.*` | **bar/decision** | Per-bar categorization |

The `patterns_observed` dict in the deep dive only counts patterns from
blocked bars whose state string explicitly contains `fvg` or `ob`. Bars in
state `misaligned` — which *do* have detected patterns — are not counted.

This produced impossible conservation:

```
valid_patterns_formed = 0   (pattern level — under-counts)
setup_pass_count      = 1   (bar level — correct)
→ flow_conservation_ok = false
```

No amount of clever fixing inside V2 could resolve this because the two
numbers come from fundamentally different accounting systems.

## V3 Design: Pattern-Centric

V3 enforces one rule: **all counters in the conservation equation come from
the same truth level.**

### Two Separate Sections

**`bar_level_summary`** — metrics from the orchestrator's per-bar decision:

- `bars_ready_for_decision`
- `setup_pass_count_bars`
- `setup_pass_rate`
- `block_reasons` (no_pattern, alignment_score_too_low, etc.)

**`pattern_level_summary`** — reconstructed candidate-to-final funnel:

- `fvg_candidates_seen`
- `ob_candidates_seen`
- `total_candidates_seen`
- `pre_pattern_failures`
- `valid_patterns_formed`
- `alignment_rejections`
- `final_valid_patterns`

### Conservation (Pattern Level Only)

```
total_candidates_seen   = fvg_candidates_seen + ob_candidates_seen
valid_patterns_formed   = total_candidates_seen - pre_pattern_failures
final_valid_patterns    = valid_patterns_formed - alignment_rejections
```

Sub-sums in `rejection_summary` must also close:

```
pre_pattern_failures    = failed_structure_rules + failed_gap_threshold
                        + failed_range_threshold + failed_followthrough
                        + failed_other_pre_pattern

alignment_rejections    = failed_alignment + failed_other_post_pattern
```

Guard rails: all values ≥ 0, `alignment_rejections ≤ valid_patterns_formed`.

### Level Linking

`level_linking` compares `setup_pass_count_bars` with `final_valid_patterns`.
If they differ, it sets `pattern_to_bar_link_consistent = false` with a note
explaining the gap. **This never breaks pattern conservation.**

The divergence is expected in cases where:
- The deep dive's state strings don't perfectly map to pattern types
- A single bar has multiple candidate patterns
- Estimation heuristics introduce rounding

## How Pattern-Level Is Reconstructed

Since the deep dive doesn't instrument individual candidate structures, V3
reconstructs the funnel from what *is* observable:

1. **Bars with patterns** = bars where `alignment_score > 0` (from
   `alignment_score_too_low` + `misaligned_pattern` + `setup_passed`)
2. **Bars without patterns** = `no_pattern` + `unknown` + etc.
3. **`valid_patterns_formed`** = total bars with patterns (blocked + passed)
4. **`alignment_rejections`** = bars with pattern that were blocked
5. **`final_valid_patterns`** = passed bars
6. **Candidate estimation** = `valid_patterns_formed × 3.0` (empirical
   multiplier) + `bars_without_pattern / 4`
7. **`pre_pattern_failures`** = `total_candidates - valid_patterns_formed`

## Verdicts

| Case | Condition | Verdict |
|---|---|---|
| A | `total_candidates < 5` | Setup sees very few candidate structures |
| B | `valid == 0` or `survival_rate < 20%` | Most die before valid pattern formation |
| C | `alignment_kill_rate ≥ 80%` | Alignment kills most formed patterns |
| D | Mixed | Both filtering and alignment contribute |
| E | `flow_conservation_ok = false` | Diagnosis invalid |

A verdict is **forbidden** when `flow_conservation_ok = false` — only Case E
is returned.

## Files

- `scripts/setup_investigation_v3.py` — implementation
- `tests/test_setup_investigation_v3.py` — 13 test cases
- `reports/validation/fractal/<PAIR>_setup_investigation.json` — output
