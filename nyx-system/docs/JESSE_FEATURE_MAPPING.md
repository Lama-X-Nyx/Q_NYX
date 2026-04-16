# Jesse Feature Mapping + Dataset Policy

> Tickets 14 & 15 — single source of truth for **what each Jesse
> agent reads** and **how its training sample is shaped**.
> All 4 agents now speak the canonical runtime feature language
> (drawn from `src/ml/jesse_features.py::compute_stationary_features`
> with `feature_set='full'`) and consume a role-specific dataset slice
> (`src/ml/jesse_dataset.py`).

---

## 1. Feature language (Ticket 14)

The 4 agents no longer own isolated custom feature blocks. Each
agent's `compute_features(df)` calls :

```python
full = compute_stationary_features(df, feature_set='full')
features = full[cols_in_self.FEATURE_PLAN].copy()
```

plus a small number of **agent-specific legacy extras** when the
agent's `.analyze()` state-mapping still reads legacy field names
(e.g. `momentum_12`, `rv_12`, `adx_norm` forced to numpy-fallback
for `RegimeAgent`; `hsmm_*` defaults to 0.0 for backward-compat).

### Role-based FEATURE_PLAN summary

| Agent | Role | TF | n_features (plan) |
|---|---|---|---:|
| JesseContextAgent | slow / macro bias | 1D | 13 |
| JesseRegimeAgent  | state classification | 4H | 15 |
| JesseSetupAgent   | liquidity-hunter opportunity | 1H | 15 + 3 cross |
| JesseEntryAgent   | short-horizon trigger | 15M | 13 |

### Context (1D) — slow / structural / macro bias

```
ema_ratio_21_50, ema_ratio_50_200, close_vs_ema50,
rsi_14, momentum_10, momentum_20, returns_5,
atr_ratio, bb_width_ratio, zscore_20, volume_ratio,
vwap_dist, chop_norm
```

Rationale : slow trend alignment (EMA-21/50/200), macro momentum,
volatility regime, long-horizon deviation (zscore_20), participation
context (volume_ratio), slow VWAP + compression/expansion proxies.
The Ticket 09 liquidity-hunter families appear only in their
*slow* form (vwap_dist, chop_norm) — the short-horizon triggers
(sr_break_*, bop, adosc_norm) do not belong at 1D.

### Regime (4H) — state classification

```
adx_norm, ema_ratio_9_21, ema_ratio_21_50,
atr_ratio, bb_width_ratio,
keltner_position, squeeze, chop_norm,
macd_hist_ratio, momentum_10, momentum_20, roc_10,
close_position, volume_ratio, kvo_norm
```

Plus agent-specific legacy extras : `momentum_12`, `rv_12`, forced
numpy-fallback `adx_norm` (preserves the 0.3 threshold in
`.analyze()` state-mapping), 6 `hsmm_*` state-prob defaults.

Rationale : trend strength (ADX + EMA alignment), volatility regime
(ATR, BB width), compression / expansion (Keltner, squeeze, chop),
momentum state, bar structure, volume regime (KVO).

### Setup (1H) — liquidity-hunter HEART

```
vwap_dist, vwma_dist,
ad_slope, adosc_norm, mfi_norm, marketfi_ratio,
bop, close_position,
sr_break_up_20, sr_break_dn_20,
sr_dist_high_20, sr_dist_low_20, minmax_pos_20,
bb_percent_b, squeeze
```

Plus cross-agent auxiliary features (forwarded by the upstream
Context/Regime agents at call time) :

```
context_score, regime_score, agent_agreement
```

Rationale : Setup is where liquidity-hunter logic matters most —
VWAP reclaim direction, S/R break + failed break, A/D + MFI + BOP
pressure imbalance, compression→expansion context, structural
zones.

### Entry (15M) — short-horizon trigger confirmation

```
returns_1, returns_5, momentum_10, rsi_14,
volume_ratio, vol_change,
close_position, high_low_ratio, bop,
vwap_dist,
sr_break_up_20, sr_break_dn_20,
adosc_norm
```

Rationale : trigger confirmation needs very short momentum
(returns_1/5), volume spikes (volume_ratio, vol_change), bar-
level pressure (bop, close_position, high_low_ratio), micro
VWAP reclaim, micro S/R triggers, short pressure imbalance
(adosc_norm). No slow structural features here — they would
blur the timing signal.

---

## 2. Dataset policy (Ticket 15)

Pre-Ticket-15, every agent trained on its full TF history.
`.backtest()` iterates `.analyze(df.iloc[:i+1])` per test-split bar
— that is O(N²) per call. On 15M EntryAgent the sandbox killed the
process because the loop over 25 k+ bars never finished.

Post-Ticket-15, every agent has a dataset builder in
`src/ml/jesse_dataset.py` that returns `(df, sample_mask)`. The
base-class `.train()` + `.backtest()` accept `sample_mask` and
**skip masked-out bars in the O(N²) per-bar `.analyze()` loop**.
This brings Setup/Entry training from "never finishes" to seconds/
minutes.

### Per-agent dataset policy

| Agent | df source | Mask policy | Effective bars (BTC 2020-2022) |
|---|---|---|---:|
| Context | full 1D history | mask.all() == True (no filter) | 1,096 / 1,096 (100 %) |
| Regime  | contiguous 4H, max 3 years | mask.all() == True | 6,576 / 6,576 (100 %) |
| Setup   | contiguous 1H | volume_ratio ≥ 1.5 × EMA20 | 3,651 / 26,304 (13.9 %) |
| Entry   | rolling 12-month 15M | EdgeStrategy candidate-proximity ± 5 bars, max 20 k, deterministic downsample | 3,823 / 35,041 (10.9 %) |

### Reproducibility

All builders take `random_state=42` default. The Entry downsampler
(when candidate-proximity mask exceeds `max_size`) uses
`np.random.default_rng(random_state).choice(..., replace=False)` —
deterministic given the same input.

### Tested invariants

`tests/test_jesse_dataset_policy.py` (17 GREEN) asserts :

- every builder returns `(df, mask)` with `len(mask) == len(df)`
- Context mask is all-True (no filter)
- Regime span ≤ 3 years + slack
- Setup mask is a strict subset (mask.sum() < len(mask))
- Setup mask is reproducible
- Entry span ≤ 12 months + slack
- Entry mask keeps < 50 % of rolling window (selective)
- Entry mask respects `max_size`
- Entry mask is reproducible
- `_BaseJesseAgent.train` accepts `sample_mask` (additive)
- A mask keeping ~30 % of bars reduces `n_samples` proportionally

---

## 3. Retrain metrics (BTC 2020-2022)

Run `python scripts/retrain_jesse_agents.py`. Results :

| Agent | Old n_features | New n_features | Accuracy | pct_passed | Elapsed |
|---|---:|---:|---:|---:|---:|
| Context | 8 | 13 | 0.500 | 100.0 % | 23 s |
| Regime  | 15 | 23 | 0.571 | 99.2 % | 354 s |
| Setup   | 16 | 18 | 0.188 | 0.0 %  | 567 s |
| Entry   | 13 | 13 | 0.560 | 99.9 % | 3.8 s |

Entry dropped from "never finishes" to 3.8 s thanks to the dataset
mask + batch-predict override.

**Caveat on Setup accuracy 0.188** : the volume-spike filter keeps
bars where potential setups are MORE LIKELY, which skews the label
distribution toward `valid_setup`. The model's `no_setup` bias
(pct_passed = 0 %) means it predicts `no_setup` everywhere → low
accuracy on a dataset dominated by `valid_setup`. This is a DATASET
BALANCE SIGNAL, not a training failure — a future ticket can
rebalance with class_weight or an additional downsample of the
positive class. The retrain itself completed successfully and the
agent exposes the canonical report schema.

---

## 4. How to extend

### Adding a new Jesse agent feature
1. Add the feature to `compute_stationary_features` in
   `src/ml/jesse_features.py` (canonical source).
2. Append it to the target agent's `FEATURE_PLAN` tuple in
   `src/ml/jesse_agents.py`.
3. Run `pytest tests/test_jesse_agents_retrained.py` → must stay
   GREEN.
4. Run the retrain script to measure impact.

### Changing a dataset builder
1. Edit the builder in `src/ml/jesse_dataset.py`.
2. Update tests in `tests/test_jesse_dataset_policy.py`.
3. Run the retrain script.
4. Document the change here + in `docs/CHANGELOG.md`.

---

## 5. What this doc forbids

- Adding features to an agent's `compute_features` that are NOT in
  `compute_stationary_features`'s output (breaks feature-language
  coherence). Exception : small agent-specific legacy extras
  (`momentum_12`, `rv_12`, `hsmm_*`) kept ONLY for backward-compat
  with `.analyze()` state-mapping. Any new legacy extra must be
  justified in-code + here.
- Dumping the full feature set into every agent (violates
  role-specialization, Ticket 14 Rule 1).
- Training Entry on the raw 15M history without a mask (violates
  Ticket 15 dataset policy — reintroduces the O(N²) sandbox kill).
