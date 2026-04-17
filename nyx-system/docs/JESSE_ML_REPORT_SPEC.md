# Jesse ML Report Specification

> Tickets 18A + 18B — single source of truth for what "ML-native
> Jesse report" means in this project.

---

## 1. What ML-native means

An ML-native Jesse agent is one where the **trained model's
`predict_proba` output** is the PRIMARY owner of :

- the state selection (`bullish`/`bearish`/`neutral` for Context,
  `trend_plus`/`trend_minus`/`range`/`squeeze` for Regime,
  `valid_setup`/`no_setup` for Setup, `ready`/`not_ready` for Entry)
- the score (= model probability, not a hand-crafted blend)
- the probability fields in metadata (`p_bull`, `p_trend`, etc.)
- the `passed` decision (based on a calibrated threshold on the
  model probability, not on heuristic indicators)

**The model decides. The code implements that decision.**

---

## 2. Allowed residual heuristics

Heuristics are allowed ONLY in these roles :

| Role | Example | Allowed? |
|---|---|---|
| Direction sign within a trending state | Regime reads `momentum_12` sign for trend_plus vs trend_minus | ✓ (direction is not a decision — it's the sign of the model's trend) |
| Safety veto for impossible physics | Squeeze detection via `rv < 0.002 AND adx < 0.2` (cannot have a breakout with zero vol) | ✓ |
| Fallback when model is unavailable | Return neutral defaults if `_is_trained == False` | ✓ |
| Cross-agent gate | Setup passes only if `context_score >= 0.4 AND regime_score >= 0.3` | ✓ (composition rule, not heuristic signal) |

**NOT allowed** :

| Role | Example | Allowed? |
|---|---|---|
| Primary signal blend | `p = 0.5 * p_ml + 0.5 * h_heuristic` | ✗ (removed in T18A) |
| State determined by indicator | `if ADX > 0.3: state = trend_plus` regardless of model | ✗ (removed in T18A) |
| Score from heuristic recipe | 7-component liquidity-signal sum | ✗ (removed in T18A; the features ARE in the model's training set so the RandomForest already learned from them) |

---

## 3. Report probability semantics

### Context (1D)
- `p_bull` : P(macro bullish direction) from model class +1
- `p_bear` : P(macro bearish) from model class -1 (or 1 - p_bull for binary)
- `p_neutral` : 1 - p_bull - p_bear
- `context_confidence` : score = max probability

### Regime (4H)
- `p_trend` : P(trending) from model class +1. Threshold ≥ 0.55 → pass
- `dominant_state` : `trend_plus` if p_trend ≥ 0.55 AND momentum ≥ 0,
  `trend_minus` if p_trend ≥ 0.55 AND momentum < 0,
  `squeeze` if rv < 0.002 AND adx < 0.2 (safety veto),
  `range` otherwise
- `regime_confidence` : score = p_trend

### Setup (1H)
- `p_setup_ml` : P(valid setup) from model class +1. Threshold ≥ 0.55 → pass
- `setup_confidence` : score = p_setup_ml
- Cross-agent gate : requires context_score ≥ 0.4 AND regime_score ≥ 0.3

### Entry (15M)
- `p_up` : P(price goes up from here) from model class +1
- `p_down` : P(price goes down) from model class -1
- `p_neutral` : P(flat / stopped out) from model class 0
- `entry_confidence` : score = max(p_up, p_down)
- Decision : p_up ≥ 0.45 AND p_up > p_down + 0.15 → ready long,
  p_down ≥ 0.45 AND p_down > p_up + 0.15 → ready short, else not ready

---

## 4. `passed` semantics

`passed = True` means the agent sees an actionable signal at this
bar. It does NOT mean "the agent recommends a trade" — that is
the Meta-GBM's role (Tickets 06/07). `passed` is a **local opinion**
that the fractal-level conditions are met for this agent's role.

---

## 5. Confidence semantics

`score` in the `FractalReport` = the agent's model-derived
confidence in its own state selection. It is NOT a trading score.
Values cluster around the model's probability distribution
(typically 0.3 – 0.7 on BTC data for most agents).

---

## 6. Current calibration (BTC 2020-2022, T18B)

| Agent | Acc | pct_passed | avg_score | Status |
|---|---:|---:|---:|---|
| Context | 0.503 | 83.6 % | 0.503 | ✓ Stable, bullish-biased on BTC bull period |
| Regime | 0.571 | 55.0 % | 0.571 | ✓ Discriminant, ML-native threshold 0.55 |
| Setup | 0.470 | 19.4 % | 0.470 | ✓ Selective, ML-native threshold 0.55 |
| Entry | 0.560 | 99.9 % | 0.596 | ✓ Permissive by design (candidate mask pre-filters) |

### Runtime-readiness verdict per agent

| Agent | Ready for future runtime? | Note |
|---|---|---|
| Context | ✓ Yes | Stable, meaningful directional bias |
| Regime | ✓ Yes | Discriminant post-threshold-calibration |
| Setup | ✓ Yes | Selective, ML-native, within target zone |
| Entry | ⚠ Conditional | pct_passed 99.9 % means it barely filters — useful only as probability enrichment, not as gate |
