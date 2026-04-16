# Architecture canonique NYX

> **This is the ONE canonical architecture for NYX.** Before any
> retraining or optimisation, the team uses this document to agree on
> what the runtime path is, what the offline path is, and what every
> module does inside that path.
>
> Ticket 01 — Freeze the canonical NYX architecture. Related docs :
> `STATE_OF_PROJECT.md` (what is validated today) and
> `OPERATING_RULES.md` (7 test-enforced technical invariants).

---

## Why we need this doc

The repository carries layers of history : `NYXEngine v0.8` → refonte
v0.2.5 (`edge_strategy`, `ml_filter_v2`, `threshold_optimizer`,
feedback / reality-check layers) → pipeline unifié v0.3.1
(`NYXEngine`, originally named `NYXPipeline` = monolithic GBM) →
hub-and-spoke multi-asset →
paper-live infra → real-time live decider.

Without a canonical statement, every session re-debates :

- "Is `edge_strategy` a runtime strategy or a calibration helper?"
- "Do the 5 Jesse agents run in production?"
- "Is the Meta-GBM something new, or the GBM already inside `NYXEngine`?"
- "Where does `threshold_optimizer` live at runtime?"

This doc freezes the answer. When a future PR disagrees with any
sentence here, either the PR is wrong or **this doc must be updated
in the same commit** — never silently.

---

## Canonical runtime path

```
Data MTF (15m + 1h + 4h + 1d OHLCV)
   │
   ▼
MTFFeatureStack — 4 rolling buffers + 15m→1h/4h/1d aggregation
   │
   ▼
Fractal reporters (Jesse Context / Regime / Setup / Entry)
   │       (CURRENT IMPLEMENTATION: proxied by hand-crafted
   │        rule_context / rule_regime / rule_setup / disagreement
   │        scalars inside NYXEngine — see "Honest status" below)
   ▼
Meta-GBM strategy brain — GradientBoostingClassifier (threshold 0.60)
   │                      owned by NYXEngine.run
   ▼
Risk manager — conditional_dial + bear_risk_dial
   │           (per-bar conditional threshold + cooldown + size mult)
   ▼
Execution layer — PostOnlyPaperBroker (maker fees, no taker fallback)
   │              + HubSpokeRunner + PortfolioAllocator (multi-asset)
   ▼
Logging / feedback — PersistentDecisionLogger (append-only SQLite)
                     + missed_trades.db + heartbeat
                     + feedback_loop.py (champion-challenger)
```

### Canonical runtime entrypoint

Per Ticket 04, the canonical runtime entrypoint is `NYXEngine` at
`src/core/nyx_engine.py` :

```python
from src.core.nyx_engine import NYXEngine
engine = NYXEngine()
result = engine.run(mtf_data, mtf_features, train_end, test_start, test_end)
```

Live inference uses the equivalent per-bar engine :

```python
from src.ml.nyx_live_decider import NYXLiveDecider
decider = NYXLiveDecider(symbol='ETHUSDT', artifact_dir=...)
sig = decider.on_15m_bar(bar)
```

A regression guard (`tests/test_nyx_equivalence_replay_vs_live.py`)
asserts that `NYXLiveDecider` never drifts from `NYXEngine.run`
on the same historical window.

**Deprecation shim**: `src/ml/nyx_pipeline.py` re-exports
`NYXEngine as NYXPipeline` so the ~45 pre-ticket-04 callers keep
working unchanged through the migration. New code imports `NYXEngine`
directly from `src/core/nyx_engine.py`.

**Legacy v0.8 engine** (HSMM + SMC + macro) lives at
`src/core/nyx_engine_v08.py` for its 9 legacy callers
(`src/runner/run_paper.py`, 5 × `src/validation/*.py`,
`scripts/run_backtest.py`). Not canonical. Do not import it from new
code.

---

## Canonical offline path

```
Training data (parquet 15m/1h/4h/1d OHLCV + features)
   │
   ▼
compute_stationary_features — src/ml/jesse_features.py
   │
   ▼
NYXEngine._generate_candidates — hard gate + outcome labels
   │
   ▼
calibration — threshold_optimizer.py (offline-only sweep,
   │                                   CV on net PnL)
   ▼
training — train_asset_model.train_and_save (persists
   │        ml_filter_v1.pkl + scaler.pkl + feature_names.json)
   ▼
OOS / walk-forward — tests/test_oos_final.py,
   │                  tests/test_walk_forward_trio.py
   ▼
reality checks — Monte Carlo shuffle, block bootstrap,
                 post-only miss rate, taker sensitivity,
                 daily Sharpe (not per-trade), forced-stop B&H
```

---

## Component ownership — runtime vs offline

| Component | Role | File | Runtime or Offline |
|---|---|---|---|
| `MTFFeatureStack` | Data layer — 4 rolling TF buffers | `src/ml/mtf_feature_stack.py` | runtime |
| `IncrementalFeatureBuffer` | Per-TF sliding-window features | `src/ml/feature_buffer.py` | runtime |
| **`JesseContextAgent`** (fractal reporter, 1D) | Fractal reporter | `src/ml/jesse_agents.py` + `src/agents/context_agent.py` | runtime (DORMANT — see honest status) |
| **`JesseRegimeAgent`** (fractal reporter, 1H) | Fractal reporter | `src/ml/jesse_agents.py` + `src/agents/regime_agent.py` | runtime (DORMANT) |
| **`JesseSetupAgent`** (fractal reporter, 15M) | Fractal reporter | `src/ml/jesse_agents.py` + `src/agents/setup_agent.py` | runtime (DORMANT) |
| **`JesseEntryAgent`** (fractal reporter, 15M) | Fractal reporter | `src/ml/jesse_agents.py` + `src/agents/entry_agent.py` | runtime (DORMANT) |
| `JesseOrchestrator` | Meta rule-based combiner — ALTERNATIVE to Meta-GBM | `src/ml/jesse_agents.py` + `src/agents/orchestrator.py` | runtime (DORMANT) |
| **Meta-GBM (strategy brain)** | `GradientBoostingClassifier`, threshold 0.60, 84 features | inside `NYXEngine.run` | **runtime — ACTIVE** |
| `conditional_dial` | Per-bar conditional risk dial (bear triggers) | `src/ml/conditional_dial.py` | runtime |
| `bear_risk_dial` | Risk-parameter table (bull/range/bear) | `src/ml/bear_risk_dial.py` | runtime |
| `soft_gate` | Size factor + disagreement | `src/ml/soft_gate.py` | runtime |
| `PostOnlyPaperBroker` | Honest execution (no taker fallback) | `src/paper_live/post_only_broker.py` | runtime |
| `HubSpokeRunner` + `PortfolioAllocator` | Multi-asset hub | `src/assets/` | runtime |
| `PersistentDecisionLogger` | Append-only SQLite decisions | `src/paper_live/persistent_decisions.py` | runtime |
| `StateManager` + `Heartbeat` | Atomic state + liveness | `src/paper_live/*` | runtime |
| `feedback_loop` | Champion-challenger + outcome eval | `src/ml/feedback_loop.py` | runtime |
| `edge_strategy` | Historical trend+volume edge validator (walk-forward only) | `src/ml/edge_strategy.py` | **offline-only** — not a standalone runtime strategy |
| `threshold_optimizer` | CV threshold sweep | `src/ml/threshold_optimizer.py` | **offline-only** |
| `ml_filter_v2` | Legacy GBM + CV threshold (consumed by `threshold_optimizer`) | `src/ml/ml_filter_v2.py` | **offline-only** (not imported at runtime) |
| `realistic_backtest` | Legacy standalone backtester | `src/ml/realistic_backtest.py` | offline-only |
| `monte_carlo`, `bootstrap` | OOS simulation | `src/ml/*` | offline-only |
| `jesse_features`, `jesse_labeler` | Stationary features + triple-barrier labels | `src/ml/*` | offline (reused by runtime for inference-time features) |

---

## Honest status of the 4 Jesse fractal reporters (CRITICAL)

The 4 fractal reporters (`JesseContextAgent`, `JesseRegimeAgent`,
`JesseSetupAgent`, `JesseEntryAgent`) are **implemented, tested (45
GREEN tests + 60 new Ticket 05 tests = 105 GREEN), but NOT WIRED into
the canonical runtime today.**

Since Ticket 05, every Jesse agent (mono-file `src/ml/jesse_agents.py`
+ per-file `src/agents/*.py`) exposes the canonical reporter :

```python
agent.report(df, asset, timestamp, **analyze_kwargs) -> FractalReport
```

with class-level `REPORT_AGENT` and `REPORT_TIMEFRAME` constants
(Context=1d, Regime=4h, Setup=1h, Entry=15m) enforcing the fractal
timeframe stack. See `JESSE_AGENTS_STATUS.md` for the full reporter
API contract and the test matrix.

Instead, `NYXEngine._generate_candidates` computes **hand-crafted
proxy scalars** with identical semantic roles :

| Canonical fractal reporter | Proxy scalar today |
|---|---|
| `JesseContextAgent.report().score` | `rule_context` (clip of `(ema9 − ema50) / ema50 × 20 + 0.5`) |
| `JesseRegimeAgent.report().score` | `rule_regime` (clip of `atr / close × 200`) |
| `JesseSetupAgent.report().score` | `rule_setup` (clip of `|ema9 − ema21| / ema21 × 100`) |
| `JesseEntryAgent.report().score` | implicit — carried by `direction` + `hour_norm` + `volume_spike` |
| Reporter disagreement | `disagreement` (from `compute_disagreement` in `soft_gate.py`) |

These proxies are fed to the Meta-GBM **exactly like the reporter
scores would be**. The swap path — replacing the proxy scalars with
live calls to the 4 Jesse reporters — is documented in
`JESSE_AGENTS_STATUS.md`. Until the swap is deliberate and tested, the
canonical runtime uses the proxy scalars.

**Do not read this as "the Jesse agents are deprecated".** They are
the canonical *interface*; the proxies are the canonical
*implementation* today. The GBM contract (feature names, threshold
0.60, training labels) is identical in both cases.

---

## Meta-GBM — the strategy brain

"Meta-GBM" in this architecture refers to the single
`GradientBoostingClassifier` owned by `NYXEngine` (or loaded from
`models/<SYMBOL>/ml_filter_v1.pkl` for live inference). It is *meta*
because it consumes the outputs of the 4 fractal reporters (or their
proxy scalars) together with the full MTF feature block and emits
the strategy decision probability.

- **Estimator** : `GradientBoostingClassifier(n_estimators=300,
  max_depth=3, random_state=42)`
- **Input** : 84 features — 24 (15m) + 17 (`h1_*`) + 17 (`h4_*`) +
  17 (`d1_*`) + `ctx_1d_*` + `reg_1h_*` + `rule_*` + `disagreement` +
  5 extras
- **Output threshold** : `0.60` by default (bull), `0.68` in
  conditional bear dial, calibrated by `threshold_optimizer` offline
- **Label** : sign of `outcome_net` after TP/SL/TIME + maker fees +
  slippage

There is no "Meta-GBM" class distinct from NYXEngine's GBM. The
name "Meta-GBM" names the **role** (strategy brain), not a new
module.

---

## What this doc forbids

- Referring to `edge_strategy.py` as a runtime strategy. It is a
  historical walk-forward validation helper. Not wired at runtime.
- Referring to `threshold_optimizer.py` or `ml_filter_v2.py` as the
  live classifier. They are offline calibration. The live classifier
  is `NYXEngine`'s internal GBM / persisted `ml_filter_v1.pkl`.
- Adding a "v2" pipeline alongside `NYXEngine` (see CLAUDE.md
  rule 6 — No architectural improvisation).
- Silently re-wiring Jesse agents into the runtime without updating
  the "Honest status" section above AND adding a matching regression
  test (`tests/test_nyx_equivalence_replay_vs_live.py` or similar).

---

## Acceptance criteria (ticket 01)

- ☑ A single canonical runtime path is documented (see above).
- ☑ A single canonical offline path is documented (see above).
- ☑ `edge_strategy` is no longer described as a standalone strategy
  (table explicit : **offline-only**).
- ☑ `threshold_optimizer` is explicitly marked offline-only.
- ☑ The 4 Jesse agents are explicitly defined as fractal reporters,
  with an honest "not wired today, proxied by rule_* scalars" status.
- ☑ The Meta-GBM is explicitly defined as the strategy brain —
  `GradientBoostingClassifier`, threshold 0.60, inside `NYXEngine`.

---

## Maintenance rule

Any commit that :

- adds a runtime-time strategy, or
- promotes an offline-only module to runtime, or
- wires Jesse agents into production, or
- replaces the Meta-GBM with a different strategy brain

**must update this doc in the same commit** — and must update the
matching assertion in `tests/test_architecture_canonical.py`. Breaking
that test = breaking the canonical architecture contract.
