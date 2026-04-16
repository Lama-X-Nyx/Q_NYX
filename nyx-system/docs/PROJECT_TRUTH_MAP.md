# Project Truth Map — Q_NYX

> High-level **layered view** of everything that lives in this repo.
> Drill-down detail lives in `docs/RUNNER_INVENTORY.md`.
> Architectural contract lives in `docs/ARCHITECTURE_CANONIQUE.md`.
> What is validated today lives in `docs/STATE_OF_PROJECT.md`.
>
> This map exists so a new contributor (or a new session) can answer,
> in under 2 minutes, the question **"where does this file fit?"**

---

## The 4 layers

```
┌────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — canonical runtime                                      │
│  The ONE approved path from OHLCV to decision to fill to log.     │
│  This is what runs in paper-live / will run in production.        │
└────────────────────────────────────────────────────────────────────┘
            ▲
            │ validated by
            │
┌───────────┴────────────────────────────────────────────────────────┐
│  LAYER 2 — offline calibration                                    │
│  Trains the artefacts. Validates OOS numbers. Never runs live.    │
└────────────────────────────────────────────────────────────────────┘
            ▲
            │ can be revived only by a deliberate ticket
            │
┌───────────┴────────────────────────────────────────────────────────┐
│  LAYER 3 — legacy                                                 │
│  v0.8 NYXEngine, legacy paper runner, old validation suite.       │
│  Not the approved runtime. Still on disk because deleting it      │
│  would break unrelated legacy code. Do NOT call from new code.   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — research / experimental                                │
│  Fractal agents, HSMM, SMC, setup-investigations, alternative ML  │
│  stacks, feedback-loop prototype. Not in any runtime path.        │
└────────────────────────────────────────────────────────────────────┘
```

The 4 layers are **disjoint by design**. No file is both canonical
runtime AND legacy. Every file under `src/` / `scripts/` / `tests/`
belongs to exactly one layer.

---

## Layer 1 — canonical runtime

This is the approved path. If a bug is fixed here, production
improves. If a test lands here, the system is safer.

### Entry points

| Entry point | Runs what |
|---|---|
| `python -m src.paper_live.main` | paper-live loop (Docker entry) |
| `NYXLiveDecider.on_15m_bar(bar)` | per-bar live inference |
| `NYXEngine.run(...)` | batch run over a historical window (pre-Ticket 04 called `NYXPipeline.run`) |
| `HubSpokeRunner.on_bars(bars)` | multi-asset orchestration tick |

### Decision engine

- `src/core/nyx_engine.py` — `NYXEngine` batch engine (Ticket 04 rename of NYXPipeline). Since Ticket 07 it delegates scoring to `src/core/meta_gbm.py::MetaGBM`, which now owns the decision path and encapsulates the trained `GradientBoostingClassifier` + scaler + 84-feature contract. Bear dial, soft gate sizing, cooldown, maker fees + slippage all stay downstream and consume `MetaDecision.probability`. Deprecation shim at `src/ml/nyx_pipeline.py` re-exports `NYXEngine as NYXPipeline` for pre-ticket-04 callers.
- `src/core/meta_gbm.py` — `MetaGBM` canonical strategy brain (Tickets 06 + 07). Wired as owner in NYXEngine.run() and NYXLiveDecider.on_15m_bar(). Emits `MetaDecision` per bar/candidate with 3 probability sources (precomputed, trained GBM via encapsulated artefact, heuristic aggregate). TRANSITIONAL status — encapsulates the validated GBM pending Option B retraining on FractalReports.
- `src/ml/nyx_live_decider.py` — `NYXLiveDecider` per-bar equivalent. Parity guarded by `tests/test_nyx_equivalence_replay_vs_live.py`.

### Data & feature layer

- `src/ml/mtf_feature_stack.py`, `src/ml/feature_buffer.py`, `src/ml/jesse_features.py`.

### Risk

- `src/ml/conditional_dial.py`, `src/ml/bear_risk_dial.py`, `src/ml/soft_gate.py`, `src/ml/execution_policy.py`.

### Multi-asset

- `src/assets/hub_spoke_runner.py`, `portfolio_allocator.py`, `nyx_live_pod.py`, `nyx_pipeline_pod.py`, `signal.py`, `registry.py`, `combined_portfolio.py`, `execution_profile.py`.

### Execution + logging + alerting

- `src/paper_live/*` (14 modules : `runner.py`, `post_only_broker.py`, `decision_logger.py`, `missed_trade_logger.py`, `state_manager.py`, `heartbeat.py`, `data_validator.py`, `event_alerter.py`, `multi_alerter.py`, `telegram_alerter.py`, `discord_alerter.py`, `resync_manager.py`, `multi_pair_runner.py`, `main.py`).

### Scripts that validate the runtime

- `scripts/validate_abc.py`, `validate_ab.py`, `validate_abc_via_hubspoke.py`, `walk_forward_trio.py`, `oos_live_replay.py`, `run_reality_checks.py`.

---

## Layer 2 — offline calibration

Runs during training, calibration, OOS validation. Never called from
any live path. Its outputs (artefacts, reports, parquets) are read by
Layer 1, but Layer 2 itself does not run in production.

### Training

- `src/ml/train_asset_model.py` (`train_and_save` persists
  `models/<SYMBOL>/ml_filter_v1.pkl` + scaler + names + metadata).
- `scripts/train_eth_model.py`, `scripts/train_sol_model.py`.

### Threshold & parameter calibration

- `src/ml/threshold_optimizer.py` (CV sweep on net PnL, offline only).
- `src/ml/ml_filter_v2.py` (consumed only by `threshold_optimizer`).
- `scripts/optimize.py`.

### Validation

- `src/ml/monte_carlo.py`, `bootstrap.py`, `walk_forward_splitter.py`, `reality_check.py`, `triple_barrier.py`, `jesse_labeler.py`, `realistic_backtest.py`, `feature_engine.py`.

### Data preparation

- `scripts/download_data.py`, `precompute_features.py`, `compute_4h_features.py`.

### Feedback loop (built, not yet wired)

- `src/ml/feedback_loop.py` — `DecisionLogger` + `OutcomeEvaluator` + `ChampionChallenger`. Reserved for future activation; currently zero runtime imports.

---

## Layer 3 — legacy

Historical v0.8 code. Still present so scripts referencing it don't
crash, and so we can always diff against it, but must NOT be called
from new code.

| File / module | Still imported by |
|---|---|
| `src/core/nyx_engine.py` | `src/runner/run_paper.py`, 5 × `src/validation/*`, `scripts/run_backtest.py` |
| `src/core/nyx_engine_mtf.py` | legacy runners |
| `src/core/precomputed_runner.py` | `scripts/backtest_mtf.py` |
| `src/core/risk_manager_mtf.py` | legacy runners |
| `src/runner/run_paper.py` | — (entry script) |
| `src/execution/paper_engine.py` | legacy runner |
| `src/validation/{walk_forward, oos_report, benchmarks, smc_diagnostics, signal_funnel, pattern_quality, order_block_audit, hsmm_deep_dive, reporting, metrics, manifest}.py` | scripts/run_validation.py |
| `src/ml/ml_filter.py` | tests only (v1, superseded) |
| `src/ml/edge_strategy.py` | **0 runtime imports — dead in runtime**, kept as walk-forward historical proof |
| `src/agents/orchestrator.py` | `scripts/backtest_mtf.py` |
| `src/paper_live/multi_pair_runner.py` | older main variant (v0.8 wrapper) |
| `scripts/run_backtest.py`, `run_validation.py`, `walk_forward_validation.py`, `optimize.py`, `backtest_mtf.py` | — |

**Rule** : any new code importing anything from this layer is a
regression. Block at review.

---

## Layer 4 — research / experimental

Exploratory code. Tests may be GREEN, but no production path uses it.

### Alternative agent stacks

- `src/ml/jesse_agents.py` (5 Jesse agents in one file) + `src/agents/*.py` (same 5 agents per-file). 45 GREEN tests. **Not wired** ; `NYXEngine` uses proxy `rule_*` scalars. Swap path in `docs/JESSE_AGENTS_STATUS.md`.
- `src/ml/ml_agents.py`, `ml_entry_agent.py`, `ml_orchestrator.py` (LightGBM + River alternative).
- `src/ml/jesse_strategy.py`, `jesse_backtest.py`, `jesse_ab_runner.py`, `jesse_research.py`, `jesse_utils.py`.

### Fractal / HSMM / SMC research

- `src/core/hsmm.py`, `src/core/smc.py`, `src/core/fractal_cached_runner.py`.
- `scripts/fractal_{check, geometry, bottleneck, cache_check, cache_benchmark}.py`.
- `scripts/setup_investigation*.py` (3 versions), `scripts/setup_{bottleneck, coverage_multi, deep_dive, detector_check}.py`.
- `scripts/smc_autopsy.py`, `regime_feature_check.py`, `regime_tuning.py`, `bullish_audit.py`, `component_edge_test.py`, `ale_analysis.py`, `test_parity.py`.
- `scripts/train_ml_ecosystem.py` (trains the LightGBM alt stack).

### Monitoring prototype

- `src/ml/model_monitor.py`.

---

## Decision rules for contributors

| You want to... | Rule |
|---|---|
| add a new runtime strategy | **no new layer** — patch `NYXEngine` (src/core/nyx_engine.py) or document deliberate replacement in a ticket |
| run a new backtest | use `NYXEngine.run` or `scripts/validate_abc*.py` — never the legacy v0.8 `NYXEngine` at `src/core/nyx_engine_v08.py` |
| calibrate a threshold / parameter | add to Layer 2 (`threshold_optimizer` pattern), not to the runtime |
| experiment with a new agent / ML stack | Layer 4 — keep it off any import path from Layer 1 |
| delete legacy code | open a dedicated ticket — must prove nothing in Layer 1 / 2 imports it |
| add a new script | add it to `docs/RUNNER_INVENTORY.md` with a status tag |

---

## How this map stays honest

- `tests/test_runner_inventory.py` asserts every ticket-minimum path
  is named in `RUNNER_INVENTORY.md` with a status tag.
- `TestNoMysteryRunner` walks `scripts/` and fails if any
  `if __name__ == '__main__':` script is missing from the inventory.
- `tests/test_architecture_canonical.py` asserts the canonical
  layer's entrypoint contract.
- Any commit that adds a runner and forgets to update the inventory
  breaks CI. By design.
