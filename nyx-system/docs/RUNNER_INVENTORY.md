# Runner Inventory — Q_NYX

> **Ticket 02 — every executable, every decision engine, every
> model-producing module in this repo, tagged with one of 4 status
> values.** Companion doc : `docs/PROJECT_TRUTH_MAP.md` (high-level
> layered map). Canonical definitions come from
> `docs/ARCHITECTURE_CANONIQUE.md`.

## Status tag legend

| Tag | Meaning |
|---|---|
| **canonical runtime** | Part of the ONE approved runtime path per `ARCHITECTURE_CANONIQUE.md`. Actively used to produce live signals / official backtest numbers. |
| **offline calibration** | Runs during training / calibration / validation only. Never called by a runtime path. |
| **legacy** | Older code still on disk and possibly still imported by other legacy modules, but **not** the approved runtime. Must not be promoted without a new ticket. |
| **research/experimental** | Exploratory code — HSMM, SMC, fractal, setup-investigations, ML alternative agents. Not a production path. |

---

## 1. Scripts (`scripts/*.py`)

Every Python file under `scripts/` with `if __name__ == '__main__':`
is invocable. All are listed. 35 scripts total.

### Canonical / production validation

| Script | Purpose | Engine called | Status |
|---|---|---|---|
| `scripts/validate_abc.py` | A/B/C portfolio OOS + MC + bootstrap (BTC / BTC+ETH / BTC+ETH+SOL) | `NYXEngine.run` (via shim) | **canonical runtime** (offline validation of the runtime engine) |
| `scripts/validate_ab.py` | A/B subset of the above (BTC / BTC+ETH) | `NYXEngine.run` (via shim) | canonical runtime (offline validation) |
| `scripts/validate_abc_via_hubspoke.py` | A/B/C re-run through `HubSpokeRunner` + `PostOnlyPaperBroker` | `NYXPipelinePod` + `HubSpokeRunner` | canonical runtime (offline validation) |
| `scripts/walk_forward_trio.py` | 4-year walk-forward BTC+ETH+SOL | `NYXEngine.run` (via shim) | canonical runtime (offline validation) |
| `scripts/oos_live_replay.py` | Replay 2023 bar-by-bar through `NYXLiveDecider`, PnL via `NYXEngine._generate_candidates` lookup | `NYXLiveDecider` + `NYXEngine._generate_candidates` | **canonical runtime** (live replay) |
| `scripts/run_reality_checks.py` | 6 reality-check corrections (miss-rate, taker, daily Sharpe, block bootstrap, OOS 2024+, forced-stop B&H) | `NYXEngine.run` (via shim) | canonical runtime (offline validation) |
| `scripts/train_eth_model.py` | Train + persist ETH artefact (`models/ETHUSDT/`) | `train_asset_model.train_and_save` | **offline calibration** |
| `scripts/train_sol_model.py` | Train + persist SOL artefact (`models/SOLUSDT/`) | `train_asset_model.train_and_save` | offline calibration |
| `scripts/precompute_features.py` | Precompute parquet feature files per TF | `jesse_features.compute_stationary_features` | offline calibration |

### Legacy (v0.8 NYXEngine path — not canonical)

| Script | Purpose | Engine called | Status |
|---|---|---|---|
| `scripts/run_backtest.py` | v0.8 backtest CLI | `src.core.nyx_engine.NYXEngine` | **legacy** |
| `scripts/backtest_mtf.py` | v1.0 MTF backtest (v5.5M spec) | `src.agents.orchestrator.Orchestrator` + `src.core.precomputed_runner.PrecomputedStates` | **legacy** (different agent stack, parallel to canonical) |
| `scripts/walk_forward_validation.py` | Walk-forward validation using HSMM | `src.core.hsmm.SemiMarkovHMM` | legacy |
| `scripts/run_validation.py` | Generic validation CLI | various | legacy |
| `scripts/optimize.py` | Grid-search over v0.8 parameters | `src.core.nyx_engine.NYXEngine` (implicit) | legacy |
| `scripts/download_data.py` | Fetch raw OHLCV CSVs | — | offline calibration (data prep) |
| `scripts/compute_4h_features.py` | One-off compute of 4h parquet | `compute_stationary_features` | offline calibration (one-off data prep) |

### Research / experimental

| Script | Purpose | Status |
|---|---|---|
| `scripts/fractal_cache_benchmark.py` | Benchmark fractal-cache runner | research/experimental |
| `scripts/fractal_cache_check.py` | Cache-correctness sanity check | research/experimental |
| `scripts/fractal_check.py` | Fractal-agent sanity check | research/experimental |
| `scripts/fractal_bottleneck.py` | Profiling fractal runner | research/experimental |
| `scripts/fractal_geometry.py` | Geometry/topology experiments | research/experimental |
| `scripts/setup_bottleneck.py` | Setup-layer profiling | research/experimental |
| `scripts/setup_coverage_multi.py` | Setup coverage explorer | research/experimental |
| `scripts/setup_deep_dive.py` | Setup-rule forensic analysis | research/experimental |
| `scripts/setup_detector_check.py` | Setup-detector test harness | research/experimental |
| `scripts/setup_investigation.py` | Setup-bug exploration v1 | research/experimental |
| `scripts/setup_investigation_v2.py` | Setup-bug exploration v2 | research/experimental |
| `scripts/setup_investigation_v3.py` | Setup-bug exploration v3 | research/experimental |
| `scripts/smc_autopsy.py` | SMC (smart-money-concepts) post-mortem | research/experimental |
| `scripts/regime_feature_check.py` | Regime-agent feature diagnostics | research/experimental |
| `scripts/regime_tuning.py` | Regime-threshold tuning | research/experimental |
| `scripts/bullish_audit.py` | Bull-bias audit | research/experimental |
| `scripts/component_edge_test.py` | Per-component edge decomposition | research/experimental |
| `scripts/ale_analysis.py` | ALE analysis (accumulated local effects) | research/experimental |
| `scripts/test_parity.py` | Parity check between paths | research/experimental |
| `scripts/train_ml_ecosystem.py` | Train alternative `MLContextAgent`/`MLRegimeAgent`/`MLSetupAgent` stack | research/experimental (uses `src/ml/ml_agents.py`, not wired) |

---

## 2. Core runtime modules (`src/core/`)

Post-Ticket-04, the canonical batch engine `NYXEngine` lives at
`src/core/nyx_engine.py`. Live per-bar inference is still
`src/ml/nyx_live_decider.py::NYXLiveDecider`. Everything else in
`src/core/` remains v0.8-era legacy.

| File | Role | Who calls it | Status |
|---|---|---|---|
| `src/core/nyx_engine.py` | **Canonical batch engine** — `NYXEngine.run` is the production entrypoint (Ticket 04 rename of NYXPipeline). | canonical callers (via `src/ml/nyx_pipeline.py` shim while migrating) + `src/ml/nyx_live_decider.py` + `src/assets/nyx_pipeline_pod.py` + scripts/tests | **canonical runtime** |
| `src/core/nyx_engine_v08.py` | Legacy v0.8 `NYXEngine` — HSMM + SMC + macro. Frozen for its legacy callers only. | `src/runner/run_paper.py`, `src/validation/{walk_forward,oos_report,benchmarks,smc_diagnostics,signal_funnel,pattern_quality}.py`, `scripts/run_backtest.py` | **legacy** |
| `src/core/nyx_engine_mtf.py` | 4-TF fractal swing agent (v5.5M spec) | research scripts | legacy |
| `src/core/precomputed_runner.py` | Cached-state fractal runner | `scripts/backtest_mtf.py` | legacy |
| `src/core/fractal_cached_runner.py` | Fractal agents with state caching | research scripts | research/experimental |
| `src/core/hsmm.py` | Hidden semi-Markov model for regimes | `scripts/ale_analysis.py`, `walk_forward_validation.py`, `component_edge_test.py` | research/experimental |
| `src/core/smc.py` | Smart-money-concepts detector | `src/core/nyx_engine.py` | research/experimental |
| `src/core/risk_manager_mtf.py` | v0.8 MTF risk manager | legacy runners | legacy |

---

## 3. ML modules (`src/ml/`)

### Canonical runtime

| File | Role | Status |
|---|---|---|
| `src/ml/nyx_pipeline.py` | **Deprecation shim** — re-exports `NYXEngine as NYXPipeline` so pre-ticket-04 callers keep working. See `src/core/nyx_engine.py` for the real engine. | canonical runtime (shim) |
| `src/ml/nyx_live_decider.py` | **Canonical live engine** — `NYXLiveDecider.on_15m_bar` per-bar inference | **canonical runtime** |
| `src/ml/mtf_feature_stack.py` | 4-TF rolling buffers + 15m→1h/4h/1d aggregation | canonical runtime |
| `src/ml/feature_buffer.py` | Sliding-window feature buffer per TF | canonical runtime |
| `src/ml/conditional_dial.py` | Per-bar conditional risk dial | canonical runtime |
| `src/ml/bear_risk_dial.py` | Risk-parameter table (bull/range/bear) | canonical runtime |
| `src/ml/soft_gate.py` | Size factor + disagreement | canonical runtime (**imports `realistic_backtest.py`**) |
| `src/ml/jesse_features.py` | 24 stationary features (EMA/ATR/RSI/ADX ratios) | canonical runtime (shared with offline) |
| `src/ml/train_asset_model.py` | Per-asset train + persist (`train_and_save`) | offline calibration (runtime reads its artefacts) |
| `src/ml/execution_policy.py` | Execution filter (`execution_check`) | canonical runtime |

### Offline calibration

| File | Role | Status |
|---|---|---|
| `src/ml/threshold_optimizer.py` | CV threshold sweep on `ml_filter_v2` | **offline calibration** |
| `src/ml/ml_filter_v2.py` | Legacy GBM + threshold (consumed only by `threshold_optimizer`) | **offline calibration** |
| `src/ml/ml_filter.py` | v1 legacy ML filter | legacy |
| `src/ml/realistic_backtest.py` | `RealisticBacktester` — imported by `soft_gate.py` for calibration-time sizing | offline calibration (imported by canonical `soft_gate` but only at training time) |
| `src/ml/reality_check.py` | Reality-check helpers | offline calibration |
| `src/ml/jesse_labeler.py` | Triple-barrier labels (+1 / −1 / 0) | offline calibration |
| `src/ml/monte_carlo.py` | Monte-Carlo OOS shuffle | offline calibration |
| `src/ml/bootstrap.py` | Block bootstrap | offline calibration |
| `src/ml/walk_forward_splitter.py` | WF splits | offline calibration |
| `src/ml/triple_barrier.py` | TP/SL/TIME barrier helper | offline calibration |
| `src/ml/feature_engine.py` | Parquet-feature generator | offline calibration |
| `src/ml/feedback_loop.py` | `DecisionLogger` + `OutcomeEvaluator` + `ChampionChallenger` | offline calibration (not wired in current runtime — reserved for feedback loop activation) |

### Research / experimental (alternative architectures)

| File | Role | Status |
|---|---|---|
| `src/ml/jesse_agents.py` | 5 Jesse agents (Context / Regime / Setup / Entry / Orchestrator) — alternative to monolithic `NYXEngine` | research/experimental — **not wired** (proxied by `rule_*` scalars; see `JESSE_AGENTS_STATUS.md`) |
| `src/ml/edge_strategy.py` | Historical trend+volume edge validator (walk-forward 14/14 quarters) | legacy (not a standalone runtime strategy) |
| `src/ml/ml_agents.py` | `MLContextAgent` / `MLRegimeAgent` / `MLSetupAgent` (LightGBM 4-agent fractal) | research/experimental |
| `src/ml/ml_entry_agent.py` | `MLEntryAgent` (blend 0.7×LGB + 0.3×River) | research/experimental |
| `src/ml/ml_orchestrator.py` | `MLOrchestrator` combining the 4 ML agents | research/experimental |
| `src/ml/jesse_ab_runner.py` | A/B harness for Jesse variants | research/experimental |
| `src/ml/jesse_backtest.py` | Standalone Jesse backtest | research/experimental |
| `src/ml/jesse_research.py` | Jesse research notebook-style utils | research/experimental |
| `src/ml/jesse_strategy.py` | Jesse strategy object | research/experimental |
| `src/ml/jesse_utils.py` | Jesse utilities | research/experimental |
| `src/ml/model_monitor.py` | Drift monitor prototype | research/experimental |

---

## 4. Agents layer (`src/agents/`)

| File | Role | Status |
|---|---|---|
| `src/agents/contracts.py` | `AgentResult`, `OrchestratorDecision` | research/experimental (dormant — used by Jesse agents + legacy `backtest_mtf.py`) |
| `src/agents/context_agent.py` | 1D fractal reporter | research/experimental (dormant) |
| `src/agents/regime_agent.py` | 1H fractal reporter | research/experimental (dormant) |
| `src/agents/setup_agent.py` | 15M fractal reporter | research/experimental (dormant) |
| `src/agents/entry_agent.py` | 15M fractal reporter | research/experimental (dormant) |
| `src/agents/orchestrator.py` | Meta rule-based combiner — used by legacy `backtest_mtf.py` | legacy |

**All 5 agents exist in BOTH `src/ml/jesse_agents.py` (single-file
module) AND `src/agents/*.py` (per-file)** — a double implementation
from history. Neither is wired into the canonical runtime. Kept as
alternative; swap path in `JESSE_AGENTS_STATUS.md`.

---

## 5. Assets layer (`src/assets/`) — canonical multi-asset hub

| File | Role | Status |
|---|---|---|
| `src/assets/signal.py` | `Signal` dataclass (pod output contract) | **canonical runtime** |
| `src/assets/hub_spoke_runner.py` | `HubSpokeRunner` — orchestrator | **canonical runtime** |
| `src/assets/portfolio_allocator.py` | `PortfolioAllocator` — risk caps + no-pyramiding + cluster veto | **canonical runtime** |
| `src/assets/nyx_live_pod.py` | Live pod wrapping `NYXLiveDecider` | **canonical runtime** |
| `src/assets/nyx_pipeline_pod.py` | Replay pod wrapping `NYXEngine` (batch) via shim | canonical runtime (replay) |
| `src/assets/registry.py` | Asset registry / YAML loader | canonical runtime |
| `src/assets/execution_profile.py` | Per-asset execution profile | canonical runtime |
| `src/assets/combined_portfolio.py` | Portfolio aggregator | canonical runtime |

---

## 6. Paper-live layer (`src/paper_live/`) — canonical infra

| File | Role | Status |
|---|---|---|
| `src/paper_live/runner.py` | `PaperLiveRunner` | **canonical runtime** |
| `src/paper_live/multi_pair_runner.py` | Multi-pair wrapper | canonical runtime |
| `src/paper_live/post_only_broker.py` | `PostOnlyPaperBroker` (no taker fallback) | canonical runtime |
| `src/paper_live/decision_logger.py` | Append-only decision log | canonical runtime |
| `src/paper_live/missed_trade_logger.py` | Missed-trade SQLite | canonical runtime |
| `src/paper_live/state_manager.py` | Atomic state (tmp→fsync→rename) | canonical runtime |
| `src/paper_live/heartbeat.py` | Heartbeat liveness | canonical runtime |
| `src/paper_live/data_validator.py` | Bar integrity checks | canonical runtime |
| `src/paper_live/event_alerter.py` | 6-event alerter | canonical runtime |
| `src/paper_live/telegram_alerter.py` | Telegram backend | canonical runtime |
| `src/paper_live/discord_alerter.py` | Discord backend | canonical runtime |
| `src/paper_live/multi_alerter.py` | Fan-out to Telegram+Discord | canonical runtime |
| `src/paper_live/resync_manager.py` | Reconcile state on reconnect | canonical runtime |
| `src/paper_live/main.py` | Entry binary (`python -m src.paper_live.main`) | **canonical runtime** |

---

## 7. Legacy runtime paths still present (flagged)

These directories contain legacy code that is **still imported by
other legacy code or scripts** — deleting them would break the legacy
scripts. They are NOT part of the canonical runtime.

| Path | Status | Deletion blocker |
|---|---|---|
| `src/core/` (all files) | legacy | `src/runner/run_paper.py`, 5 `src/validation/*` modules, `scripts/run_backtest.py` still import `NYXEngine` |
| `src/runner/run_paper.py` | **legacy v0.8 paper runner — parallel to canonical `src/paper_live/main.py`** | imports `NYXEngine` |
| `src/execution/paper_engine.py` | legacy v0.8 paper engine | imported by legacy runner |
| `src/validation/{walk_forward, oos_report, benchmarks, smc_diagnostics, signal_funnel, pattern_quality, order_block_audit, hsmm_deep_dive, reporting, metrics, manifest}.py` | legacy validation suite wrapping NYXEngine | depends on `NYXEngine` |
| `src/backtest/`, `src/strategy/`, `src/risk/` | empty dirs (placeholders) | none |

---

## Summary counts

| Status | Scripts | Modules | Total |
|---|---:|---:|---:|
| canonical runtime | 7 | ~30 (ml + assets + paper_live) | ~37 |
| offline calibration | 3 | ~15 (training + reality + MC/bootstrap) | ~18 |
| legacy | 5 | ~15 (core + runner + execution + validation) | ~20 |
| research/experimental | 20 | ~15 (jesse_* + ml_agents + setup_investigations) | ~35 |

**110+ files mapped. No mystery runner left unnamed.**

---

## Maintenance rule

When a new `scripts/*.py` with `if __name__ == '__main__':` lands,
`tests/test_runner_inventory.py::TestNoMysteryRunner` will fail
until it is added here with an explicit status. Same for any new
`src/**/nyx_engine*.py` or `src/**/*runner*.py`.
