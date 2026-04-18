# Legacy Module Manifest

> These modules are NOT part of the canonical production system.
> They exist as historical reference, offline tools, or deprecated code.
> **No canonical module should import from these.**

---

## src/core/ (legacy files)

| File | What it does | Why legacy | Replaced by |
|---|---|---|---|
| `nyx_engine_v08.py` | HSMM/SMC-based engine | Replaced by GBM monolith | `NYXRuntime` + `MetaGBM` |
| `nyx_engine_mtf.py` | Old MTF engine | Replaced by NYXRuntime | `src/live/nyx_runtime.py` |
| `nyx_engine.py` | Batch engine (training + OOS) | Offline-only, not runtime | Used by training scripts |
| `precomputed_runner.py` | Pre-computed state runner | Legacy optimization | N/A |
| `fractal_cached_runner.py` | Cached fractal pipeline | Legacy optimization | N/A |
| `risk_manager_mtf.py` | Old risk manager | Replaced by Ticket 25 | `src/live/risk_engine.py` |
| `hsmm.py` | Semi-Markov HMM | Used by agents (support) | Still used by SetupAgent/RegimeAgent |
| `smc.py` | SMC detector | Used by SetupAgent (support) | Still used by SetupAgent |

## src/ml/ (legacy files)

| File | What it does | Why legacy | Replaced by |
|---|---|---|---|
| `nyx_pipeline.py` | Shim: re-exports NYXEngine as NYXPipeline | Backward compatibility | `src/core/nyx_engine.py` |
| `edge_strategy.py` | Candidate generator (hard gate) | Component of NYXEngine | Used by NYXEngine internally |
| `ml_filter.py` | v1 ML trade filter | Replaced | `train_asset_model.py` |
| `ml_filter_v2.py` | v2 ML filter + CV threshold | Offline-only | `threshold_optimizer.py` |
| `threshold_optimizer.py` | CV threshold sweep | Offline-only | N/A |
| `ml_agents.py` | River-based ML agents | Pre-Jesse | `src/agents/*.py` |
| `ml_entry_agent.py` | River-based entry agent | Pre-Jesse | `src/agents/entry_agent.py` |
| `ml_orchestrator.py` | ML agent orchestrator | Pre-Jesse | `src/agents/orchestrator.py` |
| `soft_gate.py` | Rule-based soft gate | Used by NYXEngine | Support module |
| `bear_risk_dial.py` | Bull/range/bear risk params | Used by NYXEngine | Support module |
| `conditional_dial.py` | Per-bar conditional risk dial | Used by NYXEngine | Support module |
| `realistic_backtest.py` | Legacy backtester | Offline research | N/A |
| `jesse_backtest.py` | Fast backtester | Offline research | N/A |
| `monte_carlo.py` | Monte Carlo simulation | Offline validation | N/A |
| `bootstrap.py` | Bootstrap validation | Offline validation | N/A |
| `reality_check.py` | Reality check suite | Offline validation | N/A |
| `feedback_loop.py` | Champion-challenger | Not wired in production | N/A |
| `execution_policy.py` | Old execution policy | Replaced by T32 | `execution_optimizer.py` |
| `feature_engine.py` | Old feature engine | Replaced | `jesse_features.py` |
| `walk_forward_splitter.py` | Walk-forward CV | Offline tool | N/A |
| `model_monitor.py` | Model monitoring | Not wired | N/A |
| `jesse_strategy.py` | Jesse ML strategy | Research | N/A |
| `jesse_ab_runner.py` | A/B comparison | Research | N/A |
| `jesse_research.py` | Feature research | Research | N/A |
| `jesse_utils.py` | Jesse utilities | Research | N/A |
| `triple_barrier.py` | Triple barrier labels | Replaced by jesse_labeler | `jesse_labeler.py` |

## src/paper_live/ (entire package — legacy)

Old paper trading system. Replaced by `NYXRuntime` (Ticket 22R).
Includes: PostOnlyPaperBroker, PaperLiveRunner, StateManager,
Heartbeat, DecisionLogger, Discord/Telegram alerters, etc.

## src/assets/ (legacy files, except signal.py)

| File | What it does | Why legacy | Replaced by |
|---|---|---|---|
| `hub_spoke_runner.py` | Multi-asset hub-spoke | Replaced by T33 multi-runtime | `run_multi_asset.py` |
| `portfolio_allocator.py` | Old portfolio allocator | Replaced by T35 | `src/live/portfolio_allocator.py` |
| `combined_portfolio.py` | Portfolio combiner | Replaced by T35 | `src/live/portfolio_allocator.py` |
| `execution_profile.py` | Execution profiling | Not used in production | N/A |
| `nyx_live_pod.py` | Live pod wrapper | Replaced by NYXRuntime | `src/live/nyx_runtime.py` |
| `nyx_pipeline_pod.py` | Pipeline pod wrapper | Replaced by NYXRuntime | `src/live/nyx_runtime.py` |
| `registry.py` | Asset registry | Not used by canonical path | N/A |

## src/validation/ (entire package — legacy)

Offline validation tools: benchmarks, walk-forward, signal funnel,
SMC diagnostics, HSMM deep dive, OOS reports, pattern quality.
Not part of the production runtime.

## src/data/, src/execution/, src/cache/, src/macro/, src/runner/

Small legacy packages. `data/` has MTF loader + live feed (replaced by
BarBuilder + BinanceKlineStream). `execution/` has PaperEngine (replaced
by OMS). Others are empty or single-file utilities.

## src/agents/orchestrator.py

Old rule-based orchestrator. Replaced by CentralOrchestrator (T36).
