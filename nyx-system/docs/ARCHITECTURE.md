# NYX System Architecture

> Read this first. Understand the system in 5 minutes.

---

## What is NYX?

An institutional-grade multi-asset crypto trading system.
One GBM model decides. Four Jesse agents contextualize.
A central orchestrator arbitrates across assets.

---

## Canonical Pipeline (the ONLY production path)

```
15m bar arrives (Binance WebSocket)
  │
  ▼
NYXRuntime.on_bar(bar)                    ← src/live/nyx_runtime.py
  │
  ├── 1. GBM SIGNAL                       ← src/ml/nyx_live_decider.py
  │     NYXLiveDecider.on_15m_bar()             + src/core/meta_gbm.py
  │     → hard gate (EMA + vol + hour)
  │     → MetaGBM.decide() → p_trade ∈ [0,1]
  │     → threshold 0.60 + bear dial
  │
  ├── 2. JESSE FRACTAL REPORTS            ← src/agents/{context,regime,setup,entry}_agent.py
  │     Context (1D) + Regime (4H)
  │     + Setup (1H) + Entry (15M)
  │
  ├── 3. FRACTAL QUALITY                  ← src/core/fractal_quality.py
  │     compute_fractal_quality()
  │     → size_multiplier or SKIP
  │
  ├── [candidate mode — Ticket 36]
  │     If candidate_store set:
  │     → emit CandidateDecision → STOP
  │     CentralOrchestrator collects all assets,
  │     runs dependency + allocator on FULL set
  │
  ├── 3b. DEPENDENCY LAYER                ← src/live/inter_asset_dependency.py
  │     Lead/lag + contagion + regime conditioning
  │
  ├── 3c. PORTFOLIO ALLOCATOR             ← src/live/portfolio_allocator.py
  │     Capital constraints + scoring + ranking
  │
  ├── 4. RISK ENGINE                      ← src/live/risk_engine.py + var_cvar.py
  │     Static rules + VaR/CVaR gates
  │
  ├── 5. OMS                              ← src/live/oms.py
  │     Submit order → state machine
  │
  ├── 6-8. FILL → PORTFOLIO → PERSIST     ← src/live/portfolio_state.py
  │                                            + src/live/state_store.py
  │
  ├── 9. MONITORING                       ← src/live/monitoring.py
  │
  └── 10. AUDIT                           ← src/live/audit_trail.py
```

---

## Multi-Asset Architecture (Ticket 36)

```
Thread BTCUSDT → NYXRuntime(candidate_store) → CandidateDecision ─┐
Thread ETHUSDT → NYXRuntime(candidate_store) → CandidateDecision ─┤
Thread SOLUSDT → NYXRuntime(candidate_store) → CandidateDecision ─┘
                                                                   │
                              CandidateStore (time-bucketed) ◄─────┘
                                        │
                            CentralOrchestrator.run_cycle()
                             ├── InterAssetDependencyLayer
                             ├── PortfolioAllocator (FULL ranking)
                             └── Approved trades → per-asset Risk → OMS
```

**Entrypoint:** `scripts/run_multi_asset.py`

---

## Canonical Module Map (30 files = the entire production system)

### `src/nyx/` — Clean import namespace (re-exports)

```python
from src.nyx import NYXRuntime, CentralOrchestrator
from src.nyx.decision import MetaGBM, NYXLiveDecider
from src.nyx.portfolio import PortfolioAllocator, Portfolio
from src.nyx.execution import OMS, RiskEngine
from src.nyx.data import BarBuilder, compute_stationary_features
from src.nyx.state import StateStore, AuditStore
from src.nyx.contracts import FractalReport, Signal, CandidateDecision
```

### Source locations

| Layer | Files | Package |
|---|---|---|
| Runtime | nyx_runtime.py, central_orchestrator.py | src/live/ |
| Decision | meta_gbm.py, fractal_quality.py | src/core/ |
| Decision | nyx_live_decider.py | src/ml/ |
| Agents | contracts.py, context/regime/setup/entry_agent.py | src/agents/ |
| Portfolio | portfolio_allocator.py, inter_asset_dependency.py, portfolio_state.py | src/live/ |
| Execution | oms.py, risk_engine.py, var_cvar.py, execution_optimizer.py | src/live/ |
| Data | bar_builder.py, feed_health.py, binance_ws.py | src/live/ |
| Data | jesse_features.py, mtf_feature_stack.py, feature_buffer.py | src/ml/ |
| State | state_store.py, monitoring.py, audit_trail.py | src/live/ |
| Training | train_asset_model.py, jesse_agents.py, jesse_labeler.py, jesse_dataset.py, model_registry.py | src/ml/ |
| Contract | signal.py | src/assets/ |

### Legacy (everything else)

See `legacy/MANIFEST.md` for the full inventory. Key rule:
**no canonical module imports from legacy packages.**

Legacy packages: `src/paper_live/`, `src/validation/`, `src/data/`,
`src/execution/`, `src/cache/`, `src/macro/`, `src/runner/`.

Legacy files in canonical packages: `src/core/nyx_engine_v08.py`,
`src/ml/ml_filter*.py`, `src/agents/orchestrator.py`, etc.

---

## Trained Models

```
models/BTCUSDT/   — 128 features, GBM, Sharpe ~10
models/ETHUSDT/   — 128 features, GBM, Sharpe ~7.4
models/SOLUSDT/   — 128 features, GBM, Sharpe ~2.6
```

Each contains: `ml_filter_v1.pkl`, `scaler.pkl`, `feature_names.json`,
`training_metadata.json`.

---

## Entrypoints

| Script | What it does |
|---|---|
| `scripts/run_multi_asset.py` | Multi-asset live trading (synchronized cycles) |
| `scripts/run_live_feed.py` | Single-asset live trading |
| `scripts/train_btc_model.py` | Train BTC model |
| `scripts/train_eth_model.py` | Train ETH model |
| `scripts/train_sol_model.py` | Train SOL model |

---

## Where to Start (new developer)

1. Read this file (5 min)
2. Read `src/live/nyx_runtime.py` — THE orchestrator, 10 numbered steps
3. Read `src/nyx/__init__.py` — clean import namespace
4. Run `python -m pytest tests/test_architecture_ticket38.py -v` — verify structure
5. Run `python -m pytest tests/test_multi_asset_runtime_ticket33.py -v` — verify runtime works
