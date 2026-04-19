# Q_NYX — Operating System for Claude Code (VS Code Edition)

> **Read this entire file before EVERY task.** This is Rule 7.
> If you skip this, you WILL break the system.

---

## What is NYX?

An **institutional-grade multi-asset crypto trading system** with:
- **GBM monolith** (GradientBoostingClassifier, 128 features) = the ONLY decision brain
- **4 Jesse agents** (Context 1D / Regime 4H / Setup 1H / Entry 15M) = post-decision modulators
- **3 assets**: BTC / ETH / SOL — each with its own NYXRuntime instance
- **Full live infrastructure**: OMS, RiskEngine, Portfolio, StateStore, AuditTrail, ExecutionMonitor
- **Cockpit UI**: Next.js + FastAPI operator dashboard with charts, controls, security
- **OOS Engine**: parametric, evaluator-driven research platform

**211 commits. 50+ tickets. 300+ tests. This is NOT a prototype.**

---

## The ONE canonical pipeline

```
15m bar → NYXRuntime.on_bar()
  ├── 1. GBM signal           (src/ml/nyx_live_decider.py)
  ├── 2. Jesse fractal reports (src/agents/*_agent.py)
  ├── 3. Fractal quality       (src/core/fractal_quality.py)
  ├── 3b. Inter-asset dependency (src/live/inter_asset_dependency.py)
  ├── 3c. Portfolio allocator  (src/live/portfolio_allocator.py)
  ├── 3d. Execution monitor    (src/live/execution_monitor.py)
  ├── 4. Risk engine           (src/live/risk_engine.py)
  ├── 5. OMS                   (src/live/oms.py)
  └── 6-10. Fill → Portfolio → State → Monitoring → Audit
```

**If it's not in this pipeline, it doesn't exist in production.**

---

## Non-negotiable rules

### 1) Integration-first
This is **one product**. No parallel pipelines, no duplicate modules, no `*_v2.py`, no `*_new.py`. Before writing code:
- Identify the current execution path
- Identify the owner module
- Explain how the change fits the existing path

### 2) TDD only
For EVERY change:
1. Write a failing test FIRST
2. Implement the smallest patch
3. Run the test
4. Refactor only if tests stay green

**No "test later". No "works in theory".**

### 3) Multi-timeframe is mandatory
Every change must respect: 1D context → 4H regime → 1H setup → 15M entry.

### 4) Document every iteration
Update `docs/SESSION_LOG.md` and `docs/CHANGELOG.md` at every meaningful step.

### 5) Minimal surface area
Patch the smallest possible area. Don't refactor nearby code "while you're here".

### 6) No architectural improvisation
Don't rename, move, split, or replace modules unless explicitly asked.

### 7) Read this file first — EVERY session
Before acting, summarize: current execution path, owner module, TDD plan.

---

## Repository structure

### Canonical production modules (THE system)

```
src/live/
  nyx_runtime.py           ← THE single orchestrator
  central_orchestrator.py   ← synchronized multi-asset cycles (T36)
  oms.py                    ← order management state machine
  risk_engine.py            ← sovereign risk gate (%-based, T40)
  var_cvar.py               ← VaR/CVaR distribution risk
  portfolio_state.py        ← positions + PnL
  portfolio_allocator.py    ← capital-constrained allocation (T35)
  inter_asset_dependency.py ← cross-asset lead/lag/contagion (T34)
  execution_monitor.py      ← adaptive risk (v2 smoothed, T46.1)
  execution_optimizer.py    ← fill probability + dynamic offset (T32)
  state_store.py            ← atomic JSON persistence
  monitoring.py             ← metrics + alerts
  audit_trail.py            ← structured decision history (T37)
  bar_builder.py            ← kline normalization
  feed_health.py            ← stale detection
  binance_ws.py             ← WebSocket adapter
  oos_engine.py             ← canonical OOS engine + evaluator framework (T42/42.1)
  oos_cache.py              ← caching + profiling (T45)

src/core/
  meta_gbm.py               ← MetaGBM strategy brain
  fractal_quality.py         ← post-decision modulation (T20)

src/agents/
  contracts.py               ← FractalReport, MetaDecision, etc.
  context_agent.py           ← 1D context
  regime_agent.py            ← 4H regime
  setup_agent.py             ← 1H setup
  entry_agent.py             ← 15M entry

src/ml/
  nyx_live_decider.py        ← per-bar live inference
  jesse_features.py          ← canonical 37-col feature engine
  train_asset_model.py       ← model training + persistence
  mtf_feature_stack.py       ← MTF buffer
  feature_buffer.py          ← incremental features
  jesse_agents.py            ← mono-file Jesse ML-native agents

src/data_pipeline/
  bar_store.py               ← canonical bar store (DATA-1)
  raw_market_store.py        ← raw event lake, append-only (DATA-1.1)

src/ml_pipeline/
  dataset_builder.py         ← dataset snapshots (ML-1)
  model_registry_v2.py       ← model versioning + promotion (ML-1)

src/live/evaluators/
  stability.py               ← T43 + T43.1 (regime, tail risk, classification)
  capacity.py                ← T44 (capital ladder)
  capacity_execution.py      ← T44.1 (execution friction stress)
```

### Cockpit (operator UI)

```
cockpit/
  api/
    server.py                ← FastAPI backend (REST + WebSocket)
    paper_control.py         ← paper trading state machine (UI-2)
    security.py              ← JWT auth, RBAC, audit, rate limiting (SEC-1)
  ui/
    src/app/page.tsx          ← main dashboard page
    src/components/charts.tsx ← 12 recharts components
    src/lib/api.ts            ← API client + types
  run_cockpit.py             ← launcher (API + UI)
```

### Clean namespace (src/nyx/ — re-exports)

```python
from src.nyx import NYXRuntime, CentralOrchestrator
from src.nyx.decision import MetaGBM, NYXLiveDecider
from src.nyx.execution import OMS, RiskEngine
from src.nyx.contracts import FractalReport, Signal
```

### Legacy (see legacy/MANIFEST.md)

Everything in `src/paper_live/`, `src/validation/`, `src/data/`, `src/execution/`, old engine versions. **Do NOT import legacy from canonical modules.**

---

## Key models

| Asset | Features | Sharpe (2023 OOS realistic) | Path |
|---|---|---|---|
| BTC | 128 | 4.20 | models/BTCUSDT/ |
| ETH | 128 | 4.09 | models/ETHUSDT/ |
| SOL | 128 | 0.97 | models/SOLUSDT/ |

**Models are FROZEN.** Do not retrain unless explicitly asked.

---

## Running NYX

### Paper trading (multi-asset)
```bash
python scripts/run_multi_asset.py
```

### Cockpit (UI + API)
```bash
python cockpit/run_cockpit.py --with-ui
```

### OOS engine
```python
from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
cfg = OOSConfig(assets=['BTCUSDT'], start_date='2023-01-01',
                end_date='2023-12-31', initial_capital=10_000_000)
result = CanonicalOOSEngine().run_oos(cfg)
```

### Tests
```bash
cd nyx-system && python -m pytest tests/ -v
```

---

## Current status

| Layer | Status | Key file |
|---|---|---|
| GBM signal | ✅ frozen | nyx_live_decider.py |
| Jesse modulation | ✅ post-decision | fractal_quality.py |
| Risk engine | ✅ %-based, scalable | risk_engine.py |
| OMS | ✅ state machine | oms.py |
| Portfolio | ✅ multi-asset | portfolio_state.py |
| Execution monitor | ✅ v2 smoothed | execution_monitor.py |
| Audit trail | ✅ per-bar + per-cycle | audit_trail.py |
| Inter-asset dependency | ✅ lead/lag + contagion | inter_asset_dependency.py |
| Portfolio allocator | ✅ capital-constrained | portfolio_allocator.py |
| Central orchestrator | ✅ synchronized cycles | central_orchestrator.py |
| OOS engine | ✅ parametric + evaluators | oos_engine.py |
| Cockpit UI | ✅ charts-first + controls | cockpit/ |
| Security | ✅ JWT + RBAC + bootstrap | security.py |
| Data pipeline | ✅ bar store + raw lake | data_pipeline/ |
| Model registry v2 | ✅ promote/rollback | ml_pipeline/ |
| Live Binance | 🔌 ready, needs network | binance_ws.py |

---

## What NOT to do

1. **Don't move files.** The import graph has 1,400+ references.
2. **Don't create _v2, _new, _clean files.** Patch the natural owner.
3. **Don't modify GBM or Jesse logic.** Models are frozen.
4. **Don't import from legacy** (src/paper_live/, src/validation/) in canonical modules.
5. **Don't skip tests.** Every change gets a test first.
6. **Don't make grand architectural changes.** Small patches, one behavior at a time.
7. **Don't add features beyond what was asked.** No "while I'm here" cleanup.
8. **Don't create documentation files unless asked.** Work from conversation context.

---

## Environment variables

```bash
NYX_JWT_SECRET=<random-long-string>    # auth token signing
NYX_ENV=paper                           # paper / testnet / live
BINANCE_API_KEY=<read-only-key>         # for live data feed
BINANCE_API_SECRET=<secret>             # for live data feed
```

---

## Git workflow

- Branch: `claude/run-pyright-system-qroCy`
- Commit style: `feat(ticket-XX): Short description`
- Always commit + push when done
- Never force-push
- Never amend published commits

---

## When in doubt

1. Read `docs/ARCHITECTURE.md` (5-minute system overview)
2. Read `docs/CHANGELOG.md` (ticket-by-ticket history)
3. Read `src/live/nyx_runtime.py` (THE orchestrator, numbered steps)
4. Run `python -m pytest tests/ -v` (verify nothing is broken)
5. Ask the user before making architectural decisions

---

## Priority (what the user wants next)

1. **Live paper trading on Binance** — connect WS, run 24/7, observe behavior
2. **Historical data backfill** — 10+ years via Binance REST API
3. **Walk-forward CV** — purged multi-fold validation
4. **Cockpit improvements** — as needed during paper trading

**The model is frozen. The priority is operational, not research.**
