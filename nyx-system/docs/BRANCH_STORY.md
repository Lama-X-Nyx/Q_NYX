# The story of branch `claude/run-pyright-system-qroCy`

> Chronological account of what was built, what was learned, and what
> the honest current state is. This doc is written **last**, looking
> back at ~60 commits.
>
> For quick navigation, see [`README.md`](README.md).
> For operating rules, see [`OPERATING_RULES.md`](OPERATING_RULES.md).
> For the honest numbers, see [`REALITY_CHECK.md`](REALITY_CHECK.md).

---

## Part 1 — Foundations (pyright + paper-live + hub-and-spoke)

### Starting point

Branch was spun up with a single instruction: **"run pyright on the
system"**. Initial pyright run against `src/ + tests/`:

```
417 errors, 0 warnings, 0 informations
```

Most of those were pandas/numpy stub false-positives — not real bugs —
but some were real: possibly-unbound variables, optional subscripts
without guards, missing imports.

### Commit `f6bb230` — pyright hygiene

Created `pyrightconfig.json` in `basic` mode:

| Check | Level | Rationale |
|---|---|---|
| `reportPossiblyUnbound` | **error** | real bug |
| `reportUnboundVariable` | **error** | typos |
| `reportOptionalSubscript` / `reportOptionalMemberAccess` | **error** | None guard missing |
| `reportArgumentType` / `reportAttributeAccessIssue` | none | pandas stub noise |
| `reportMissingImports` / `reportIndexIssue` | warning | often optional deps |

Fixed the real issues:
- `src/ml/jesse_features.py` — `ta: Any = None` + hoisted `candles` out of conditional
- `src/ml/jesse_utils.py`, `src/ml/jesse_research.py` — same pattern
- `src/ml/ml_filter.py` — guarded `self._scaler is None`
- `tests/test_jesse_orchestrator.py` — assert `decision.risk_analysis is not None`
- `tests/test_bear_risk_dial.py` — initialize `static_trades/adaptive_trades`

End state: **0 errors**. This became Rule #6 in
[`OPERATING_RULES.md`](OPERATING_RULES.md).

### Commit `377c0f2` — paper-live infrastructure (67/67 GREEN TDD)

Asked: what's the minimum infrastructure needed to run a strategy 24/7
without silent data loss? Built, TDD strict (RED → GREEN), 9 modules:

| Module | Tests | Purpose |
|---|---:|---|
| `PersistentDecisionLogger` | 11/11 | SQLite WAL, append-only, idempotent on (ts, pair) |
| `StateManager` | 8/8 | atomic JSON snapshot (tmp → fsync → rename), corruption detect |
| `Heartbeat` | 7/7 | file-based liveness, Docker HEALTHCHECK-friendly |
| `TelegramAlerter` | 7/7 | injectable HTTP, cooldown dedup |
| `DataValidator` | 11/11 | rejects NaN/inf/geometry-broken/stale bars |
| `ReSyncManager` | 9/9 | gap detection + replay missed bars on restart |
| `PaperLiveRunner` | 6/6 | wires everything, warm-restart |
| `Dockerfile` + `docker-compose.yml` | — | production deployment |

All live under `src/paper_live/`. Full doc:
[`PAPER_LIVE_INFRASTRUCTURE.md`](PAPER_LIVE_INFRASTRUCTURE.md).

### Commit `b010dab` — Telegram + Discord semantic events (34/34 GREEN)

Three-layer alerting:

```
EventAlerter (6 semantic events)
    ↓
MultiAlerter (fan-out, failure isolation)
    ↓
TelegramAlerter + DiscordAlerter
```

Six events, routed to the right severity:

| Event | Severity | Dedup | Trigger |
|---|---|---|---|
| `service_down` | critical | yes | watchdog |
| `reconnect_exchange` | warn | yes | adapter |
| `trade_decision` | info | **never** | auto on BUY/SELL |
| `order_unfilled` | warn | yes (once per oid) | auto on REJECTED/TIMED_OUT |
| `api_error` | error | yes | auto on strategy crash |
| `restart` | info | yes | main.py at boot |

`trade_decision` is never rate-limited — every trade must surface.

### Commit `659fffe` — honest post-only execution (30/30 GREEN)

**This was an architectural turn.** The earlier `MakerFirstBroker`
silently fell back to taker after `max_wait_bars`. That hides the
fact that many post-only orders would simply not fill in live.

Replaced with `PostOnlyPaperBroker`:

```
NEW → POSTED → FILLED         (bar low ≤ limit → maker fee)
            ↘ TIMED_OUT   (max_wait reached → MISSED TRADE)
            ↘ CANCELLED   (client cancel)
     ↘ REJECTED        (post-only crosses at placement)
```

**No taker fallback.** Ever. Enforced by
`test_no_taker_fallback` in `tests/test_post_only_broker.py`.

Paired with `MissedTradeLogger` (SQLite WAL, append-only, idempotent
on `oid`) so every REJECTED/TIMED_OUT is auditable. Miss rate is
exposed in the heartbeat context as `broker_miss_rate`.

This became Rule #3 in [`OPERATING_RULES.md`](OPERATING_RULES.md).

### Commit `7eef9d4` — MultiPairRunner (9/9 GREEN)

First stab at multi-asset: N isolated `PaperLiveRunner` instances,
one per pair, with per-pair storage:

```
storage/
├── ETHUSDT/
│   ├── decisions.db
│   ├── missed_trades.db
│   ├── state.json
│   └── heartbeat.json
├── XRPUSDT/ ...
└── SOLUSDT/ ...
```

Crash in one pair does not affect the others. Useful, but this was
"3 clones running in parallel", not a real portfolio layer.

### Commit `6280d13` — hub-and-spoke architecture (45/45 GREEN)

**Second architectural turn.** Feedback: "3 clones in parallel isn't
an architecture, it's a deployment trick. Build a proper hub-and-spoke."

Result:

```
Market data
  ↓
Feature engine (shared, asset-aware)
  ↓
[ETH Pod]  [XRP Pod]  [SOL Pod]       — spokes, emit Signal
     ↓         ↓          ↓
     └──────── Portfolio Allocator ────┘ — hub, ranks + risk caps
              ↓
       PostOnlyPaperBroker              — shared execution
              ↓
       Logger + Feedback                — per asset + global
```

5 modules under `src/assets/`:

| Module | Tests | Role |
|---|---:|---|
| `AssetRegistry` | 10/10 | loads `config/assets.yaml`, validates enums |
| `ExecutionProfile` | 7/7 | session_profile → max_wait; slippage_model → bps + viability threshold |
| `Signal` | 10/10 | typed dataclass (symbol, direction, conviction, edge, viability, regime, cluster) |
| `PortfolioAllocator` | 12/12 | max_total_risk 4 %, max_asset_risk 1.5 %, max_cluster_risk {alts: 2 %}, no-pyramiding, correlation veto |
| `HubSpokeRunner` | 6/6 | wires pods → allocator → per-asset broker |

Pods are **pure producers** of Signals. They don't place orders. The
allocator is the single decision point. This enforces "central
portfolio layer" properly.

Full doc: [`HUB_AND_SPOKE_ARCHITECTURE.md`](HUB_AND_SPOKE_ARCHITECTURE.md).

### State after Part 1

- **177 TDD tests GREEN** (paper-live + hub-and-spoke)
- **Pyright: 0 errors**
- Paper execution: honest post-only, never silent taker
- Alerting: 6 events, Telegram + Discord, failure-isolated fan-out
- Multi-asset: proper hub-and-spoke, not parallel clones

Next: **train real models** on ETH, SOL. Part 2.
