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

---

## Part 2 — Training real models (ETH, SOL, and the 4-TF rule)

### Commit `de86a1d` — ETH training on real 15m data (10/10 GREEN)

Earlier ETH tests used 1h resampled as 15m (stand-in). The real ETH
15m CSV (`data/raw/ETHUSDT_15m.csv`, 143k bars, Dec 2019 → Jan 2024)
was now available. Built:

- `scripts/train_eth_model.py` — end-to-end driver
- `src/ml/train_asset_model.py` — generic `train_and_save()` for any asset
- `tests/test_eth_training.py` — 10/10 GREEN incl. BTC regression guard

Artefacts now persisted per asset:

```
models/ETHUSDT/
  ├── ml_filter_v1.pkl         GBM, 300 trees
  ├── scaler.pkl               StandardScaler
  ├── feature_names.json       ordered feature list
  └── training_metadata.json   { n_features, accuracy, ... }
```

ETH OOS 2023 first numbers (with the 3-TF feature block that was in
place at the time):

```
n_trades=74   Sharpe=4.38   PnL=+$1,814   DD=1.58%
```

### Commit `c03619f` — the "it wasn't really MTF" discovery (6/6 GREEN)

Feedback: "ETH doit être entraîné sur le multi-timeframe."

Audit of `_generate_candidates` revealed the truth:

| TF | What the model actually saw |
|---|---|
| 15m | 24 full stationary features ✓ |
| 1h | only 4 scalars (adx, atr_pct, mom12, trending) + ≤ 4 parquet columns |
| 1d | only 3 hand-crafted scalars (trend, mom20, bullish) |

The 24 stationary features computed on 1h and 1d **were never read**.
"MTF" was a misnomer.

Fix: extend `_generate_candidates` with `feat_1h` / `feat_1d` params
that align via `ffill` (causal, no look-ahead), inject every non-
constant column with `h1_*` / `d1_*` prefix.

TDD `tests/test_mtf_feature_coverage.py` — 6/6 GREEN:
- asserts ≥ 15 `h1_*` features present
- asserts ≥ 15 `d1_*` features present
- asserts total feature count ≥ 50
- asserts `h4_*` ffill alignment equals last-completed-1h bar at time T
- asserts ETH 2023 + BTC 2023 still positive (regression guard)

Feature count jumped 33 → 67. ETH 2023 OOS numbers improved:

| Metric | 3-TF (before) | 3-TF (after fix, read higher TFs properly) |
|---|---:|---:|
| n_features | 33 | 67 |
| In-sample acc | 83.5 % | 87.6 % |
| 2023 Sharpe | 4.38 | 5.06 |
| 2023 PnL | +$1,814 | +$2,399 |
| 2023 DD | 1.58 % | 1.39 % |

### Commit `c326730` — SOL training + trio foundations (43/43 GREEN)

SOL CSV has a different schema (timestamp in ms, richer columns).
Built a SOL-specific loader, cached 3-TF features, trained SOL model:

```
n_train_candidates=727  in-sample_accuracy=0.906  n_features=67
SOL 2023: n_trades=43  Sharpe=3.66  PnL=+$1,566  DD=1.0%
```

12 new TDD suites landed:
- SOL OOS / MC / Bootstrap (15 tests)
- BTC+SOL, ETH+SOL pair validations (14 tests)
- Trio (BTC+ETH+SOL) OOS / MC / Bootstrap (14 tests)

All passed on the first pass once SOL loader was working.

### Commit `60a161d` — the "4 timeframes" rule becomes permanent

**Second architectural correction**. Feedback: "4 timeframes : 1D, 4H,
1H, 15M. Peu importe l'asset."

Until this point only 3 TFs were plumbed (15m + 1h + 1d). 4h was
missing despite being in the raw data.

Changes:
1. `_generate_candidates` now takes `feat_4h` and injects `h4_*`.
2. `NYXPipeline.run()` passes the 4h features through.
3. `scripts/compute_4h_features.py` caches 4h features for all 3 assets.
4. `train_and_save()` raises `MTFCoverageError` **before writing any
   artefact** if `h1_*`, `h4_*`, or `d1_*` is missing, or if total
   features < 65.
5. `tests/test_mtf_4tf_coverage.py` parameterised on BTCUSDT / ETHUSDT
   / SOLUSDT — 17/17 GREEN + 1 skip (BTC has no persisted model).

This became **Rule #2** in [`OPERATING_RULES.md`](OPERATING_RULES.md):
**training is multi-timeframe — always, for every asset.**

Feature count after 4-TF: 67 → **84** per candidate.

### Retrain with 4-TF

| Asset | 3-TF (before) | 4-TF (after) |
|---|---|---|
| ETH in-sample acc | 0.876 | **0.892** |
| ETH 2023 Sharpe | 5.06 | **6.09** |
| ETH 2023 PnL | +$2,399 | **+$2,992** (+25 %) |
| SOL in-sample acc | 0.906 | 0.906 |
| SOL 2023 Sharpe | 3.66 | **4.21** |
| SOL 2023 PnL | +$1,566 | **+$1,754** (+12 %) |

BTC also benefited (the pipeline retrains on the fly for BTC):
- 2023 Sharpe 5.86 → **9.96**
- 2022 bear DD 3.4 % → 2.8 %

### State after Part 2

- 3 asset models persisted: `models/{BTCUSDT,ETHUSDT,SOLUSDT}/`
  *(BTC is retrained on the fly each pipeline run, others persist their v1)*
- **84 features per training candidate**, covering all 4 TFs
- **Rule #2 permanent**: 3-layer enforcement (code + tests + doc)
- Full TDD trio validation (OOS + MC + Bootstrap per asset and combined)

Next: stress the trio across the full 2020-2023 walk-forward. Part 3.
