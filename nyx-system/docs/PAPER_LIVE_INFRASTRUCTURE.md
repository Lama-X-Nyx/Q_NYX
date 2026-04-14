# Paper-Live Infrastructure — v0.3.2

All modules built TDD-strict. Test counts are current as of the commit
this document ships with.

```
paper-live suite: 132/132 GREEN
pyright: 0 errors / 25 warnings (legacy non-used)
```

## Table of contents

1. [Overview](#overview)
2. [Pyright setup](#pyright-setup)
3. [Module catalog](#module-catalog)
4. [Execution semantics (honest post-only)](#execution-semantics)
5. [Alerting architecture](#alerting-architecture)
6. [Warm restart + durability](#warm-restart--durability)
7. [Docker deployment](#docker-deployment)
8. [What's NOT in scope yet](#whats-not-in-scope-yet)

---

## Overview

The paper-live stack is the infrastructure that turns a backtested
strategy into a supervisable paper trading process. It sits BETWEEN
the ML pipeline (`NYXPipeline`) and a future live exchange adapter.

Design principles enforced by the test suite:

| Principle | Enforced by |
|---|---|
| No silent data loss | `PersistentDecisionLogger` (SQLite WAL, immediate commit, idempotent key) |
| No stealth taker fallback | `PostOnlyPaperBroker` + `test_no_taker_fallback` |
| Every missed trade is auditable | `MissedTradeLogger` (REJECTED + TIMED_OUT) |
| No half-written state on crash | `StateManager` (tmp → fsync → rename) |
| No corrupt bar reaches strategy | `DataValidator` (NaN/inf/geom/stale) |
| Liveness observable externally | `Heartbeat` (file age, Docker HEALTHCHECK) |
| Every op-relevant event reaches ops | `EventAlerter` over `MultiAlerter` (Telegram + Discord) |
| One failing channel never silences the others | `MultiAlerter` failure isolation |
| Every change is typed | `pyrightconfig.json` with real-bug detectors ON |

---

## Pyright setup

**Config**: `nyx-system/pyrightconfig.json`

Stance: `basic` mode with pandas/numpy stub noise silenced (those checks
produce hundreds of false positives on legitimate pandas code), but real
bug detectors are ON:

| Check | Level | Rationale |
|---|---|---|
| `reportPossiblyUnbound` | **error** | conditional imports / branches |
| `reportUnboundVariable` | **error** | typos / dead branches |
| `reportOptionalSubscript` | **error** | `.foo[x]` on possibly-None |
| `reportOptionalMemberAccess` | **error** | `x.attr` on Optional[X] |
| `reportMissingImports` | warning | some optional deps are expected |
| `reportIndexIssue` | warning | scipy spmatrix false positives |
| `reportArgumentType` | none | pandas stubs are too aggressive |
| `reportAttributeAccessIssue` | none | `Index.date` etc. |

Run: `cd nyx-system && pyright` → expected **0 errors**.

---

## Module catalog

All modules live under `src/paper_live/`. Each module has a companion
test file `tests/test_<module>.py`.

### `decision_logger.py` — `PersistentDecisionLogger`

Append-only SQLite log of every bar decision.

- WAL mode (concurrent readers while strategy writes)
- UNIQUE `(timestamp, pair)` → idempotent writes
- Columns cover: agent states, scores, passed flags; orchestrator
  score + size factor; price, atr, volume ratio; trade intent
  (entry, stop, tp, direction, size); features blob JSON; model version.
- Helpers: `log()`, `load_all()`, `load_range(start, end)`, `count()`,
  `last_timestamp()`.

**Tests**: 11/11.

### `state_manager.py` — `StateManager`

Atomic JSON snapshot for warm restart.

- Write path: `tempfile.mkstemp` → `fsync` → `os.replace`. Never leaves a
  half-written state file on disk.
- Load path: raises `StateCorruptionError` on malformed JSON,
  `StateSchemaError` on incompatible `schema_version`. **Never** silently
  returns empty state — that would zero positions on a corrupt file.
- Refuses to save non-serializable objects (fails fast before touching disk).

**Tests**: 8/8.

### `post_only_broker.py` — `PostOnlyPaperBroker`

Honest maker-first paper simulator. Replaces the earlier
`MakerFirstBroker` (deleted) which silently fell back to taker after
max_wait — that behavior overstated alpha.

**Order state machine**:

```
NEW → POSTED → FILLED       (bar trades through limit → maker)
            ↘ TIMED_OUT  (max_wait_bars reached → MISSED trade)
            ↘ CANCELLED  (client cancel)
     ↘ REJECTED       (post-only would have crossed at placement)
```

API:

- `place_post_only(pair, side, qty, limit_price, mark_price, placed_at, max_wait_bars=...)`
  → returns `oid`. State becomes `REJECTED` immediately if:
    - `side='buy'  and limit_price >= mark_price`, or
    - `side='sell' and limit_price <= mark_price`.
- `on_bar(pair, ohlcv, bar_ts)` advances POSTED orders. Fills as maker
  when the bar's low ≤ buy-limit or high ≥ sell-limit. Otherwise increments
  `bars_waited`; if it reaches `max_wait_bars`, state → `TIMED_OUT`.
- `cancel(oid)`, `get(oid)`, `all_orders()`, `orders_in_state(state)`,
  `missed_trades()`, `filled_orders()`, `rejected_orders()`, `miss_rate()`.

Fee model:
- Maker: `limit_price × qty × maker_fee` (default 0.02%).
- **No taker path.** Taker fills in backtest must now come from the
  strategy explicitly asking for a market order (not modeled yet).

**Tests**: 16/16. The non-negotiable contract test is
`test_no_taker_fallback`: after timeout, `fills` must not contain any
`{role: 'taker'}`.

### `missed_trade_logger.py` — `MissedTradeLogger`

Persistent SQLite log of REJECTED + TIMED_OUT orders. This is the
audit trail "alpha lost to execution reality".

- Schema: `oid (UNIQUE), pair, side, qty, limit_price, mark_price,
  placed_at, final_state, reason, bars_waited, timed_out_at`.
- `log(order)` idempotent on `oid`.
- Queries: `count()`, `count_by_pair(pair)`, `count_by_state(state)`,
  `load_all()` → DataFrame.

**Tests**: 9/9.

### `data_validator.py` — `DataValidator`

Rejects corrupt or stale OHLCV bars before they can poison the
strategy state.

Raises `DataValidationError` on:
- Non-positive `open/high/low/close`.
- NaN or inf in any numeric field.
- `high < low` or `close/open` outside `[low, high]`.
- Negative volume.
- Stale timestamp (≤ last accepted).
- Zero volume only in strict mode (`require_volume=True`).

Stateful: remembers last accepted timestamp.

**Tests**: 11/11.

### `heartbeat.py` — `Heartbeat`

File-based liveness signal. Docker HEALTHCHECK reads the file age.

- `tick(context={...})` atomically rewrites a JSON blob with
  `ts` (epoch) + `iso` + arbitrary context.
- `age_seconds()` → float, `inf` if file missing.
- `is_alive(max_age_s)` → bool.

**Tests**: 7/7.

### `resync_manager.py` — `ReSyncManager`

On restart, compute which bars were missed and replay them.

- `needs_resync(last_seen, current, tf)` → True if gap > 1 bar.
- `missing_bars(last_seen, current, tf)` → list of ISO timestamps.
- `replay(last_seen, current, tf, fetch_fn)` → list of bars via the
  injected `fetch_fn(ts)` callback.
- Supports `15m`, `1h`, `4h`, `1d`.

**Tests**: 9/9.

### `telegram_alerter.py` — `TelegramAlerter`

- `send(text)`, `info/warn/error/critical(text)`.
- Injectable `http_post` (unit tests never talk to the real Bot API).
- `cooldown_s` dedup by message content.
- Missing `token` or `chat_id` → no-op returning False.

**Tests**: 7/7.

### `discord_alerter.py` — `DiscordAlerter`

Same contract as Telegram but for a Discord webhook URL.
POSTs `{"content": text}` as JSON.

**Tests**: 7/7.

### `multi_alerter.py` — `MultiAlerter`

Fan-out to N backends. A failing backend never prevents the others from
being called. Returns True if **any** backend returned True.

**Tests**: 8/8.

### `event_alerter.py` — `EventAlerter`

Semantic events routed to the right severity. Every message is prefixed
`[NYX] <event>:` so ops grep easily.

| Event | Severity | Dedup | Trigger |
|---|---|---|---|
| `service_down` | critical | yes | watchdog → `runner.notify_service_down()` |
| `reconnect_exchange` | warn | yes | adapter → `runner.notify_reconnect_exchange()` |
| `trade_decision` | info | **never** | auto on BUY/SELL |
| `order_unfilled` | warn | yes (once per oid) | auto on REJECTED/TIMED_OUT |
| `api_error` | error | yes | auto on strategy crash + `notify_api_error()` |
| `restart` | info | yes | `main.py` calls `notify_restart()` at boot |

`trade_decision` is **never** rate-limited — every trade must surface.

**Tests**: 12/12.

### `runner.py` — `PaperLiveRunner`

Wires everything. Per-bar flow:

```
validate(bar)                   # DataValidator
  ↓
broker.on_bar(pair, bar, ts)    # PostOnlyPaperBroker advances orders
  ↓
_sweep_missed()                  # log + alert newly REJECTED/TIMED_OUT
  ↓
strategy.decide(pair, bar)       # → {action, order_intent, agent_results}
  ↓
if action in (BUY, SELL) with order_intent:
  broker.place_post_only(...)    # may be REJECTED immediately
  _handle_missed(new_order)      # catches immediate reject
  ↓
decision_logger.log(row)         # persist decision
  ↓
events.trade_decision(...)       # Telegram + Discord
  ↓
state_manager.save({...})        # atomic snapshot
  ↓
heartbeat.tick({
  bars_processed, last_pair,
  last_ts, broker_miss_rate,    # live miss-rate exposed
})
```

External helpers:
- `notify_restart()`, `notify_service_down(reason)`,
  `notify_reconnect_exchange(reason)`, `notify_api_error(reason)`.

**Tests**: 6 single-pair + 7 event integration + 5 post-only integration = 18/18.

### `multi_pair_runner.py` — `MultiPairRunner`

Runs N fully isolated `PaperLiveRunner` instances in parallel (one per
pair). Used for the ETH / XRP / SOL paper trio.

Storage is per-pair:

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

API:

- `MultiPairRunner(config, shared_alerter=None)` — builds one sub-runner
  per pair. `shared_alerter` is optional; if provided, every sub-runner's
  `EventAlerter` routes through it (single ops stream).
- `on_bar(pair, bar)` routes to the matching sub-runner. Unknown pair
  raises `UnknownPairError` (fail loudly).
- `on_bars({pair: bar, ...}, raise_on_error=False)` dispatches a batch.
  With `raise_on_error=False`, a crash in one pair is captured and
  returned in the errors dict without aborting the other pairs.
- `miss_rate_all()` → `{pair: post-only miss rate}`.
- `heartbeat_status(max_age_s)` → `{pair: bool}`.
- `bars_processed_all()` → `{pair: int}`.
- `notify_restart_all()`, `notify_reconnect_exchange(reason)` fan-out.
- `shutdown()` closes every sub-runner.

**Tests**: 9/9. Key contracts:
- Isolated storage per pair (paths contain pair name, no sharing).
- Crash in ETH does not prevent XRP / SOL from processing their bars.
- Trade-decision messages always include the pair name.

---

## Execution semantics (honest post-only)

The strategy must emit an `order_intent` with every BUY/SELL decision:

```python
{
    'action': 'BUY',
    'trade_size': 0.01,
    'order_intent': {
        'side': 'buy',
        'qty': 0.01,
        'limit_price': 44_900.0,    # sub-market for a buy → posts as maker
        'mark_price':  45_000.0,    # snapshot of book at decision time
        'max_wait_bars': 3,          # timeout budget
    },
    # ... agent_results etc.
}
```

What happens next:

| Condition at placement | State | Observable |
|---|---|---|
| `limit_price >= mark` (buy) | `REJECTED` | missed log + `order_unfilled` alert |
| `limit_price <  mark` (buy), bar never crosses within `max_wait` | `TIMED_OUT` | missed log + `order_unfilled` alert |
| Bar crosses the limit within `max_wait` | `FILLED` | `trade_decision` alert, fill recorded |

The paper miss-rate (`broker.miss_rate() = TIMED_OUT / (TIMED_OUT + FILLED)`)
is written to the heartbeat context every bar — operators see in
near-real-time whether the strategy is actually getting filled.

---

## Alerting architecture

```
                 ┌──────────────────────┐
                 │      runner.events    │
                 │   (EventAlerter)      │
                 └──────────┬───────────┘
                            │ uses
                            ▼
                 ┌──────────────────────┐
                 │   runner.alerter     │
                 │   (MultiAlerter)      │
                 └──────────┬───────────┘
                            │ fan-out
             ┌──────────────┴──────────────┐
             ▼                             ▼
    ┌────────────────────┐       ┌────────────────────┐
    │  TelegramAlerter   │       │  DiscordAlerter    │
    │  (Bot API POST)    │       │  (webhook POST)    │
    └────────────────────┘       └────────────────────┘
```

Either backend can be disabled independently (no env var = no-op).
Failure in one never prevents the other from sending.

Env vars (consumed by `main.py`):

| Var | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Target chat |
| `DISCORD_WEBHOOK_URL` | Discord webhook |

---

## Warm restart + durability

Survival of a crash / container restart:

1. **Decisions** → SQLite WAL commits on every `log()`.
2. **Missed trades** → same, separate DB.
3. **State** (capital, equity, positions, last_bar_ts per pair) →
   atomic `state.json` via `tmp → fsync → rename`.
4. **Heartbeat** → updated every bar; Docker HEALTHCHECK reads its age.

On boot, `PaperLiveRunner._load_warm_state()` reads `state.json`. If the
file is corrupt, a `StateCorruptionError` surfaces (fail loudly, don't
silently zero positions). Schema mismatch raises `StateSchemaError`.

`ReSyncManager` computes the gap between `last_bar_ts` and wall-clock
current time, so the loader can replay the missed bars through the
strategy before going live.

---

## Docker deployment

Files: `nyx-system/Dockerfile`, `nyx-system/docker-compose.yml`.

- `python:3.11-slim` base.
- `VOLUME /app/nyx-system/storage` persists decisions DB, missed DB,
  state file, heartbeat.
- `HEALTHCHECK` polls `Heartbeat.is_alive(300)`.
- `restart: unless-stopped` on the compose service.
- Log rotation via docker json-file driver (50m × 5).

Minimum `.env`:

```bash
NYX_PAIR=BTCUSDT
NYX_MODEL_VERSION=v0.3.2
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DISCORD_WEBHOOK_URL=...
```

Start (single pair — BTC by default):

```bash
cd nyx-system
docker compose up -d
docker compose logs -f nyx-paper-live
```

### Multi-pair: ETH + XRP + SOL in parallel

`docker-compose.multi.yml` spins up three independent containers, each
with its own storage volume and heartbeat. A crash in one pair does not
affect the others, and Docker auto-restarts any container whose
heartbeat goes stale (> 5 min).

```bash
cd nyx-system
docker compose -f docker-compose.multi.yml up -d
docker compose -f docker-compose.multi.yml logs -f nyx-eth
```

Storage layout on the host:

```
storage/
├── ETHUSDT/   ← nyx-paper-eth writes here
├── XRPUSDT/
└── SOLUSDT/
```

Running BTC plus the alt trio at the same time is supported: start both
compose files. Each container has a distinct `container_name` so there
is no port / volume collision.

---

## What's NOT in scope yet

These are deliberately out of the paper-live infrastructure because the
user flagged them as "peut attendre":

- Grafana dashboards
- Prometheus scrape endpoint
- Kubernetes manifests

These are explicitly out of the paper-live infrastructure because they
are the **next integration step** and depend on operational environment:

- Live Binance adapter (WebSocket bar feed + REST post-only adapter
  that respects our `order_intent` contract)
- Real strategy replacing `_NoopStrategy` in `main.py`
- Docker HEALTHCHECK-triggered `service_down` alert (currently the
  infrastructure exposes it, but no external watchdog calls it yet)

---

## Change log (paper-live only)

| Commit | What |
|---|---|
| chore(types) | pyrightconfig + real type-error fixes |
| feat(paper-live) infra | DecisionLogger, StateManager, Heartbeat, DataValidator, ReSyncManager, PaperLiveRunner (67/67) |
| feat(paper-live) alerts | Discord + MultiAlerter + EventAlerter + 6 events (34/34) |
| feat(paper-live) post-only | PostOnlyPaperBroker (strict, no taker fallback), MissedTradeLogger (30/30) |
| feat(paper-live) multi-pair | MultiPairRunner + docker-compose.multi.yml for ETH/XRP/SOL (9/9) |
