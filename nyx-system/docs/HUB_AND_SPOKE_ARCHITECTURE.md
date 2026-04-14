# Hub-and-Spoke Multi-Asset Architecture

> Le scalable, ici, ce n'est pas "un modèle pour tout".
> C'est "une machine unique capable de produire et valider des modèles
> propres pour plusieurs assets".

This document describes the multi-asset architecture, built TDD-strict.
It replaces the naive "3 clones of NYX" approach with a proper
hub-and-spoke design: common core + per-asset spokes + portfolio layer.

```
asset suite (new in this iteration): 45/45 GREEN
paper-live suite total: 177/177 GREEN
pyright: 0 errors
```

## Diagram

```
                 ┌──────────────────────────────┐
                 │      Market Data Layer       │
                 │   ETH / XRP / SOL MTF data   │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │   Shared Feature Engine       │
                 │   Common schema, asset-aware  │
                 └──────────────┬───────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
│ ETH Signal Pod │    │ XRP Signal Pod  │    │ SOL Signal Pod  │
│ edge + ml +    │    │ edge + ml +     │    │ edge + ml +     │
│ bear dial      │    │ bear dial       │    │ bear dial       │
└───────┬────────┘    └────────┬────────┘    └────────┬────────┘
        │                      │                      │
        └──────────────┬───────┴──────────┬──────────┘
                       │                  │
              ┌────────▼──────────────────▼────────┐
              │      Portfolio Allocator           │   ← hub
              │ risk caps / correlation / ranking  │
              └────────────────┬───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Execution Engine   │   ← shared
                    │ maker-first / cancel │   (PostOnlyPaperBroker)
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Logger + Feedback    │   ← per asset + global
                    │ decisions / missed   │
                    └──────────────────────┘
```

## The three layers

### Layer 1 — Per-asset spokes (brains, not hands)

Each asset has a `SignalPod`. Pods are pure producers: they take an
OHLCV bar and emit a `Signal`. They do **not** place orders.

The `Signal` typed contract (`src/assets/signal.py`):

```python
@dataclass(frozen=True)
class Signal:
    symbol: str
    timestamp: str
    direction: int                 # -1 / 0 / +1
    conviction: float              # [0, 1]
    expected_edge_net: float       # bps after fees (can be negative)
    maker_viability: float         # [0, 1]
    regime_tag: str
    bull_bear_tag: str             # 'bull' | 'bear' | 'range'
    size_suggestion: float         # fraction of asset risk_cap
    cluster_group: str             # 'majors' | 'alts' | ...
```

A negative-edge signal or `direction=0` results in `score()==0` — it
will never rank above FLAT. Enforced by `Signal.__post_init__`.

Tests: `tests/test_signal_contract.py` — 10/10 GREEN.

### Layer 2 — Central hub (`PortfolioAllocator`)

The hub takes the N signals emitted on a given bar and returns the
subset that will actually trade. It applies the architecture's risk
rules (`src/assets/portfolio_allocator.py`):

| Rule | Default | Overridable |
|---|---|---|
| `max_total_risk` | 4% | yes |
| `max_asset_risk` | 1.5% | yes |
| `max_cluster_risk` | `{'alts': 2.0%}` | yes |
| `max_open_positions` | 2 | yes |
| no-pyramiding | ON | (always) |

Ranking: `Signal.score() = max(0, edge) × viability × conviction`,
descending. Deterministic tie-break by symbol ascending.

Correlation veto: same-cluster + same-direction share the cluster cap.
Opposite-direction in the same cluster do not correlate → both allowed.

Tests: `tests/test_portfolio_allocator.py` — 12/12 GREEN.

### Layer 3 — Shared execution (`PostOnlyPaperBroker`)

Same broker we already built. Each asset's pod gets its own
`PaperLiveRunner` for isolated storage (decisions DB, missed DB,
heartbeat, state file), but they all share the same broker semantics:

- Post-only strict (REJECTED if limit crosses at placement).
- Timeout without fill → TIMED_OUT (missed trade, logged).
- No taker fallback.

See [`PAPER_LIVE_INFRASTRUCTURE.md`](PAPER_LIVE_INFRASTRUCTURE.md) for
the full broker spec.

## Asset registry (`config/assets.yaml`)

Single source of truth for every per-asset parameter. Adding a symbol
is a yaml-only change.

```yaml
assets:
  ETHUSDT:
    enabled: true
    group: majors
    price_precision: 2
    qty_precision: 3
    maker_fee: 0.0002
    taker_fee: 0.0004
    slippage_model: medium          # low / medium / medium_high / high
    session_profile: liquid         # liquid / event_sensitive / momentum_fast
    risk_cap: 0.40
    model_profile: eth_v1           # locates per-asset model folder

  XRPUSDT: { ... slippage_model: medium_high, session_profile: event_sensitive, ... }
  SOLUSDT: { ... slippage_model: high,        session_profile: momentum_fast,   ... }
```

Load it via `AssetRegistry.load('config/assets.yaml')`. Validates
required fields, enforces the enum for `slippage_model`, raises
`AssetConfigError` on malformed input. Tests: 10/10 GREEN.

## ExecutionProfile (asset → broker params)

`src/assets/execution_profile.py` derives concrete broker settings from
an `Asset`:

| Session profile | `max_wait_bars` |
|---|---|
| liquid | 4 |
| event_sensitive | 3 |
| momentum_fast | 2 |

| Slippage model | `slippage_bps` | `maker_viability_threshold` |
|---|---|---|
| low | 2 | 0.40 |
| medium | 5 | 0.50 |
| medium_high | 8 | 0.60 |
| high | 12 | 0.65 |

Intent:
- **ETH** (liquid, medium) → tolerant, best starter asset.
- **XRP** (event_sensitive, medium_high) → stricter timeout + threshold.
- **SOL** (momentum_fast, high) → tightest timeout, hardest viability bar.

Tests: 7/7 GREEN.

## HubSpokeRunner — the wire-up

`src/assets/hub_spoke_runner.py` ties it all together:

```python
pods = [ETHSignalPod(...), XRPSignalPod(...), SOLSignalPod(...)]

runner = HubSpokeRunner(
    pods=pods,
    storage_root='./storage',
    shared_alerter=multi_alerter,   # Telegram + Discord fan-out
    max_total_risk=0.04,
    max_asset_risk=0.015,
    max_cluster_risk={'alts': 0.02},
    max_open_positions=2,
)

# Each new bar per asset
runner.on_bars({
    'ETHUSDT': eth_bar,
    'XRPUSDT': xrp_bar,
    'SOLUSDT': sol_bar,
})
```

What happens inside `on_bars()`:

1. Each pod is called once → returns a Signal (crash-isolated:
   a pod exception fires an `api_error` event and skips that asset).
2. Broker advances posted orders + missed-trade sweep for every asset.
3. `PortfolioAllocator.decide(signals, open_positions)` ranks + caps.
4. Approved trades → post-only orders via each asset's broker.
5. Every approved trade emits one `trade_decision` event.
6. Rejected-by-allocator signals are NOT placed (the allocator is the
   guardrail — a good signal past its cluster cap does not trade).

Tests: 6/6 GREEN.

## Implementation phases (per architecture spec)

| Phase | Scope | Status |
|---|---|---|
| **Phase 1 — ETH** | port BTC strategy, train `eth_v1`, validate OOS/MC/BS | pod wrapper ready, model training next |
| **Phase 2 — SOL** | nervous/momentum stress test | pending |
| **Phase 3 — XRP** | event-sensitive robustness test | pending |
| **Phase 4 — Portfolio allocator live** | activate only after ≥2 assets validated | code ready, policy values tuneable via yaml |

The TDD suite guarantees the **architecture** is correct. Per-asset
**models** (MLFilter, threshold, bear dial) still need to be trained
from each asset's data — that's not an architecture concern, it's a
data-and-compute task done one asset at a time.

## What we did NOT do (by design)

- **One global multi-asset model.** The spec is explicit: train per-asset
  models first. A global model is phase 5+.
- **Copy-paste the BTC pipeline.** The hub-and-spoke design means the BTC
  `NYXPipeline` is the reference implementation that each SignalPod
  wraps with asset-specific calibration.
- **Per-pair Docker clones.** `docker-compose.multi.yml` remains useful
  for isolated deployments, but the **recommended** production form is
  one process running the `HubSpokeRunner` so the portfolio allocator
  actually sees all signals together.

## Module catalog

| Module | Tests | Purpose |
|---|---|---|
| `src/assets/registry.py` | 10/10 | load + validate `config/assets.yaml` |
| `src/assets/execution_profile.py` | 7/7 | asset → broker params |
| `src/assets/signal.py` | 10/10 | typed contract for pod output |
| `src/assets/portfolio_allocator.py` | 12/12 | risk caps + clusters + ranking |
| `src/assets/hub_spoke_runner.py` | 6/6 | end-to-end wire-up |

## Files

```
config/
├── assets.yaml                 ← per-asset config
src/
└── assets/
    ├── registry.py             ← load assets.yaml
    ├── execution_profile.py    ← per-asset broker params
    ├── signal.py               ← Signal contract
    ├── portfolio_allocator.py  ← hub
    └── hub_spoke_runner.py     ← orchestrator
tests/
├── test_asset_registry.py      10 tests
├── test_execution_profile.py    7 tests
├── test_signal_contract.py     10 tests
├── test_portfolio_allocator.py 12 tests
└── test_hub_spoke_runner.py     6 tests
```

## Next step (not in this commit)

Train + integrate a real `ETHSignalPod`:

1. Compute features from ETH MTF data (shared feature engine).
2. Train `MLFilter` on ETH → save under `models/ETHUSDT/ml_filter_v1.pkl`.
3. Wrap with the `SignalPod` interface (must implement `on_bar(bar) -> Signal`).
4. Instantiate `HubSpokeRunner(pods=[ETHSignalPod(...)])`.
5. Feed live 15m ETH bars; monitor Telegram/Discord.
6. Only after ETH passes OOS + bootstrap, repeat for SOL, then XRP.
7. Then turn on the allocator's multi-asset policy with all three pods.
