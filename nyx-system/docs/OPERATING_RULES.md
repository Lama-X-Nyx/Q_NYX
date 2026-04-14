# Operating Rules — NYX

These rules are **permanent** for any code written in this repo. They
are enforced by the test suite and by the `train_and_save()` guardrail.
Treat them the same way we treat TDD.

---

## Rule 1 — TDD strict

Every new module gets **RED tests first, then GREEN implementation**.
Tests live under `tests/`, named `test_<module>.py`. A module without a
corresponding test file is not considered shipped.

- `pytest tests/` must be GREEN before any `git push`.
- A failing red-bar test is the expected starting state of any new feature.
- Bugs are reproduced as a failing test BEFORE the fix lands.

---

## Rule 2 — Training is multi-timeframe — ALWAYS, FOR EVERY ASSET

No matter which symbol is being trained (BTC, ETH, SOL, XRP, future
additions), the ML filter is fed the **4 timeframes**:

| Prefix    | Timeframe | Role              |
|-----------|-----------|-------------------|
| *(none)*  | 15m       | Execution         |
| `h1_*`    | 1h        | Intra-day regime  |
| `h4_*`    | 4h        | Higher-order regime |
| `d1_*`    | 1d        | Macro context     |

### Why

Backtesting showed that dropping any higher TF silently reduced the
feature block from ~84 to ~33 without any error, while also lowering
OOS Sharpe. The MTF rule ensures every asset trained here sees the
same structural context the BTC reference model sees.

### Enforcement

Three complementary layers enforce this rule:

1. **Pipeline level** — `NYXPipeline._generate_candidates` accepts
   `feat_1h`, `feat_4h`, `feat_1d` and injects every non-constant column
   with the right prefix. See `src/ml/nyx_pipeline.py`.

2. **Training level** — `src.ml.train_asset_model.train_and_save`
   checks that the candidate feature dict includes **all three** higher-
   TF prefixes AND at least `MIN_FEATURES = 65` unique feature keys.
   Either check failing raises `MTFCoverageError` **before anything
   is saved** to `models/<SYMBOL>/`.

3. **Test level** — `tests/test_mtf_4tf_coverage.py` is parameterised
   over every asset (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, …). For each:
   - asserts `h1_*`, `h4_*`, `d1_*` blocks are present
   - asserts total feature count ≥ 65
   - asserts no-look-ahead via forward-fill alignment of `h4_*` to
     the 15m index
   - reads `models/<SYMBOL>/training_metadata.json` and asserts
     `n_features ≥ 65`

Adding a new asset ≡ adding a new entry in the registry + retraining.
The test suite automatically extends coverage if the new asset is
added to the parametrised fixture in `test_mtf_4tf_coverage.py`.

### How to add a new asset without violating the rule

1. Drop `<SYMBOL>_{15m,1h,4h,1d}.csv` under `data/raw/`.
2. Add an entry in `config/assets.yaml`.
3. Compute features (`compute_stationary_features` for each of the 4
   TFs, cached under `data/features/`).
4. Call `train_and_save(symbol, mtf_data, mtf_features, train_end, out_dir)`.
5. Add the symbol to the `_ASSETS` tuple in
   `tests/test_mtf_4tf_coverage.py`.
6. Run `pytest tests/test_mtf_4tf_coverage.py`. Must be GREEN.

---

## Rule 3 — Honest execution

Paper trading uses `PostOnlyPaperBroker` only:

- Post-only strict at placement (REJECTED if limit would cross the
  spread).
- Timeout after `max_wait_bars` without a fill = TIMED_OUT = **missed
  trade**, written to `missed_trades.db`. No silent taker fallback.
- Miss rate exposed live in the heartbeat context
  (`broker_miss_rate`).

Enforced by `tests/test_post_only_broker.py::TestTimeout::test_no_taker_fallback`.

---

## Rule 4 — No silent data loss

- `PersistentDecisionLogger` is append-only with `UNIQUE(timestamp,
  pair)` — duplicates are rejected, not overwritten.
- `StateManager` writes atomically (tmp → fsync → rename) and raises
  `StateCorruptionError` on malformed JSON. Never silently returns an
  empty state.
- Every semantic event is fan-out to every enabled alerter
  (Telegram + Discord). A failing backend does not silence the other.

---

## Rule 5 — Every operator-relevant event is alerted

Six events must reach Telegram + Discord:

| Event                | Severity |
|----------------------|----------|
| `service_down`       | critical |
| `reconnect_exchange` | warn     |
| `trade_decision`     | info     (never deduplicated) |
| `order_unfilled`     | warn     |
| `api_error`          | error    |
| `restart`            | info     |

Enforced by `tests/test_event_alerter.py` + `tests/test_runner_events.py`.

---

## Rule 6 — Pyright clean on new code

`pyrightconfig.json` is kept at **0 errors**. Noisy pandas/numpy stub
errors are silenced at the config level; real bug detectors stay ON:

- `reportPossiblyUnbound` — error
- `reportOptionalSubscript` / `reportOptionalMemberAccess` — error
- `reportUnboundVariable` — error

---

## Permanent guardrail tests

These must stay GREEN on every CI run. Breaking any of them blocks
merge without exception:

- `tests/test_oos_final.py::TestOOSFullPipeline` (BTC edge baseline)
- `tests/test_mtf_4tf_coverage.py` (4-TF rule for every asset)
- `tests/test_mtf_feature_coverage.py` (MTF integration test)
- `tests/test_post_only_broker.py::TestTimeout::test_no_taker_fallback`
- `tests/test_paper_live_runner.py` (wire-up integrity)
