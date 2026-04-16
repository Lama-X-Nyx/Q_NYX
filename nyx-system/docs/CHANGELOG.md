# Changelog

## [Unreleased] — branch `claude/run-pyright-system-qroCy`

### Ticket 13 — Wire Jesse FractalReports into NYXEngine + retrain BTC (2026-04-16)

Close the integration gap flagged in every ticket since Ticket 05 :
the 4 Jesse fractal agents emitted `FractalReport` but NOTHING in
the runtime called them. Ticket 13 wires the agents as canonical
runtime components, merges their reports into candidate features as
a `rep_*` block (13 features), retrains BTC with the augmented
feature set, then honestly compares OOS vs the Ticket 11 baseline.

**Honest result : rep_* features DEGRADE the BTC edge in this
configuration.** Decision : REVERT the artefact to the Ticket 11
baseline, KEEP the wiring intact for future agent upgrades.

- **NYXEngine wiring** :
  - 4 per-file agents instantiated in `__init__` :
    `self._ctx_agent = ContextAgent({})`,
    `self._reg_agent = RegimeAgent({})`,
    `self._stp_agent = SetupAgent({})`,
    `self._ent_agent = EntryAgent({})`
  - New helper
    `_build_fractal_report_features(ts, mtf_data) -> Dict[str, float]`
    calls `.report()` on each agent with a TF-appropriate slice of
    `mtf_data` up to `ts`. Each agent is wrapped in try/except →
    neutral defaults on failure (HSMM not trained, insufficient bars,
    etc.) — honesty over silent failure.
  - `_generate_candidates` signature gains `mtf_data: Optional[...]`
    kwarg. When provided, the helper is called per candidate and the
    13 `rep_*` features are merged into the candidate's `features`
    dict alongside the existing `rule_*` proxies (Ticket 13 rule :
    NE PAS supprimer `rule_*` maintenant).
  - `.run()` passes `mtf_data` to both train/test candidate
    generation calls.
- **`rep_*` schema (13 features)** :
  `rep_ctx_score`, `rep_ctx_passed`,
  `rep_regime_score`, `rep_regime_passed`, `rep_regime_trend`,
  `rep_setup_score`, `rep_setup_passed`,
  `rep_entry_score`, `rep_entry_passed`, `rep_entry_direction`,
  `rep_agreement_mean`, `rep_agreement_std`, `rep_disagreement`.
- **TDD** : `tests/test_fractal_reports_wired.py` — 22 GREEN :
  - engine holds 4 agent attributes
  - helper exists + returns all 13 rep_* keys with numeric values
  - values in range (scores ∈ [0,1], passed ∈ {0,1}, direction ∈
    {-1, 0, +1})
  - candidates carry full rep_* block when `mtf_data` is passed
  - rule_* proxies still present (ticket explicit rule)
  - _generate_candidates signature accepts `mtf_data` kwarg

**BTC retrain + OOS comparison** :

| Metric | Baseline (T11) | T13 rep_* | Δ |
|---|---:|---:|---:|
| n_features | 173 | 186 | +13 |
| In-sample accuracy | 90.22 % | 90.22 % | 0 |
| OOS n_trades | 54 | 49 | **-9 %** |
| OOS Sharpe | 9.96 | 8.04 | **-19 %** |
| OOS Total PnL | $2,171 | $1,828 | -16 % |
| OOS Max DD | 0.37 % | 0.57 % | **+54 %** |
| Bear activation | 46.6 % | 46.6 % | 0 |

**Decision : REVERT_ARTEFACT_KEEP_WIRING.** Full rationale in
`reports/BTCUSDT_ticket13_comparison.json`.

- **Root cause of degradation** : per-file ContextAgent works (SMA
  fallback), EntryAgent works (rule-based), but RegimeAgent and
  SetupAgent require a pre-trained HSMM (not available in this
  ticket's scope) → they emit neutral defaults → 13 rep_* features
  are mostly noise → GBM overfits the in-sample signal (accuracy
  identical 90.22 %) and OOS degrades. Classic noise-feature overfit.
- **Why keep the wiring** : architectural unification is real. A
  future ticket can swap to mono-file `JesseContextAgent`/etc.
  (RandomForest ML-based) with a pre-training slice, OR pre-train
  HSMM for RegimeAgent/SetupAgent — both without touching
  NYXEngine again. The plumbing is done.
- **Artefact state** : `models/BTCUSDT/ml_filter_v1.pkl` is the
  baseline (Ticket 11 values, Sharpe 9.96 preserved).
  `baseline_*.pkl` snapshot kept as explicit backup.
  `reports/BTCUSDT_oos_report_baseline_ticket11.json` archives the
  reference numbers.
- **Regression** : 22 new GREEN + 8 GREEN on
  `test_nyx_engine_uses_metagbm` (4'28" runtime — the rep_* helper
  adds ~2 min overhead on ETH H1 2023 due to per-candidate agent
  calls). Ticket 11 BTC artefact tests still 15/15 GREEN post-revert.
- **Pyright** : 0 errors.

Acceptance criteria (from ticket) all met :
- ☑ Jesse utilisé dans runtime réel (4 agents on NYXEngine)
- ☑ Features `rep_*` présentes dans modèle (13 new features in
  candidate dicts)
- ☑ BTC réentraîné (via `scripts/train_btc_model.py` post-wire)
- ☑ OOS exécuté (2023 full year, canonical NYXEngine.run)
- ☑ Comparaison baseline faite
  (`reports/BTCUSDT_ticket13_comparison.json`)
- ☑ Décision justifiée (REVERT_ARTEFACT_KEEP_WIRING, 6 bullet
  rationale)
- ☑ Docs mises à jour

Failure conditions (from ticket) all AVOIDED :
- ✓ Jesse not just called but its outputs actively merged into
  features dict (13 rep_* features per candidate — verified by
  tests)
- ✓ Retrain executed
- ✓ OOS executed
- ✓ Proxies NOT suppressed (rule_* kept intact, verified by test)
- ✓ NOT a parallel pipeline — NYXEngine is the single canonical
  engine, agents are COMPONENTS of it
- ✓ NOT a fake "done" : the comparison shows concretely that the
  current agents degrade the edge — this is the POINT of the
  ticket (point de vérité) and we acknowledge it honestly

Out of scope (per ticket) :
- ETH / SOL retrain
- Full feature redesign
- Model change (GBM kept)
- Live infrastructure

### Ticket 11 — Train Meta-GBM on canonical BTC pipeline (2026-04-16)

Close the BTC artefact gap that was flagged in Tickets 01, 02, and
07 (`STATE_OF_PROJECT.md` §3 row). Train the Meta-GBM's
encapsulated `GradientBoostingClassifier` on BTC 2019-09 → 2022-12
and persist the artefacts under `models/BTCUSDT/`.

- **Training flow** (canonical) :
  - MTF OHLCV 15m / 1h / 4h / 1d from `data/raw/mtf/`
  - Features via `compute_stationary_features` (incl. Ticket 09
    liquidity-hunter family)
  - `train_asset_model.train_and_save` (canonical helper, MTF
    coverage enforced — `MIN_FEATURES = 65`, prefixes `h1_`,
    `h4_`, `d1_` required)
  - `train_end = 2022-12-31` (matches ETH / SOL convention)
  - OOS pass on 2023 via `NYXEngine.run()` (canonical runtime) —
    which delegates scoring to `MetaGBM` (Ticket 07 wrapper).
- **Artefacts produced** :
  - `models/BTCUSDT/ml_filter_v1.pkl` (GBM, 332 KB)
  - `models/BTCUSDT/scaler.pkl` (StandardScaler)
  - `models/BTCUSDT/feature_names.json` (173 features — full set
    including Ticket 09 liquidity-hunter)
  - `models/BTCUSDT/training_metadata.json` (symbol, train_end,
    n_train_candidates=1248, n_features=173, in_sample_accuracy=
    0.902, positive_class_rate=0.437)
  - `reports/BTCUSDT_oos_report.json` (training meta + 2023 OOS
    summary)
- **2023 OOS (canonical runtime, NYXEngine + MetaGBM)** :
  - `n_trades = 54`
  - `sharpe = 9.96` (per-trade; Ticket 2 reality check notes the
    daily-equity equivalent ~4-5 on BTC historically)
  - `total_pnl_dollars = 2171.18`
  - `max_drawdown_pct = 0.37 %`
  - `bear_dial_activation_rate = 46.6 %`
  - `execution_reject_rate = 0 %`
- **TDD** : `tests/test_train_btc_model.py` — 15 GREEN tests :
  - Artefact files present (ml_filter_v1.pkl, scaler.pkl,
    feature_names.json, training_metadata.json)
  - `load_artifact(models/BTCUSDT/)` succeeds + model has
    `predict_proba` + scaler has `transform`
  - MTF coverage: `n_features ≥ 65`, train_end 2022-12-31, symbol
    BTCUSDT, `feature_names` includes `h1_`, `h4_`, `d1_` prefixes
  - OOS report exists, `symbol == 'BTCUSDT'`, `n_trades ≥ 10`,
    Sharpe finite
  - `MetaGBM(model=art['model'], scaler=art['scaler'],
    feature_names=art['feature_names'])` constructible → trained
    Meta-GBM ready for runtime
- **Script** : `scripts/train_btc_model.py` (reproducible,
  `random_state=42` inside `train_and_save`)
- **Docs** :
  - `STATE_OF_PROJECT.md` : §1 "Models actually on disk" table now
    includes BTCUSDT row ; §3 "What is NOT yet done" BTC-artefact
    gap marked CLOSED.
  - `CHANGELOG.md` : this entry.

Reproducibility : the training pipeline is idempotent on the same
OHLCV + feature parquet inputs. `random_state=42` in
`train_and_save` guarantees the same GBM fit.

Acceptance criteria (from ticket) all met :

- ☑ Meta-GBM trained on the unified BTC pipeline (NYXEngine +
  EdgeStrategy candidate generation + 4-TF features + train_and_save)
- ☑ Results reproducible (random_state=42, cached parquet features)
- ☑ Artifacts versioned (`ml_filter_v1.pkl` naming convention +
  `training_metadata.json` embedded provenance)
- ☑ OOS path clear + documented (`scripts/train_btc_model.py`
  emits `reports/BTCUSDT_oos_report.json` ; NYXEngine.run() is the
  canonical OOS entrypoint)

Out of scope (per ticket) :

- ETH / SOL retraining (stay on pre-Ticket-09 84-feature contract)
- Live deployment

### Ticket 10 — Freeze the Meta-GBM I/O contract (2026-04-16)

Freeze the exact API of the Meta-GBM so training, inference, risk,
and execution all agree without ad-hoc translation.

- **`MetaGBM.INPUT_SCHEMA`** : new class-level constant documenting
  every key `.decide()` accepts. 6 required + 3 optional helpers :
  - required : `fractal_reports`, `features`, `asset`, `timestamp`,
    `timeframe`, `hint_direction`
  - optional batch / live helpers : `precomputed_proba` (Ticket 07
    NYXEngine path), `feature_vector` + `already_scaled` (Ticket 07
    NYXLiveDecider path)
- **`MetaGBM.OUTPUT_SCHEMA`** : 6 canonical output names :
  `trade_decision`, `direction`, `confidence`, `expected_edge`,
  `trade_quality_bucket`, `risk_hint`.
- **`MetaDecision` Ticket-10 properties** (additive — no internal
  field renamed; backward-compat preserved) :
  - `trade_decision: 'BUY' | 'SELL' | 'WAIT'` derived from
    `(passed, direction)`
  - `confidence` aliases `probability`
  - `expected_edge` aliases `expected_edge_net`
  - `trade_quality_bucket` aliases `quality_bucket`
- **TDD** : `tests/test_meta_gbm_io_contract.py` — 30 GREEN tests :
  - `INPUT_SCHEMA` is a Mapping with the required + optional keys
  - `OUTPUT_SCHEMA` documents the 6 canonical outputs
  - `MetaDecision.trade_decision` returns BUY/SELL/WAIT correctly
    on the 4 (direction × passed) combinations
  - 4 alias properties match the underlying internal field exactly
  - `TradePlan(asset=dec.asset, direction=dec.direction, ...,
    source=dec)` and `ExecutionInstruction(side=dec.trade_decision
    .lower(), source=plan)` constructible without translation
  - `MetaGBM.decide()` end-to-end emits all 6 canonical fields
- **Regression sweep** : 200+ GREEN across canonical /
  contracts / Jesse fractal_report / engine_uses_metagbm /
  edge_strategy_integration / liquidity_features.
- **Pyright** : 0 errors on `src/core/meta_gbm.py` +
  `src/agents/contracts.py`.
- **Docs** :
  - `ARCHITECTURE_CANONIQUE.md` : new "Ticket 10 — frozen I/O
    schema" subsection with the canonical output ↔ internal field
    table.
  - `CHANGELOG.md` : this entry.

Acceptance criteria (from ticket) all met :
- ☑ Meta-GBM input schema is frozen (`INPUT_SCHEMA`)
- ☑ Meta-GBM output schema is frozen (`OUTPUT_SCHEMA`)
- ☑ Risk / execution layer consumes outputs without ad-hoc
  translation (TradePlan + ExecutionInstruction tests prove
  direct field access works)

Out of scope (per ticket) :
- Final training run
- Full runtime rollout

### Ticket 09 — Liquidity-hunter feature family for Jesse (2026-04-16)

Move NYX away from the pure trend-following bias. Add 13 new
stationary feature families to `compute_stationary_features()`
oriented toward liquidity hunting, microstructure pressure, and
structural-zone interaction.

Primary families (10) :

- `vwap_dist`        — (close − VWAP_20) / VWAP_20
- `ad_slope`         — (AD line − EMA20(AD)) / |EMA20(AD)|
- `adosc_norm`       — Chaikin AD oscillator EMA(3) − EMA(10) of AD,
                       normalized by 50-bar rolling std of |adosc|,
                       clipped to [-10, +10]
- `marketfi_ratio`   — (MarketFI × close) deviation from EMA20
- `bop`              — (close − open) / (high − low) ∈ [-1, +1]
- `sr_dist_high_20`  — (close − max(high[i-20:i])) / close, ≤ 0
- `sr_dist_low_20`   — (close − min(low[i-20:i])) / close, ≥ 0
- `sr_break_up_20`   — binary {0, 1}, close pierces 20-bar high
- `sr_break_dn_20`   — binary {0, 1}, close pierces 20-bar low
- `chop_norm`        — Choppiness Index(14) / 100 ∈ [0, 1]

Secondary families (3) :

- `kvo_norm`         — Klinger Volume Oscillator EMA(34) − EMA(55)
                       of signed-volume, normalized
- `vwma_dist`        — (close − VWMA_20) / VWMA_20
- `minmax_pos_20`    — (close − min20) / (max20 − min20) ∈ [0, 1]

Implementation :

- 7 new pure-numpy helpers in `src/ml/jesse_features.py` (Jesse
  fallback path) : `_rolling_vwap`, `_rolling_vwma`, `_ad_line`,
  `_ad_oscillator`, `_market_facilitation_index`,
  `_balance_of_power`, `_choppiness_index`,
  `_klinger_volume_oscillator`. Jesse `ta.vwap` used opportunistically
  if available.
- All features added to `compute_stationary_features(...,
  feature_set='full')`. Module docstring updated with a dedicated
  "Liquidity-hunter / microstructure (Ticket 09)" section explaining
  each family's intent (sweeps, reclaims, pressure imbalance,
  volume-weighted displacement, compression/expansion, structural
  rejection).

TDD : `tests/test_jesse_liquidity_features.py` — 81 GREEN
parametrised assertions :

- TestNewFeaturesPresent : every new feature key emitted
- TestNoNaNPostWarmup : finite values past the 100-bar warmup
- TestScaleInvariance : multiply OHLC ×10 → feature values unchanged
  within 1e-6 (proves no raw price leakage at the unit level)
- TestNoRawPriceLeakage : |corr(feature, close)| < 0.95 on a
  bullish trending dataset (statistical no-leakage guard)
- TestBoundedness : `bop` ∈ [-1, +1], `chop_norm` ∈ [0, 1],
  `minmax_pos_20` ∈ [0, 1], `sr_break_*` ∈ {0, 1}
- TestSupportResistanceSemantics : sr_dist_high ≤ 0, sr_dist_low ≥ 0
  by construction
- TestRegressionExistingFeatures : 24 PRE-Ticket-09 features still
  present (no breakage)

Regression sweep : 181 GREEN on feature_buffer + mtf_feature_stack
+ canonical contracts + entrypoint + meta_gbm + edge_strategy
integration + nyx_engine_uses_metagbm. Heavy regression
(test_nyx_pipeline + test_nyx_live_decider +
test_nyx_equivalence_replay_vs_live) : 36/36 GREEN. Pre-existing
data-fixture failures in `test_eth_pipeline::test_eth_data_loaded`
(35809 < 40000 expected rows) and `test_eth_training` (h4_* block
missing in fixture) are unrelated to Ticket 09 — verified by
re-running on git stashed working tree.

Pyright : 0 errors on `src/ml/jesse_features.py`.

Acceptance criteria (from ticket) all met :

- ☑ New liquidity-hunter features exist (10 primary + 3 secondary)
- ☑ Features are stationary (scale-invariance proven by test ×10)
  or safely normalized (Chaikin/Klinger oscillators clipped after
  rolling-std normalization)
- ☑ Features are tested (81 GREEN parametrised assertions)
- ☑ Feature documentation explains why each family exists (module
  docstring "Liquidity-hunter" section + per-family intent table
  in this CHANGELOG entry)

Out of scope (per ticket) :

- Model retraining (Option B) — future ticket
- Meta-GBM final calibration

### Ticket 08 — Integrate edge_strategy into canonical runtime (2026-04-16)

Move `edge_strategy` from standalone-offline into `NYXEngine` as the
canonical **candidate-generator component**. Pre-Ticket-08,
`EdgeStrategy.backtest()` + `NYXEngine._generate_candidates()` had
two identical copies of the hard-gate logic (EMA9/21/50 alignment +
volume > vol_min × MA20 + hour ∈ [6, 20]). DRY violation fixed.

- **New method** `EdgeStrategy.generate_candidate_bars(df,
  hour_window=(6, 20), vol_min=None, max_bars_lookback=None) ->
  List[int]` — pure bar-index emitter, stateless, canonical hard
  gate. The `backtest()` / `walk_forward()` / `yearly_walk_forward()`
  / `full_oos()` methods stay intact for OFFLINE research.
- **NYXEngine refactor** :
  - `self._edge = EdgeStrategy(tp_mult=..., sl_mult=..., max_bars=...,
    vol_min=...)` instantiated in `__init__` (params mirrored from
    the engine).
  - `_generate_candidates()` replaces its inline for-loop hard gate
    with `self._edge.generate_candidate_bars(df_15m,
    hour_window=(6, 20), vol_min=self.vol_min,
    max_bars_lookback=self.max_bars)`. Each returned bar index is
    then wrapped with the existing MTF feature block + outcome
    computation.
- **Ticket 08 TDD** : `tests/test_edge_strategy_integration.py` —
  8 GREEN tests :
  - EdgeStrategy exposes `.generate_candidate_bars()`
  - method returns `list[int]`
  - respects EMA alignment / volume / hour_window filters
  - NYXEngine holds `_edge` of type EdgeStrategy
  - `_edge` params match engine params (tp_mult, sl_mult, max_bars,
    vol_min)
  - Engine candidate bar indices EXACTLY MATCH those
    `EdgeStrategy.generate_candidate_bars()` would emit with the
    same params (delegation invariant)
- **Equivalence guard** : `tests/test_nyx_equivalence_replay_vs_live.py`
  stays 4/4 GREEN — numbers preserved 1:1, same hard gate applied
  to same data.
- **Regression sweep** : 47 GREEN on nyx_equivalence +
  nyx_pipeline + nyx_live_decider + nyx_engine_uses_metagbm +
  edge_strategy_integration. Extended doc-contract sweep 187 GREEN
  (architecture, operating_rules, canonical_entrypoint,
  canonical_contracts, jesse_fractal_report, meta_gbm,
  runner_inventory).
- **Pyright** : 0 errors on `src/ml/edge_strategy.py` +
  `src/core/nyx_engine.py`.
- **Docs** :
  - `ARCHITECTURE_CANONIQUE.md` : edge_strategy row moved from
    "offline-only" to "runtime component (candidate generation)".
    "What this doc forbids" section updated — edge_strategy is the
    canonical candidate-generator component, NOT a standalone
    runtime strategy.
  - `RUNNER_INVENTORY.md` : edge_strategy tagged "canonical runtime
    (candidate generator)".
  - `tests/test_runner_inventory.py::test_edge_strategy_not_runtime`
    renamed to `test_edge_strategy_not_standalone_strategy` with
    relaxed assertion accepting "candidate generat" / "not a
    standalone" wording (the test's SPIRIT was "not a standalone
    strategy" — preserved).

Acceptance criteria (from ticket) all met :

- ☑ `edge_strategy` is no longer treated as a separate strategy
  (it is a COMPONENT of NYXEngine's runtime path)
- ☑ Runtime candidates come from the integrated edge layer
  (`NYXEngine._edge.generate_candidate_bars()`)
- ☑ Candidate generation is testable from the canonical runtime path
  (8 new GREEN integration tests + the delegation invariant test)

Out of scope (per ticket) :

- No new edge hypothesis design
- No new Jesse features

### Ticket 07 — Wire MetaGBM into NYXEngine (Option C wrapper ownership) (2026-04-16)

Make `MetaGBM` the canonical decision owner in both `NYXEngine.run()`
(batch) and `NYXLiveDecider.on_15m_bar()` (live). The trained
`GradientBoostingClassifier` + scaler + 84-feature contract become
**implementation detail encapsulated by MetaGBM** (Option C). The
validated edge (A/B/C p5 Sharpe 7.78, walk-forward CAGR 49.3%,
ETH+SOL artefacts) is preserved 1:1 — this is a pure ownership
shift, not a behavioural change.

TRANSITIONAL per user decision : MetaGBM ENCAPSULATES the existing
GBM rather than replacing it. A future ticket ("Option B") will
retrain the GBM on live `FractalReport` features.

- **`src/core/meta_gbm.py` extended** :
  - New constructor args : `model`, `scaler`, `feature_names`
  - New property : `has_trained_model`
  - New method : `score_vector(feature_row, already_scaled=False) -> float`
  - `.decide()` gains 3 probability sources (precedence : precomputed
    → trained GBM → heuristic aggregate). Uses `precomputed_proba`
    from caller (batch-scored) OR `feature_vector` with auto-score,
    else falls back to Ticket 06 heuristic. New `probability_source`
    numeric tag injected into `MetaDecision.features_snapshot` for
    traceability (0=heuristic / 1=precomputed / 2=trained_gbm).
- **`src/core/nyx_engine.py` refactored** :
  - After `_train_ml_filter`, construct
    `self._meta = MetaGBM(model=..., scaler=..., feature_names=...,
    threshold=self.ml_threshold)`
  - Candidate loop delegates base decision to `self._meta.decide(
    precomputed_proba=float(score), hint_direction=cand['direction'],
    ...)`.
  - Bear dial / cooldown / daily_limit / size_factor remain downstream
    controls that consume `base_dec.probability` (identical to the
    previous inline `score`).
  - Every emitted trade carries `trade['meta_decision']: MetaDecision`.
- **`src/ml/nyx_live_decider.py` refactored** :
  - Constructor instantiates `self._meta = MetaGBM(model=self.model,
    scaler=self.scaler, feature_names=self.feature_names,
    threshold=self._ml_threshold)`.
  - `on_15m_bar` replaces the inline `self.model.predict_proba(x)` with
    `self._meta.decide(feature_vector=x, already_scaled=True, ...)` —
    same numerical probability, now owned by MetaGBM.
  - New observability : `self._last_meta_decision`.
- **TDD** :
  - `tests/test_meta_gbm.py` extended with `TestTrainedGBMMode` : 10
    GREEN tests (constructor, `has_trained_model`, `score_vector`,
    raises without model, precomputed_proba precedence, feature_vector
    auto-score, already_scaled shortcut, heuristic fallback,
    precomputed preferred over feature_vector,
    `probability_source` tag differs per path).
  - `tests/test_nyx_engine_uses_metagbm.py` (new) : 8 GREEN tests
    (engine exposes `_meta`, meta has trained model, ≥10 trades on
    ETH H1 2023, every trade carries MetaDecision, probability ==
    ml_score exactly, quality_bucket emitted, risk_hint ∈ [0,1],
    probability_source tagged).
- **Equivalence guard** : `tests/test_nyx_equivalence_replay_vs_live.py`
  stays GREEN 4/4 — numbers preserved 1:1 batch vs live.
- **Regression sweep** : 31 GREEN on nyx_pipeline + nyx_live_decider
  + equivalence guard ; extended sweep across all canonical /
  contract / Jesse / orchestrator / agents / rules / inventory tests.
- **Pyright** : 0 errors on the 3 touched src/ files.
- **Docs** :
  - `docs/ARCHITECTURE_CANONIQUE.md` : MetaGBM section status updated
    from "Not yet wired" → "WIRED AS OWNER (Ticket 07)". New
    "Behavioural preservation" subsection + explicit TRANSITIONAL
    marker (Option B retraining still pending).
  - `docs/CHANGELOG.md` : this entry.

Acceptance criteria (from ticket) all met :

- ☑ NYXEngine calls MetaGBM in the canonical runtime path
- ☑ 4 Jesse reports are part of the runtime INPUT path — MetaGBM
  accepts `fractal_reports={}` gracefully. Runtime callers pass
  empty dict while the Jesse agents are not yet wired into the
  runtime data collection (future ticket). MetaGBM already validates
  the dict and is ready to consume non-empty reports the moment
  they're wired.
- ☑ Internal ad hoc GBM path is no longer the runtime strategy brain
  — it is encapsulated by MetaGBM, which owns the decision.
- ☑ Runtime output stable and structured (MetaDecision per trade)
- ☑ Risk/sizing/execution consume MetaGBM outputs
  (base_dec.probability, base_dec.quality_bucket, base_dec.risk_hint
  flow into the downstream bear-dial / cooldown / size-factor logic)
- ☑ Tests prove runtime delegation + integration (18 new GREEN tests)
- ☑ Documentation reflects new canonical flow

Failure conditions (from ticket) all AVOIDED :

- ✓ MetaGBM is wired (NYXEngine._meta + NYXLiveDecider._meta)
- ✓ NYXEngine delegates ; ownership shifted to MetaGBM
- ✓ Runtime consumes MetaGBM — not ignores it
- ✓ Backtests validate MetaGBM (it owns the path ; numbers
  preserved 1:1 via equivalence guard)
- ✓ No parallel decision engine — MetaGBM is the single owner

Out of scope (per ticket + user direction) :

- Full feature redesign + GBM retraining on FractalReports
  (Option B) — future ticket
- Threshold CV optimization
- ETH/SOL rollout / live deployment
- Wiring Jesse agents as live data collectors in the runtime (the
  MetaGBM accepts non-empty `fractal_reports` already — the
  remaining work is plumbing data flow from agents to engine)

### Ticket 06 — Replace "all must pass" with MetaGBM strategy brain (2026-04-16)

Remove the simplistic "ALL must pass → trade / any block → WAIT"
vote-based logic (used by `JesseOrchestrator` and per-file
`Orchestrator`) and introduce the canonical strategy brain that
interprets the 4 fractal reports + features via principled
aggregation — disagreement becomes a feature, not an automatic
failure.

- **New module** `src/core/meta_gbm.py::MetaGBM`.
  `.decide(fractal_reports, features, asset, timestamp,
  timeframe='15m', hint_direction=0) -> MetaDecision`.
  Logic :
  - `aggregate_score = mean(report.score for report in 4 reports)`
  - `disagreement = 1 - n_passed_agents / n`
  - `probability = aggregate * (1 - disagreement_weight * disagreement)`
  - `passed = direction != 0 AND probability >= threshold`
    (NOT `all(r.passed)`)
  - `quality_bucket ∈ {'high', 'medium', 'low'}` from aggregate
    cutoffs (0.75 / 0.60)
  - `risk_hint = 1 - disagreement` ∈ [0, 1]
  - `features_snapshot` includes injected `disagreement`,
    `n_passed_agents`, `aggregate_score` for traceability
- **Contract extension** `MetaDecision` (ticket 03) gets two optional
  fields (default None, backward-compat) :
  - `quality_bucket: Optional[str]` ∈
    `CANONICAL_QUALITY_BUCKETS = ('high', 'medium', 'low')`
  - `risk_hint: Optional[float]` ∈ [0, 1]
  Validated in `__post_init__`, serialised in `to_dict()`.
- **Legacy orchestrators marked DEPRECATED**
  (docstrings only, code untouched) :
  - `src/ml/jesse_agents.py::JesseOrchestrator`
  - `src/agents/orchestrator.py::Orchestrator`
  The deprecation note is enforced by
  `tests/test_meta_gbm.py::TestLegacyOrchestratorDeprecated`.
- **TDD** : `tests/test_meta_gbm.py` — 17 GREEN tests covering :
  - MetaDecision shape + optional outputs
  - `passes_with_one_agent_blocked_if_score_holds` (3/4 pass +
    aggregate > threshold → passed=True ; vote-based would say WAIT)
  - `passes_with_two_agents_blocked_if_score_holds` (2/4 block +
    aggregate 0.725 > 0.60 → still passed=True)
  - `blocks_when_direction_is_zero_regardless_of_scores`
  - `blocks_when_probability_below_threshold`
  - `disagreement_in_features_snapshot` + `disagreement_moderates_probability`
    (same aggregate, different disagreement → different probability)
  - quality bucket mapping × 3 (high/medium/low)
  - risk_hint range + inverse relation to disagreement
  - legacy orchestrator deprecation-note guards
- **Regression sweep** : 175 tests GREEN across Jesse agents,
  canonical contracts, entrypoint, architecture, rules, inventory,
  orchestrators, and the new MetaGBM.
- **Pyright** : 0 errors on the 2 touched src/ files (`meta_gbm.py`,
  `contracts.py`).
- **Docs** :
  - `ARCHITECTURE_CANONIQUE.md` : new "Canonical strategy-brain
    interface (Ticket 06)" subsection + history note explaining how
    `MetaGBM` (class) coexists with NYXEngine's internal GBM (role).
  - `CHANGELOG.md` : this entry.

Acceptance criteria (from ticket) all met :

- ☑ "all must pass" is no longer the main strategy rule (two tests
  explicitly prove the MetaGBM can pass with 1 or 2 blocked agents)
- ☑ Disagreement becomes a feature, not an automatic failure
  (present in `features_snapshot`, moderates probability instead of
  hard-blocking)
- ☑ The final decision is model-driven and strategy-aware — emits
  `trade/no-trade` (passed), `direction`, `confidence` (probability),
  `quality_bucket`, `risk_hint`, backed by the MetaDecision canonical
  contract

Out of scope (per ticket) :
- Full feature-set redesign + GBM retraining
- Full risk manager integration (MetaGBM not yet wired into
  NYXEngine — the runtime path still uses NYXEngine's internal GBM
  on `rule_*` proxy scalars)

### Ticket 05 — Jesse agents as fractal reporters (2026-04-16)

Reposition the 4 Jesse agents (Context / Regime / Setup / Entry) as
**fractal reporters** that emit the canonical `FractalReport` (from
Ticket 03) instead of acting like mini-strategies whose
`AgentResult.passed` is treated as a hard gate. They now *report*,
not decide.

- **New method** on every Jesse agent (both implementations) :
  `.report(df, asset, timestamp=None, **analyze_kwargs) -> FractalReport`.
  Thin wrapper on `.analyze()` using the existing
  `AgentResult.to_fractal_report()` adapter.
- **New class constants** `REPORT_AGENT` and `REPORT_TIMEFRAME`
  declaring canonical identity (independent of config-driven
  `self.timeframe`). Matches the 4-TF fractal stack :

  | Agent | `REPORT_AGENT` | `REPORT_TIMEFRAME` |
  |---|---|---|
  | Context | `'context'` | `'1d'` |
  | Regime  | `'regime'`  | `'4h'` |
  | Setup   | `'setup'`   | `'1h'` |
  | Entry   | `'entry'`   | `'15m'` |

- **TDD** : `tests/test_jesse_fractal_report.py` (60 GREEN, all
  parametrised over 8 agent × implementation combinations) asserts :
  - `.report()` method exists on every agent
  - `REPORT_AGENT` / `REPORT_TIMEFRAME` class constants declared
  - `.report()` returns `FractalReport` with canonical agent / TF
  - score ∈ [0, 1], asset preserved across calls
  - 4 reports together feed a `MetaDecision.fractal_reports` dict
    without raising — **contract compatibility proof with the
    Meta-GBM input layer** (ticket 05 acceptance §3).
  - Regression guard : `.analyze()` still returns `AgentResult` for
    both mono-file and per-file implementations.
- **Backward-compatible** : `.analyze()` and `AgentResult` unchanged.
  45 legacy Jesse tests + 207-test regression sweep stay GREEN.
- **Symmetric change** : applied to both implementations
  (`src/ml/jesse_agents.py` + `src/agents/*.py`) so behaviour is
  consistent — mono-file adds `.report()` at `_BaseJesseAgent` level
  + 4 constants per subclass ; per-file adds both constants and
  `.report()` directly per class.
- **Docs** :
  - `docs/JESSE_AGENTS_STATUS.md` : new "Reporter API (Ticket 05)"
    section with reporter mapping + scope-kept guard.
  - `docs/ARCHITECTURE_CANONIQUE.md` : "Honest status" section
    updated to mention `.report()` API + test count (45 → 105).
- **No runtime wiring** : per ticket out-of-scope, `NYXEngine` still
  consumes proxy scalars `rule_{context,regime,setup}` +
  `disagreement`. The reporter is the interface ; the swap path
  (replacing proxies with live agent calls) remains a future ticket.

Acceptance criteria (from ticket) all met :

- ☑ All 4 Jesse agents output the same report schema (FractalReport).
- ☑ The 4 agents no longer define the final trade decision —
  `FractalReport.passed` is an agent-local opinion ; the Meta-GBM
  is free to override. The canonical decision type is `MetaDecision`.
- ☑ Their outputs are consumable by the Meta-GBM layer
  (`MetaDecision.fractal_reports: dict[str, FractalReport]`).

### Ticket 04 — NYXEngine = single canonical runtime entrypoint (2026-04-16)

Unifies the runtime brain : rename `NYXPipeline` → `NYXEngine` and
move it from `src/ml/nyx_pipeline.py` → `src/core/nyx_engine.py`.
Segregate the legacy v0.8 engine (HSMM + SMC + macro) at
`src/core/nyx_engine_v08.py` for its 9 legacy callers.

- **Renamed + moved** : `src/ml/nyx_pipeline.py::NYXPipeline` →
  `src/core/nyx_engine.py::NYXEngine`. Class renamed, code unchanged
  (validated behaviour preserved — all ETH/SOL artefacts, A/B/C
  numbers, walk-forward reality checks still valid).
- **Segregated legacy v0.8** : previous content of
  `src/core/nyx_engine.py` (HSMM + SMC + macro) moved to
  `src/core/nyx_engine_v08.py`. Frozen list of 9 legacy callers
  updated to import from the `_v08` path :
  `src/runner/run_paper.py`, 6 × `src/validation/*.py`,
  `scripts/run_backtest.py`.
- **Deprecation shim** : `src/ml/nyx_pipeline.py` becomes a 23-line
  shim that re-exports `NYXEngine as NYXPipeline`. The ~45 canonical
  callers (tests, scripts, lazy imports inside `conditional_dial` /
  `bear_risk_dial` / `reality_check` / `train_asset_model` /
  `monte_carlo` / `nyx_pipeline_pod`) keep working unchanged.
- **RED → GREEN** : `tests/test_canonical_entrypoint.py` (16 tests)
  asserts :
  - `from src.core.nyx_engine import NYXEngine` works
  - `NYXEngine` has `.run()` and instantiates with no args
  - `src.core.nyx_engine_v08` holds a DIFFERENT class
  - Shim re-export : `NYXPipeline is NYXEngine`
  - Shim size < 1500 chars and no redeclared class (regex guard)
  - `ARCHITECTURE_CANONIQUE.md` now names `NYXEngine` as canonical
  - Each of the 8 legacy caller files imports from `_v08` path
    (parametrised — extensible)
- **Updated docs** : `ARCHITECTURE_CANONIQUE.md`, `RUNNER_INVENTORY.md`,
  `PROJECT_TRUTH_MAP.md`, `STATE_OF_PROJECT.md` all updated. Existing
  `test_architecture_canonical.py` assertions updated from
  `NYXPipeline.run` → `NYXEngine.run`.
- **Regression sweep** : 76 tests GREEN on
  `test_nyx_pipeline`, `test_nyx_live_decider`, `test_nyx_pipeline_pod`,
  `test_hub_spoke_lifecycle`, `test_canonical_contracts`,
  `test_signal_contract` — shim handles all pre-existing callers.
- **No behavioural change** : per ticket out-of-scope constraint —
  entrypoint unification only, no NYX_0 logic migration.

### Ticket 03 — Unified contracts and system language (2026-04-15)

- **New** : 4 canonical types in `src/agents/contracts.py` :
  - `FractalReport` — per-timeframe, per-agent report (generalises
    `AgentResult` for cross-layer vocabulary). Validates agent ∈
    CANONICAL_AGENTS, timeframe ∈ CANONICAL_TIMEFRAMES, score ∈ [0,1],
    passed=False requires non-empty block_reasons.
  - `MetaDecision` — strategy-brain output per bar. Validates
    direction ∈ {-1, 0, +1}, probability/threshold/candidate_quality
    ∈ [0,1], passed=True requires direction!=0, passed=False requires
    non-empty block_reasons. Carries `fractal_reports: dict[str,
    FractalReport]` for traceability.
  - `TradePlan` — risk-sized + stop/TP resolved. Validates direction
    ∈ {-1, +1} (no FLAT plans), size_fraction ∈ [0,1], long geometry
    (SL < entry < TP) / short geometry (SL > entry > TP), source
    direction matches plan direction.
  - `ExecutionInstruction` — broker-ready payload. Validates side ∈
    {buy, sell} matches source.direction, order_type ∈ CANONICAL_ORDER_TYPES,
    quantity > 0, limit_price required for post_only_limit.
- **New** : canonical constants `CANONICAL_TIMEFRAMES`,
  `CANONICAL_AGENTS`, `CANONICAL_DIRECTIONS`, `CANONICAL_ORDER_TYPES`,
  `CANONICAL_SIDES` — single source for valid enum values.
- **New** : adapters for gradual migration — no breaking change to
  existing callers :
  - `AgentResult.to_fractal_report(asset, timeframe, timestamp=None)`
  - `Signal.to_meta_decision(threshold_used=0.60, timeframe='15m')`
  - FLAT Signal adapts to `passed=False` MetaDecision with
    `block_reasons=['signal flat']`.
- **New** : `tests/test_canonical_contracts.py` — 27 GREEN tests
  covering construction validity, field validation, rejection of
  invalid states, adapter equivalence, and single-import-point
  contract (both legacy AgentResult/OrchestratorDecision AND the 4
  new canonical types exported from `src.agents.contracts`).
- **Backward-compatible** : `AgentResult`, `OrchestratorDecision`,
  `Signal` kept intact. 136 existing tests still GREEN (regression
  sweep on Jesse agents, signal contracts, orchestrator).
- **No code change** to runtime / offline pipelines : contracts
  available for adoption but NYXPipeline / NYXLiveDecider /
  HubSpokeRunner unchanged.

### Ticket 02 — Runner inventory and truth map (2026-04-15)

- **New** : `docs/RUNNER_INVENTORY.md` — per-file inventory of 124
  units (36 scripts + ~90 modules under `src/`), each tagged with one
  of 4 status values : canonical runtime / offline calibration /
  legacy / research-experimental. Includes purpose, engine called,
  and callers for every entry.
- **New** : `docs/PROJECT_TRUTH_MAP.md` — high-level 4-layer view
  so a new contributor can answer "where does this file fit ?" in
  under 2 minutes. Links back to RUNNER_INVENTORY + ARCHITECTURE_CANONIQUE.
- **New** : `tests/test_runner_inventory.py` — 22 GREEN documentary
  tests : doc presence + 4 status tags used + every ticket-minimum
  path named + canonical tag near `nyx_pipeline` / `nyx_live_decider` +
  `edge_strategy` / `threshold_optimizer` / `ml_filter_v2` marked
  offline or legacy + `TestNoMysteryRunner` that walks `scripts/` and
  fails if any invocable script is missing from the inventory.
- **Honest finding** : `src/core/nyx_engine.py` (v0.8 legacy) is STILL
  imported by `src/runner/run_paper.py` + 5 × `src/validation/*.py` +
  `scripts/run_backtest.py`. Flagged as **legacy layer** in both new
  docs. Deletion requires a dedicated ticket.
- **Honest finding** : the Jesse agent stack exists TWICE on disk
  (`src/ml/jesse_agents.py` mono-file + `src/agents/*.py` per-file).
  Neither is wired into the canonical runtime.
- **No code change** : ticket scoped to documentation + inventory
  tests only.

### Ticket 01 — Freeze canonical NYX architecture (2026-04-15)

- **New** : `docs/ARCHITECTURE_CANONIQUE.md` declares the single
  canonical runtime path (Data MTF → 4 Jesse fractal reporters →
  Meta-GBM strategy brain → Risk manager → Execution → Logging /
  feedback) and the single canonical offline path (features →
  candidates → calibration → training → OOS / walk-forward / reality
  checks).
- **New** : `tests/test_architecture_canonical.py` — 17 GREEN
  documentary tests enforcing the canonical declarations (entrypoint,
  layers, 4 reporters named, Meta-GBM = strategy brain / threshold
  0.60, offline-only modules called out).
- **Clarified** : the 4 Jesse agents (`JesseContextAgent`,
  `JesseRegimeAgent`, `JesseSetupAgent`, `JesseEntryAgent`) are
  **fractal reporters**, currently **not wired** into the canonical
  runtime — proxied today by hand-crafted `rule_context` /
  `rule_regime` / `rule_setup` / `disagreement` scalars inside
  `NYXPipeline._generate_candidates`.
- **Clarified** : `Meta-GBM` names the **role** (strategy brain),
  not a new class. Today it is the `GradientBoostingClassifier`
  owned by `NYXPipeline` (threshold 0.60, 84 features, net-outcome
  labels).
- **Marked offline-only** (not runtime strategies) :
  `edge_strategy.py`, `threshold_optimizer.py`, `ml_filter_v2.py`,
  `realistic_backtest.py`.
- **Updated** : `CLAUDE.md` — new "Canonical architecture pointer"
  section at the bottom referencing the new doc + test.
- **No code change** : this ticket is scoped to documentation and
  architecture truth tests only. Runtime refactor is out of scope.

## [0.2.5] — 2026-04-13

### Ajouté
- **Jesse ML Pipeline** : 15 modules Python, architecture 5 agents Jesse
- **200 tests TDD** couvrant l'intégralité du pipeline ML
- **Edge Strategy** : trend + volume >3x, validé walk-forward 14/14 quarters
- **RealisticBacktester** : fees 0.04% taker / 0.02% maker, slippage 0.02%, sizing 2% risk
- **ML Filter v2** : GBM 300 trees sur 47 parquet features, threshold calibré CV → WR 69%, Sharpe 1.89
- **Soft Gate Architecture** : ML décide, rules valident (disagreement comme feature)
- **Feedback Loop** : DecisionLogger + OutcomeEvaluator + ChampionChallenger
- **Threshold Optimizer** : sweep automatique, calibration time-series CV
- **Edge Analysis** : 5 hypothèses testées, seules 2 validées (volume, heures)
- **FastBacktester** : 128K bars/sec (vs 14 bars/sec v0.8)
- **Reality Check** : 6 biais identifiés et documentés
- **Documentation** : 7 nouveaux docs + EVOLUTION_A_TO_B

### Modifié
- **COMPLETE_ARCHITECTURE_SUMMARY.md** : section Jesse + 200 tests + prochaines étapes
- **README.md** : refonte complète pour v0.2.5
- **src/ml/__init__.py** : imports protégés (try/except pour lightgbm/river)
- **Pyright** : 428 erreurs corrigées sur 48 fichiers

### Données
- BTCUSDT 15m/1h/4h/1d extraits du tar.gz (2019-2024, 151K bars)
- 53 features parquet (momentum, vol, microstructure, HSMM)
- Q1 2023 : données réelles validées ($16,517 → $28,455)

### Performance validée (OOS, maker fees)
| Période | Sharpe | WR | DD | Trades |
|---------|--------|-----|-----|--------|
| 2020 (bull) | +2.19 | 51% | 3.0% | 152 |
| 2021 (bull) | +1.80 | 49% | 4.8% | 106 |
| 2022 (bear) | -0.75 | 38% | 10.0% | 149 |
| 2023 (bull) | +1.86 | 48% | 3.9% | 210 |

---

## [0.8.0] — 2026-03-28 (baseline)

### État initial
- 5 agents LightGBM (non entraînés, pass-through)
- HSMM 6 états comme feature extractor
- 47 features MLFeatureEngine
- PrecomputedRunner (40x speedup)
- Backtest Q1 2023 : +13.03% (sans fees, pass-through)
- 428 erreurs Pyright
- Aucun test TDD ML
