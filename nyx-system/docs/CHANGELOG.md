# Changelog

## [Unreleased] — branch `claude/run-pyright-system-qroCy`

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
