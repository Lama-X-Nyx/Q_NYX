# Changelog

## [Unreleased] — branch `claude/run-pyright-system-qroCy`

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
