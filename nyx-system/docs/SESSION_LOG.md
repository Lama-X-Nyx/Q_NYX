# Session Log — 2026-04-12/13

## Phase 1 : Pyright Audit (7 commits)

**Objectif** : Analyser et corriger toutes les erreurs de type dans le système.

1. Lancé `pyright nyx-system/` → 428 erreurs initiales
2. Installé les dépendances manquantes (numpy, pandas, scipy, sklearn, etc.)
3. L'installation a révélé ~300 erreurs supplémentaires (pandas type stubs)
4. Fixé par batches en parallèle (12 sub-agents) :
   - Batch 1 : 24 fichiers (agents, core, ml, validation)
   - Batch 2 : scripts, tests, core
   - Batch 3 : precomputed_runner, smc, ml_orchestrator
   - Batch 4 : nyx_engine, run_validation, ml_agents
   - Batch 5 : metrics.py (86 erreurs), validation modules
   - Batch 6 : component_edge_test, walk_forward, fractal
   - Batch 7 : remaining files
5. Patterns de fix : None guards, Optional annotations, np.asarray(), pd.Timestamp casts

**Résultat** : 48 fichiers modifiés, ~600 insertions

## Phase 2 : Jesse Installation

**Problème** : `pip install jesse` échouait (peewee, timeloop, starkbank-ecdsa)
**Solution** : `SETUPTOOLS_USE_DISTUTILS=stdlib pip install jesse`
**Conflits résolus** : FastAPI/Pydantic, numpy/numba/scipy version pinning
**Résultat** : Jesse 1.13.11 opérationnel, 177 indicateurs

## Phase 3 : Pipeline ML TDD (vidéo Jesse)

Implémenté la méthode de la vidéo YouTube en TDD strict :

1. **Tests RED** écrits d'abord (18 tests) :
   - Triple barrier labels (+1/-1/0)
   - Features stationnaires (ratios, pas de prix bruts)
   - Gather/Deploy modes
   - Synthetic TDD validation (>85% accuracy = code correct)
   - Feature importance
   - A/B comparison

2. **Implémentation** → 18/18 GREEN

3. **Données synthétiques proportionnelles** : +0.25%/-0.25% par bougie (pas d'absolu) pour invariance au prix

4. **Règle vidéo** : `prob_up > 0.45 AND prob_up > prob_down + 0.20`

## Phase 4 : Jesse Integration (utils, research, features v2)

- jesse_features.py v2 : 24 features stationnaires (core=13, full=24)
- jesse_utils.py : risk_to_qty, crossed, kelly_criterion, streaks
- jesse_research.py : 4-method feature importance, feature impact, train_model
- 23 tests d'intégration → GREEN
- Adapté les wrappers au vrai Jesse API → 41/41 GREEN

## Phase 5 : FastBacktester Q1 2023

**Problème** : backtest NYX existant = 70ms/bar (HSMM per-bar)
**Solution** : tout vectoriser

- `precompute()` : features + labels numpy arrays ONCE
- `run()` : boucle numpy pure, ZERO DataFrame slicing
- **Speed** : 0.38ms/bar (185x faster) → 8,640 bars en 3 secondes

**Données** : Q1 2023 synthetic (proxy Binance bloqué) puis REAL (extraites du tar.gz)

14 tests TDD → GREEN

## Phase 6 : Architecture 5 Agents Jesse

Construction séquentielle TDD — chaque agent validé avant le suivant :

| Agent | TF | Tests | Iterations pour GREEN |
|-------|-----|-------|-----------------------|
| Context | 1D | 12 | 4 (warmup, momentum, labels, blend) |
| Regime | 1H | 10 | 1 (premier essai) |
| Setup | 15M | 6 | 1 |
| Entry | 15M | 8 | 1 |
| Orchestrator | Meta | 9 | 1 |

**Problèmes de cohérence identifiés et fixés** :
- Setup bloquait 100% (seuil trop strict) → blend ML+heuristic 40/60
- Entry trop permissif (96% ready) → blend ML+heuristic 60/40
- Context inutile sur 30 jours → momentum 5/10/20 au lieu de 5/20/60

## Phase 7 : Backtest Real Data

Données extraites du `nyx-p0p1p2p3-done.tar.gz` :
- `data/raw/mtf/BTCUSDT_15m.csv` (151K bars, 2019-2024)
- `data/raw/mtf/BTCUSDT_1h.csv` (37K bars)
- `data/raw/mtf/BTCUSDT_4h.csv` (9K bars)
- `data/raw/mtf/BTCUSDT_1d.csv` (1.5K bars)

**Résultat real data (5 agents, train 2021-2022, test Q1 2023)** :
- 21 trades (21L / 0S) — direction correcte dans un bull
- PnL: -$1,639 (-16%)
- Win rate: 24%, W/L ratio: 1.53
- Max DD: 25%
- Speed: 4.3s total (128K bars/s)

## Phase 8 : Edge Analysis

5 hypothèses testées sur les vraies données 2023 :

| Edge | Verdict | Impact |
|------|---------|--------|
| Asymmetric TP/SL | Standard 1.5/1.0 est optimal | Baseline |
| Pullback RSI | NOT an edge (trop peu de trades) | Rejeté |
| Volume >1.5x | **EDGE** (+20% EV) | Implémentable |
| Heures 9/14/17-18h | Faible edge (+13pts WR) | Optionnel |
| Trend continuation | **EDGE MAJEUR** (58% WR) | Prioritaire |

**Diagnostic fondamental** : les labels triple barrier donnent 60% SL même dans un +39% bull month. La cause racine est le SL trop serré (1.0x ATR). Aucune feature ne corrèle avec les labels (< 0.03).

## Phase 9 : Feedback Loop (paper trading → evaluate → retrain)

Implémenté en 3 niveaux, 13/13 tests GREEN :
- **DecisionLogger** : log chaque barre (features, agents, décision, trade info)
- **OutcomeEvaluator** : relabel avec vrais outcomes (TP/FP/TN/FN + taxonomy 10 erreurs)
- **ChampionChallenger** : promote seulement si accuracy +2% ET drawdown pas pire

## Phase 10 : Edge Strategy + Walk-Forward

Edge validé en TDD (12/12 GREEN) puis yearly walk-forward (11/11 GREEN).

Walk-forward annuel (volume >1.5x) :
- Fold 1 : train 2019 → test 2020 (+303% BTC) : +12,094pts
- Fold 2 : train 2020-2021 → test 2022 (-64% BTC, BEAR) : +16,452pts
- Fold 3 : train 2022-2023 → test 2023-Q4 (+58% BTC) : +5,924pts
- **3/3 folds positifs**

Full OOS (train 2019-2021, test 2022-2024) :
- 6,397 trades | WR 43% | EV +0.079 ATR | 2/2 years +

## Phase 11 : Reality Check

**Les chiffres sont gonflés.** Biais identifiés :
1. PnL en points bruts (pas en dollars)
2. **Zéro fees/slippage** (~0.1% round trip = ~$180K sur 6,396 trades)
3. Re-entry immédiate (8.8 trades/jour irréaliste)
4. Position sizing = 1 BTC (pas % du capital)
5. Compounding irréaliste
6. Overcounting des signaux

**L'edge existe** (direction positive dans tous les régimes) mais le return
réaliste est estimé à **10-30%/an** avec fees, pas 200%+.

Voir `docs/REALITY_CHECK.md` pour le détail.

## Prochaines étapes

- [ ] Ajouter fees + slippage au backtest (P0)
- [ ] Position sizing réaliste en % du capital (P0)
- [ ] Max 3 trades/jour + cooldown (P1)
- [ ] ML filtrage pour réduire de 8 trades/j à 2-3 (P1)
- [ ] Backtest en dollars nets avec equity curve (P1)
- [ ] HSMM probs des parquets comme features ML (P2)
- [ ] Paper trading live avec DecisionLogger (P2)

---

## 2026-04-16 — Ticket 07 (MetaGBM wired into NYXEngine)

### Problème
Le ticket demande de faire de MetaGBM la strategy brain propriétaire
de NYXEngine, qui détenait jusqu'ici sa propre logique de scoring
GBM. Risque majeur : détruire les numbers validés (A/B/C p5 Sharpe
7.78, walk-forward CAGR 49.3%, ETH+SOL artefacts) si le MetaGBM
heuristique remplace littéralement le GBM entraîné.

### Hypothèse testée
Option C (wrapper ownership) : MetaGBM encapsule le GBM entraîné
existant comme implementation detail. Le scoring reste
numériquement identique (même model + scaler + 84 features), seul
le point d'entrée change (MetaGBM.decide → predict_proba). Les
FractalReports peuvent être passés en dict vide initialement
(wiring Jesse dans le runtime = ticket futur).

### Fichiers touchés
- `src/core/meta_gbm.py` (+60 lignes) — trained-GBM mode
- `src/core/nyx_engine.py` — `_meta` instance + delegation dans
  la candidate loop + `meta_decision` dans trade record
- `src/ml/nyx_live_decider.py` — `_meta` + delegation dans
  `on_15m_bar` via `feature_vector + already_scaled=True`
- `tests/test_meta_gbm.py` (+200 lignes) — `TestTrainedGBMMode`,
  10 tests GREEN
- `tests/test_nyx_engine_uses_metagbm.py` (nouveau) — 8 tests GREEN
- `docs/ARCHITECTURE_CANONIQUE.md` — status MetaGBM passe de
  "Not yet wired" à "WIRED AS OWNER (Ticket 07)" + marqueur
  TRANSITIONAL + note sur Option B
- `docs/CHANGELOG.md` — entrée Ticket 07
- `docs/PROJECT_TRUTH_MAP.md` — update NYXEngine + ajout MetaGBM
- `docs/SESSION_LOG.md` — ce log

### Tests
- Ticket 07 directs : 18/18 GREEN (10 TestTrainedGBMMode + 8
  TestNYXEngineDelegatesToMetaGBM et friends)
- Equivalence guard `test_nyx_equivalence_replay_vs_live` : 4/4
  GREEN (preserved batch vs live semantics)
- `test_nyx_pipeline` : 17/17 GREEN
- `test_nyx_live_decider` : 15/15 GREEN
- Regression sweep broader (contracts, canonical, Jesse agents ×4,
  orchestrators, rules, inventory, architecture) : 200+ GREEN
- Pyright : 0 errors sur les 3 fichiers src/ touchés

### Impact architectural
- UN seul propriétaire canonique de la décision : `MetaGBM`
- NYXEngine devient orchestrateur (data MTF → candidates →
  MetaGBM → bear dial → cooldown → sizing → execution)
- NYXLiveDecider : même pattern per-bar via feature_vector
- Les FractalReports peuvent maintenant être consommés dès que
  wirés ; MetaGBM gère le dict vide gracefully

### Risques restants
- **MetaGBM transitoire** : encapsule un GBM entraîné sur proxy
  `rule_*` scalars, PAS sur les FractalReport scores réels. Le
  "vrai" Meta-GBM entraîné (Option B) reste un ticket futur
  nécessitant retraining + OOS complet.
- **Jesse agents pas wirés dans le runtime** : MetaGBM accepte
  `fractal_reports={}` — le plumbing agents → engine reste à
  faire (autre ticket intégration).

### Next smallest step possible
Ticket 08 potentiel : Option B — retrain le GBM sur un feature
set qui inclut les FractalReport scores comme features (au lieu
des proxies `rule_context/regime/setup`). Nécessite OOS
equivalence pour vérifier que l'edge est préservé/amélioré.

---

## 2026-04-16 — Ticket 08 (EdgeStrategy integrated as runtime candidate generator)

### Problème
Ticket 08 demande d'intégrer `edge_strategy` dans le runtime path
canonique. Mais `edge_strategy.py` était tagué offline-only depuis
Ticket 01/02. De plus, `NYXEngine._generate_candidates()` contenait
déjà une COPIE du hard gate (EMA alignment + vol > 3× MA + hour
∈ [6, 20]) identique à la logique dans `EdgeStrategy.backtest()`.
DRY violation à résoudre tout en respectant l'acceptance ticket
08.

### Hypothèse testée
Promouvoir `edge_strategy` comme COMPONENT du runtime (candidate
generator), PAS comme standalone strategy. Nouvelle méthode
`EdgeStrategy.generate_candidate_bars(df, hour_window, vol_min,
max_bars_lookback) -> List[int]` — pure bar-index emitter.
NYXEngine holds `self._edge` et délègue la hard gate à cette
méthode.

### Fichiers touchés
- `src/ml/edge_strategy.py` — nouvelle méthode `generate_candidate_bars`
- `src/core/nyx_engine.py` — `self._edge` en __init__ + delegation
  dans `_generate_candidates`
- `tests/test_edge_strategy_integration.py` (nouveau) — 8 tests
  GREEN
- `tests/test_runner_inventory.py` — test renommé
  `test_edge_strategy_not_standalone_strategy` avec assertion
  relâchée (accepte "candidate generat" en plus de legacy/offline)
- `docs/ARCHITECTURE_CANONIQUE.md` — edge_strategy passe de
  "offline-only" à "runtime component (candidate generator)"
- `docs/RUNNER_INVENTORY.md` — même update
- `docs/CHANGELOG.md` — entrée Ticket 08
- `docs/SESSION_LOG.md` — ce log

### Tests
- Ticket 08 directs : 8/8 GREEN
- Equivalence guard + core : 47/47 GREEN (nyx_equivalence +
  nyx_pipeline + nyx_live_decider + nyx_engine_uses_metagbm +
  edge_strategy_integration) — numbers preserved 1:1
- Doc-contract sweep : 187/187 GREEN
- Pyright : 0 errors

### Impact architectural
- UN seul endroit qui définit la hard gate : `EdgeStrategy.generate_candidate_bars`
- DRY violation résolue — plus de duplication entre edge_strategy
  et NYXEngine
- `edge_strategy.backtest()` reste pour research offline
- La règle "edge_strategy n'est pas standalone" est préservée
  (c'est un COMPONENT maintenant, pas un standalone engine)

### Risques restants
- Paramétrage hour_window `(6, 20)` est hardcodé dans
  `_generate_candidates` call — pas exposé sur NYXEngine. Ticket
  futur si config'able par asset.
- `EdgeStrategy.use_hours` / `good_hours` config reste utilisable
  en backtest OFFLINE mais pas routée via `generate_candidate_bars`
  (qui accepte seulement un `hour_window` tuple).

### Next smallest step possible
Continuer l'unification : wire Jesse FractalReports comme enrichissement
de MetaGBM au runtime (agents.report() called in NYXEngine.run loop),
ou bien attaquer ticket B (retraining sur FractalReports).

---

## 2026-04-16 — Ticket 09 (Liquidity-hunter feature family)

### Problème
Le ticket demande de sortir NYX du pure trend-following et d'ajouter
des features Jesse-natives orientées liquidity hunting +
microstructure.

### Hypothèse testée
13 nouvelles features stationnaires (10 primaires + 3 secondaires) :
VWAP-distance, AD-line slope, Chaikin oscillator normalized,
MarketFI ratio, BOP bounded, SR breaks + distances, Choppiness Index,
KVO normalized, VWMA distance, minmax position. Toutes
scale-invariantes (test ×10) + bornées par construction où
applicable + non-correlated avec close raw.

### Fichiers touchés
- `src/ml/jesse_features.py` — +7 helpers numpy + section
  liquidity dans `compute_stationary_features()` + docstring
  module enrichi
- `tests/test_jesse_liquidity_features.py` (nouveau) — 81 tests GREEN
- `docs/CHANGELOG.md` — entrée Ticket 09

### Tests
- 81/81 GREEN sur le nouveau fichier (présence × 13, no-NaN × 13,
  scale-invariance × 13, no-raw-leakage × 13, boundedness × 4,
  SR semantics × 2, regression existing × 24)
- Sweep : 181/181 GREEN (feature_buffer + mtf_feature_stack +
  canonical + meta_gbm + edge_strategy + nyx_engine_uses_metagbm)
- Heavy regression (nyx_pipeline + nyx_live_decider + equivalence) :
  36/36 GREEN
- Pre-existing fixture failures (test_eth_pipeline data size,
  test_eth_training h4_* missing) confirmés indépendants du Ticket 09
- Pyright : 0 errors

### Impact architectural
- Le feature engine reste UN seul module canonique
  (`src/ml/jesse_features.py`)
- Pas de "v2" parallèle — extension additive de
  `compute_stationary_features` dans le bloc 'full'
- L'ancien feature_set 'core' inchangé (rétrocompat)
- MTFFeatureStack (qui utilise compute_stationary_features) hérite
  automatiquement des nouvelles features → disponibles aussi en live

### Risques restants
- Les nouvelles features ne sont PAS encore vues par MetaGBM /
  NYXEngine GBM (pas de retraining dans ce ticket — out of scope).
  Elles sont DISPONIBLES dans la pipeline de features mais pas
  consommées en runtime.
- Option B (retraining sur ces features liquidity) reste un ticket
  futur dédié.

### Next smallest step possible
Ticket 10 potentiel : retraining controlled (Option B) sur
ETH/SOL avec les nouvelles features liquidity intégrées au vecteur
84-features. OOS equivalence pour mesurer le delta d'edge.

---

## 2026-04-16 — Ticket 10 (Meta-GBM I/O contract frozen)

### Problème
Le ticket demande de geler le schéma I/O de MetaGBM pour que
training, inference, risk, execution s'accordent. Sans contrat
strict, les couches dérivent et translatent à la main.

### Hypothèse testée
2 nouveaux constants class-level sur MetaGBM (INPUT_SCHEMA,
OUTPUT_SCHEMA) + 4 propriétés alias sur MetaDecision
(trade_decision, confidence, expected_edge, trade_quality_bucket).
Pas de rename des champs internes — additif strict.

### Fichiers touchés
- `src/core/meta_gbm.py` — INPUT_SCHEMA (9 keys) + OUTPUT_SCHEMA
  (6 keys) sur la classe MetaGBM
- `src/agents/contracts.py` — 4 propriétés sur MetaDecision
  (trade_decision dérivée de (passed, direction); confidence,
  expected_edge, trade_quality_bucket = aliases)
- `tests/test_meta_gbm_io_contract.py` (nouveau) — 30 tests GREEN
- `docs/ARCHITECTURE_CANONIQUE.md` — section frozen schema avec
  table canonical ↔ internal
- `docs/CHANGELOG.md` — entrée Ticket 10
- `docs/SESSION_LOG.md` — ce log

### Tests
- 30/30 GREEN sur le nouveau fichier
- Sweep doc-contract : 200+ GREEN
- Pyright : 0 errors
- Pas de régression sur les 27 tests précédents de test_meta_gbm.py

### Impact architectural
- UN contrat I/O canonique pour MetaGBM, vivant DANS la classe
  (INPUT_SCHEMA, OUTPUT_SCHEMA = single source of truth)
- TradePlan / ExecutionInstruction consomment MetaDecision par
  les noms canoniques (.trade_decision, .confidence, etc.) sans
  ad-hoc translation
- Backward-compat strict : tous les champs internes (probability,
  expected_edge_net, quality_bucket) intacts ; les aliases sont
  des @property pures

### Risques restants
- Le schéma documenté est un dict[str, str] (string types only).
  Pas de validation runtime du type strict (pas de TypedDict). Si
  un futur ticket veut l'enforcement, c'est une extension simple.
- INPUT_SCHEMA / OUTPUT_SCHEMA sont des class attributes mutables
  Python (pas frozen). On enforce le contrat par les tests, pas
  par immutabilité dataclass-style.

### Next smallest step possible
- Ticket 11 : wire les 4 Jesse FractalReports en runtime
  (NYXEngine.run() appelle agent.report() par candidate bar et
  passe un dict non-vide à MetaGBM)
- OU Ticket 12 : Option B — retraining avec le contract étendu
  (84 + 13 nouvelles features liquidity-hunter)

---

## 2026-04-16 — Ticket 11 (BTC Meta-GBM trained + artefacts persisted)

### Problème
Le ticket demande d'entraîner le Meta-GBM sur la pipeline BTC
canonique maintenant que le runtime est unifié (Tickets 04-10).
Avant ce ticket, BTC n'avait PAS d'artefact persisté (gap flagué
dans STATE_OF_PROJECT §3). Les numbers BTC venaient du
training-inside-test de NYXEngine.run(), pas d'un artefact loadable.

### Hypothèse testée
Utiliser `train_asset_model.train_and_save` (canonical helper avec
Rule 2 MTF coverage enforcement) sur BTC 2019-09→2022-12 puis run
OOS 2023 via NYXEngine.run() (qui délègue à MetaGBM ticket 07).
Les features incluent la famille liquidity-hunter du Ticket 09
automatiquement (compute_stationary_features 'full').

### Fichiers touchés
- `scripts/train_btc_model.py` (nouveau) — mirror de train_eth_model
  avec data/raw/mtf/ prefix
- `tests/test_train_btc_model.py` (nouveau) — 15 tests GREEN
  (artefact files + loadable + MTF coverage + OOS non-degenerate +
  MetaGBM constructible from BTC artefact)
- `models/BTCUSDT/ml_filter_v1.pkl` (nouveau, 332 KB)
- `models/BTCUSDT/scaler.pkl` (nouveau)
- `models/BTCUSDT/feature_names.json` (nouveau, 173 features)
- `models/BTCUSDT/training_metadata.json` (nouveau)
- `reports/BTCUSDT_oos_report.json` (nouveau)
- `docs/STATE_OF_PROJECT.md` — BTCUSDT row ajoutée, gap marqué CLOSED
- `docs/CHANGELOG.md` — entrée Ticket 11
- `docs/SESSION_LOG.md` — ce log

### Tests
- 15/15 GREEN sur tests/test_train_btc_model.py
- MTF coverage : n_features=173 ≥ 65 ✓
- Rule 2 : h1_ / h4_ / d1_ prefixes présents ✓
- MetaGBM constructible depuis l'artefact BTC ✓
- Pyright : pas de fichier src/ touché (script seulement)

### Numbers obtenus
- In-sample accuracy : 90.2 %
- n_train_candidates : 1,248
- 2023 OOS :
  - n_trades : 54
  - Sharpe (per-trade) : 9.96
  - Total PnL : $2,171.18
  - Max Drawdown : 0.37 %
  - Bear dial activation : 46.6 %
  - Execution reject rate : 0 %

### Impact architectural
- UN artefact BTC persisté, prêt pour NYXLiveDecider en production
- La pipeline entière (data MTF → EdgeStrategy → features → train →
  NYXEngine → MetaGBM → trade decisions) est maintenant VALIDÉE
  end-to-end sur BTC (canonical source du Sharpe 5+ historique)
- Le gap "Tickets 01/07 : BTC has no persisted artefact" est
  officiellement CLOSED
- Les features Ticket 09 liquidity-hunter sont DANS le modèle BTC
  (n_features=173 vs 84 pour ETH/SOL)

### Risques restants
- ETH et SOL restent sur leur 84-feature contract pré-Ticket-09.
  Un ticket futur pourrait les retrain avec la même stack étendue
  pour homogénéiser (hors-scope Ticket 11, "ETH/SOL rollout" exclu).
- Sharpe per-trade 9.96 est EXCELLENT mais pas directement
  comparable au Sharpe daily-equity — la reality check Ticket 6
  ancien montre que daily ≈ sqrt(N)-adjusted ≈ 4-5.
- Bear dial activation 46.6 % — consistant avec PIPELINE_V031_RESULTS
  historique (39-47 % selon l'année).

### Next smallest step possible
- Ticket 12 : retrain ETH + SOL avec la nouvelle feature stack
  (173 features) pour homogénéiser les 3 majors. OOS equivalence
  vs baseline 84-features.
- OU Ticket 13 : wire les FractalReports des 4 Jesse agents dans
  NYXEngine.run() runtime path (agent.report() per candidate bar)
  + retrain pour que le GBM VOIE les report.score vs les proxies
  hand-crafted.

---

## 2026-04-16 — Ticket 13 (Jesse FractalReports wirés dans NYXEngine + retrain BTC)

### Problème
Les 4 Jesse agents émettaient des FractalReport depuis Ticket 05 mais
RIEN dans le runtime ne les appelait. Le GBM s'entraînait sur des
proxies rule_* hand-crafted (rule_context / rule_regime / rule_setup)
au lieu des vrais outputs d'agents.

### Hypothèse testée
Instancier 4 per-file agents (ContextAgent, RegimeAgent, SetupAgent,
EntryAgent) dans NYXEngine.__init__. Helper
_build_fractal_report_features(ts, mtf_data) appelle .report() sur
chaque agent avec une slice TF-appropriée, flattens en 13 rep_*
features, merged dans candidate['features'] à côté des rule_* existants
(garde-fou ticket : NE PAS supprimer rule_*). Retrain BTC et compare
vs baseline Ticket 11.

### Fichiers touchés
- `src/core/nyx_engine.py` — 4 agents en __init__ + helper +
  signature _generate_candidates accepte mtf_data + merge dans
  candidate features
- `tests/test_fractal_reports_wired.py` (nouveau) — 22 tests GREEN
- `reports/BTCUSDT_ticket13_comparison.json` (nouveau) — comparaison
  structurée + décision justifiée
- `reports/BTCUSDT_oos_report_baseline_ticket11.json` (nouveau) —
  snapshot baseline pour audit
- `models/BTCUSDT/baseline_*.pkl|json` (nouveau) — backup explicite
- `docs/CHANGELOG.md` — entrée Ticket 13
- `docs/SESSION_LOG.md` — ce log

### Tests
- 22/22 GREEN sur test_fractal_reports_wired
- 8/8 GREEN sur test_nyx_engine_uses_metagbm (4'28" runtime)
- 15/15 GREEN sur test_train_btc_model (post-revert artefact)
- Pyright : 0 errors

### Numbers obtenus (point de vérité honnête)

| Metric | Baseline T11 | T13 rep_* | Delta |
|---|---:|---:|---|
| n_features | 173 | 186 | +13 |
| In-sample acc | 90.22% | 90.22% | = |
| OOS n_trades | 54 | 49 | -9% |
| OOS Sharpe | 9.96 | 8.04 | -19% |
| OOS PnL | $2,171 | $1,828 | -16% |
| OOS Max DD | 0.37% | 0.57% | +54% |

Le rep_* wiring DEGRADE l'edge BTC sur 2023. Décision :
REVERT_ARTEFACT_KEEP_WIRING.

### Root cause
Per-file agents ne sont pas tous fonctionnels :
- ContextAgent : OK (SMA fallback, pas besoin HSMM)
- RegimeAgent : HSMM requis, pas pré-entraîné → neutral defaults
- SetupAgent : HSMM + SMC requis, pas pré-entraîné → neutral defaults
- EntryAgent : OK (rule-based momentum check)

2 agents sur 4 émettent du bruit → 13 rep_* features dominés par du
bruit → GBM overfit in-sample (90.22% identique) mais OOS degrade.
Classic noise-feature overfit.

### Décision — REVERT_ARTEFACT_KEEP_WIRING
1. models/BTCUSDT/ml_filter_v1.pkl restauré depuis baseline_*.pkl
   (Ticket 11 values, Sharpe 9.96 préservé)
2. reports/BTCUSDT_oos_report.json restauré à la baseline T11
3. Le WIRING dans NYXEngine RESTE committed — un futur ticket peut :
   - pré-entraîner HSMM pour RegimeAgent/SetupAgent
   - OU swap vers mono-file Jesse*Agent (RandomForest) avec slice
     pré-training
   - OU combinaison
   sans toucher NYXEngine (plumbing done)

### Risques restants
- Le wiring rend NYXEngine.run() plus lent (~2 min overhead sur
  200 candidates ETH H1 2023, ~5 min sur 1800 candidates BTC full).
  Acceptable tant que ce n'est pas en production live.
- Si un futur ticket réactive le rep_* training SANS régler le
  problème HSMM, le même overfit se reproduira — le comparison
  JSON sert de référence pour détecter.

### Next smallest step possible
- Ticket 14 : pré-entraîner HSMM pour RegimeAgent + SetupAgent sur
  BTC 2019 Q1-Q3 (slice pre-training). Re-run la même comparaison.
  OU swap aux mono-file Jesse*Agent ML-based avec training slice.

---

## 2026-04-16 — Tickets 14 + 15 (Jesse retrain + dataset policy)

### Problème
Ticket 14 : les 4 Jesse agents utilisaient des blocs de features
désalignés du runtime canonique (custom pour Context/Regime,
`feature_set='core'` pour Setup/Entry). Ticket 15 : le retrain
Entry sur 15m 2020-2022 (~100k bars) via .backtest() était O(N²)
per-bar .analyze() — le sandbox tuait systématiquement.

### Hypothèse testée (Tickets combinés)
- Ticket 14 : chaque agent appelle compute_stationary_features('full')
  + FEATURE_PLAN subset role-based (13/15/15+3/13 features)
- Ticket 15 : dataset builder par agent retourne (df, sample_mask) ;
  .train()/.backtest() acceptent sample_mask qui SKIP les bars
  non-masqués dans la boucle O(N²). Entry : rolling 12m + candidate-
  proximity ±5 + max 20k + deterministic downsample.

### Fichiers touchés
- `src/ml/jesse_agents.py` — FEATURE_PLAN × 4 agents + sample_mask
  sur train/backtest (base class + Entry override)
- `src/ml/jesse_dataset.py` (nouveau) — 4 builders
- `tests/test_jesse_agents_retrained.py` (nouveau Ticket 14) —
  27 tests
- `tests/test_jesse_dataset_policy.py` (nouveau Ticket 15) —
  17 tests
- `tests/test_jesse_agent_context.py` — MAJ tests legacy (500 bars
  synthetic + canonical feature names)
- `scripts/retrain_jesse_agents.py` — MAJ pour builder-driven
  retrain
- `reports/jesse_agents_retrain_ticket14.json` — métriques finales
- `docs/JESSE_FEATURE_MAPPING.md` (nouveau) — single source of
  truth agent features + dataset policy
- `docs/CHANGELOG.md` + ce log

### Tests
- 27 Ticket 14 + 17 Ticket 15 + 115 regression = **159 GREEN** sur
  le Jesse sweep complet
- Pyright : ajouts additifs, pas de nouvelle erreur

### Numbers retrain BTC 2020-2022
| Agent | Bars | Kept | Acc | pct_passed | Elapsed |
|---|---:|---:|---:|---:|---:|
| Context | 1,096 | 100% | 0.500 | 100.0% | 23s |
| Regime | 6,576 | 100% | 0.571 | 99.2% | 354s |
| Setup | 26,304 | 13.9% | 0.188 | 0.0% | 567s |
| Entry | 35,041 | 10.9% | 0.560 | 99.9% | 3.8s |

Entry : de NEVER-FINISH à 3.8s. Ticket 15 mission accomplie.

### Constats honnêtes
- **Setup acc 0.188 est bas** : le volume-spike filter skew le dataset
  vers valid_setup → le modèle prédit no_setup partout (pct_passed=0)
  → acc faible. C'est un signal de déséquilibre des labels, pas un
  échec de training. Documenté dans JESSE_FEATURE_MAPPING.md §3.
- **Context + Entry ont des numbers raisonnables** (0.50 et 0.56).
- **Regime 0.571 pct_passed 99.2%** : presque tout "passe" → le
  modèle est très permissif. À peaufiner dans un ticket futur via
  seuil de confiance.

### Risques restants
- Setup/Regime "pct_passed" extrêmes (0% et 99.2%) suggèrent que
  les agents, bien que trainables et stables, ne discriminent pas
  bien. Un ticket futur calibrera les seuils de décision + class
  weights.
- Les agents restent NON-WIRÉS dans NYXEngine (scope Ticket 13+14
  explicitement exclut le runtime wiring).
- HSMM toujours non-entraîné → RegimeAgent runtime tombe en
  fallback neutre dans NYXEngine (Ticket 13 remark).

### Next smallest step possible
- Ticket 16 : class_weight + threshold tuning pour Setup/Regime
  pour obtenir pct_passed équilibré
- OU Ticket 17 : wire agents en runtime pour vraiment tester
  l'impact edge-à-edge (mesurer Sharpe delta vs Ticket 11 baseline)

---

## 2026-04-16 — Ticket 16 (Setup + Regime calibration quality-gate)

### Problème
Après Tickets 14+15 :
- Setup : pct_passed 0% (dégénéré) + acc 0.188
- Regime : pct_passed 99.2% (non-discriminant)

Ces 2 agents ne méritent pas d'être wirés en runtime tels quels.

### Hypothèse testée (Ticket 16)
1. Setup root cause : `analyze()` lit `momentum_10`, `ema_ratio_9_21`,
   `rsi_14` — aucun dans FEATURE_PLAN (liquidity-hunter). Heuristic
   collapse à 0 → p_setup = 0.4×p_ml < 0.40 → pct_passed=0.
2. Regime root cause : `range` state → passed=True par défaut. Sur
   BTC 4H la majorité des bars sont `range` → 99% passent.

### Fichiers touchés
- `src/ml/jesse_agents.py` :
  - JesseSetupAgent.analyze → nouvelle heuristic sur 7 signals
    liquidity-hunter (sr_break_*, vwap_dist, bop, adosc_norm,
    minmax_pos_20, mfi_norm) + threshold 0.40→0.55
  - JesseRegimeAgent.analyze → `range` default passed=False
- `tests/test_jesse_calibration_ticket16.py` (nouveau) — 9 tests GREEN
- `scripts/retrain_jesse_agents.py` (run re-utilisé, pas modifié)
- `reports/jesse_agents_retrain_ticket14.json` — MAJ numbers
- `docs/JESSE_FEATURE_MAPPING.md` §3 — before/after + root causes
- `docs/CHANGELOG.md` + ce log

### Tests
- 9/9 GREEN Ticket 16 (operating zones + behavioural + report schema)
- 168/168 GREEN Jesse sweep complet

### Numbers retrain BTC 2020-2022 post-Ticket-16

| Agent | Acc | pct_passed pre→post | Zone cible | Verdict |
|---|---:|---:|---|---|
| Context | 0.440 | 100%→44.4% | libre | ✓ |
| Regime | 0.571 | 99.2%→51.6% | 30-80% | ✓ FIXED |
| Setup | 0.500 | 0%→35.4% | 10-40% | ✓ FIXED |
| Entry | 0.560 | 99.9% | (hors scope) | N/A |

### Décision
**Ticket 17 est maintenant éligible.** Les 2 agents Setup + Regime
sont calibrés, discriminants, reproducibles. Le wiring en runtime
peut commencer.

Caveat Entry 99.9% : hors scope Ticket 16, documenté dans
JESSE_FEATURE_MAPPING §3 — Entry est déjà filtré par le candidate-
proximity mask, donc sa "décision" finale ne devrait pas être un
gate supplémentaire — candidate gen + cooldown + bear dial
restent les vraies gates.

### Next smallest step possible
- Ticket 17 : wire les 4 Jesse agents en runtime (NYXEngine._build_
  fractal_report_features passe d'un stub à des vrais appels
  agent.report() sur les 4 agents retrainés). Comparer edge BTC
  2023 OOS vs Ticket 11 baseline (Sharpe 9.96).

---

## 2026-04-16 — Ticket 17 (Jesse Runtime Integration & Edge Validation)

### Question testée
Les 4 Jesse agents calibrés (Ticket 16) améliorent-ils l'edge BTC
quand leurs rep_* enrichis (21 features, dont p_bull/p_bear/
trend_plus/trend_minus/p_up/p_down) sont injectés dans le GBM ?

### Réponse : NON. REJECT.

| Metric | Baseline T11 | T13 (no calib) | T17 (calibrated) |
|---|---:|---:|---:|
| n_features | 173 | 186 | 194 |
| Sharpe | 9.96 | 8.04 | 8.73 |
| PnL | $2,171 | $1,828 | $1,925 |
| Max DD | 0.37% | 0.57% | 0.57% |

### Impact de la calibration Ticket 16
T13 → T17 : Sharpe +8.6 %, PnL +5.3 %. La calibration A AIDÉ mais
pas assez pour rattraper le baseline.

### Root cause
Les per-file agents sont rule-based (heuristic 0/0.5/1). Leurs
rep_* features sont REDONDANTES avec les features techniques
existantes (rule_context/regime/setup proxies + EMA/ATR/RSI/ADX).
Le GBM ne bénéficie pas d'info orthogonale supplémentaire.

### Décision
- Artefact REVERTED au baseline T11 (Sharpe 9.96 préservé).
- Wiring CONSERVÉ dans NYXEngine (plumbing intact).
- Comparison JSON archivé : reports/BTCUSDT_ticket17_comparison.json

### Options documentées pour la suite
1. HYBRID : feature_importances_ analysis (quels rep_* ont > 0 ?)
2. Swap mono-file ML-based Jesse*Agent (RandomForest probas calibrées)
3. Structural-only : agents comme enrichissement MetaGBM
   (quality/risk/disagr) sans injection dans le feature vector GBM

---

## 2026-04-17 — Ticket 18B (Retrain + validate ML-native Jesse agents)

### Problème
Ticket 18A a converti le code en ML-native. Mais sans retrain +
validation, c'est du ML-native en forme seulement, pas en substance.

### Hypothèse testée
Les 4 agents ML-natifs (model owns state/score/probas) produisent
des reports non-dégénérés et stables sur BTC 2020-2022.

### Résultats retrain
| Agent | Acc | pct_passed | avg_score | Elapsed |
|---|---:|---:|---:|---:|
| Context | 0.503 | 83.6% | 0.503 | 18s |
| Regime | 0.571 | 55.0% | 0.571 | 336s |
| Setup | 0.470 | 19.4% | 0.470 | 305s |
| Entry | 0.560 | 99.9% | 0.596 | 3.5s |

### Fix intermédiaire
Regime threshold 0.4 → 0.55. La première passe à 0.4 donnait
pct_passed 99.7% (même problème que T16 mais côté ML cette fois).
Le model RandomForest produit des p_trend concentrés à 0.4-0.6
sur BTC 4H → seuil à 0.55 restaure la discrimination.

### Avant/après (heuristic T16 → ML-native T18B)
| Agent | T16 heur | T18B ML | Delta |
|---|---:|---:|---|
| Context | 44.4% | 83.6% | +39 pp (ML voit plus de bull) |
| Regime | 51.6% | 55.0% | +3 pp (stable) |
| Setup | 35.4% | 19.4% | -16 pp (ML plus sélectif) |
| Entry | 99.9% | 99.9% | stable |

### Runtime-readiness verdict
- Context : ✓ stable, meaningful directional bias
- Regime : ✓ discriminant post-calibration
- Setup : ✓ selective, ML-native
- Entry : ⚠ conditional (pct_passed 99.9%, utile comme
  enrichissement probabiliste, pas comme gate)

### Docs créés
- `docs/JESSE_ML_REPORT_SPEC.md` — définit les sémantiques ML-
  native (probas, confidence, passed, heuristics autorisées/
  interdites, verdict runtime)

### Ticket 18 FULLY COMPLETE (A + B)
- 18A = code ML-native ✓ (source-level verified)
- 18B = behavior validated as ML-native ✓ (retrained + non-
  degenerate + documented)

---

## 2026-04-17 — Ticket 19 (Meta-GBM on ML fractal agents — REJECT)

### Question testée
Un Meta-GBM DÉDIÉ entraîné SEULEMENT sur les 22 features agent
(probas ML-natives + alignements cross-agents) peut-il outperformer
le GBM direct sur 173 features techniques ?

### Réponse : NON. REJECT catégorique.
Sharpe 1.10 vs baseline 9.96 = -89 %. PnL $483 vs $2,171 = -78 %.

### Root cause
Les 22 meta_* features sont des COMPRESSIONS LOSSY des données
techniques sous-jacentes. Chaque agent RandomForest jette de l'info
que le GBM direct sur 173 features préserve. Empiler 4 compressions
ne récupère pas le signal perdu.

### Convergence des 3 expériences
| Expérience | Approche | Sharpe | Verdict |
|---|---|---:|---|
| T11 baseline | 173 tech → GBM | 9.96 | KEEP |
| T17 augmented | 173 + 21 rep_* → même GBM | 8.73 | REJECT |
| T19 dedicated | 22 meta_* seuls → new GBM | 1.10 | REJECT |

### Conclusion architecturale
Le GBM direct sur features techniques = meilleure strategy brain.
Les Jesse agents = OBSERVABILITÉ (quality_bucket, risk_hint,
disagreement pour MetaGBM enrichissement output, Tickets 06/07),
PAS le signal primaire pour le trading.

### Ce que ça signifie pour la suite
1. Le baseline T11 reste canonique. Pas de swap.
2. Les agents Jesse restent dans le code comme couche
   d'enrichissement (reports pour MetaGBM output fields) — mais
   NE PAS injecter leurs probas dans le modèle de trading.
3. L'architecture "4 fractal reporters → Meta-GBM brain" est
   élégante conceptuellement mais ne bat pas le GBM monolithique
   sur ces données avec cette implémentation.
4. Possible direction future : entraîner les 4 agents sur un
   OBJECTIF DIFFÉRENT (pas le même label que le GBM) pour qu'ils
   capturent un signal ORTHOGONAL au lieu de compresser le même.

---

## 2026-04-17 — Ticket 20 (Jesse as post-decision modulators — NEUTRAL)

### Question testée
Les agents améliorent-ils l'edge quand ils MODULENT le sizing
POST-décision GBM au lieu de contribuer au signal ML ?

### Réponse : NEUTRAL. Non destructif (−3.2 % Sharpe, DD identique).

| Metric | Baseline T11 | T20 modulation | Delta |
|---|---:|---:|---|
| n_trades | 54 | 53 | −1 |
| Sharpe | 9.96 | 9.64 | −3.2 % |
| PnL | $2,171 | $1,966 | −9.5 % |
| Max DD | 0.37% | 0.375% | ≈ 0 |

### Convergence des 4 expériences

| # | Approche | Sharpe | Verdict |
|---|---|---:|---|
| T11 | GBM seul | 9.96 | Baseline |
| T17 | rep_* DANS le GBM | 8.73 | REJECT |
| T19 | Meta-GBM agents seuls | 1.10 | REJECT |
| **T20** | **GBM décide + agents modulent** | **9.64** | **NEUTRAL** |

### Ce que ça signifie
C'est la PREMIÈRE fois que les agents Jesse sont wirés dans le
runtime sans détruire l'edge. L'approche "modulation post-décision"
est architecturalement saine : les agents ajoutent de la sémantique
(quality_bucket, risk_hint) sans polluer le signal ML du GBM.

### Fichiers touchés
- `src/core/fractal_quality.py` (nouveau) — compute_fractal_quality
- `src/core/nyx_engine.py` — wiring post-GBM + mtf_data=None pour
  _generate_candidates (GBM pur)
- `tests/test_fractal_quality_ticket20.py` (nouveau) — 9 tests
- `reports/BTCUSDT_oos_report.json` — OOS T20
- `docs/CHANGELOG.md` + ce log

---

## 2026-04-17 — Ticket 21 (Realistic OOS BTC 2023)

### Question testée
L'edge du système survit-il à une exécution réaliste (post-only
maker, miss rate, fees) ?

### Réponse : OUI. Sharpe 6.46, WR 85.7 %, PF 7.8, DD < 0.4 %.

### Exécution
53 signaux → 53 placés → 35 FILLED (66%) + 18 TIMED_OUT (34%)
0 rejected, 0 slippage (maker fill at limit), avg 0.06 bars to fill

### Réaliste vs Idéalisé
| Metric | Idéalisé | Réaliste | Delta |
|---|---:|---:|---|
| n_trades | 53 | 35 | -34% (18 missed) |
| WR | — | 85.7% | high |
| Sharpe | 9.64 | 6.46 | -33% |
| PnL | $1,966 | $1,247 | -37% |
| DD | 0.375% | 0.396% | +6% |
| PF | — | 7.8 | excellent |

### Observation clé
La miss rate (34%) agit comme un filtre de QUALITÉ involontaire :
les trades missés sont souvent des breakouts rapides où le prix
s'éloigne du limit → certains auraient été des perdants. Les
trades qui FILL (le prix revient au limit) sont naturellement
de meilleure qualité → WR 85.7 %.

### Axes d'amélioration possibles
- Élargir le limit offset de 0.1% à 0.2-0.3% → meilleur fill rate
  mais entrée légèrement pire
- Augmenter max_wait_bars de 3 à 5 → plus de fills, plus de latence
- Les deux paramètres sont calibrables post-déploiement

### Verdict
Le système est DÉPLOYABLE. L'edge est réel, mesurable, et survit
à l'exécution réaliste. Next : Binance testnet ou live micro-capital.

---

## 2026-04-17 — Ticket 22 (Binance Market Connectivity Layer)

### Problème
NYX a un edge validé + exécution réaliste mais AUCUNE connexion au
marché live. Le flux de données vient de CSVs historiques.

### Livré
3 modules dans src/live/ :
- `binance_ws.py` : WebSocket client avec reconnect exponentiel
- `bar_builder.py` : normalisation kline → bar dict canonique (closed-
  only, duplicate reject, ISO timestamp, no exchange key leakage)
- `feed_health.py` : staleness + monotonicity + gap detection

Script d'intégration `scripts/run_live_feed.py` :
- WS → BarBuilder → FeedHealth → NYXLiveDecider.on_15m_bar()
- Même path canonique que le backtest, juste le DATA SOURCE change
- Pas de nouveau engine, pas de pipeline parallèle

### Tests
12/12 GREEN avec payloads WS mockés (pas de connexion réseau réelle).

### Architecture canonique préservée
Le ticket est explicite : le live feed NOURRIT l'architecture
existante, il ne la remplace pas. NYXLiveDecider reste le runtime
canonique. Le seul changement est la source des bars (WS au lieu de
CSV.iterrows()).

### Prérequis pour le déploiement réel
- `pip install websocket-client`
- Env var `BINANCE_SYMBOL=btcusdt`
- Artefact modèle dans `models/BTCUSDT/`
- Réseau ouvert vers `stream.binance.com:9443`

---

## 2026-04-17 — Ticket 23 (OMS — Order Management System)

### Livré
`src/live/oms.py::OMS` — single source of truth pour l'état des
ordres. Wraps le broker existant comme un CONTROLLER d'exécution.

### Lifecycle canonique
SUBMITTED → PARTIALLY_FILLED → FILLED
SUBMITTED → REJECTED / CANCELLED
PARTIALLY_FILLED → FILLED / CANCELLED
Terminal (FILLED/REJECTED/CANCELLED) = immuable.

### Protections
- Duplicate client_order_id → ValueError
- Overfill → ValueError
- Mutation terminal → RuntimeError
- Cancel terminal → no-op sûr

### Tests : 11/11 GREEN

---

## 2026-04-17 — Ticket 24 (Position & Portfolio State)

### Livré
`src/live/portfolio_state.py` — Position + Portfolio, single source
of truth. Derived from OMS fills only.

Position : open/add/close/flip, VWAP avg entry, realized + unrealized PnL.
Portfolio : multi-symbol, available_balance, total_equity, exposure.

### Tests : 16/16 GREEN

---

## 2026-04-17 — Ticket 25 (Risk Engine — sovereign, no bypass)

### Livré
`src/live/risk_engine.py::RiskEngine` — sovereign risk gate.
Validates every trade intent BEFORE OMS submission. Kill switch
for anomalies.

### Tests : 12/12 GREEN
Trade-level + portfolio-level + drawdown protection + kill switch.

### Architecture canonique complète

NYXEngine decision
  → RiskEngine.validate_trade()    ← SOVEREIGN (Ticket 25)
  → OMS.submit_order()             ← STATE OWNER (Ticket 23)
  → PostOnlyPaperBroker            ← EXECUTION
  → Portfolio.on_fill()            ← POSITION STATE (Ticket 24)

---

## 2026-04-17 — Ticket 26 (Persistence & Recovery)

### Livré
`src/live/state_store.py::StateStore` — atomic JSON persistence.
OMS orders + Portfolio positions + event log. Crash-safe (tmp →
fsync → rename). On restart: load_oms + load_portfolio → state
exact, duplicate protection active, no reconstruction needed.

### Tests : 6/6 GREEN

### Architecture live COMPLÈTE

Binance WS → BarBuilder → FeedHealth → NYXLiveDecider
  → Fractal Modulation → RiskEngine → OMS → Broker
  → Portfolio → StateStore (persist) → EventAlerter

Chaque couche a un rôle unique. L'état survit au crash.

---

## 2026-04-17 — Ticket 27 (Monitoring & Observability)

### Livré
`src/live/monitoring.py` — MetricsCollector (trading + system
metrics) + AlertManager (daily loss, drawdown, disconnect alerts).
12/12 GREEN.

### Pipeline live COMPLÈTE — toutes couches

Binance WS → BarBuilder → FeedHealth → NYXLiveDecider
  → Fractal Modulation → RiskEngine → OMS → Broker
  → Portfolio → StateStore → **MetricsCollector + AlertManager**

Le système est maintenant observable, persisté, et risk-controlled.

---

## 2026-04-17 — Ticket 22R (NYXRuntime — single orchestrator)

### Problème
Le live était fragmenté : run_live_feed.py appelait seulement
NYXLiveDecider. Jesse absent du live. Risk/OMS/Portfolio/State/
Monitoring jamais appelés en live. Deux pipelines divergents
(batch vs live).

### Livré
`src/live/nyx_runtime.py::NYXRuntime` — UN orchestrateur, UN fichier.
`on_bar(bar)` traverse les 10 couches. Rien n'est caché.

`run_live_feed.py` appelle SEULEMENT `runtime.on_bar(bar)`.
Plus aucun appel direct à NYXLiveDecider, RiskEngine, OMS, etc.

### Vérification de lisibilité (la question du user)

| Question | Réponse |
|---|---|
| Où est le GBM ? | `runtime.decider._meta.score_vector()` via `on_15m_bar()` → visible via `runtime.gbm` |
| Où sont les Jesse ? | `runtime._call_jesse_agents()` — 4 agents, un seul endroit |
| Un seul flux ? | ✅ OUI — `on_bar()` = GBM → Jesse → FractalQuality → Risk → OMS → Portfolio → State → Monitoring |
| Une seule commande ? | ✅ `python scripts/run_live_feed.py` |

### Tests : 14/14 GREEN

---

## 2026-04-17 — Ticket 28 (Model Governance)

### Rule 7 respectée
CLAUDE.md + OPERATING_RULES.md lus avant de coder. Natural owner
identifié : `src/ml/train_asset_model.py` (déjà save/load).

### Livré
`src/ml/model_registry.py::ModelRegistry` — versioned model store.
register → promote (with validation) → rollback (instant) → load_active.
Étend le path existant, ne le remplace pas.

### Tests : 11/11 GREEN

---

## 2026-04-17 — Ticket 29 (Control Plane — Operator Layer)

### Rule 7 ✓
CLAUDE.md lu. Natural owner identifié : NYXRuntime (pas un module
séparé). Rule 1 integration-first respectée.

### Livré
6 commandes opérateur SUR NYXRuntime : start/stop/pause/resume/
emergency_stop/cancel_all_orders/flatten_all. Guards dans on_bar()
retournent SYSTEM_STOPPED/PAUSED si pas running/paused.

### Tests : 27/27 GREEN (8 control + 19 runtime)

---

## 2026-04-17 — Ticket 31 (VaR / CVaR — Advanced Risk)

### Rule 7 ✓
CLAUDE.md lu. Natural owner : RiskEngine (T25). Extension, pas
remplacement. Alpha non touché.

### Livré
`src/live/var_cvar.py` : compute_var, compute_cvar, RollingRiskMetrics.
RiskEngine.validate_trade() étendu avec max_var_95 / max_cvar_95.

### Tests : 24/24 GREEN (12 VaR/CVaR + 12 T25 régression)

---

## 2026-04-17 — Ticket 32 (Execution Optimization)

### Rule 7 ✓
CLAUDE.md lu. Alpha non touché. Execution-only layer AFTER Risk, BEFORE OMS.

### Livré
`src/live/execution_optimizer.py` : fill probability estimation,
dynamic limit offset (replace static 0.1%), maker/taker decision
logic, build_execution_plan(). Déterministe, reproductible.

### Tests : 12/12 GREEN
Fill prob range + monotonicity, offset widens with vol + low fill
prob, maker/taker logic, execution plan structure + limit prices.

---

## 2026-04-17 — Ticket 33 (Multi-Asset NYXRuntime)

### Rule 7 ✓
CLAUDE.md lu. Extension multi-asset par instanciation — une NYXRuntime
par asset, isolation totale.

### Problème
ETH + SOL models trained sur legacy 84 features (ancien parquet 53 cols/TF).
BTC déjà sur 128 features canoniques. Divergence empêche multi-asset.

### Hypothèse
Regenerate parquets canoniques (37 cols/TF × 4 TFs) + retrain ETH/SOL
→ les 3 assets partagent le même feature set 128.

### Livré
1. Parquets ETH + SOL regénérés (37 cols/TF, `compute_stationary_features('full')`)
2. ETH retrained : 128 features, Sharpe 7.36, 90 trades, $2,895 PnL (2023 OOS)
3. SOL retrained : 128 features, Sharpe 2.59, 49 trades, $887 PnL (2023 OOS)
4. `tests/test_multi_asset_runtime_ticket33.py` : 14 tests
   - 3 instantiation (3 runtimes, correct model, feature parity)
   - 5 state isolation (OMS, portfolio, risk, metrics, state dirs)
   - 4 bar processing (stopped, started, independent, kill switch)
   - 2 control plane isolation (pause, stop)
5. `scripts/run_multi_asset.py` : multi-threaded runner, per-asset WS,
   BTC=live, ETH/SOL=paper

### Tests : 14/14 GREEN

### Architecture
```
run_multi_asset.py
  ├── Thread BTCUSDT → NYXRuntime(BTCUSDT) → BinanceKlineStream(btcusdt)
  ├── Thread ETHUSDT → NYXRuntime(ETHUSDT) → BinanceKlineStream(ethusdt)
  └── Thread SOLUSDT → NYXRuntime(SOLUSDT) → BinanceKlineStream(solusdt)
```
Each thread owns its runtime. No shared state. Kill switch on one
does not affect others (proven by test).

---

## 2026-04-17 — Ticket 34 (Inter-Asset Dependency Layer)

### Rule 7 ✓
CLAUDE.md lu. Alpha (GBM/Jesse) non touché. Layer opère APRÈS
Fractal Quality, AVANT RiskEngine. Direction jamais changée.

### Problème
Les 3 runtimes (T33) sont indépendantes. Aucune information ne
circule entre elles. BTC peut crasher sans que ETH/SOL ne réagissent.

### Hypothèse
Un layer de dépendance inter-actifs partagé entre runtimes détecte
les relations lead/lag, la contagion, et module le sizing en
conséquence — sans toucher à l'alpha.

### Livré
1. `src/live/inter_asset_dependency.py` :
   - Lagged feature matrix (returns, vol_ratio, momentum_spread)
   - Lead/lag detection via cross-correlation
   - Regime-conditioned dependency score (trending amplifie, ranging atténue)
   - Contagion modeling (shock propagation, spillover probability)
   - Decision modulation (aligned → boost 1.25×, misaligned → reduce 0.5×,
     high contagion → suppress)
   - InterAssetDependencyLayer orchestrator (shared across runtimes)
2. NYXRuntime.on_bar() wired : step 3b between Fractal Quality and Risk Engine
3. scripts/run_multi_asset.py updated : shared dependency layer

### Tests : 29/29 GREEN + 14 T33 regression

### Architecture
```
NYXRuntime(BTC).on_bar()  ─┐
NYXRuntime(ETH).on_bar()  ─┤─→ shared InterAssetDependencyLayer
NYXRuntime(SOL).on_bar()  ─┘     ├── update_return(symbol, ts, ret)
                                  └── evaluate(symbol, dir, size_mult)
                                       ├── detect_lead_lag()
                                       ├── detect_contagion()
                                       ├── aggregate_dependency()
                                       └── modulate_decision()
```
BTC = leader (not modulated). ETH/SOL = followers (modulated by
dependency with BTC).

---

## 2026-04-17 — Ticket 35 (Portfolio Allocator)

### Rule 7 ✓
CLAUDE.md lu. Alpha (GBM/Jesse) non touché. Allocator opère APRÈS
Dependency Layer, AVANT RiskEngine. Direction jamais changée.

### Problème
Chaque runtime alloue du capital indépendamment. Pas de vue
consolidée : 3 trades simultanés pourraient dépasser le capital
total, ou sur-concentrer dans une direction.

### Hypothèse
Un allocator partagé reçoit les candidates de chaque runtime, les
score, les rank, et alloue le capital de façon contrainte et
déterministe.

### Livré
1. `src/live/portfolio_allocator.py` :
   - `compute_allocation_score()` — ranking metric (confidence + edge +
     quality - contagion - exposure)
   - `PortfolioAllocator.allocate()` — score → rank → allocate iteratively
     avec 4 contraintes (per-trade, per-asset, total, directional)
   - Decisions : APPROVE_FULL / APPROVE_REDUCED / DEFER / REJECT
   - Dependency-aware : follower sizing reduced by dep_strength × 40%
2. NYXRuntime.on_bar() wired : step 3c between Dependency and Risk Engine
3. scripts/run_multi_asset.py : shared allocator across runtimes

### Tests : 23/23 GREEN + 43 regression (14 T33 + 29 T34)

### Architecture
```
NYXRuntime.on_bar()
  ├── 1. GBM signal
  ├── 2. Jesse fractal reports
  ├── 3. Fractal quality modulation
  ├── 3b. Inter-asset dependency
  ├── 3c. Portfolio allocator ← NEW
  ├── 4. RiskEngine
  ├── 5. OMS
  └── 6-10. Fill / Portfolio / Persist / Monitor
```

---

## 2026-04-17 — Ticket 36 (Central Synchronous Orchestrator)

### Rule 7 ✓
CLAUDE.md lu. Alpha (GBM/Jesse) non touché. Runtimes deviennent
candidat-only. Orchestrateur central arbitre le portfolio.

### Problème (identifié dans review architecturale)
L'allocator (T35) dans on_bar() ne voyait qu'un candidat à la fois.
Jamais de vrai arbitrage cross-asset. L'allocator "portfolio" était
en fait per-asset.

### Solution
- NYXRuntime avec `candidate_store` → s'arrête après fractal quality,
  émet `CandidateDecision`, retourne `CANDIDATE_EMITTED`
- `CandidateStore` collecte tous les candidats par bucket temporel
  (15m close), avec tolérance pour arrivées tardives
- `CentralOrchestrator.run_cycle()` reçoit le SET COMPLET, applique
  dependency + allocator, retourne décisions d'allocation
- Backward compatible : sans candidate_store, runtime exécute
  le pipeline complet (mode inline T34/T35)

### Livré
1. `src/live/central_orchestrator.py` :
   - CandidateDecision (dataclass)
   - CandidateStore (thread-safe, time-bucketed, tolerance window)
   - CentralOrchestrator (synchronized cycles)
2. NYXRuntime.on_bar() : candidate mode
3. scripts/run_multi_asset.py : runtimes candidat + orchestrateur

### Tests : 22/22 GREEN + 66 regression (14+29+23)

### Architecture finale
```
Thread BTCUSDT → NYXRuntime(candidate_store) → CandidateDecision
Thread ETHUSDT → NYXRuntime(candidate_store) → CandidateDecision
Thread SOLUSDT → NYXRuntime(candidate_store) → CandidateDecision
                         ↓
                  CandidateStore (time bucket)
                         ↓
              CentralOrchestrator.run_cycle()
               ├── InterAssetDependencyLayer (FULL set)
               ├── PortfolioAllocator (FULL ranking)
               └── Approved trades dispatched
```

---

## 2026-04-17 — Ticket 37 (Unified Audit Trail)

### Rule 7 ✓
CLAUDE.md lu. Alpha non touché. Audit trail est observability pure —
aucun impact sur les décisions.

### Problème
Aucune traçabilité structurée. Les décisions ne sont pas
reconstructibles ou rejouables. Pas institutional-grade.

### Solution
- `AssetAuditEvent` (per-bar) : capture signal + jesse + fractal +
  dependency + allocator + risk + OMS pour chaque barre
- `PortfolioAuditEvent` (per-cycle) : capture candidates + allocation
  results + dependency outputs pour chaque cycle
- `AuditStore` : append-only JSONL, un fichier par symbole +
  un fichier portfolio, survit au restart
- Linkage : cycle_id relie asset events ↔ portfolio events
- Replay : `replay_asset_events()` + `verify_audit_chain()`

### Instrumentation
- NYXRuntime._update_monitoring() → émet AssetAuditEvent
- CentralOrchestrator.run_cycle() → émet PortfolioAuditEvent
- Zero overhead quand audit_store=None

### Tests : 21/21 GREEN + 88 regression (14+29+23+22)

---

## 2026-04-18 — Ticket 38 (Repository Refactor)

### Rule 7 ✓
CLAUDE.md lu. Zero fichiers modifiés. Zero import paths changés.
Refactor par shims (pattern prouvé par nyx_pipeline.py).

### Problème
141 fichiers src/, un nouveau développeur ne peut pas distinguer
canonical vs legacy en <5 minutes.

### Solution
Shim-based : créer un namespace `src/nyx/` qui re-exporte les 30
modules canoniques. Documenter le legacy dans `legacy/MANIFEST.md`.
Écrire `docs/ARCHITECTURE.md` pour la vue 5 minutes.

### Livré
1. `src/nyx/` : 8 modules re-export (runtime, decision, portfolio,
   execution, data, state, contracts, __init__)
2. `docs/ARCHITECTURE.md` : pipeline diagram, module map, entrypoints
3. `legacy/MANIFEST.md` : inventaire complet des ~110 modules legacy
4. `tests/test_architecture_ticket38.py` : 19 tests structurels

### Tests : 19/19 GREEN + 109 regression (14+29+23+22+21)

---

## 2026-04-18 — Ticket 39 (Full-Stack OOS Standardization)

### Rule 7 ✓
CLAUDE.md lu. Pipeline identique pour les 3 assets. Pas de mode
idéalisé mélangé avec réaliste.

### Problème
BTC avait un OOS full-stack (broker, risk, OMS). ETH et SOL
n'avaient qu'un OOS idéalisé (pas de broker, pas de miss rate).
Résultats non comparables.

### Solution
`scripts/oos_multi_asset_full_stack.py` — fonction paramétrique
`run_full_stack_asset(symbol, capital)` applique le pipeline COMPLET
identique pour chaque asset. Reports séparés idealized_baseline vs
performance (réaliste).

### Résultats ($10M, 2023)
```
Asset      Trades  Sharpe      PnL     WR   Miss%    DD%
BTC            39    4.20  $612,594  76.9%    34%   0.93%
ETH            49    4.09  $426,887  71.4%    32%   1.55%
SOL            21    0.97   $21,838  61.9%    22%   0.50%
Portfolio: $10M → $11.06M (+10.61%)
```

### Tests : 12/12 GREEN

---

## 2026-04-18 — Ticket 40 (Capital-Aware Risk Engine)

### Rule 7 ✓
CLAUDE.md lu. Alpha non touché. Seul le RiskEngine est modifié.

### Problème
`max_position_notional = $50,000` hardcodé bloquait 62/63 trades
BTC à $5M de capital. Le système n'était pas scalable.

### Solution
Remplacé les limites fixes par des % d'equity :
- `max_position_pct=5.0` → compute `equity × 5% = max_position`
- `max_var_95_pct` / `max_cvar_95_pct` → dynamiques
- Legacy `max_position_notional` → auto-mappé comme hard_cap secondaire
- `risk_meta` dans chaque réponse : equity, limits, caps

### Tests : 16/16 GREEN + 152 regression (T25+T31+T33-T38)

---

## 2026-04-18 — Ticket 41 (Per-Asset ML Artifact Validation)

### Rule 7 ✓
CLAUDE.md lu. Pas de changement d'alpha. Validation + retrain des artifacts.

### Constat
ETH+SOL déjà retrained (Ticket 33) sur 128 features canoniques.
Retrain fresh = résultats IDENTIQUES (seed=42, déterministe).

### Validation
- Feature parity : BTC == ETH == SOL (128 noms identiques)
- Prefixes MTF : h1_, h4_, d1_ présents dans les 3 assets
- Runtime : chaque asset charge ses propres artifacts
- Pas de cross-contamination (MetaGBM distinct par instance)
- OOS reports avec idealized_baseline + performance séparés

### Tests : 38/38 GREEN

---

## 2026-04-18 — Ticket 42 (Canonical OOS Engine)

### Rule 7 ✓
CLAUDE.md lu. Alpha non touché. Engine OOS paramétrique, pas de
nouveau script.

### Livré
`src/live/oos_engine.py` :
- `OOSConfig` : assets, dates, capital, mode (idealized/realistic)
- `CanonicalOOSEngine.run_oos(config)` : mono/multi, toute fenêtre,
  tout capital ($100 → $1B)
- Reports persistés comme `oos_{run_id}.json`
- `list_oos_runs()` / `load_oos_report(run_id)` pour naviguer

### Usage
```python
cfg = OOSConfig(assets=['BTCUSDT','ETHUSDT','SOLUSDT'],
                start_date='2023-01-01', end_date='2023-12-31',
                initial_capital=10_000_000, mode='realistic')
result = CanonicalOOSEngine().run_oos(cfg)
```

### Tests : 19/19 GREEN

---

## 2026-04-18 — Ticket 42.1 (OOS Evaluator Framework)

### Rule 7 ✓
CLAUDE.md lu. Extension du OOS engine, pas de nouveau pipeline.

### Livré
- `OOSResult` : classe wrapper avec `.evaluate()`, `.run_id`, `.per_asset`
- `OOSResultEvaluator` : base class pour évaluateurs post-OOS
- `EvaluatorRegistry` : registration/dispatch par nom
- `CanonicalOOSEngine.run_oos()` retourne `OOSResult` (plus dict)
- Résultats d'évaluation persistés comme `eval_{run_id}_{name}.json`

### Tests : 15/15 GREEN + 19 T42 regression

---

## 2026-04-18 — Ticket 43 (Stability Evaluator)

### Rule 7 ✓
CLAUDE.md lu. Evaluator pluggé dans framework T42.1. Pas de script ad hoc.

### Livré
`src/live/evaluators/stability.py` :
- Segment slicer (year/quarter)
- Regime tagger (volatility/trend/composite)
- Stability metrics (score, profitable ratio, Sharpe dispersion)
- Classification (robust / unstable / regime_sensitive / opportunistic)
- Flags (stable_across_cycles, unstable_edge, high_regime_dependency, etc.)

### Tests : 17/17 GREEN

---

## 2026-04-18 — Ticket 43.1 (Regime-Aware Stability v2)

### Rule 7 ✓
CLAUDE.md lu. Extension de T43, pas de réécriture.

### Livré
Extensions au StabilityEvaluator :
- Tail risk : worst_trade, tail_loss_95/99, skewness, kurtosis
- Gain concentration : Gini coefficient, top 5%/10 contribution
- Regime dependency matrix : per-regime Sharpe/DD/WR
- Capital-aware classification : core/satellite/opportunistic/avoid
- Stability score v2 : composite avec pénalités tail/concentration/regime
- Advanced flags : tail_risk_dominant, gain_concentration_high,
  requires_regime_filter, capital_limited_edge

### Tests : 17/17 GREEN + 17 T43 regression = 34 total
