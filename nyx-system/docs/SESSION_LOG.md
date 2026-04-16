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
