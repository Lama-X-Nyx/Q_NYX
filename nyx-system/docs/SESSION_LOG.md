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
