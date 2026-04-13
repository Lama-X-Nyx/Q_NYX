# Évolution A → B — Du système heuristique au ML validé

> Comment NYX est passé d'un système de trading rule-based sans edge mesurable
> à un pipeline ML avec edge validé walk-forward sur 4 ans.

---

## Point A — NYX v0.8 (Mars 2026)

### Ce qu'on avait

Un système fonctionnel mais sans preuve d'edge :

- **5 agents** rule-based + LightGBM (non entraînés)
- **HSMM 6 états** comme extracteur de features
- **47 features** custom (momentum, vol, microstructure)
- **Backtest** : 70ms/bar, sans fees
- **Résultat** : +13% Q1 2023 en pass-through (sans ML, sans fees)
- **Problèmes** : pas de TDD, pas de walk-forward, pas de fees, agents non entraînés

### Ce qui manquait

1. Aucune preuve que l'edge existe après fees
2. Aucun test automatisé
3. Pas de feedback loop (pas d'apprentissage des erreurs)
4. Pas de validation out-of-sample
5. 428 erreurs Pyright (code fragile)

---

## Le raisonnement — Pourquoi ce chemin

### Étape 1 : Nettoyer le code (Pyright)

**Pourquoi** : On ne peut pas construire du ML fiable sur du code fragile.
428 erreurs de type → des bugs potentiels partout.

**Action** : Audit Pyright complet, 48 fichiers corrigés.
**Résultat** : Code type-safe, base solide pour construire.

### Étape 2 : Adopter Jesse (framework trading ML)

**Pourquoi** : L'auteur de la vidéo Jesse ML explique la bonne méthode :
- TDD avec données synthétiques (si >85% accuracy = code OK)
- Features STATIONNAIRES (ratios, jamais de prix bruts)
- Triple barrier labeling (+1/-1/0)
- Gather/Deploy modes (pas d'apprentissage en live)

**Action** : Intégrer Jesse (177 indicateurs), créer le pipeline TDD.
**Résultat** : 41 tests GREEN, pipeline validé sur synthétique.

### Étape 3 : Reconstruire l'architecture 5 agents

**Pourquoi** : Un seul modèle ne suffit pas. L'architecture originale NYX
(Context → Regime → Setup → Entry → Orchestrator) est le bon cadre,
mais il faut la valider agent par agent.

**Action** : TDD séquentiel — chaque agent validé individuellement AVANT
d'être intégré. Backtest isolé par agent pour identifier les incohérences.

**Résultat** : 
- Context (1D) : détecte bullish/bearish correctement
- Regime (1H) : 55% range, 29% trend+, 16% squeeze (cohérent)
- Setup (15M) : bloquait 100% → fixé avec blend ML+heuristic
- Entry (15M) : trop permissif (96% ready) → fixé avec seuil
- **Conclusion** : les agents individuels sont valides, le problème était dans l'orchestration

### Étape 4 : FastBacktester vectorisé

**Pourquoi** : Le backtest v0.8 prend 10 minutes pour 8,640 bars.
Impossible d'itérer rapidement sans backtest rapide.

**Action** : Tout vectoriser — features + labels numpy ONCE, boucle pure numpy.
**Résultat** : 0.03ms/bar (2,333x plus rapide), 8,640 bars en 3 secondes.

### Étape 5 : Données réelles (le tournant)

**Pourquoi** : Les données synthétiques validaient le code mais pas l'edge.
Le modèle shortait dans un bull market → catastrophe.

**Action** : Extraire les vraies données du tar.gz (BTCUSDT 15m/1h/4h/1d, 2019-2024).

**Résultat** : Avec les vraies données, le système détecte correctement
la direction (100% long en bull Q1 2023). Mais les résultats étaient
"trop beaux" → Reality Check nécessaire.

### Étape 6 : Reality Check + Fees

**Pourquoi** : Les résultats bruts montrent +$50K sur 4 ans.
C'est suspect. Il faut compter les fees, slippage, sizing.

**Découvertes** :
- Sans fees : Sharpe 3.76 → FAUX
- Avec taker fees (0.04%) : Sharpe -4.62 → edge détruit
- Avec maker fees (0.02%) : Sharpe +1.26 → edge modeste mais réel
- Position sizing sans cap → 28x levier → irréaliste

**Conclusion** : L'edge existe mais il est MINCE. Chaque trade coûte ~$6-10
en fees. Il faut trader MOINS, pas plus.

### Étape 7 : Trouver l'edge réel (walk-forward)

**Pourquoi** : Il faut prouver que l'edge n'est pas du overfitting.

**Action** : Walk-forward sur 14 quarters (2020-2023), 5 hypothèses d'edge testées.

**Résultats** :
| Edge | Quarters + | Verdict |
|------|-----------|---------|
| Trend + Vol >1.5x | 14/14 | **VALIDÉ** |
| Pullback RSI | 0 | Rejeté |
| Volume >3x | 14/14 | **BEST** |
| Hours 8-18 | 13/14 | Fort |
| Cooldown | 8/14 | Artefact |

**Volume > 3x + heures de session** = l'edge robuste.

### Étape 8 : Soft gate (ML décide, rules valident)

**Pourquoi** : Les agents rules ne doivent pas bloquer le ML.
Ils doivent être des validateurs structurels, pas des décideurs.

**Principe** :
- ML = moteur principal
- Non-ML = garde-fous + détecteurs de désaccord
- Hard veto = RARE (data invalide seulement)
- Disagreement = FEATURE (pas un blocage)

**Résultat** : Le soft gate permet plus de trades mais dilue la qualité.
Le hard gate (filtres stricts) gagne grâce à la sélectivité.

### Étape 9 : ML Filter v2 (la percée)

**Pourquoi** : Le hard gate produit ~150 trades/an.
Le ML peut identifier les meilleurs 60% et rejeter le reste.

**Action** : GBM trained on net outcomes (après fees) avec les 47 features
parquet (microstructure, risk-adjusted) + rule scores comme features.
Threshold calibré par CV time-series.

**Résultat** :
- **WR 69%** (vs 48% sans filtre)
- **Sharpe 1.89** (vs 1.26 hard gate)
- **DD 0.7%** (vs 3.9%)
- **Rejette 97%** des candidats → ne garde que les trades à haute conviction

Les features clés : buy_pressure, vol_surprise, volume_spike, amihud_z.
→ La microstructure (L2 proxy) ajoute du vrai signal.

---

## Point B — NYX v2.5.0 (Avril 2026)

### Ce qu'on a maintenant

| Composant | Status |
|-----------|--------|
| 200 tests TDD GREEN | ✅ |
| Edge validé walk-forward 14/14 quarters | ✅ |
| Backtest avec fees/slippage/sizing réalistes | ✅ |
| ML Filter v2 avec 47 parquet features | ✅ |
| Threshold calibration auto | ✅ |
| Feedback loop (log → evaluate → retrain) | ✅ |
| Champion/Challenger model promotion | ✅ |
| Soft gate architecture | ✅ |
| Reality check documenté | ✅ |

### Résultats honnêtes

| Métrique | Hard Gate seul | + ML Filter v2 |
|----------|---------------|----------------|
| Sharpe (2023) | +1.86 | **+1.89** |
| Win Rate | 48% | **69%** |
| Max DD | 3.9% | **0.7%** |
| Trades/an | ~150 | ~15-60 |
| Return réaliste/an | 10-15% | 10-30% |

---

## Les leçons apprises

### 1. "Si c'est trop beau, ça l'est"
Les premiers résultats montraient +50K PnL → c'était sans fees.
Avec fees, tout change. Toujours vérifier : fees, slippage, sizing, overcounting.

### 2. "Moins de trades = mieux"
Avec des fees de ~$6/trade, chaque trade doit rapporter plus de $6 net.
Un edge de +$0.77/trade brut est mangé par les fees.
Filtrer de 700 trades/an à 150 → edge positif.

### 3. "Le ML ne remplace pas les règles, il les complète"
Le ML seul overfit. Les règles seules sont trop rigides.
La combinaison (ML décide + rules valident + disagreement comme feature) est optimale.

### 4. "Le TDD sauve du temps"
Chaque bug trouvé par le TDD sur synthétique aurait coûté des semaines
de debugging sur données réelles. Le TDD synthétique (>85% accuracy = code OK)
est la meilleure idée de la vidéo Jesse.

### 5. "Les features de microstructure font la différence"
Les features classiques (EMA, RSI, momentum) n'ajoutent presque rien en edge.
Les features de microstructure (buy_pressure, amihud, vol_surprise) sont
les prédicteurs les plus importants pour la qualité d'un trade.

---

## Prochaines étapes

| Priorité | Action |
|----------|--------|
| P0 | Paper trading live avec DecisionLogger |
| P0 | Feedback loop en production (evaluate + retrain hebdo) |
| P1 | Optimiser threshold sur données plus récentes |
| P1 | Ajouter données L2 réelles (order book) |
| P2 | Multi-pair (ETH, SOL) |
| P2 | Cross-asset features (funding, OI) |
