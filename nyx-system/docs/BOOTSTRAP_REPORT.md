# Bootstrap Stress Test — Rapport Détaillé

> NYX v0.3 | 3 méthodes × 3000 simulations | 286 trades OOS 2020-2023
> Le bootstrap avec remise est plus conservateur que le shuffle :
> certains trades sont dupliqués, d'autres absents → vraie variance.

---

## Méthode 1 : Bootstrap Standard (iid avec remise)

Chaque simulation tire 286 trades au hasard AVEC remise parmi les 286 réels.
Un trade peut apparaître 0, 1, 2 ou 3 fois. Crée de la variance réelle.

### Return
| Percentile | Valeur |
|------------|--------|
| 5th | **+30.8%** |
| 10th | +35.7% |
| 25th | +42.7% |
| **Median** | **+51.4%** |
| 75th | +59.2% |
| 95th | +71.2% |
| Std | 12.3% |

### Sharpe
| Percentile | Valeur |
|------------|--------|
| **5th** | **+2.31** |
| 10th | +2.69 |
| **Median** | **+3.91** |
| 95th | +5.55 |
| Std | 0.98 |

### Max Drawdown
| Percentile | Valeur |
|------------|--------|
| Median | 4.2% |
| 75th | 5.4% |
| 90th | 6.7% |
| **95th** | **7.6%** |
| 99th | 9.8% |
| Worst | 14.5% |

### Risque
| Métrique | Valeur |
|----------|--------|
| **P(loss)** | **0.0%** |
| P(DD > 10%) | 0.7% |
| P(DD > 20%) | 0.0% |
| Losing streak 95th | 8 trades |

---

## Méthode 2 : Block Bootstrap (taille 5)

Échantillonne des **blocs consécutifs** de 5 trades au lieu de trades individuels.
Préserve l'autocorrélation temporelle (séries de pertes, momentum runs).

**C'est le test le plus important** : si le block bootstrap s'effondre par rapport au standard, l'edge dépend de clusters temporels chanceux.

### Return
| Percentile | Valeur |
|------------|--------|
| 5th | **+33.0%** |
| 10th | +37.2% |
| **Median** | **+51.5%** |
| 95th | +71.0% |

### Sharpe
| Percentile | Valeur |
|------------|--------|
| **5th** | **+2.43** |
| 10th | +2.79 |
| **Median** | **+3.92** |
| 95th | +5.61 |

### Max Drawdown
| Percentile | Valeur |
|------------|--------|
| Median | 4.0% |
| **95th** | **6.9%** |
| Worst | 14.3% |

### Risque
| Métrique | Valeur |
|----------|--------|
| **P(loss)** | **0.0%** |
| P(DD > 10%) | 0.4% |
| Losing streak 95th | 8 trades |

---

## Méthode 3 : Regime Bootstrap (stratifié bull/bear)

Échantillonne séparément dans chaque régime de marché,
puis recombine proportionnellement. Teste si l'edge est porté
par un seul régime ou s'il est distribué.

### Return global
| Percentile | Valeur |
|------------|--------|
| 5th | **+31.4%** |
| **Median** | **+51.6%** |
| 95th | +72.6% |

### Sharpe global
| Percentile | Valeur |
|------------|--------|
| **5th** | **+2.33** |
| **Median** | **+3.96** |

### Par régime (bootstrap séparé)

| Régime | Trades | Return médian | Sharpe | DD 95th | **P(loss)** |
|--------|--------|--------------|--------|---------|-------------|
| **Bull** (2020, 2021, 2023) | 189 | +41.7% | +4.30 | 5.9% | **0.0%** |
| **Bear** (2022) | 97 | +9.2% | +1.23 | 10.1% | **10.2%** |

**Interprétation critique** : Le bear est le point faible.
10.2% de probabilité de perte en bear-only. Sharpe 1.23 (acceptable mais fragile).
DD 95th à 10.1% en bear — le seul scénario où le DD frôle les 10%.

Mais l'edge **existe quand même en bear** (médiane +9.2%, Sharpe +1.23).
Le système n'est pas un pur long-bias qui crash en bear.

---

## Comparaison des 3 méthodes

> Le point clé : si le standard tient mais que le block s'écroule,
> l'edge dépend de clusters. Si les 3 tiennent → edge robuste.

| Métrique | Standard | Block | Regime | **Verdict** |
|----------|----------|-------|--------|-------------|
| Median return | +51.4% | +51.5% | +51.6% | **Consistant** |
| 5th pctile return | +30.8% | **+33.0%** | +31.4% | **Block ≥ Standard** |
| Median Sharpe | 3.91 | 3.92 | 3.96 | **Consistant** |
| 5th pctile Sharpe | 2.31 | **2.43** | 2.33 | **Block ≥ Standard** |
| DD 95th | 7.6% | **6.9%** | 7.4% | **Block meilleur** |
| P(loss) | 0.0% | 0.0% | 0.0% | **Identique** |
| P(DD>10%) | 0.7% | **0.4%** | 0.8% | **Block meilleur** |

**Résultat clé : le block bootstrap est MEILLEUR que le standard.**
C'est le meilleur signal possible — l'autocorrélation temporelle
**aide** plutôt qu'elle ne nuit. Les gains viennent en séries,
les pertes sont isolées. Pas de cluster de pertes systématique.

---

## Test Institutionnel : Le Bas de la Distribution

> Le vrai test n'est pas "le scénario moyen est beau".
> C'est "le bas de distribution reste acceptable".

| Critère | Worst-case (3 méthodes) | Seuil | Status |
|---------|------------------------|-------|--------|
| Perte rare | **P(loss) = 0.0%** | < 20% | **PASS** |
| DD stressé | **95th = 7.6%** | < 25% | **PASS** |
| Sharpe dégradé | **5th = +2.31** | > 0 | **PASS** |
| Return dégradé | **5th = +30.8%** | > -10% | **PASS** |
| Pas un seul régime qui porte tout | **Bull +41.7%, Bear +9.2%** | Les deux > 0 | **PASS** |

### Ce qui est fort
- 0% de probabilité de perte sur 9000 simulations (3 × 3000)
- Le block bootstrap est meilleur que le standard (pas de cluster de pertes)
- L'edge existe en bull ET en bear (Sharpe > 1 dans les deux)
- Le 5th percentile Sharpe est à +2.31 (même le pire scénario bat la plupart des fonds)

### Ce qui est fragile
- Bear-only P(loss) = 10.2% — 1 simulation sur 10 perd de l'argent en bear market pur
- DD 95th en bear = 10.1% — s'approche du seuil de douleur
- Seulement 97 trades en bear → faible significance statistique
- Recovery worst case = 136 trades (~5 mois) — long psychologiquement

---

## Verdict Final

**L'architecture est validée pour paper trading.**

Le système n'est pas un artefact statistique :
- L'edge survit au reshuffling, au bootstrap, au bruit, aux fees doublées
- Il est positif en bull ET en bear
- Le bas de la distribution (5th percentile) reste largement positif
- Aucun cluster temporel de pertes (block ≥ standard)

Le point de vigilance en live : le **bear market**.
10% de probabilité de perte, DD pouvant aller à 10%.
C'est là que le feedback loop (DecisionLogger + OutcomeEvaluator)
sera le plus critique pour détecter une dégradation en temps réel.
