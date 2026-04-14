# Reality Check — Pourquoi les chiffres sont gonflés

> Mise à jour : 2026-04-13

## Le problème

Les résultats du walk-forward montrent un edge **positif sur 14/14 quarters**
et PnL de +50K points OOS. C'est suspect. Voici pourquoi.

## 6 biais identifiés

### 1. PnL en points, pas en dollars
Le backtest compte `pnl_pts = direction * (exit - entry)`.
Un "point" à $20K n'a pas la même valeur qu'un point à $40K.
**Impact** : PnL gonflé en période de prix élevé.

### 2. Fees + slippage = 0
Aucun frais compté.
- Fee taker Binance Futures : 0.04% (maker 0.02%)
- Round trip : ~0.08% minimum
- Slippage estimé : 0.02-0.05%
- **Coût réel par trade : ~0.10-0.15% du notional**

Avec 6,396 trades × $28K prix moyen × 0.1% = **~$180K de fees**
→ Mange une grosse partie du PnL brut.

### 3. Re-entry immédiate
Le backtest re-entre à la barre suivante après un exit.
En réalité : latence, spread, validation, cooldown.
**Impact** : ~8.8 trades/jour est irréaliste. Max réaliste : 2-5/jour.

### 4. Position sizing = 1 BTC
Le backtest compte 1 unité par trade quel que soit le capital.
En réalité avec $10K et 2% risk :
- ATR moyen ~$108
- Qty = $200 / $108 = 0.00185 BTC
- PnL par trade ≈ $0.06 (pas $7.9)

### 5. Pas de compounding réaliste
Le calcul dollar avec compounding explose exponentiellement.
$10K → $24M sur 6,396 trades est un artefact mathématique.
En réalité, le slippage et les fees grandissent avec la position.

### 6. Overcounting des signaux
Chaque barre 15m avec un trend aligné + volume est un trade potentiel.
8.8 trades/jour = on ne filtre presque rien.
Un vrai filtre ML doit réduire à 1-3 trades/jour max.

## Ce qui est VRAI malgré tout

| Affirmation | Validité |
|-------------|----------|
| L'edge trend+volume existe | **OUI** — direction positive dans tous les régimes |
| 14/14 quarters positifs (points) | **OUI** — mais en points bruts, avant fees |
| WR ~43% avec TP/SL 1.5:1 | **OUI** — mathématiquement cohérent (EV = 0.43×1.5 - 0.57×1.0 = +0.075) |
| Volume >1.5x améliore l'EV | **OUI** — +0.077 vs +0.039, robuste sur 14Q |
| Edge survit au bear 2022 | **OUI** — short fonctionne en downtrend |

## Ce qu'il faut corriger

| Correction | Priorité |
|------------|----------|
| Ajouter fees 0.04% taker au backtest | **P0** |
| Ajouter slippage 0.02% estimé | **P0** |
| Position sizing réaliste (% du capital) | **P0** |
| Cooldown minimum 4 barres (1h) | **P1** |
| Max 3 trades/jour | **P1** |
| PnL en % du capital, pas en points | **P1** |
| Max drawdown en dollars | **P1** |

## Estimation réaliste

Avec les corrections (fees 0.1%, max 3 trades/jour, sizing 2% risk) :
- ~730 trades/an (3/jour × 244 jours de trading)
- EV par trade après fees : ~0.03 ATR (vs 0.077 brut)
- Return annuel estimé : **10-30%** (pas 200%+)
- Max drawdown estimé : **15-25%**

C'est un edge modeste mais réel. Pas un Graal.

## Règle

> Si les résultats semblent trop beaux, ils le sont.
> Toujours vérifier : fees, slippage, sizing, overcounting, compounding.

---

## Update 2026-04 — walk-forward trio (BTC+ETH+SOL) reste suspect

Les corrections P0/P1 listées plus haut ont été implémentées :
- fee_rate=0.0002 maker / 0.0004 taker (tests existants)
- slippage_rate=0.0001
- sizing risk_pct=0.02 du capital, position calculée via `risk_to_qty`
- bear dial conditionnel actif
- post-only `MakerFirstBroker` + missed-trade logger côté paper-live

ET POURTANT le walk-forward trio 2020-2023 affiche :
- CAGR **+49.31 %**
- 4/4 années positives
- max DD année **3.2 %**
- bootstrap p5 return **+183 %** sur 4500 sims

→ **C'est encore trop beau.** Voici les 6 nouvelles raisons concrètes :

### A. In-sample accuracy 88-90 %

`models/{ETH,SOL}USDT/training_metadata.json` : 0.892 et 0.906.
Baseline binary triple-barrier en crypto = 50-55 %.
**Différentiel de 35-40 pp = overfit fort** sur 2020-2022.

Même si l'OOS 2023 marche, ça ne dit RIEN sur 2024+ (régime jamais
vu). BTC + ETH s'arrêtent au 2024-01-01 → on n'a pas de vraie OOS
ultime sur ces deux.

### B. Sharpe annualisé sur trades, pas sur capital quotidien

`src/ml/nyx_pipeline.py:268` :
```python
sharpe = float(np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(min(nt, 252)))
```

C'est `per_trade × √n_trades`. Pour 200 trades/an : `√200 ≈ 14.1` →
Sharpe rapporté 11.37, qui correspond à un ratio par-trade de 0.80.

Le **Sharpe quotidien** (comparable aux fonds) serait probablement
3-5, pas 8-11. Toujours bon, mais 2-3× moins spectaculaire.

### C. Win rate 70-80 % à 1.5R = anormal

| Année | WR combiné |
|---|---:|
| 2021 | 72 % |
| 2022 | 71.9 % |
| 2023 | **80 %** |

Pros : 55 % WR à 1.5R OU 40 % WR à 2.5R typiquement.
**80 % WR à 1.5R = soit fuite subtile, soit régime spécifique 2020-2023
qui ne durera pas.**

### D. NYXPipeline backtest n'utilise PAS le PostOnlyPaperBroker

Le `PostOnlyPaperBroker` (que j'ai construit pour le paper-live) tracke
honnêtement les missed trades (REJECTED + TIMED_OUT). **Le NYXPipeline
backtest ignore ça** : il assume tous les trades fillent au close.

En réalité (vu sur la branche paper-live) : **30-60 % des limits
post-only ne fillent pas** dans les régimes nerveux (SOL surtout).

→ **Ces trades manqués ne sont pas comptés** dans les stats walk-forward.
Si on les comptait correctement (en supprimant les trades non-fillés),
le n_trades baisserait de 30-50 %, l'edge total avec.

### E. Bootstrap IID sur-confiant

`standard_bootstrap` rééchantillonne 560 trades **IID avec
remplacement**. Trades crypto **ne sont pas IID** :
- Régime persiste (autocorrélation positive)
- Bull 2021 ≠ Bear 2022 ≠ Bull 2023 (non-stationnaire)

`block_bootstrap(block_size=5)` aide à peine — 5 trades = ~3 jours
de marché, ≪ une période de régime.

**La vraie incertitude est probablement 5-10× plus large.** Le
prob_loss "0 %" est presque certainement faux. Réaliste : 5-15 %.

### F. Pas de vrai OOS sur régime jamais vu

- BTC + ETH : data s'arrête 2024-01-01 → 2024 + 2025 jamais testés.
- SOL : data va jusqu'à 2026-04 → **2024-2026 NEVER TESTED**, alors
  que ce sont 2 ans de data fraîche.

**À FAIRE EN PRIORITÉ** : `train_end=2023-12-31, test_start=2024-01-01,
test_end=2026-04-13` sur SOL. C'est le seul vrai jugement.

---

## Estimation réaliste après correction des biais ci-dessus

| Métrique | Walk-forward reporté | Estimation honnête live |
|---|---:|---:|
| CAGR | +49 % | **+15-25 %** |
| Max DD année | 3 % | **8-15 %** |
| Sharpe (annualisé capital) | 8-11 | **2-3** |
| Prob_loss année | 0 % | **5-15 %** |
| 4/4 années positives | oui | **probablement 3/4** |

Toujours un **edge réel** mais **pas un Graal**.

---

## Ce que je propose comme prochaine itération

1. **OOS pur SOL 2024-2026** (data dispo, jamais testé)
2. **Re-runner le trio walk-forward avec PostOnlyPaperBroker en
   boucle** pour mesurer la vraie miss-rate
3. **Re-runner avec taker fees** (test existant `test_taker_fees.py`)
   sur le walk-forward trio
4. **Sharpe quotidien** sur l'equity curve, pas sur les trades
5. **Block bootstrap n=30** au lieu de 5 → meilleur
   préservation des régimes
6. **Comparaison vs benchmark naïf** : buy & hold trio sur la même
   fenêtre

Si après ces 6 corrections le CAGR survit à >15 % : la stratégie
est *vraiment* bonne. Sinon, c'est du backtest porn et il faut
recalibrer.

Pour l'instant, **les chiffres walk-forward trio sont du backtest
porn jusqu'à preuve du contraire**. L'architecture est solide, les
tests TDD sont propres, mais les niveaux de performance restent à
valider OOS-réel.
