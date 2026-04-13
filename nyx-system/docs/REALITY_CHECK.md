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
