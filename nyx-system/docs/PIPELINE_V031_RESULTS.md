# NYX Pipeline v0.3.1 — Résultats OOS & Conditional Bear Dial

> Pipeline unifié : MTF (15m+1h+1d) + ML Filter (threshold 0.60) + Conditional Bear Dial + Maker Fees
> 264 tests GREEN | 4/4 années OOS positives | Sharpe moyen 2.59

---

## OOS 4 Ans — Conditional vs Static

| Année | BTC | | Conditional | | | Static | | |
|-------|-----|---|------------|---|---|--------|---|---|
| | | T | WR | Sharpe | DD | T | WR | Sharpe | DD |
| **2020** | +303% | 57 | 51% | **+0.38** | **3.2%** | 68 | 47% | +0.16 | 5.5% |
| **2021** | +61% | 57 | 63% | +3.05 | 2.4% | 65 | 63% | +3.29 | 2.4% |
| **2022** | -64% | 83 | **59%** | **+1.76** | **2.6%** | 97 | 57% | +1.23 | 4.9% |
| **2023** | +156% | 42 | 79% | +5.18 | 0.6% | 56 | 80% | +6.77 | 0.7% |
| **Total** | | **239** | | **+2.59** | | **286** | | **+2.86** | |

---

## Impact du Conditional Bear Dial

| Année | Sharpe Δ | DD amélioration | Bear dial actif |
|-------|----------|-----------------|-----------------|
| 2020 | **+0.22** | **+2.3pts** | 39% |
| 2021 | -0.24 | ±0 | 41% |
| **2022** | **+0.52** | **+2.3pts** | 45% |
| 2023 | -1.59 | ±0 | 47% |

### Ce que le conditional dial améliore
- **2020** : Sharpe +0.22, DD réduit de 5.5% à 3.2% (protection bear héritée)
- **2022 bear** : Sharpe +0.52 (1.23 → 1.76), DD réduit de 4.9% à 2.6%
- Les années où il y a du stress, le dial protège

### Ce qu'il coûte
- **2021** : -0.24 Sharpe (bear dial actif 41% alors que bull clair)
- **2023** : -1.59 Sharpe (même problème — activation encore trop fréquente)
- **PnL total** : 43.3% vs 51.2% (perd ~8pts de return)

### Verdict
Le conditional dial **protège le bear** (Sharpe 1.23 → 1.76 en 2022, DD -2.3pts)
mais **coûte en bull** (~8pts de return total). Le bear dial s'active encore trop
(39-47% même en bull) — les seuils des triggers doivent être affinés.

**Pour le paper trading : utiliser le conditional dial** car la protection du
drawdown en bear vaut plus que les quelques % de return perdus en bull.

---

## Architecture unifiée v0.3.1

```
OHLCV 15m + 1h + 1d
       │
       ▼
┌─────────────────────┐
│  Edge Candidates     │  Trend aligned + Vol > 3x + Hours 6-20
│  (~150/an)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ML Filter GBM      │  47 parquet features + MTF context (1h+1d)
│  Threshold 0.60     │  Rule scores + disagreement as features
│  Trained on net PnL │  Rejects ~60-90% of candidates
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Conditional Dial    │  Per-trade regime check (2+ triggers)
│  Bear: threshold↑   │  Triggers: 1H bear regime, ATR stress,
│        size↓         │           weak trend, high disagreement
│        cooldown↑     │  Bull clear → full static params
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Soft Gate Sizing    │  ML confidence × rule agreement
│  + Realistic Fees    │  Maker 0.02%, slippage 0.01%
│  + Position Sizing   │  2% risk (1% in bear), 1x capital max
└──────────┬──────────┘
           │
           ▼
       TRADE
```

---

## Tests TDD : 264 GREEN

| Suite | Tests | Scope |
|-------|-------|-------|
| Pipeline unified | 12 | MTF input, ML filter, soft gate, OOS |
| Pipeline merged | 7 | Conditional dial default, 4yr OOS, bear DD |
| Conditional dial | 11 | Activation logic, pipeline, walk-forward |
| Bear risk dial | 12 | Detection, params, adaptive, bootstrap |
| Monte Carlo | 9 | Shuffle, noise, fee stress |
| Bootstrap | 13 | Standard, block, regime |
| + tous les précédents | 200 | Pipeline ML, agents, features, etc. |
| **Total** | **264** | |

---

## Prochaines étapes

| Priorité | Action |
|----------|--------|
| **P0** | Paper trading live avec DecisionLogger |
| P0 | Feedback loop actif (evaluate + retrain hebdo) |
| P1 | Affiner les triggers du conditional dial (réduire activation en bull) |
| P1 | Données 5m pour entry timing plus précis |
| P2 | Multi-pair (ETH, SOL) |
| P2 | Données L2 réelles (order book) |
