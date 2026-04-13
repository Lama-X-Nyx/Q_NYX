# Monte Carlo Final — Pipeline v0.3.1 (Conditional Dial)

> 3000 simulations | 239 trades | 2020-2023 OOS
> Pipeline unifié MTF + ML Filter + Conditional Bear Dial + Maker Fees

---

## 1. Distribution du Sharpe

| Méthode | Médian | 5th pctile | 10th pctile | Std |
|---------|--------|------------|-------------|-----|
| **Standard bootstrap** | **+4.20** | **+2.42** | +2.80 | 0.99 |
| **Block bootstrap** | +4.21 | **+2.64** | +2.95 | 0.95 |
| **Regime bootstrap** | +4.22 | +2.44 | +2.77 | 1.02 |

**Le 5th percentile Sharpe est à +2.42** — même le pire scénario bat les fonds institutionnels.
Le block bootstrap donne un p5 MEILLEUR (+2.64 vs +2.42) → pas de clustering de pertes.

---

## 2. Max Drawdown Stressé

| Méthode | Médian | 75th | 90th | **95th** | 99th | Worst |
|---------|--------|------|------|----------|------|-------|
| Shuffle | 3.7% | 4.5% | 5.3% | **5.9%** | 7.3% | 10.3% |
| Standard | 3.6% | 4.6% | 5.5% | **6.5%** | 8.3% | 13.3% |
| Block | 3.2% | 4.0% | 4.7% | **5.2%** | 6.5% | 10.9% |
| Regime | 3.7% | 4.7% | 5.7% | **6.6%** | 8.5% | 14.1% |

| Probabilité | Standard | Block | Regime |
|-------------|----------|-------|--------|
| **P(DD > 10%)** | **0.3%** | 0.1% | 0.5% |
| P(DD > 20%) | 0.0% | 0.0% | 0.0% |

**DD 95th percentile = 6.6% (worst-case).** 0.3% de chance de dépasser 10%.

---

## 3. Probabilité de Finir Négatif

| Méthode | P(loss) |
|---------|---------|
| Shuffle | **0.0%** |
| Standard | **0.0%** |
| Block | **0.0%** |
| Regime | **0.0%** |

**0 simulation sur 9000 finit négative.** (3 méthodes × 3000 sims)

---

## 4. Dépendance à l'Ordre (Shuffle)

| Métrique | Valeur |
|----------|--------|
| Return (invariant) | +43.3% |
| DD varie de | 0.5% à 10.3% |
| P(profit) | **100%** |

L'edge ne dépend PAS de la séquence des trades.

---

## 5. Block vs Standard (le test critique)

| Métrique | Standard | Block | Verdict |
|----------|----------|-------|---------|
| Sharpe p5 | +2.42 | **+2.64** | **Block meilleur** |
| DD p95 | 6.5% | **5.2%** | **Block meilleur** |
| P(DD>10%) | 0.3% | **0.1%** | **Block meilleur** |

**Le block bootstrap est MEILLEUR que le standard sur toutes les métriques.**
Pas de clustering temporel de pertes. L'edge est structurel.

---

## 6. Par Régime

| Régime | Trades | Return médian | Sharpe | DD 95th | P(loss) |
|--------|--------|--------------|--------|---------|---------|
| **Bull** | 156 | +31.7% | +3.97 | 5.4% | **0.0%** |
| **Bear** | 83 | +11.3% | +1.73 | 7.9% | **4.4%** |

Le bear dial conditionnel réduit le P(loss) bear de 10.2% (avant) à **4.4%**.
Le Sharpe bear est passé de +1.23 (sans dial) à **+1.73** (avec dial).

---

## 7. Pain Profile

| Métrique | Valeur |
|----------|--------|
| Losing streak 95th | **8 trades** |
| Losing streak worst | 14 trades |
| DD recovery p95 | ~60 trades (~2 mois) |

---

## Institutional Summary (worst-case)

| Critère | Valeur | Seuil | Status |
|---------|--------|-------|--------|
| Median return | +43.3% | > 0% | **PASS** |
| Median Sharpe | +4.20 | > 1.0 | **PASS** |
| 5th pctile return | +25.7% | > -10% | **PASS** |
| **5th pctile Sharpe** | **+2.42** | > 0 | **PASS** |
| 95th pctile DD | 6.6% | < 25% | **PASS** |
| **P(loss)** | **0.0%** | < 20% | **PASS** |
| P(DD > 10%) | 0.3% | < 10% | **PASS** |
| P(DD > 20%) | 0.0% | < 5% | **PASS** |
| Block ≥ Standard | +2.64 ≥ +2.42 | Block pas pire | **PASS** |
| Bear P(loss) | 4.4% | < 15% | **PASS** |

**Tous les critères institutionnels PASS. Architecture validée pour paper trading.**
