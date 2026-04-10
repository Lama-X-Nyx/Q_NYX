# NYX — ML Ecosystem Architecture

**Version:** 1.0 — Avril 2026  
**Statut:** Implémenté, agents en mode pass-through (non entraînés)

---

## Principe fondamental

Chaque agent répond à **une seule question** sur **un seul timeframe**.
Les sorties s'empilent en features pour les agents suivants (stacking).

```
Question              Agent              Timeframe  Modèle
──────────────────────────────────────────────────────────
Direction macro ?     MLContextAgent     1D         LGB (bull/bear/neutral)
Quel régime ?         MLRegimeAgent      1H         LGB + HSMM proba features
Setup valide ?        MLSetupAgent       15M        LGB + SMC + cross-agent
Bon timing ?          MLEntryAgent       15M        LGB + River online
P(profit) global ?    MLOrchestrator     --         meta-LGB (stacking)
```

---

## MLContextAgent (1D)

**Question** : "Quelle est la direction macro du marché ?"

**Features** :
- Momentum 5/20/60/200 jours + acceleration
- Réalisée vol 5/20j + ratio court/long
- EWMA vol GARCH (λ=0.94)
- RSI(14) Wilder's EMA
- Position vs SMA50 + SMA200 (trend strength)
- Buy pressure (close - low) / (high - low)
- Amihud illiquidity 20j
- Garman-Klass vol 20j

**Labels** :
- bullish : max(close t+1..t+5) / close_t - 1 > 3% AND NOT bearish
- bearish : close_t / min(close t+1..t+5) - 1 > 3% AND NOT bullish
- neutral : sinon (pas de trade)

**Sortie** : AgentResult(agent='context', state='bullish'|'bearish'|'neutral', score=P(état))

**Pass-through** : SMA200 rule-based si non entraîné.

---

## MLRegimeAgent (1H)

**Question** : "Quel est le régime de marché actuel ?"

**Features** :
- Momentum 4/12/48h
- Vol réalisée 8/24/96h + EWMA GARCH
- Buy pressure (raw + MA)
- Amihud + Kyle's lambda
- Eff. spread ratio
- HSMM 6 états proba (p_trend_plus, p_range, p_trend_minus, p_squeeze, p_distribution, p_liquidation)
- HSMM dominant state index + entropie Shannon
- RSI(14) Wilder
- Vol of vol (vov_24)

**Labels** : 1 si max(high t+1..t+4) / close_t - 1 > 1% (direction haussière dans 4h)

**Note clé** : Les probabilités HSMM deviennent des **features** pour LGB, pas des règles.
LGB apprend quels états HSMM *en combinaison avec la microstructure* prédisent vraiment une direction.

**Sortie** : AgentResult(agent='regime', state=dominant_state, score=P(bull), ...)

**Pass-through** : argmax HSMM proba, dom_prob comme score.

---

## MLSetupAgent (15M)

**Question** : "Y a-t-il un setup d'entrée valide ?"

**Features** :
- Momentum 4/8/16/32 barres 15M
- Vol + GARCH + OB proxies (buy pressure, Amihud, Kyle, spread)
- HSMM 15M proba (6 états + dominant + entropie)
- SMC patterns : bullish/bearish OB, FVG, CHoCH (binaires + scores)
- Cross-agent : regime_score (float), context_dir (+1/-1/0)
- Interaction : regime_score × context_dir

**Labels** : 1 si prix > 0.8% dans la direction du contexte sur 8 barres 15M

**Note clé** : regime_score et context_dir créent des **features cross-agents**.
Le modèle apprend "setup valide si régime fort ET context bull ET SMC pattern présent".

**Sortie** : AgentResult(agent='setup', state='valid_setup'|'misaligned'|'no_pattern', score=P(setup))

---

## MLEntryAgent (15M)

**Question** : "Est-ce le bon moment précis d'entrer ?"

**Features** : 47 features MLFeatureEngine (voir feature_engine.py)

**Modèle** : blend 0.7×LGB + 0.3×River
- LGB : walk-forward CV, structure lente (vol regime, momentum persistence)
- River : adapte aux dynamics récentes (mise à jour après chaque trade résolu)

**Sortie** : AgentResult(agent='entry', score=ml_prob, passed=(ml_prob >= 0.55))

---

## MLOrchestrator (meta-LGB)

**Question** : "Étant donné les 4 signaux, P(trade profitable) = ?"

**Features** :
```
# Agent signals
p_context_bull, p_context_bear     # Direction proba
p_regime_trend, p_regime_range, p_regime_squeeze  # Régime proba
p_setup                             # Setup proba
p_entry                             # Entry proba

# Agreement metrics (key feature: détecte le consensus vs divergence)
agent_std   = std([p_ctx, p_reg, p_setup, p_entry])  # bas = consensus
agent_min   = min(...)   # signal le plus faible
agent_mean  = mean(...)

# Microstructure snapshot (dernière barre 15M)
vol_ratio, buy_pressure, amihud, hour_sin/cos, dow_sin/cos

# Flags individuels
ctx_passed, reg_passed, stp_passed, ent_passed (booleans)
is_trending, is_range, is_squeeze, has_smc_pattern
```

**Label training** : prix > 0.8% dans la direction du contexte sur 8 barres
→ bar-level labels → 50-100× plus d'échantillons que les trades réels

**Size factor** (scaling de position) :
```
P(profit) >= 0.75  →  1.5×  (très haute conviction)
P(profit) >= 0.65  →  1.2×
P(profit) >= 0.60  →  1.0×  (baseline)
P(profit) <  0.60  →  0.75× (basse conviction mais signal présent)
```

**Fallback** si non entraîné : score = moyenne pondérée (context 30% + regime 30% + setup 25% + entry 15%), seuil 0.78.

---

## Pipeline d'entraînement

```
2019 ──────────────────── 2022-12-31 | 2023 ──────────────────────► 
     Train                           | OOS Validation
```

```bash
python scripts/train_ml_ecosystem.py --train-end 2022-12-31
```

Ordre d'entraînement (chaque agent dépend du précédent) :
1. HSMM EM → gamma_1h, gamma_15m (feature extractors)
2. SMA200 → context_arr
3. SMC rolling → smc_list
4. MLContextAgent.pretrain(df_1d)
5. MLRegimeAgent.pretrain(df_1h, gamma_1h)
6. MLSetupAgent.pretrain(df_15m, gamma_15m, smc_list, context_series)
7. MLOrchestrator.generate_training_data(...)
8. MLOrchestrator.pretrain(meta_dataset)

---

## Mode pass-through (backward compatibility)

Chaque agent fonctionne sans modèle entraîné :
- MLContextAgent → SMA200 rule-based
- MLRegimeAgent → argmax HSMM proba
- MLSetupAgent → combinaison linéaire HSMM proba
- MLEntryAgent → score=0.5, passed=True (ne bloque jamais)
- MLOrchestrator → weighted average scores (seuil 0.78)

Cela garantit la compatibilité avec l'historique des backtests.

---

## Performances actuelles (pass-through)

Q1 2023 : 11 LONGs / 0 SHORTs | +13.03% | Sharpe 3.82 | MaxDD 6.94% | PF 4.02

Avec ML entraîné (objectifs) :
- Sharpe > 1.5 sur année OOS
- MaxDD < 10%
- Profit Factor > 1.8
- t-stat Sharpe > 2.0
