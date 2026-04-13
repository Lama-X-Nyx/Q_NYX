# NYX Trading System v0.2.5 — Architecture Complète

> Mise à jour : 13 Avril 2026 — 200 tests TDD GREEN, edge validé walk-forward

---

## Vue d'ensemble

NYX est un système de trading algorithmique multi-timeframe (MTF) pour BTC/USDT perpetual futures.
La décision d'entrée se fait sur barres 15M avec 4 niveaux de contexte temporel imbriqués.

**Philosophie** : HSMM et SMA200 sont des **extracteurs de features**, pas des décideurs.
Les modèles LightGBM apprennent quelles combinaisons de signaux prédisent vraiment des trades profitables.

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Détection de régime | Semi-Markov HMM (6 états) |
| Patterns de structure | Smart Money Concepts (OB, FVG, CHoCH) |
| ML agents | LightGBM walk-forward CV (5 splits) |
| Online learning | River LogisticRegression (post-trade) |
| Features | 47 (momentum, vol GARCH, OB proxies, saisonnalité) |
| Speedup backtest | PrecomputedRunner streaming forward (40×) |

---

## Architecture décisionnelle

```
RAW DATA (OHLCV) : 1D / 4H / 1H / 15M
        │
        │  MLFeatureEngine → 47 features par barre
        │  HSMM (1H, 15M)  → streaming forward → gamma (T×6)
        │  SMA200 (1D)      → context_arr
        │
        ▼
COUCHE 1 — MLContextAgent (1D)
  → P(bullish / bearish / neutral)
  features : momentum 5/20/60/200j, vol, RSI, Amihud
  label    : max return > 3% sur 5 jours

COUCHE 2 — MLRegimeAgent (1H)
  → P(trend+), état dominant, confiance
  features : momentum 4/12/48h + vol + HSMM proba (6)
  label    : max high > 1% dans 4 barres

COUCHE 3 — MLSetupAgent (15M)
  → P(valid_setup)
  features : momentum + HSMM 15M + SMC + cross-agent scores
  label    : prix > 0.8% en 8 barres

COUCHE 4 — MLEntryAgent (15M)
  → P(entry_ok)  0.7×LGB + 0.3×River online
  features : 47 features MLFeatureEngine

   4 proba + microstructure + agreement
        │
        ▼
MLOrchestrator (meta-LGB)
  → P(profit) + BUY/SELL/WAIT + size_factor
  size_factor : P>0.75 → 1.5× | P>0.65 → 1.2× | P<0.60 → 0.75×
  label training : prix > 0.8% en 8 barres (bar-level)
```

---

## HSMM 6 états

| État | Description |
|------|-------------|
| Trend+ | Tendance haussière structurée |
| Range | Consolidation latérale |
| Trend- | Tendance baissière |
| Squeeze | Compression vol — breakout imminent |
| Distribution | Topping institutionnel |
| Liquidation | Flush violent — entrées bloquées |

---

## Features (47 total)

| Catégorie | Features |
|-----------|----------|
| Momentum | mom_4/8/16/32/96 + acceleration |
| Vol classique | rv_8/32/96, parkinson, vol_ratio |
| Vol GARCH | ewma_vol_94, ewma_vol_97, ewma_vol_ratio |
| Order-book proxies | buy_pressure (×3), amihud + z-score, kyle_lambda, eff_spread (×3), vol_surprise |
| Garman-Klass | gk_24h |
| Risk-adjusted | sharpe_8/32, sortino_8/32 (fixés) |
| Indicators | rsi_14/28 (Wilder), macd/signal/hist |
| Volume | vol_change, vol_ratio_6_24 |
| Saisonnalité | hour_sin/cos, dow_sin/cos |
| Context | context float |

---

## PrecomputedRunner (40× speedup)

Avant : 70ms/barre (HSMM 22ms + SMC 21ms + prep 14ms + autres)
Après : ~1ms/barre

```
Phase precompute (~30s pour 3 mois) :
  streaming forward causal O(T·N²) total (équivalent forward-backward)
  SMC uniquement sur le window backtest (2880 barres, pas 151K)
  SMA200 vectorisé pandas

Hot loop :
  O(1) numpy row lookup
  decision = states.decide_fast(aligned_idx[bar_i], current_price)
```

---

## Gestionnaire de risque

```
dollar_risk  = capital × 2%
stop_dist    = max(k × ATR(50), price × 1.5%)
notional     = min(dollar_risk / stop_dist × price, capital × 10×)
notional    *= size_factor  (MLOrchestrator)

Trail stop   : suit le prix (jamais en arrière)
Fixed TP     : entry ± 2.5 × k × ATR  (ratio 2.5:1)
Daily DD     : kill switch à -5% vs capital du matin
Cooldown     : 96 barres (24h) après chaque clôture
```

Filtres post-signal : score, momentum 15M, pullback 0.6%, volume 1.2×.

---

## Résultats backtests (pass-through — agents ML non entraînés)

| Période | Trades | Return | Sharpe | Sortino | MaxDD | PF |
|---------|--------|--------|--------|---------|-------|----|
| Mars 2023 | 4 L / 0 S | +11.48% | 7.03 | 2.43 | 3.41% | 9.11 |
| Q1 2023 | 11 L / 0 S | +13.03% | 3.82 | 1.48 | 6.94% | 4.02 |

BTC B&H Q1 2023 : +72.21% (bull run post-FTX)
Runtime Q1 2023 : 2.4s (precomputed)

---

## Bugs corrigés

| Bug | Impact | Fix |
|-----|--------|-----|
| else:SHORT dans backtest | 184 SHORTs en bull year | if action not in ('BUY','SELL'): pass |
| RSI = SMA | RSI biaisé | ewm(alpha=1/period) Wilder |
| Sortino broken | NaN-heavy | clip(upper=0).pow(2).rolling().mean().pow(0.5) |

---

## Fichiers clés

| Fichier | Description |
|---------|-------------|
| src/core/hsmm.py | Semi-Markov HMM (forward-backward, EM) |
| src/core/smc.py | SMC (Order Blocks, FVG, CHoCH) |
| src/core/precomputed_runner.py | Streaming forward + PrecomputedStates |
| src/agents/contracts.py | AgentResult + OrchestratorDecision |
| src/agents/orchestrator.py | Pipeline séquentiel 4 agents |
| src/ml/feature_engine.py | 47 features + IncrementalFeatureEngine |
| src/ml/ml_agents.py | MLContextAgent, MLRegimeAgent, MLSetupAgent |
| src/ml/ml_entry_agent.py | LGB batch + River online |
| src/ml/ml_orchestrator.py | Meta-LGB + size_factor |
| src/ml/model_monitor.py | KS drift + calibration + AUC |
| scripts/backtest_mtf.py | Runner MTF (--precompute, --use-cache) |
| scripts/train_ml_ecosystem.py | Pipeline entraînement 8 étapes |

---

## Jesse ML Integration (Avril 2026)

Architecture parallèle Jesse avec 5 agents, 100+ tests TDD.
Voir `docs/JESSE_ARCHITECTURE.md` pour le détail.

### Modules Jesse

| Fichier | Rôle |
|---------|------|
| src/ml/jesse_agents.py | 5 agents Jesse (Context/Regime/Setup/Entry/Orchestrator) |
| src/ml/jesse_features.py | 24 features stationnaires (ratios, jamais bruts) |
| src/ml/jesse_labeler.py | Triple barrier (+1/-1/0) |
| src/ml/jesse_strategy.py | Gather/Deploy + RandomForest |
| src/ml/jesse_backtest.py | FastBacktester vectorisé (128K bars/sec) |
| src/ml/edge_strategy.py | Edge walk-forward validé |
| src/ml/feedback_loop.py | DecisionLogger + OutcomeEvaluator + ChampionChallenger |
| src/ml/jesse_research.py | Feature importance 4 méthodes |
| src/ml/jesse_utils.py | risk_to_qty, crossed, kelly |

### Edge validé (walk-forward, 3 folds annuels)

| Edge | WR | EV/trade | Quarters+ | Anti-overfit |
|------|-----|----------|-----------|-------------|
| Trend + Volume >1.5x | 43% | +0.079 ATR | 14/14 | **Validé** |
| + Hours 8-18 UTC | 46% | +0.154 ATR | 14/14 | **Validé** |

**⚠ REALITY CHECK** : Les PnL bruts sont gonflés.
Sans fees, slippage, sizing réaliste, le return est estimé à 10-30%/an.
Voir `docs/REALITY_CHECK.md`.

### Tests TDD

| Suite | Tests |
|-------|-------|
| Pipeline ML | 18 |
| Integration Jesse | 23 |
| FastBacktester | 14 |
| 5 Agents (Context/Regime/Setup/Entry/Orchestrator) | 45 |
| Feedback Loop | 13 |
| Edge Walk-Forward | 12 |
| Yearly WF + OOS | 11 |
| **Total** | **136** |

---

## Prochaines étapes

| Priorité | Tâche | Status |
|----------|-------|--------|
| P0 | Ajouter fees + slippage au backtest | TODO |
| P0 | Position sizing réaliste (% capital) | TODO |
| P1 | Max 3 trades/jour + cooldown | TODO |
| P1 | Backtest en dollars nets avec equity curve | TODO |
| P1 | ML filtrage (réduire de 8 trades/j à 2-3) | TODO |
| P2 | HSMM probs des parquets comme features ML | TODO |
| P2 | Paper trading live avec DecisionLogger | TODO |
| P2 | Données order book L2 réelles | TODO |
