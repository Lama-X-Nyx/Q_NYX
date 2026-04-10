# Q_NYX — NYX Trading System v1.0

Système de trading algorithmique institutionnel pour BTC/USDT perpetual futures.  
**Décision sur barres 15M — No look-ahead — 4 TF alignés (1D / 4H / 1H / 15M)**

---

## Architecture ML Ecosystem

```
Raw OHLCV  (1D / 4H / 1H / 15M)
       │
       ▼  47 features : momentum, vol GARCH, OB proxies, saisonnalité
┌──────────────────────────────────┐
│       MLFeatureEngine            │
└──────────────┬───────────────────┘
               │  HSMM + SMA200 → feature extractors (pas décideurs)
    ┌──────────┼──────────────────────┐
    ▼          ▼                      ▼
MLContextAgent  MLRegimeAgent    MLSetupAgent    MLEntryAgent
  (1D daily)     (1H + HSMM)    (15M+HSMM+SMC)   (15M)
  P(bull/bear)   P(trend)        P(setup)         P(entry)
    └──────────┬──────────────────────┘
               │  4 proba + micro-features
               ▼
       ┌───────────────────┐
       │   MLOrchestrator  │  meta-LGB
       │   → P(profit)     │  → size_factor 0.75–1.5×
       │   → BUY/SELL/WAIT │  → direction
       └───────────────────┘
```

**Clé** : chaque agent ML apprend *quelles combinaisons* de signaux HSMM + microstructure prédisent vraiment des trades profitables — y compris les interactions non-linéaires que les seuils règles ratent.

---

## Résultats (pass-through, agents ML non entraînés)

| Période | Barres | Trades | Return | Sharpe | MaxDD | Win Rate | Profit Factor |
|---------|--------|--------|--------|--------|-------|----------|---------------|
| Mars 2023 | 3072 | 4 L / 0 S | **+11.48%** | 7.03 | 3.41% | 75% | 9.11 |
| Q1 2023 | 8736 | 11 L / 0 S | **+13.03%** | 3.82 | 6.94% | 54.5% | 4.02 |

*BTC Buy-and-Hold Q1 2023 : +72.21% (bull run post-FTX — le système filtre, ne chase pas)*  
*Mode précompilé : 2.4s pour 8736 barres vs ~4min standard (40× speedup)*

---

## Quick Start

```bash
cd nyx-system

# Backtest Q1 2023 (mode rapide)
python scripts/backtest_mtf.py \
    --start 2023-01-01 --end 2023-04-01 \
    --pretrain-all --use-cache --precompute

# Entraîner le ML ecosystem sur 2019-2022
python scripts/train_ml_ecosystem.py --train-end 2022-12-31

# Backtest avec ML entraîné
python scripts/backtest_mtf.py \
    --start 2023-01-01 --end 2023-12-31 \
    --pretrain-all --use-cache --precompute
```

---

## Structure

```
nyx-system/
├── src/
│   ├── agents/          # Orchestrateur + agents HSMM (contrats AgentResult)
│   ├── core/
│   │   ├── hsmm.py              # Semi-Markov HMM (6 états)
│   │   ├── smc.py               # Smart Money Concepts (OB, FVG, CHoCH)
│   │   └── precomputed_runner.py # Streaming forward + SMC rolling (40×)
│   └── ml/
│       ├── feature_engine.py    # 47 features (vol, OB proxies, saisonnalité)
│       ├── ml_agents.py         # MLContextAgent, MLRegimeAgent, MLSetupAgent
│       ├── ml_entry_agent.py    # MLEntryAgent (LGB + River online)
│       ├── ml_orchestrator.py   # Meta-learner MLOrchestrator
│       └── model_monitor.py     # Drift + calibration + AUC monitoring
├── scripts/
│   ├── backtest_mtf.py          # Runner principal (--precompute, --use-cache)
│   └── train_ml_ecosystem.py    # Pipeline d'entraînement ML complet
├── data/
│   ├── raw/mtf/                 # OHLCV CSV (1D, 4H, 1H, 15M)
│   └── pretrain_cache/          # HSMM params + ML models
└── docs/                        # Notes d'investigation
```
