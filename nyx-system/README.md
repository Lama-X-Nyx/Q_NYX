# NYX Trading System v1.0

Système de trading algorithmique institutionnel — BTC/USDT perpetual futures.  
Décisions sur barres 15M avec contexte MTF aligné (1D / 4H / 1H / 15M). Aucun look-ahead.

---

## Principes de design

| Principe | Implémentation |
|----------|----------------|
| No look-ahead | `searchsorted` strict avant chaque barre |
| Un agent = un TF = une question | Context 1D / Regime 1H / Setup 15M / Entry 15M |
| HSMM = feature extractor | Les proba HSMM sont des features ML, pas des décideurs |
| Precompute max | O(T·N²) total au lieu de O(T·window·N²) par barre |
| Online learning | River LogisticRegression après chaque trade résolu |

---

## Installation

```bash
pip install -r requirements.txt
python -c "import lightgbm, river, scipy; print('OK')"
```

---

## Usage

### Backtest rapide (mode précompilé)

```bash
python scripts/backtest_mtf.py \
    --start 2023-01-01 --end 2023-04-01 \
    --pretrain-all --use-cache \
    --precompute --precompute-cache
```

| Flag | Description |
|------|-------------|
| `--pretrain-all` | EM Baum-Welch sur tout l'historique avant `--start` |
| `--use-cache` | Charge/sauve les params HSMM depuis `data/pretrain_cache/` |
| `--precompute` | Active le PrecomputedRunner (streaming forward + SMC rolling) |
| `--precompute-cache` | Cache le résultat precompute sur disque |

### Entraîner le ML ecosystem

```bash
# Entraîne les 4 agents ML + orchestrateur sur 2019-2022
python scripts/train_ml_ecosystem.py --train-end 2022-12-31

# Avec force-retrain (ignore les caches)
python scripts/train_ml_ecosystem.py --train-end 2022-12-31 --force-retrain
```

---

## Architecture

### Couches de décision

```
1D  →  MLContextAgent   → P(bullish / bearish / neutral)
1H  →  MLRegimeAgent    → P(trend+) + HSMM 6 états comme features
15M →  MLSetupAgent     → P(valid_setup) + SMC + scores agents amont
15M →  MLEntryAgent     → P(entry_ok) LGB + River online
         │
         ▼  meta-features (4 proba + microstructure + agreement)
     MLOrchestrator  →  P(profit) + BUY/SELL/WAIT + size_factor
```

### HSMM 6 états

| État | Description |
|------|-------------|
| `Trend+` | Tendance haussière structurée |
| `Range` | Consolidation / marché latéral |
| `Trend-` | Tendance baissière structurée |
| `Squeeze` | Volatilité compressée (pre-breakout) |
| `Distribution` | Distribution institutionnelle (topping) |
| `Liquidation` | Flush violent — entrées bloquées |

### Features (47 total)

**Momentum** : mom_4/8/16/32/96 barres + acceleration  
**Volatilité** : RV rolling, Parkinson, Garman-Klass, EWMA λ=0.94/0.97, vol ratio  
**Order-book proxies** : Amihud illiquidity (+ z-score), Kyle's lambda, buy pressure, eff. spread ratio, vol surprise  
**Risk-adjusted** : Sharpe/Sortino rolling (Wilder's EMA, downside RMS)  
**Momentum indicators** : RSI Wilder's EMA, MACD normalisé  
**Saisonnalité** : hour_sin/cos, dow_sin/cos  
**Context** : float direction encodé

### PrecomputedRunner (40× speedup)

| Étape | Méthode |
|-------|---------|
| HSMM 1H + 15M | Streaming causal forward O(T·N²) total |
| SMC 15M | Restreint au window de backtest uniquement |
| SMA200 1D | Vectorisé pandas sur dataset complet |
| Hot loop | O(1) lookup numpy row |

**Résultat** : 2.4s pour 8736 barres (Q1 2023) vs ~4min en mode standard.

---

## Performances backtests (pass-through — agents ML non entraînés)

### Q1 2023

```
Capital        $10,000 → $11,303    (+13.03%)
BTC B&H        +72.21%

Sharpe   3.82  ✅    Sortino  1.48  ⚠️
MaxDD    6.94% ✅    Profit Factor  4.02  ✅

Trades : 11 LONGs / 0 SHORTs
Win Rate : 54.5%  |  Gain moyen $319 / Perte moyenne $95
```

### Mars 2023

```
4 LONGs / 0 SHORTs  |  +11.48%  |  Sharpe 7.03  |  MaxDD 3.41%
Runtime : 1.2s pour 3072 barres
```

---

## Corrections critiques appliquées

| Bug | Symptôme | Correction |
|-----|----------|-----------|
| `else: SHORT` dans backtest_mtf.py | 184 SHORTs en année bull | Guard `if action not in ('BUY','SELL'): pass` |
| RSI = SMA | RSI biaisé | Wilder's EMA `ewm(alpha=1/period)` |
| Sortino = `rolling.std()` subset | NaN-heavy | `clip(upper=0).pow(2).rolling().mean().pow(0.5)` |

---

## Structure fichiers

```
src/
├── agents/
│   ├── contracts.py         # AgentResult, OrchestratorDecision
│   ├── orchestrator.py      # Orchestrateur pipeline séquentiel
│   ├── context_agent.py     # SMA200 rule-based
│   ├── regime_agent.py      # HSMM 1H
│   └── setup_agent.py       # HSMM 15M + SMC
├── core/
│   ├── hsmm.py              # Semi-Markov HMM — forward-backward, EM
│   ├── smc.py               # Order Blocks, FVG, CHoCH
│   ├── precomputed_runner.py # Streaming forward + PrecomputedStates
│   └── risk_manager_mtf.py  # Sizing vol-adjusted, R:R check
└── ml/
    ├── feature_engine.py    # MLFeatureEngine (47 feat) + Incremental
    ├── ml_agents.py         # MLContextAgent, MLRegimeAgent, MLSetupAgent
    ├── ml_entry_agent.py    # LGB batch + River online
    ├── ml_orchestrator.py   # Meta-LGB + size_factor
    └── model_monitor.py     # KS drift, calibration, rolling AUC

scripts/
├── backtest_mtf.py          # Runner principal MTF
└── train_ml_ecosystem.py    # Pipeline entraînement ML complet

data/
├── raw/mtf/                 # BTCUSDT_1d/4h/1h/15m.csv
└── pretrain_cache/          # HSMM params + ML models
```

---

## Prochaines étapes

1. Entraîner ML ecosystem sur 2019-2022 (`train_ml_ecosystem.py`)
2. Backtest OOS 2023 avec agents ML entraînés
3. Validation 2021/2022 (bull fort + bear fort)
4. Intégration données order book réelles (L2)
