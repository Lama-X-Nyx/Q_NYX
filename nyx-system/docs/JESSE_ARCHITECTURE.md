# Jesse ML Integration — Architecture & Progress

> Dernière mise à jour : 2026-04-13

## Vue d'ensemble

Migration du système NYX vers une architecture 5 agents Jesse ML avec TDD strict.
Chaque agent est validé individuellement avant intégration dans le pipeline.

## Architecture

```
RAW DATA (OHLCV) : 1D / 4H / 1H / 15M
        │
        │  jesse_features.py → 24 features stationnaires (ratios, jamais bruts)
        │  jesse_labeler.py  → triple barrier (+1/-1/0)
        │
        ▼
COUCHE 1 — JesseContextAgent (1D)
  → bullish / bearish / neutral
  Features : momentum 5/10/20d, realized vol, RSI, Amihud, EMA20/50
  Label    : 5-day net return > adaptive threshold
  Decision : 50% ML (RF) + 50% heuristic (EMA + momentum)

COUCHE 2 — JesseRegimeAgent (1H)
  → trend_plus / range / trend_minus / squeeze
  Features : momentum 4/12/48h + vol + buy_pressure + ADX + 6 HSMM probs
  Label    : max high > 1% dans 4 barres
  Decision : ADX + momentum heuristic, ML pour scoring

COUCHE 3 — JesseSetupAgent (15M)
  → valid_setup / no_setup
  Features : 13 jesse features + context_score + regime_score + agreement
  Label    : prix > 0.8% en 8 barres
  Decision : 40% ML + 60% heuristic (momentum + EMA + RSI alignment)

COUCHE 4 — JesseEntryAgent (15M)
  → ready / not_ready + direction (+1/-1)
  Features : 13 jesse features stationnaires
  Label    : triple barrier (+1/-1/0)
  Decision : 60% ML + 40% heuristic, règle vidéo: prob > 0.45 AND prob > opposite + 0.15

COUCHE 5 — JesseOrchestrator (Meta)
  → BUY / SELL / WAIT + size_factor
  Input    : 4 AgentResults
  Logic    : ALL must pass → trade. Any block → WAIT.
  Sizing   : score > 0.75 → 1.5x | > 0.65 → 1.2x | < 0.60 → 0.75x
```

## Contrats

Tous les agents retournent `AgentResult` (src/agents/contracts.py) :
- `agent`: nom ('context', 'regime', 'setup', 'entry')
- `state`: état textuel
- `score`: [0, 1]
- `passed`: bool (gate decision)
- `metadata`: dict avec probabilités et détails

L'orchestrateur retourne `OrchestratorDecision` :
- `action`: 'BUY' / 'SELL' / 'WAIT'
- `score`: [0, 1]
- `blocked_by`: list d'agents bloquants
- `components`: dict des 4 AgentResults
- `risk_analysis`: {'size_factor', 'direction'}

## Features stationnaires (24)

**Principe fondamental** : JAMAIS de valeurs brutes, TOUJOURS des ratios.
`(EMA21 - EMA50) / EMA50` — pas `EMA50` directement.
Garantit l'invariance au niveau de prix ($100 ou $50,000).

### Core (13)
| Feature | Formule | Catégorie |
|---------|---------|-----------|
| ema_ratio_9_21 | (EMA9-EMA21)/EMA21 | Trend |
| ema_ratio_21_50 | (EMA21-EMA50)/EMA50 | Trend |
| close_vs_ema50 | (close-EMA50)/EMA50 | Trend |
| rsi_14 | (RSI-50)/50 | Momentum |
| atr_ratio | ATR/close | Volatilité |
| volume_ratio | volume/EMA(vol,20) | Volume |
| momentum_10 | (close-close[10])/close[10] | Momentum |
| momentum_20 | (close-close[20])/close[20] | Momentum |
| high_low_ratio | (H-L)/close | Structure |
| close_position | (close-L)/(H-L) | Structure |
| returns_1 | pct_change(1) | Returns |
| returns_5 | pct_change(5) | Returns |
| vol_change | volume pct_change | Volume |

### Extended (+11 = 24 total)
| Feature | Formule | Source |
|---------|---------|--------|
| adx_norm | ADX/100 | jesse.indicators |
| macd_hist_ratio | MACD_hist/close | jesse.indicators |
| bb_percent_b | (close-BB_low)/(BB_up-BB_low) | jesse.indicators |
| bb_width_ratio | (BB_up-BB_low)/close | jesse.indicators |
| mfi_norm | (MFI-50)/50 | jesse.indicators |
| obv_slope | (OBV-EMA(OBV))/EMA(OBV) | Volume flow |
| roc_10 | ROC 10 bars | Momentum |
| keltner_position | (close-Kelt_low)/(Kelt_up-Kelt_low) | Volatilité |
| squeeze | BB inside Keltner (0/1) | Volatilité |
| ema_ratio_50_200 | (EMA50-EMA200)/EMA200 | Long-term trend |
| zscore_20 | (close-SMA20)/std20 | Statistical |

## Fichiers

| Fichier | Lignes | Rôle |
|---------|--------|------|
| src/ml/jesse_agents.py | ~500 | 5 agents + base class |
| src/ml/jesse_backtest.py | ~230 | FastBacktester vectorisé |
| src/ml/jesse_features.py | ~290 | 24 features stationnaires |
| src/ml/jesse_labeler.py | ~80 | Triple barrier (+1/-1/0) |
| src/ml/jesse_strategy.py | ~200 | Gather/Deploy + RandomForest |
| src/ml/jesse_ab_runner.py | ~120 | A/B comparison |
| src/ml/jesse_utils.py | ~170 | Position sizing, crossovers, Kelly |
| src/ml/jesse_research.py | ~250 | Feature importance 4 méthodes |

## Dépendances

- Jesse 1.13.11 (177 indicateurs techniques)
- scikit-learn (RandomForest, StandardScaler, metrics)
- numpy, pandas (vectorisation)
- Fallback numpy automatique si Jesse non installé
