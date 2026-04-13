# Feedback Loop — Paper Trading & Retraining

> Principe : on n'apprend pas dans le trade. On apprend après le trade.

## Architecture

```
┌─────────────────────────────────────────┐
│  A. SESSION (modèle GELÉ)               │
│     Backtest ou Paper Trading           │
│     → log chaque décision (trades + WAIT)│
│     → paper_decisions_log.parquet       │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  B. ÉVALUATION BATCH                    │
│     evaluate_paper_outcomes.py          │
│     → relabellise avec vrai outcome     │
│     → catégorise les erreurs            │
│     → calcule métriques par agent       │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  C. RETRAINING CONTRÔLÉ                 │
│     retrain_from_paper.py               │
│     → train challenger sur historique   │
│       + paper récent                    │
│     → compare champion vs challenger    │
│     → promote seulement si meilleur OOS │
└─────────────────────────────────────────┘
```

## Niveau 1 — paper_decisions_log

Chaque barre 15m, une ligne complète :

### Colonnes obligatoires

| Colonne | Type | Description |
|---------|------|-------------|
| timestamp | datetime | Timestamp de la barre |
| pair | str | BTCUSDT |
| bar_index | int | Index dans la session |
| **Features snapshot** | | |
| features_json | str(json) | Dict complet des features à cet instant |
| **Sorties agents** | | |
| context_state | str | bullish/bearish/neutral |
| context_score | float | [0,1] |
| context_passed | bool | |
| context_p_bull | float | |
| context_p_bear | float | |
| regime_state | str | trend_plus/range/squeeze/... |
| regime_score | float | |
| regime_passed | bool | |
| setup_state | str | valid_setup/no_setup |
| setup_score | float | |
| setup_passed | bool | |
| entry_state | str | ready/not_ready |
| entry_score | float | |
| entry_passed | bool | |
| entry_direction | int | +1/-1/0 |
| entry_p_up | float | |
| entry_p_down | float | |
| **Orchestrator** | | |
| action | str | BUY/SELL/WAIT |
| orch_score | float | |
| size_factor | float | |
| blocked_by | str | CSV des agents bloquants |
| **Contexte marché** | | |
| price | float | Close courant |
| atr | float | ATR(14) |
| volume_ratio | float | vol/vol_ma20 |
| **Si trade pris** | | |
| trade_entry_price | float | NaN si WAIT |
| trade_stop | float | |
| trade_tp | float | |
| trade_direction | int | |
| trade_size | float | |
| **Model version** | | |
| model_version | str | Hash ou ID du modèle gelé |

### Colonnes ajoutées APRÈS (évaluation)

| Colonne | Type | Description |
|---------|------|-------------|
| realized_return_1h | float | Return 1h après décision |
| realized_return_4h | float | |
| realized_return_24h | float | |
| triple_barrier_label | int | +1/-1/0 recalculé avec vrai outcome |
| trade_pnl_net | float | PnL net (fees inclus) — NaN si WAIT |
| max_favorable_excursion | float | Meilleur prix atteint avant exit |
| max_adverse_excursion | float | Pire prix atteint avant exit |
| did_stop_hit | bool | |
| did_tp_hit | bool | |
| was_decision_correct | bool | TP=True, SL=False, WAIT=voir ci-dessous |

### Classification des décisions

| Type | Condition | Description |
|------|-----------|-------------|
| TRUE_POSITIVE | Trade pris + PnL > 0 | Bon trade |
| FALSE_POSITIVE | Trade pris + PnL ≤ 0 | Mauvais trade |
| TRUE_NEGATIVE | WAIT + le move n'existait pas | Bon WAIT |
| FALSE_NEGATIVE | WAIT + un beau move raté | WAIT qui aurait dû trader |

### Taxonomy d'erreurs

| error_type | Description |
|------------|-------------|
| bad_context | Context agent a mal lu la direction macro |
| bad_regime | Regime agent a mal identifié le régime |
| setup_false_positive | Setup a validé un mauvais setup |
| setup_false_negative | Setup a bloqué un bon setup |
| entry_too_early | Entry a signalé trop tôt |
| entry_too_late | Entry a signalé trop tard |
| size_too_large | Size factor trop agressif |
| size_too_small | Size factor trop conservateur |
| wait_should_have_traded | WAIT alors qu'il y avait un move |
| traded_should_have_waited | Trade pris mais devait attendre |

## Niveau 2 — evaluate_paper_outcomes.py

Calcule après chaque session :
- Win/loss par agent
- Expectancy globale
- Erreur dominante par agent
- Qualité des WAIT (true negative rate)
- Top false positives (pires trades)
- Top false negatives (meilleurs moves ratés)
- Drift detection (distribution features vs train)

## Niveau 3 — retrain_from_paper.py

1. Charge logs paper + historique
2. Ajoute les nouveaux labels
3. Réentraîne les 4 agents + orchestrator
4. Compare champion vs challenger (OOS)
5. Promote seulement si :
   - Accuracy meilleure
   - Drawdown pas pire
   - Calibration meilleure
   - Stable sur 3+ périodes

## Fréquence

| Composant | Fréquence |
|-----------|-----------|
| Logging | Chaque barre (temps réel) |
| Évaluation | Fin de session / quotidien |
| Retraining RF/LGB | Hebdomadaire |
| River online | Post-trade (léger) |
| Promotion | Seulement si improvement OOS |
