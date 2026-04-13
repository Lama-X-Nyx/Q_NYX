# TDD Test Registry — 200 Tests GREEN

> NYX v2.5.0 — Tous les tests et leur statut.

## Résumé

| Suite | Fichier | Tests | Status |
|-------|---------|-------|--------|
| Pipeline TDD | test_jesse_ml_tdd.py | 18 | GREEN |
| Integration | test_jesse_integration.py | 23 | GREEN |
| FastBacktester | test_jesse_backtest.py | 14 | GREEN |
| Agent 1 Context | test_jesse_agent_context.py | 12 | GREEN |
| Agent 2 Regime | test_jesse_agent_regime.py | 10 | GREEN |
| Agent 3 Setup | test_jesse_agent_setup.py | 6 | GREEN |
| Agent 4 Entry | test_jesse_agent_entry.py | 8 | GREEN |
| Agent 5 Orchestrator | test_jesse_orchestrator.py | 9 | GREEN |
| Feedback Loop | test_feedback_loop.py | 13 | GREEN |
| Edge Walk-Forward | test_edge_walkforward.py | 12 | GREEN |
| Yearly WF + OOS | test_yearly_walkforward.py | 11 | GREEN |
| Realistic Backtest | test_realistic_backtest.py | 15 | GREEN |
| Soft Gate | test_soft_gate.py | 18 | GREEN |
| ML Filter v1 | test_ml_filter.py | 12 | GREEN |
| ML Filter v2 | test_ml_filter_v2.py | 10 | GREEN |
| Threshold Optimizer | test_threshold_optimizer.py | 7 | GREEN |
| **Total** | **16 fichiers** | **200** | **GREEN** |

## Catégories de tests

### Validation synthétique (TDD core)
- Données synthétiques avec patterns évidents → >85% accuracy = code correct
- Invariance au prix ($1K et $50K → même features)
- Features stationnaires (bornées, pas de NaN après warmup)

### Contrats agents
- Chaque agent retourne `AgentResult` valide
- States, scores, metadata conformes aux specs
- Backtest isolé par agent (détecte les incohérences)

### Anti-overfit
- Walk-forward : train/test jamais overlap
- Survit au bear 2022
- Aucun quarter ne domine >40% du PnL
- Consistant première/deuxième moitié des données

### Réalisme
- Fees tracked et déduites du PnL
- Position sizing capped (1x capital)
- Max trades/jour respecté
- Cooldown entre trades
- Equity jamais négative

### Feedback loop
- Décisions logguées (BUY + WAIT)
- Outcomes évalués post-session (TP/FP/TN/FN)
- Taxonomy d'erreurs (10 types)
- Champion/Challenger : promote seulement si meilleur
