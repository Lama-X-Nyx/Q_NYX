# TDD Test Registry — 100 Tests

> Tous les tests Jesse ML et leur statut.

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
| **Total** | | **100** | **GREEN** |

## Détail par suite

### test_jesse_ml_tdd.py (18)
Validation TDD synthétique — si >85% accuracy sur données synthétiques → code correct.

- TestTripleBarrierLabeler (4) : labels +1 bull, -1 bear, 0 range, longueur
- TestStationaryFeatures (5) : borné, indicateurs requis, EMA ratio ~0, NaN, invariance prix
- TestGatherDeployModes (3) : dataset (X,y), predictions, seuil confiance
- TestSyntheticTDDValidation (2) : >85% accuracy synthétique, cross-price $1K↔$50K
- TestFeatureImportance (2) : dict retourné, top features significatives
- TestABComparison (2) : métriques valides pour 2 stratégies

### test_jesse_integration.py (23)
Features étendues, utilities, research ML.

- TestExtendedFeatures (8) : 24+ colonnes, features présentes, bornées, NaN, ADX [0,1], BB, squeeze
- TestJesseUtils (10) : risk_to_qty, size_to_qty, crossed above/below/sequential, kelly, anchor_timeframe, streaks
- TestJesseResearch (4) : 4-method importance, feature impact, train_model complet, accuracy synthétique
- TestFullPipelineExtended (1) : full features >= core accuracy

### test_jesse_backtest.py (14)
FastBacktester vectorisé sur Q1 2023.

- TestDataAvailability (3) : fichier existe, 8K+ lignes, prix $16K→$28K
- TestPrecomputation (3) : numpy arrays, shape, speed <2s
- TestBacktestSpeed (1) : 8,640 bars < 5 secondes
- TestNoLookAheadBias (1) : trades après train period uniquement
- TestPnLCorrectness (2) : PnL = somme trades, equity >= 0
- TestWalkForward (2) : train/test split correct, train accuracy > 50%
- TestFullPipeline (2) : résultat complet, métriques présentes

### test_jesse_agent_context.py (12)
Agent 1 — filtre directionnel macro (1D).

- TestContextContract (4) : AgentResult, state valide, score [0,1], metadata (p_bull, p_bear)
- TestContextDetection (3) : bullish sur uptrend, bearish sur downtrend, neutre sur range
- TestContextFeatures (2) : stationnaire, momentum + vol présents
- TestContextBacktest (3) : 35%+ bullish sur bull, 40%+ blocked sur bear, métriques complètes

### test_jesse_agent_regime.py (10)
Agent 2 — détection de régime (1H).

- TestRegimeContract (3) : AgentResult, state valide, metadata regime
- TestRegimeDetection (3) : trend_plus sur uptrend, trend_minus sur downtrend, range sur sideways
- TestRegimeFeatures (2) : 6 colonnes HSMM, stationnaire
- TestRegimeBacktest (2) : 30%+ trend_plus sur bull, métriques complètes

### test_jesse_agent_setup.py (6)
Agent 3 — validation de setup (15M).

- TestSetupContract (2) : AgentResult, state valid_setup/no_setup
- TestSetupDetection (2) : score > 0 avec bon contexte, no_setup bloque
- TestSetupFeatures (1) : cross-agent scores (context_score, regime_score, agreement)
- TestSetupBacktest (1) : métriques complètes

### test_jesse_agent_entry.py (8)
Agent 4 — timing d'entrée (15M).

- TestEntryContract (4) : AgentResult, state, direction, probabilités
- TestEntryDetection (2) : score > 0 sur trend, confiance respectée
- TestEntryBacktest (2) : accuracy, distribution directions

### test_jesse_orchestrator.py (9)
Agent 5 — méta-décision.

- TestOrchestratorContract (2) : OrchestratorDecision, action valide
- TestOrchestratorLogic (4) : all passed+long→BUY, all passed+short→SELL, 1 block→WAIT, multi blocks
- TestOrchestratorSizing (3) : score>0.75→1.5x, score<0.60→0.75x, components préservés
