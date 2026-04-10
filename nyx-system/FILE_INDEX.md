# NYX v1.0 — Index des fichiers

> Mis à jour : Avril 2026

---

## Scripts principaux

| Fichier | Usage | Flags clés |
|---------|-------|------------|
| `scripts/backtest_mtf.py` | Backtest MTF complet | `--start`, `--end`, `--pretrain-all`, `--use-cache`, `--precompute`, `--precompute-cache` |
| `scripts/train_ml_ecosystem.py` | Entraînement ML ecosystem | `--train-end`, `--em-iters`, `--n-splits`, `--force-retrain` |

---

## Source — Agents

| Fichier | Rôle |
|---------|------|
| `src/agents/contracts.py` | AgentResult + OrchestratorDecision (contrats) |
| `src/agents/orchestrator.py` | Pipeline séquentiel Context → Regime → Setup → Entry |
| `src/agents/context_agent.py` | SMA200 rule-based (passthrough pour MLContextAgent) |
| `src/agents/regime_agent.py` | HSMM 1H (proba → features pour MLRegimeAgent) |
| `src/agents/setup_agent.py` | HSMM 15M + SMC (proba → features pour MLSetupAgent) |
| `src/agents/entry_agent.py` | EntryAgent stub (remplacé par MLEntryAgent) |

---

## Source — Core

| Fichier | Rôle |
|---------|------|
| `src/core/hsmm.py` | Semi-Markov HMM — forward-backward, Baum-Welch EM, 6 états |
| `src/core/smc.py` | SMC detector — Order Blocks, Fair Value Gaps, CHoCH, BOS |
| `src/core/precomputed_runner.py` | Streaming forward causal + PrecomputedStates (40×) |
| `src/core/risk_manager_mtf.py` | Sizing vol-adjusted, R:R check, Daily DD kill switch |
| `src/core/nyx_engine_mtf.py` | Engine MTF wrapper |

---

## Source — ML Ecosystem

| Fichier | Rôle | État |
|---------|------|------|
| `src/ml/feature_engine.py` | 47 features + IncrementalFeatureEngine | Opérationnel |
| `src/ml/ml_agents.py` | MLContextAgent, MLRegimeAgent, MLSetupAgent | Pass-through (non entraîné) |
| `src/ml/ml_entry_agent.py` | MLEntryAgent LGB batch + River online | Pass-through |
| `src/ml/ml_orchestrator.py` | MLOrchestrator meta-LGB + size_factor | Pass-through |
| `src/ml/model_monitor.py` | KS drift + calibration + rolling AUC | Opérationnel |
| `src/ml/__init__.py` | Exports ML ecosystem | OK |

---

## Caches (data/pretrain_cache/)

| Fichier | Contenu | Généré par |
|---------|---------|-----------|
| `<hash>.pkl` | Params HSMM (EM Baum-Welch) | `backtest_mtf.py --use-cache` |
| `<hash>_v1_precomp.pkl` | PrecomputedStates arrays | `backtest_mtf.py --precompute-cache` |
| `gamma_1h_train.npy` | Forward proba HSMM 1H (training) | `train_ml_ecosystem.py` |
| `gamma_15m_train.npy` | Forward proba HSMM 15M (training) | `train_ml_ecosystem.py` |
| `smc_all_train.pkl` | SMC patterns toutes barres training | `train_ml_ecosystem.py` |
| `context_ml.pkl` | MLContextAgent LGB models | `train_ml_ecosystem.py` |
| `regime_ml.pkl` | MLRegimeAgent LGB model | `train_ml_ecosystem.py` |
| `setup_ml.pkl` | MLSetupAgent LGB model | `train_ml_ecosystem.py` |
| `orchestrator_ml.pkl` | MLOrchestrator meta-LGB | `train_ml_ecosystem.py` |

---

## Données (data/raw/mtf/)

| Fichier | Barres | Période |
|---------|--------|---------|
| `BTCUSDT_1d.csv` | ~1577 | 2019-09 → 2024 |
| `BTCUSDT_4h.csv` | ~9452 | 2019-09 → 2024 |
| `BTCUSDT_1h.csv` | ~37808 | 2019-09 → 2024 |
| `BTCUSDT_15m.csv` | ~151226 | 2019-09 → 2024 |

---

## Documentation

| Fichier | Contenu |
|---------|---------|
| `README.md` | Quick start, usage, performances |
| `COMPLETE_ARCHITECTURE_SUMMARY.md` | Architecture complète, features, résultats |
| `FILE_INDEX.md` | Ce fichier |
| `docs/FRACTAL_AGENT_ARCHITECTURE.md` | Architecture ML ecosystem détaillée |
| `docs/HSMM_DEEP_DIVE.md` | Détails HSMM, états, calibration |
| `docs/FRACTAL_BOTTLENECK.md` | Analyse perf + stratégie precompute |
| `docs/BULLISH_AUDIT.md` | Audit du bug else:SHORT |
| `docs/*.md` | Notes d'investigation historiques |

---

## Tickets en cours (roadmap)

| Ticket | Description | Priorité |
|--------|-------------|----------|
| Ticket 1 | Feature engineering + fenêtres contexte | Fait (47 features) |
| Ticket 2 | Triple Barrier labelling | A faire |
| Ticket 3 | Walk-Forward Splitter dédié | A faire |
| Ticket 4 | Entraînement agents ML niveau 1 | A faire (pipeline prêt) |
| Ticket 5 | Entraînement MLOrchestrator stacking | A faire (pipeline prêt) |
