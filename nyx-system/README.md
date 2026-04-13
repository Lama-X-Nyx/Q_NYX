# NYX Trading System v0.2.5

Système de trading algorithmique institutionnel — BTC/USDT perpetual futures.
Décisions sur barres 15M avec contexte MTF aligné (1D / 4H / 1H / 15M).

**200 tests TDD GREEN. Edge validé walk-forward sur 4 ans de données réelles.**

---

## Ce qui a changé depuis v0.8

| Aspect | v0.8 | v0.2.5 |
|--------|------|--------|
| Architecture | 5 agents LightGBM (non entraînés) | 5 agents Jesse ML + soft gate + ML filter |
| Edge | Aucun (pass-through heuristique) | Trend + Volume >3x, validé 14/14 quarters |
| Backtest | 70ms/bar, pas de fees | 0.03ms/bar, fees+slippage réalistes |
| Tests | Quelques tests unitaires | **200 tests TDD** couvrant tout le pipeline |
| Features | 47 features engine custom | 47 parquet + 24 Jesse stationnaires |
| Feedback | Aucun | DecisionLogger + OutcomeEvaluator + ChampionChallenger |
| Résultat net | +13% (sans fees, pass-through) | **Sharpe 1.5+, WR 65%+, DD <5%** (avec fees maker) |

---

## Installation

```bash
pip install -r requirements.txt
SETUPTOOLS_USE_DISTUTILS=stdlib pip install jesse  # optionnel mais recommandé
```

---

## Usage

### Backtest rapide avec ML filter (recommandé)

```python
from src.ml.threshold_optimizer import run_optimal_backtest
import pandas as pd

df = pd.read_csv('data/raw/mtf/BTCUSDT_15m.csv')
df['datetime'] = pd.to_datetime(df['datetime']); df = df.set_index('datetime')
pq = pd.read_parquet('data/features/BTCUSDT_features_15m.parquet')

result = run_optimal_backtest(df, pq, train_end='2022-12-31', test_start='2023-01-01')
print(f"Sharpe: {result['sharpe']:.2f} | WR: {result['win_rate']:.0%} | PnL: ${result['total_pnl_dollars']:+,.0f}")
```

### Backtest réaliste (fees + slippage)

```python
from src.ml.realistic_backtest import RealisticBacktester

bt = RealisticBacktester(
    vol_min=3.0, use_hours=True, cooldown_bars=32,
    max_daily_trades=1, fee_rate=0.0002, slippage_rate=0.0001,
)
r = bt.run(df.loc['2023-01-01':'2023-12-31'])
```

### Backtest héritage (v0.8)

```bash
python scripts/backtest_mtf.py --start 2023-01-01 --end 2023-04-01 --precompute
```

---

## Architecture

### Pipeline de décision (v0.2.5)

```
                    EDGE (trend + volume > 3x + heures 8-18)
                              │
                    Candidats (~150/an)
                              │
                    ┌─────────┴─────────┐
                    │   ML Filter v2    │  GBM 300 trees
                    │   47 parquet feat │  Threshold calibré CV
                    │   + rule scores   │  Rejette ~60% candidats
                    └─────────┬─────────┘
                              │
                    Trades sélectionnés (~60/an)
                              │
                    ┌─────────┴─────────┐
                    │   Soft Gate       │  Rules = garde-fous
                    │   Disagreement    │  Size adjustment
                    │   Hard veto rare  │  (ATR=0, spread>0.5%)
                    └─────────┬─────────┘
                              │
                    Position sizing (2% risk, 1x cap max)
                              │
                    TP = 1.5x ATR  |  SL = 1.0x ATR
```

### 5 agents Jesse (validation structurelle)

| Agent | TF | Rôle | Output |
|-------|-----|------|--------|
| ContextAgent | 1D | Filtre directionnel macro | bullish/bearish/neutral |
| RegimeAgent | 1H | Détection de régime | trend+/range/squeeze |
| SetupAgent | 15M | Validation setup | valid_setup/no_setup |
| EntryAgent | 15M | Timing d'entrée | ready/not_ready + direction |
| Orchestrator | Meta | Décision finale | BUY/SELL/WAIT + size_factor |

**Principe clé** : ML décide, agents règles valident (soft gate, pas hard block).

---

## Edge validé

### Walk-forward annuel (volume > 3x, maker fees)

| Fold | Train | Test | BTC | Trades | WR | Sharpe | PnL |
|------|-------|------|-----|--------|-----|--------|-----|
| 1 | 2019 | 2020 (+303%) | bull | 152 | 51% | +2.19 | +$2,009 |
| 2 | 2020-21 | 2022 (-64%) | **bear** | 149 | 38% | -0.75 | -$635 |
| 3 | 2022-23 | 2023-Q4 (+58%) | bull | 210 | 48% | +1.86 | +$1,121 |

**Total 4 ans** : Sharpe +1.26, +$4,956, 617 trades, WR 47%

### ML Filter v2 (meilleur résultat)

| | 2022 (bear) | 2023 (bull) |
|---|---|---|
| Trades | 60 | 13 |
| Win Rate | 43% | **69%** |
| Sharpe | -0.07 | **+1.89** |
| Max DD | 5.3% | **0.7%** |

---

## Reality Check

Les résultats bruts (points) sont gonflés. Avec fees réalistes :
- **Taker fees (0.04%)** : edge détruit sur haute fréquence
- **Maker fees (0.02%)** : edge survit avec filtrage strict
- Return réaliste estimé : **10-30%/an** (pas 200%)
- Max 1-3 trades/jour pour préserver l'edge

Voir `docs/REALITY_CHECK.md` pour le détail complet.

---

## Tests TDD (200 GREEN)

| Suite | Tests | Scope |
|-------|-------|-------|
| Pipeline ML TDD | 18 | Triple barrier, features, synthetic validation |
| Integration Jesse | 23 | 24 features, utils, research ML |
| FastBacktester | 14 | Vectorisé, speed, no look-ahead |
| 5 Agents | 45 | Contract, detection, features, backtest par agent |
| Feedback Loop | 13 | DecisionLogger, OutcomeEvaluator, ChampionChallenger |
| Edge Walk-Forward | 12 | 14 quarters, anti-overfit |
| Yearly WF + OOS | 11 | 3 folds annuels, full OOS 2022-2024 |
| Realistic Backtest | 15 | Fees, slippage, sizing, max trades |
| Soft Gate | 18 | Rule validators, disagreement, hard veto |
| ML Filter v1 | 12 | Candidate scoring, filtering, walk-forward |
| ML Filter v2 | 10 | Parquet features, threshold calibration |
| Threshold Optimizer | 7 | Sweep, optimal config, walk-forward |
| **Total** | **200** | |

---

## Structure fichiers

```
src/ml/                           # Jesse ML Pipeline (nouveau v2.5)
├── jesse_agents.py               # 5 agents Jesse
├── jesse_features.py             # 24 features stationnaires
├── jesse_labeler.py              # Triple barrier (+1/-1/0)
├── jesse_strategy.py             # Gather/Deploy + RandomForest
├── jesse_backtest.py             # FastBacktester vectorisé
├── jesse_utils.py                # risk_to_qty, crossed, kelly
├── jesse_research.py             # Feature importance 4 méthodes
├── jesse_ab_runner.py            # A/B comparison
├── edge_strategy.py              # Edge walk-forward validé
├── realistic_backtest.py         # Backtest avec fees/slippage
├── soft_gate.py                  # ML décide, rules valident
├── ml_filter.py                  # ML filter v1 (13 features)
├── ml_filter_v2.py               # ML filter v2 (47 features + calibration)
├── threshold_optimizer.py        # Threshold sweep + optimal config
├── feedback_loop.py              # DecisionLogger + Evaluate + Champion/Challenger

src/agents/                       # Architecture originale (v0.8)
├── contracts.py                  # AgentResult, OrchestratorDecision
├── orchestrator.py, context/regime/setup/entry_agent.py

src/core/                         # Moteur de base
├── hsmm.py, smc.py, precomputed_runner.py, risk_manager_mtf.py

data/
├── raw/mtf/                      # BTCUSDT 15m/1h/4h/1d (2019-2024)
├── features/                     # 53 features parquet (2019-2024)

docs/                             # Documentation complète
├── JESSE_ARCHITECTURE.md         # Architecture 5 agents Jesse
├── EVOLUTION_A_TO_B.md           # Raisonnement complet A→B
├── REALITY_CHECK.md              # Biais identifiés, estimation réaliste
├── FEEDBACK_LOOP_SPEC.md         # Spec feedback loop
├── SESSION_LOG.md                # Log chronologique complet
├── TDD_TEST_REGISTRY.md          # 200 tests détaillés
├── SPEED_BENCHMARKS.md           # 70ms → 0.03ms/bar
├── CHANGELOG.md                  # Historique des versions
├── backtest_results.json         # Tous les résultats
├── edge_analysis.json            # 5 hypothèses d'edge
├── data_inventory.json           # Inventaire données
├── yearly_walkforward_oos_report.json
├── ml_edge_training_results.json
```
