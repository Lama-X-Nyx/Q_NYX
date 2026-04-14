# Status of the 5 Jesse Agents

> This is the single source of truth for "what is the relationship
> between `src/ml/jesse_agents.py` and `src/ml/nyx_pipeline.py`?"

## TL;DR

Two systems, same branch, same philosophy, distinct code paths :

| System | Location | What it is | Used in prod? |
|---|---|---|---|
| **5 Jesse agents** | `src/ml/jesse_agents.py` | modular RF-based chain | **no** |
| **NYXPipeline** | `src/ml/nyx_pipeline.py` | monolithic GBM + hard rules | **yes** |

All A/B/C, walk-forward, reality-check, and live-decider numbers in
this branch come from `NYXPipeline`. The 5 Jesse agents are an
**alternative modular architecture** kept as reference — they have 45
GREEN tests but no wiring into the production pipeline.

## The 5 agents

From `src/ml/jesse_agents.py` (683 lines) :

| Agent | Timeframe | Output | Model |
|---|---|---|---|
| `JesseContextAgent`  | 1D  | {bullish, bearish, neutral} + score | RandomForest |
| `JesseRegimeAgent`   | 1H  | {trend_plus, range, trend_minus, squeeze, distribution, liquidation} | RandomForest |
| `JesseSetupAgent`    | 15M | {valid_setup, no_setup} | RandomForest |
| `JesseEntryAgent`    | 15M | {ready, not_ready} | RandomForest |
| `JesseOrchestrator`  | meta | {BUY, SELL, WAIT} + size_factor | rule-based meta |

Contracts: `src/agents/contracts.py` (`AgentResult`, `OrchestratorDecision`).

Tests (all passing, 45 total) :
- `tests/test_jesse_agent_context.py`
- `tests/test_jesse_agent_regime.py`
- `tests/test_jesse_agent_setup.py`
- `tests/test_jesse_agent_entry.py`
- `tests/test_jesse_orchestrator.py`

## NYXPipeline (the production system)

From `src/ml/nyx_pipeline.py` (536 lines) :

- Candidate generation: EMA9>EMA21>EMA50 (or inverse), volume > 3× MA20,
  hour ∈ [6, 20]
- 84-feature vector: 24 (15m) + 17 h1_* + 17 h4_* + 17 d1_* + rule +
  extras (Rule #2 — 4-TF always)
- Single `GradientBoostingClassifier` (n_estimators=300, max_depth=3)
- Threshold 0.60 + conditional bear dial (stricter threshold + smaller
  size in bear-regime bars)
- Soft gate: `rule_context`, `rule_regime`, `rule_setup` → disagreement
  → `compute_size_factor`
- Execution sim: maker fee 0.02 %, slippage 0.01 %, cooldown 32 bars,
  max 1 trade/day, ATR-based position sizing

The live inference path (task b.B) :

- `src/ml/nyx_live_decider.py` — per-bar decider loading the persisted
  artefact + MTF feature stack
- `src/ml/feature_buffer.py` + `src/ml/mtf_feature_stack.py` — rolling
  buffers per TF with aggregation on close boundaries
- `src/assets/nyx_live_pod.py` — SignalPod wrapper for HubSpokeRunner

## How the two systems relate

Both implement the same **trading philosophy** (context → regime →
setup → entry → orchestrate), but via **different code paths** :

### What they share

- `src/ml/jesse_features.py` — stationary feature engine (EMA, ATR,
  RSI, ADX, MACD, Bollinger, MFI, etc.). Both systems compute
  features through this module.
- `src/ml/soft_gate.py` — `compute_disagreement` + validator helpers.
  NYXPipeline uses it directly; the agents' rule scores reuse the
  same conceptual formulae.

### What differs

- **Modularity**: Jesse splits decision across 5 cooperating models;
  NYXPipeline fuses everything into one GBM + hard rule scalars.
- **Training labels**: each Jesse agent trains on its own TF-specific
  label (e.g. `ContextAgent` uses max-return > 3 % over 5 days);
  NYXPipeline trains a single GBM on net-outcome labels.
- **Inference**: Jesse runs 4 models then a meta-rule; NYXPipeline
  runs 1 model + a threshold + 3 hard-rule scalars.
- **Execution**: Jesse's Orchestrator emits `OrchestratorDecision`;
  NYXPipeline emits a trade record directly.

## Why keep the 5 agents if they're not wired?

1. **Preserve 45 GREEN tests** of legitimate prior work on this branch.
2. **Alternative** — if NYXPipeline ever regresses or we want a more
   explainable architecture, the agents are a plausible fallback.
3. **Documentation value** — the agent split makes the "context /
   regime / setup / entry" decomposition explicit in a way the
   monolith doesn't.

## How to wire the 5 agents (if we ever decided to)

To **integrate** the 5 agents in place of (or in parallel to)
NYXPipeline :

1. Build a `FiveAgentPod` class that satisfies the `SignalPod`
   Protocol expected by `HubSpokeRunner`:
   - `symbol: str`
   - `on_bar(bar) -> Signal`
2. In `FiveAgentPod.on_bar`:
   - Update per-TF feature buffers (already exists via
     `MTFFeatureStack`)
   - Run `JesseContextAgent.analyze(df_1d_tail)`
   - Run `JesseRegimeAgent.analyze(df_1h_tail)`
   - Run `JesseSetupAgent.analyze(df_15m_tail, context_score, regime_score)`
   - Run `JesseEntryAgent.analyze(df_15m_tail, all_features)`
   - Run `JesseOrchestrator.decide(context, regime, setup, entry)`
   - Convert `OrchestratorDecision` into a `Signal`
3. Train each agent offline once (analog of `train_asset_model.py`
   for the monolith) and persist artefacts under
   `models/<SYMBOL>/agents/`:
   - `context.pkl`, `regime.pkl`, `setup.pkl`, `entry.pkl`
4. Wire into `HubSpokeRunner(pods=[FiveAgentPod('ETHUSDT', ...), ...])`.
5. Re-run validate_abc_via_hubspoke.py with the agent pod — compare
   results to the NYXPipeline-based pod.

**This work is NOT currently planned.** It is documented here so a
future session has a concrete starting point if the decision flips.

## Decision criteria

We would flip to the 5-agent architecture if one of the following
happened :

- NYXPipeline's monolithic GBM showed signs of overfitting on an
  out-of-sample window, while the modular agents maintained edge.
- A specific sub-problem (e.g. regime detection) needed a bespoke
  model that the monolith can't accommodate cleanly.
- Interpretability became a hard requirement and the monolith's
  single GBM wasn't explainable enough.

Until then, **the status stays alternative** and this doc is the
clear acknowledgement that both systems exist by design, not by
accident.
