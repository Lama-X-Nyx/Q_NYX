"""
MLBacktestAdapter
=================

Drops into backtest_mtf.py as a direct replacement for the rule-based
Orchestrator.  Provides the identical interface:

    adapter.decide(mtf_data, current_price) → OrchestratorDecision

Architecture
------------
The rule-based HSMM agents are kept SOLELY as feature extractors that
produce the per-bar regime-probability (gamma) vector used as input
features by the LightGBM ML agents.  They no longer make trading
decisions — that role is fully handed to the ML pipeline:

    OHLCV  ──▶  HSMM (regime_agent / setup_agent)
                      │  gamma (6-state probs)
                      ▼
               MLContextAgent  ──┐
               MLRegimeAgent   ──┼──▶  MLOrchestrator  ──▶  OrchestratorDecision
               MLSetupAgent    ──┤        (LightGBM
               MLEntryAgent    ──┘         meta-learner)

All ML models are loaded from data/pretrain_cache/ at construction time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

from src.agents.contracts import AgentResult, OrchestratorDecision
from src.ml.ml_agents import MLContextAgent, MLRegimeAgent, MLSetupAgent
from src.ml.ml_entry_agent import MLEntryAgent
from src.ml.ml_orchestrator import MLOrchestrator
import src.ml.ml_orchestrator as _ml_orch_module


# Canonical order must match the HSMM state list in RegimeAgent / SetupAgent
_HSMM_STATE_NAMES = [
    'Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation'
]

_FALLBACK_GAMMA = np.ones(6, dtype=float) / 6.0   # uniform when HSMM not ready


def _hsmm_dict_to_array(hsmm_states: dict) -> np.ndarray:
    """Convert {'Trend+': 0.6, 'Range': 0.3, ...} → ordered (6,) array."""
    arr = np.array(
        [hsmm_states.get(s, 0.0) for s in _HSMM_STATE_NAMES], dtype=float
    )
    total = arr.sum()
    return arr / total if total > 1e-9 else _FALLBACK_GAMMA.copy()


class MLBacktestAdapter:
    """
    Drop-in replacement for the rule-based Orchestrator in backtest_mtf.py.

    Parameters
    ----------
    config : dict
        Same config dict passed to Orchestrator.
    regime_agent : RegimeAgent
        Pre-trained HSMM regime agent (from pretrain_agents()).
        Used as gamma feature extractor only.
    setup_agent : SetupAgent
        Pre-trained HSMM setup agent (from pretrain_agents()).
        Used as gamma + SMC pattern extractor only.
    """

    def __init__(self, config: dict, regime_agent, setup_agent):
        self.config = config

        # HSMM feature extractors (rule-based, already pretrained)
        self.regime_agent = regime_agent
        self.setup_agent  = setup_agent

        fractal = config.get('fractal', {})
        self._context_tf   = fractal.get('context_tf',   '1d')
        self._regime_tf    = fractal.get('regime_tf',    '1h')
        self._setup_tf     = fractal.get('setup_tf',     '15m')
        self._structure_tf = fractal.get('structure_tf', '4h')

        # ML agents — load from data/pretrain_cache/ at construction
        self._ctx    = MLContextAgent()
        self._regime = MLRegimeAgent()
        self._setup  = MLSetupAgent()
        self._entry  = MLEntryAgent(config)
        self._orch   = MLOrchestrator()

        # Explicitly load cached models (agents only auto-load inside pretrain())
        self._load_all_from_cache()

        n_trained = sum([
            self._ctx._trained,
            self._regime._trained,
            self._setup._trained,
            self._entry._lgb_trained,
            self._orch._trained,
        ])
        print(f'  [MLAdapter] {n_trained}/5 ML models loaded from cache')

    # ------------------------------------------------------------------
    # Cache loader
    # ------------------------------------------------------------------

    def _load_all_from_cache(self) -> None:
        """
        Load each ML agent from its pre-trained cache file.

        MLContextAgent / MLRegimeAgent / MLSetupAgent / MLOrchestrator all expose
        a CACHE_FILE class/module attribute and a _load() instance method.
        MLEntryAgent has no pre-trained file; it falls back to pass-through mode
        (returns passed=True, score=0.5) when _lgb_trained=False.
        """
        # MLContextAgent / MLRegimeAgent / MLSetupAgent expose CACHE_FILE as a
        # class attribute; MLOrchestrator defines it at module level.
        agents = [
            ('MLContextAgent',  self._ctx,    MLContextAgent.CACHE_FILE),
            ('MLRegimeAgent',   self._regime, MLRegimeAgent.CACHE_FILE),
            ('MLSetupAgent',    self._setup,  MLSetupAgent.CACHE_FILE),
            ('MLOrchestrator',  self._orch,   _ml_orch_module.CACHE_FILE),
        ]
        for name, agent, cache_file in agents:
            try:
                if cache_file.exists():
                    agent._load()
                else:
                    print(f'  [MLAdapter] WARNING: {name} cache not found at {cache_file}')
            except Exception as exc:
                print(f'  [MLAdapter] WARNING: {name} _load() failed — {exc}')

    # ------------------------------------------------------------------
    # Public API — identical to Orchestrator.decide()
    # ------------------------------------------------------------------

    def decide(self,
               mtf_data: Dict[str, pd.DataFrame],
               current_price: Optional[float] = None) -> OrchestratorDecision:
        """
        Run the full ML pipeline and return a trading decision.

        Steps
        -----
        1. Extract HSMM gamma from pre-trained HSMM agents (feature only).
        2. Run all 4 ML agents on their respective timeframe data.
        3. Run MLOrchestrator meta-model for the final BUY/SELL/WAIT signal.
        """
        df_1d  = mtf_data.get(self._context_tf, pd.DataFrame())
        df_1h  = mtf_data.get(self._regime_tf,  pd.DataFrame())
        df_15m = mtf_data.get(self._setup_tf,   pd.DataFrame())
        df_4h  = mtf_data.get(self._structure_tf, None)

        if current_price is None and not df_15m.empty:
            current_price = float(df_15m['close'].iloc[-1])

        # ------------------------------------------------------------------
        # Step 1 — HSMM gamma extraction (rule-based, no decision authority)
        # ------------------------------------------------------------------
        hsmm_1h  = _FALLBACK_GAMMA.copy()
        hsmm_15m = _FALLBACK_GAMMA.copy()
        smc_pats: dict = {}

        try:
            rb_regime = self.regime_agent.analyze(
                df_1h, context_state=None, df_htf=df_4h
            )
            hsmm_1h = _hsmm_dict_to_array(
                rb_regime.metadata.get('hsmm_states', {})
            )
        except Exception:
            pass

        try:
            rb_setup = self.setup_agent.analyze(
                df_15m, context_state=None, df_htf=df_1h
            )
            hsmm_15m = _hsmm_dict_to_array(
                rb_setup.metadata.get('hsmm_states', {})
            )
            smc_pats = rb_setup.metadata.get('patterns', {})
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Step 2 — ML agents
        # ------------------------------------------------------------------
        ctx_result    = self._ctx.analyze(df_1d)
        regime_result = self._regime.analyze(df_1h, hsmm_1h)
        setup_result  = self._setup.analyze(
            df_15m, hsmm_15m, smc_pats, regime_result, ctx_result
        )
        entry_result  = self._entry.analyze(
            df_15m,
            setup_state=setup_result.state,
            context_state=ctx_result.state
        )

        # ------------------------------------------------------------------
        # Step 3 — MLOrchestrator meta-decision
        # ------------------------------------------------------------------
        df_15m_row = df_15m.iloc[-1] if not df_15m.empty else pd.Series(dtype=float)
        return self._orch.decide(
            ctx_result, regime_result, setup_result, entry_result,
            df_15m_row, current_price or 0.0
        )

    # Expose regime_agent so backtest can still access hsmm for diagnostics
    # (e.g. risk manager's emission_params in the rule-based risk check)
    # In ML mode the risk manager is bypassed, but the attribute must exist.
    @property
    def context_agent(self):
        return self._ctx

    @property
    def entry_agent(self):
        return self._entry
