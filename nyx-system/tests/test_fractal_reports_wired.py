"""
TDD Tests — 4 Jesse FractalReports wired into the canonical runtime
(Ticket 13).

Pre-Ticket-13, `NYXEngine._generate_candidates` emitted
`rule_context / rule_regime / rule_setup / disagreement` as HAND-
CRAFTED proxy scalars. The 4 Jesse agents (Ticket 05) existed +
emitted `FractalReport` via `.report()` but were NEVER called by
the runtime. Ticket 13 closes that loop :

- `NYXEngine` instantiates the 4 per-file Jesse agents
  (`ContextAgent`, `RegimeAgent`, `SetupAgent`, `EntryAgent`) in
  its constructor.
- For every candidate bar, `.report()` is called on each agent with
  a TF-appropriate slice of `mtf_data` up to the candidate timestamp.
- The resulting 4 `FractalReport`s are flattened into a dedicated
  `rep_*` feature block merged into the candidate's `features` dict
  alongside the existing `rule_*` proxies (NOT replaced — Ticket 13
  explicitly keeps the proxies for baseline comparison).

The `rep_*` block follows a stable, documented schema :

  - rep_ctx_score       (float ∈ [0, 1])
  - rep_ctx_passed      (0 / 1)
  - rep_regime_score    (float ∈ [0, 1])
  - rep_regime_passed   (0 / 1)
  - rep_regime_trend    (1.0 iff regime state is trending)
  - rep_setup_score     (float ∈ [0, 1])
  - rep_setup_passed    (0 / 1)
  - rep_entry_score     (float ∈ [0, 1])
  - rep_entry_passed    (0 / 1)
  - rep_entry_direction (-1 / 0 / +1)
  - rep_agreement_mean  (mean of the 4 scores)
  - rep_agreement_std   (std dev of the 4 scores)
  - rep_disagreement    (1 − n_passed / 4)

Agents that fail (missing HSMM training, insufficient bars) return
neutral defaults — honesty over silent failure.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest


REP_KEYS = [
    'rep_ctx_score', 'rep_ctx_passed',
    'rep_regime_score', 'rep_regime_passed', 'rep_regime_trend',
    'rep_setup_score', 'rep_setup_passed',
    'rep_entry_score', 'rep_entry_passed', 'rep_entry_direction',
    'rep_agreement_mean', 'rep_agreement_std', 'rep_disagreement',
]


# ===========================================================================
class TestEngineHoldsAgents:
    """Ticket 13 — NYXEngine owns the 4 Jesse agents as runtime
    components (like EdgeStrategy per Ticket 08)."""

    def test_engine_has_ctx_agent(self):
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        assert hasattr(engine, '_ctx_agent')

    def test_engine_has_reg_agent(self):
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        assert hasattr(engine, '_reg_agent')

    def test_engine_has_stp_agent(self):
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        assert hasattr(engine, '_stp_agent')

    def test_engine_has_ent_agent(self):
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        assert hasattr(engine, '_ent_agent')


# ===========================================================================
class TestFractalReportFeatureHelper:
    """`_build_fractal_report_features(ts, mtf_data) -> dict` must
    return the stable `rep_*` schema even when an agent fails
    (neutral defaults instead of raising)."""

    def test_helper_exists(self):
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        assert hasattr(engine, '_build_fractal_report_features')

    def test_helper_returns_all_rep_keys(self, eth_mtf_data):
        """Even with empty / minimal data, the helper must return all
        13 canonical `rep_*` keys with numeric values."""
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        ts = pd.Timestamp('2023-06-15T10:00:00')
        out = engine._build_fractal_report_features(ts, eth_mtf_data)
        assert isinstance(out, dict)
        for k in REP_KEYS:
            assert k in out, f'helper must always emit {k!r}'
            assert isinstance(out[k], (int, float)), (
                f'{k} must be numeric, got {type(out[k]).__name__}'
            )

    def test_helper_values_in_range(self, eth_mtf_data):
        """Scores in [0, 1], passed in {0, 1}, direction in {-1, 0, +1}."""
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        ts = pd.Timestamp('2023-06-15T10:00:00')
        out = engine._build_fractal_report_features(ts, eth_mtf_data)
        for k in ('rep_ctx_score', 'rep_regime_score',
                  'rep_setup_score', 'rep_entry_score',
                  'rep_agreement_mean'):
            assert 0.0 <= out[k] <= 1.0
        for k in ('rep_ctx_passed', 'rep_regime_passed',
                  'rep_setup_passed', 'rep_entry_passed',
                  'rep_regime_trend'):
            assert out[k] in (0.0, 1.0)
        assert out['rep_entry_direction'] in (-1.0, 0.0, 1.0)
        assert 0.0 <= out['rep_disagreement'] <= 1.0


# ===========================================================================
class TestCandidatesCarryRepFeatures:
    """The candidates emitted by `_generate_candidates` on real data
    MUST carry the full `rep_*` block."""

    @pytest.fixture(scope='class')
    def candidates(self, eth_mtf_data, eth_mtf_features):
        from src.core.nyx_engine import NYXEngine
        engine = NYXEngine()
        ctx_1d = engine._build_1d_context(
            eth_mtf_data['1d'], eth_mtf_features.get('1d', pd.DataFrame()),
        )
        ctx_1h = engine._build_1h_context(
            eth_mtf_data['1h'], eth_mtf_features.get('1h', pd.DataFrame()),
        )
        df_15m = eth_mtf_data['15m'].loc['2023-01-01':'2023-01-31']
        feat_15m = eth_mtf_features['15m'].loc['2023-01-01':'2023-01-31']
        return engine._generate_candidates(
            df_15m=df_15m, feat_15m=feat_15m,
            ctx_1d=ctx_1d, ctx_1h=ctx_1h,
            feat_1h=eth_mtf_features.get('1h'),
            feat_1d=eth_mtf_features.get('1d'),
            feat_4h=eth_mtf_features.get('4h'),
            mtf_data=eth_mtf_data,
        )

    @pytest.mark.parametrize('key', REP_KEYS)
    def test_every_candidate_has_rep_key(self, candidates, key):
        if not candidates:
            pytest.skip('no candidates in slice')
        missing = [i for i, c in enumerate(candidates)
                   if key not in c['features']]
        assert not missing, (
            f'{len(missing)} candidates missing {key!r} — '
            'Ticket 13 requires every candidate to carry the full '
            'rep_* block'
        )

    def test_rule_proxies_still_present(self, candidates):
        """Ticket 13 explicit rule: NE PAS supprimer rule_* proxies."""
        if not candidates:
            pytest.skip('no candidates')
        c0 = candidates[0]
        for k in ('rule_context', 'rule_regime', 'rule_setup',
                  'disagreement'):
            assert k in c0['features'], (
                f'rule_* proxy {k!r} disappeared — Ticket 13 keeps '
                'both rule_* AND rep_* for baseline comparison'
            )


# ===========================================================================
class TestGenerateCandidatesAcceptsMTFData:
    """The `_generate_candidates` signature must accept `mtf_data`
    (needed to call agents per-TF)."""

    def test_mtf_data_is_kwarg(self):
        import inspect
        from src.core.nyx_engine import NYXEngine
        sig = inspect.signature(NYXEngine._generate_candidates)
        assert 'mtf_data' in sig.parameters, (
            '_generate_candidates must accept mtf_data kwarg (Ticket 13)'
        )
