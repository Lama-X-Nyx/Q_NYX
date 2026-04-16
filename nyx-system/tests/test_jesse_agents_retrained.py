"""
TDD Tests — Ticket 14.

All 4 Jesse agents must:
  1. Publish a documented `FEATURE_PLAN` class attribute — a tuple of
     canonical feature-name strings drawn from
     `compute_stationary_features(df, feature_set='full')`.
  2. `compute_features(df)` output must CONTAIN every feature in
     `FEATURE_PLAN`.
  3. Trainability on synthetic data must remain green.
  4. `.analyze(df)` must still return a valid `AgentResult` (schema
     preserved) post-retraining.
  5. `.report(df, asset, ts)` must still return a valid
     `FractalReport` (Ticket 05 contract preserved).
  6. Setup + Entry can no longer be 'core'-only — they must expose
     liquidity-hunter / short-horizon features from the
     Ticket 09 family.

This is enforcement of the ticket's "Forbidden — no retrain first
clean later, no feature dump into all agents without rationale".
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimum required features per agent (Ticket 14 role-based plan).
# Tests assert these are a SUBSET of each agent's FEATURE_PLAN.
# ---------------------------------------------------------------------------

REQUIRED_CONTEXT = {
    'ema_ratio_21_50', 'close_vs_ema50', 'ema_ratio_50_200',
    'rsi_14', 'atr_ratio', 'bb_width_ratio',
    'zscore_20', 'chop_norm',
}

REQUIRED_REGIME = {
    'adx_norm', 'atr_ratio', 'bb_width_ratio',
    'keltner_position', 'squeeze',
    'chop_norm', 'macd_hist_ratio',
    'ema_ratio_9_21', 'ema_ratio_21_50',
}

REQUIRED_SETUP = {
    # Liquidity-hunter heart (Ticket 09).
    'vwap_dist', 'vwma_dist',
    'ad_slope', 'adosc_norm', 'mfi_norm', 'bop', 'marketfi_ratio',
    'sr_break_up_20', 'sr_break_dn_20',
    'sr_dist_high_20', 'sr_dist_low_20',
    'minmax_pos_20',
}

REQUIRED_ENTRY = {
    # Short-horizon trigger confirmation.
    'returns_1', 'returns_5',
    'rsi_14', 'volume_ratio', 'vol_change',
    'close_position', 'high_low_ratio',
    'bop', 'vwap_dist',
    'sr_break_up_20', 'sr_break_dn_20',
    'adosc_norm',
}


# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------
def _synthetic_df(n: int = 400, start: float = 1000.0,
                   kind: str = 'trend') -> pd.DataFrame:
    """OHLCV 15m synthetic with the target 'kind' (trend vs range)."""
    np.random.seed(42)
    if kind == 'trend':
        drift = 0.001
    else:
        drift = 0.0
    pct = drift + np.random.randn(n) * 0.002
    close = start * np.exp(np.cumsum(pct))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.002 + 0.001)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.002 - 0.001)
    open_ = np.roll(close, 1); open_[0] = start
    volume = np.abs(np.random.randn(n)) * 1000.0 + 500.0
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }, index=pd.date_range('2023-01-01', periods=n, freq='15min'))


# ===========================================================================
class TestFeaturePlanPresent:
    """Every agent must publish a `FEATURE_PLAN` constant."""

    AGENTS_AND_REQS = [
        ('JesseContextAgent', REQUIRED_CONTEXT),
        ('JesseRegimeAgent',  REQUIRED_REGIME),
        ('JesseSetupAgent',   REQUIRED_SETUP),
        ('JesseEntryAgent',   REQUIRED_ENTRY),
    ]

    @pytest.mark.parametrize('agent_name,required',
                             AGENTS_AND_REQS,
                             ids=[c[0] for c in AGENTS_AND_REQS])
    def test_feature_plan_exists(self, agent_name, required):
        import src.ml.jesse_agents as mod
        cls = getattr(mod, agent_name)
        assert hasattr(cls, 'FEATURE_PLAN'), (
            f'{agent_name} must publish FEATURE_PLAN class attribute '
            '(Ticket 14)'
        )

    @pytest.mark.parametrize('agent_name,required',
                             AGENTS_AND_REQS,
                             ids=[c[0] for c in AGENTS_AND_REQS])
    def test_feature_plan_contains_required(self, agent_name, required):
        import src.ml.jesse_agents as mod
        cls = getattr(mod, agent_name)
        plan = set(getattr(cls, 'FEATURE_PLAN', ()))
        missing = required - plan
        assert not missing, (
            f'{agent_name}.FEATURE_PLAN missing required features: '
            f'{sorted(missing)} — Ticket 14 role-based feature plan '
            'broken'
        )


# ===========================================================================
class TestComputeFeaturesExposesPlan:
    """`agent.compute_features(df)` must emit every feature listed in
    `FEATURE_PLAN` (the agent may add extras but cannot drop planned
    features)."""

    @pytest.fixture(scope='class')
    def df(self):
        return _synthetic_df(400)

    @pytest.mark.parametrize('agent_factory,cls_name', [
        (lambda: _make_mono('JesseContextAgent'), 'JesseContextAgent'),
        (lambda: _make_mono('JesseRegimeAgent'),  'JesseRegimeAgent'),
        (lambda: _make_mono('JesseSetupAgent'),   'JesseSetupAgent'),
        (lambda: _make_mono('JesseEntryAgent'),   'JesseEntryAgent'),
    ])
    def test_compute_features_contains_plan(self, agent_factory,
                                             cls_name, df):
        agent = agent_factory()
        feats = agent.compute_features(df)
        plan = list(agent.FEATURE_PLAN)
        missing = [f for f in plan if f not in feats.columns]
        assert not missing, (
            f'{cls_name}.compute_features() missing planned features: '
            f'{missing[:5]}'
        )


def _make_mono(name: str):
    import src.ml.jesse_agents as mod
    return getattr(mod, name)()


# ===========================================================================
class TestTrainingSucceeds:
    """Each agent trains without error on synthetic data post-refactor."""

    @pytest.mark.parametrize('cls_name', [
        'JesseContextAgent', 'JesseRegimeAgent',
        'JesseSetupAgent',   'JesseEntryAgent',
    ])
    def test_train_returns_metrics(self, cls_name):
        agent = _make_mono(cls_name)
        df = _synthetic_df(500, kind='trend')
        result = agent.train(df)
        assert isinstance(result, dict)
        assert 'accuracy' in result
        assert 0.0 <= result['accuracy'] <= 1.0
        assert result.get('n_samples', 0) > 0


# ===========================================================================
class TestReportSchemaPreserved:
    """Post-refactor: `.analyze()` + `.report()` still return the
    Ticket 05 canonical `FractalReport`."""

    @pytest.mark.parametrize('cls_name,timeframe', [
        ('JesseContextAgent', '1d'),
        ('JesseRegimeAgent',  '4h'),
        ('JesseSetupAgent',   '1h'),
        ('JesseEntryAgent',   '15m'),
    ])
    def test_report_contract(self, cls_name, timeframe):
        from src.agents.contracts import FractalReport
        agent = _make_mono(cls_name)
        df = _synthetic_df(400, kind='trend')
        agent.train(df)
        r = agent.report(df, asset='BTCUSDT',
                         timestamp='2023-06-15T10:00:00')
        assert isinstance(r, FractalReport)
        assert r.timeframe == timeframe
        assert 0.0 <= r.score <= 1.0


# ===========================================================================
class TestSetupAndEntryNotOnlyCore:
    """Ticket 14 explicit : Setup + Entry must NOT be core-only.
    They must include the Ticket 09 liquidity-hunter family."""

    LIQUIDITY_KEYS = ('vwap_dist', 'ad_slope', 'mfi_norm', 'bop',
                      'sr_break_up_20')

    def test_setup_includes_liquidity_family(self):
        agent = _make_mono('JesseSetupAgent')
        plan = set(agent.FEATURE_PLAN)
        intersect = plan & set(self.LIQUIDITY_KEYS)
        assert len(intersect) >= 3, (
            f'JesseSetupAgent must expose ≥ 3 liquidity features, '
            f'got {sorted(intersect)}'
        )

    def test_entry_includes_short_horizon_triggers(self):
        agent = _make_mono('JesseEntryAgent')
        plan = set(agent.FEATURE_PLAN)
        short_horizon = {'returns_1', 'returns_5', 'vol_change',
                          'close_position', 'bop', 'vwap_dist'}
        intersect = plan & short_horizon
        assert len(intersect) >= 4, (
            f'JesseEntryAgent must expose ≥ 4 short-horizon features, '
            f'got {sorted(intersect)}'
        )


# ===========================================================================
class TestRoleSpecializationPreserved:
    """Rule 3 (Ticket 14) — agents must remain distinct. No two
    agents may share the SAME FEATURE_PLAN."""

    def test_four_plans_are_not_identical(self):
        import src.ml.jesse_agents as mod
        plans = {
            name: tuple(getattr(mod, name).FEATURE_PLAN)
            for name in ('JesseContextAgent', 'JesseRegimeAgent',
                         'JesseSetupAgent', 'JesseEntryAgent')
        }
        # No pair of plans may be identical
        names = list(plans.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                assert plans[a] != plans[b], (
                    f'{a}.FEATURE_PLAN == {b}.FEATURE_PLAN — '
                    'role specialization violated'
                )


# ===========================================================================
class TestWarmupBehavior:
    """Post-refactor, features must remain finite past the agent's
    warmup window (Ticket 14 Rule 4 — no NaN catastrophe)."""

    @pytest.mark.parametrize('cls_name', [
        'JesseContextAgent', 'JesseRegimeAgent',
        'JesseSetupAgent',   'JesseEntryAgent',
    ])
    def test_no_nan_past_warmup(self, cls_name):
        agent = _make_mono(cls_name)
        df = _synthetic_df(500, kind='trend')
        feats = agent.compute_features(df)
        tail = feats.iloc[100:][list(agent.FEATURE_PLAN)].values
        n_bad = int(np.sum(~np.isfinite(tail.astype(float))))
        assert n_bad == 0, (
            f'{cls_name} has {n_bad} non-finite values past warmup'
        )
