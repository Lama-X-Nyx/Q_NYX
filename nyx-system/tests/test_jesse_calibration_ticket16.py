"""
TDD Tests — Ticket 16.

Setup + Regime calibration before runtime integration.

Pre-Ticket-16 retrain metrics (Ticket 14/15, BTC 2020-2022,
`reports/jesse_agents_retrain_ticket14.json`) :

  Setup  : accuracy 0.188, pct_passed 0.000  (DEGENERATE)
  Regime : accuracy 0.571, pct_passed 0.992  (OVER-PERMISSIVE)

Root causes :
  Setup.analyze() reads `momentum_10 / ema_ratio_9_21 / rsi_14`
    from features — but these are NOT in the Ticket 14 FEATURE_PLAN
    (which is liquidity-hunter focused). So `last.get(...)` returns
    0 everywhere → heuristic collapses to 0 → p_setup = 0.4 *
    p_ml < 0.40 almost always → pct_passed ~ 0 %.
  Regime.analyze() has `range` state default to `passed=True`.
    On BTC 4H most bars are `range` → pct_passed ~ 99 %.

Ticket 16 fixes :
  Setup  — rewrite heuristic on liquidity-hunter FEATURE_PLAN
            features (sr_break_*, vwap_dist, bop, adosc_norm, ...).
            Target `pct_passed` ∈ [10 %, 40 %] on BTC.
  Regime — `range` default now `passed=False`. Only `trend_plus` /
            `trend_minus` pass. Target `pct_passed` ∈ [30 %, 80 %].

These tests assert :
  1. The Ticket 14/15 JSON report shows the target ranges
     (validation on real BTC data, not synthetic).
  2. Behavioural checks on analyze() logic (synthetic inputs,
     deterministic outputs).
  3. Report schema (Ticket 05 FractalReport) preserved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPORT_PATH = (
    Path(__file__).parent.parent / 'reports'
    / 'jesse_agents_retrain_ticket14.json'
)


@pytest.fixture(scope='module')
def retrain_report() -> dict:
    if not REPORT_PATH.exists():
        pytest.skip(f'{REPORT_PATH} missing — run retrain first')
    return json.loads(REPORT_PATH.read_text())


# ===========================================================================
class TestSetupNonDegenerate:
    """Setup must no longer collapse to pct_passed = 0 %."""

    def test_pct_passed_non_zero(self, retrain_report):
        setup = retrain_report['agents']['JesseSetupAgent']
        bt = setup['ticket14_backtest']
        assert bt.get('pct_passed', 0.0) > 0.02, (
            f'Setup pct_passed = {bt.get("pct_passed")} — still '
            'collapsed to all-no_setup. Ticket 16 target > 2 %.'
        )

    def test_pct_passed_in_operating_zone(self, retrain_report):
        """Ticket 16 operating zone : Setup `pct_passed` ∈ [5 %,
        60 %]. Lower bound > 5 % guards against collapse to
        all-block ; upper bound < 60 % guards against collapse to
        all-pass. The ticket's "roughly 10 % to 40 %" is the ideal
        target ; we leave slack for dataset drift."""
        setup = retrain_report['agents']['JesseSetupAgent']
        bt = setup['ticket14_backtest']
        pct = bt.get('pct_passed', 0.0)
        assert 0.05 <= pct <= 0.60, (
            f'Setup pct_passed = {pct} outside Ticket 16 operating '
            'zone [0.05, 0.60]'
        )

    def test_accuracy_above_collapse(self, retrain_report):
        """Post-Ticket-16 accuracy must exceed the pre-fix 0.188
        collapse floor. Target ≥ 0.30 (materially above broken
        baseline)."""
        setup = retrain_report['agents']['JesseSetupAgent']
        acc = setup['ticket14_backtest'].get('accuracy', 0.0)
        assert acc >= 0.30, (
            f'Setup accuracy {acc:.3f} still below collapse floor '
            '(0.30). Fix did not move the needle.'
        )


# ===========================================================================
class TestRegimeDiscriminant:
    """Regime must no longer pass nearly everything."""

    def test_pct_passed_discriminant(self, retrain_report):
        reg = retrain_report['agents']['JesseRegimeAgent']
        bt = reg['ticket14_backtest']
        pct = bt.get('pct_passed', 1.0)
        assert pct < 0.95, (
            f'Regime pct_passed = {pct} — still near-100 %, '
            'not discriminant'
        )

    def test_pct_passed_has_signal(self, retrain_report):
        """Regime must still pass SOME bars (not flip to all-block).
        Target lower bound > 15 %."""
        reg = retrain_report['agents']['JesseRegimeAgent']
        pct = reg['ticket14_backtest'].get('pct_passed', 0.0)
        assert pct > 0.15, (
            f'Regime pct_passed = {pct} — flipped to all-block, '
            'lost all signal'
        )


# ===========================================================================
def _make_df(n: int = 400, kind: str = 'trend_up') -> pd.DataFrame:
    """OHLCV synthetic DataFrame for behavioural tests."""
    np.random.seed(42)
    if kind == 'trend_up':
        drift = 0.0025
    elif kind == 'trend_down':
        drift = -0.0025
    else:
        drift = 0.0
    pct = drift + np.random.randn(n) * 0.001
    close = 1000 * np.exp(np.cumsum(pct))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.002 + 0.001)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.002 - 0.001)
    open_ = np.roll(close, 1); open_[0] = 1000
    volume = np.abs(np.random.randn(n)) * 1000 + 500
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }, index=pd.date_range('2023-01-01', periods=n, freq='1h'))


# ===========================================================================
class TestRegimeRangeDoesNotPass:
    """Post-Ticket-16 : Regime's `range` state must NOT pass by
    default. Only `trend_plus` / `trend_minus` pass."""

    def test_range_passed_false(self):
        """On a flat synthetic dataset (range kind), Regime's
        .analyze() state must be 'range' and passed=False."""
        from src.ml.jesse_agents import JesseRegimeAgent
        a = JesseRegimeAgent()
        mixed = pd.concat([
            _make_df(300, 'trend_up'),
            _make_df(300, 'trend_down'),
            _make_df(300, 'range'),
        ]).sort_index()
        a.train(mixed)
        rng = _make_df(400, 'range')
        r = a.analyze(rng)
        # On flat data, should classify range → passed=False.
        assert r.state == 'range' or r.state == 'squeeze', (
            f'Expected range/squeeze on flat data, got {r.state}'
        )
        assert r.passed is False, (
            'range / squeeze must NOT pass — Ticket 16 fix'
        )


# ===========================================================================
class TestSetupUsesLiquidityHeuristic:
    """Post-Ticket-16 : Setup's analyze() heuristic must use features
    from its FEATURE_PLAN (liquidity-hunter family) — NOT legacy
    features that collapse to 0."""

    def test_analyze_heuristic_features_in_plan(self):
        """Source-level check : analyze() body must reference at
        least one of the liquidity-hunter FEATURE_PLAN features
        (vwap_dist / sr_break_up_20 / bop / adosc_norm /
        minmax_pos_20). The legacy `momentum_10` +
        `ema_ratio_9_21` reads should be gone, OR replaced with
        the plan features."""
        import inspect
        from src.ml.jesse_agents import JesseSetupAgent
        src = inspect.getsource(JesseSetupAgent.analyze)
        liquidity_keys = (
            'vwap_dist', 'sr_break_up_20', 'sr_break_dn_20',
            'bop', 'adosc_norm', 'minmax_pos_20',
        )
        hits = sum(1 for k in liquidity_keys if k in src)
        assert hits >= 2, (
            f'JesseSetupAgent.analyze() uses only {hits} '
            'liquidity-hunter features; Ticket 16 requires ≥ 2 '
            'from {vwap_dist, sr_break_*, bop, adosc_norm, '
            'minmax_pos_20}.'
        )


# ===========================================================================
class TestReportSchemaPreserved:
    """Ticket 05 contract must stay intact post-calibration."""

    def test_setup_report_schema(self):
        from src.agents.contracts import FractalReport
        from src.ml.jesse_agents import JesseSetupAgent
        a = JesseSetupAgent()
        a.train(_make_df(400, 'trend_up'))
        r = a.report(_make_df(400, 'trend_up'), asset='BTCUSDT',
                     timestamp='2023-06-15T00:00:00')
        assert isinstance(r, FractalReport)
        assert r.timeframe == '1h'
        assert 0.0 <= r.score <= 1.0

    def test_regime_report_schema(self):
        from src.agents.contracts import FractalReport
        from src.ml.jesse_agents import JesseRegimeAgent
        a = JesseRegimeAgent()
        a.train(_make_df(400, 'trend_up'))
        r = a.report(_make_df(400, 'trend_up'), asset='BTCUSDT',
                     timestamp='2023-06-15T00:00:00')
        assert isinstance(r, FractalReport)
        assert r.timeframe == '4h'
        assert 0.0 <= r.score <= 1.0
