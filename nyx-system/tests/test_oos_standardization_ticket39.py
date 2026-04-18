"""
TDD Tests — Ticket 39 — Full-Stack OOS Standardization.

Ensures all assets are evaluated through the IDENTICAL pipeline:
GBM → Jesse → Fractal Quality → Risk Engine → Execution Optimizer
→ OMS → PostOnlyPaperBroker → Portfolio → Monitoring.

No mixed idealized/realistic modes. No asset bypasses execution.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
REPORTS_DIR = HERE / 'reports'


class TestStandardizedPipelineExists:

    def test_multi_asset_oos_script_exists(self):
        path = HERE / 'scripts' / 'oos_multi_asset_full_stack.py'
        assert path.exists(), 'scripts/oos_multi_asset_full_stack.py missing'

    def test_run_full_stack_asset_function_importable(self):
        from scripts.oos_multi_asset_full_stack import run_full_stack_asset
        assert callable(run_full_stack_asset)


class TestReportFormat:

    @pytest.fixture(scope='class')
    def reports(self):
        out = {}
        for sym in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT'):
            p = REPORTS_DIR / f'{sym}_full_stack_oos.json'
            if p.exists():
                out[sym] = json.loads(p.read_text())
        return out

    def test_all_three_reports_exist(self, reports):
        for sym in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT'):
            assert sym in reports, f'{sym} report missing'

    def test_reports_have_execution_section(self, reports):
        for sym, r in reports.items():
            assert 'execution' in r, f'{sym}: missing execution section'
            ex = r['execution']
            for field in ('idealized_signals', 'placed', 'filled',
                          'timed_out', 'fill_rate', 'miss_rate'):
                assert field in ex, f'{sym}: missing execution.{field}'

    def test_reports_have_performance_section(self, reports):
        for sym, r in reports.items():
            assert 'performance' in r, f'{sym}: missing performance'
            perf = r['performance']
            for field in ('n_trades', 'win_rate', 'sharpe',
                          'total_pnl', 'max_drawdown_pct'):
                assert field in perf, f'{sym}: missing performance.{field}'

    def test_no_asset_has_zero_placed(self, reports):
        """Every asset must have gone through broker — no idealized shortcut."""
        for sym, r in reports.items():
            placed = r['execution']['placed']
            assert placed > 0, f'{sym}: 0 orders placed — execution bypassed'

    def test_all_assets_have_miss_rate(self, reports):
        for sym, r in reports.items():
            mr = r['execution']['miss_rate']
            assert mr >= 0.0, f'{sym}: miss_rate invalid'

    def test_reports_have_risk_blocked_count(self, reports):
        for sym, r in reports.items():
            assert 'blocked_risk' in r['execution'] or 'skipped_quality' in r['execution']

    def test_idealized_and_realistic_separated(self, reports):
        for sym, r in reports.items():
            assert 'idealized_baseline' in r, f'{sym}: missing idealized_baseline'
            assert 'performance' in r, f'{sym}: missing realistic performance'
            assert r['idealized_baseline']['sharpe'] != r['performance']['sharpe'], \
                f'{sym}: idealized and realistic Sharpe should differ'

    def test_unified_summary_exists(self):
        p = REPORTS_DIR / 'MULTI_ASSET_full_stack_oos.json'
        assert p.exists(), 'unified summary report missing'
        data = json.loads(p.read_text())
        assert 'per_asset' in data
        assert len(data['per_asset']) == 3


class TestPipelineConsistency:

    @pytest.fixture(scope='class')
    def reports(self):
        out = {}
        for sym in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT'):
            p = REPORTS_DIR / f'{sym}_full_stack_oos.json'
            if p.exists():
                out[sym] = json.loads(p.read_text())
        return out

    def test_all_assets_use_same_modules(self, reports):
        if len(reports) < 3:
            pytest.skip('not all reports generated yet')
        modules_sets = [set(r.get('modules_tested', [])) for r in reports.values()]
        assert modules_sets[0] == modules_sets[1] == modules_sets[2], \
            'assets use different module sets'

    def test_all_assets_use_same_broker(self, reports):
        if len(reports) < 3:
            pytest.skip('not all reports generated yet')
        for sym, r in reports.items():
            assert 'PostOnlyPaperBroker' in r.get('modules_tested', []), \
                f'{sym}: not using PostOnlyPaperBroker'
