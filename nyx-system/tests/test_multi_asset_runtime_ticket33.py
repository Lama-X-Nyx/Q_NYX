"""
TDD Tests — Ticket 33 — Multi-asset NYXRuntime instances.

Proves:
1. Three NYXRuntime instances (BTC/ETH/SOL) instantiate independently
2. Each loads its own model artefact (128 canonical features)
3. No cross-contamination between instances (state isolation)
4. Each instance processes bars independently
5. Per-asset state directories are separate
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
HERE = Path(__file__).resolve().parent.parent


class TestMultiAssetInstantiation:
    """Each symbol gets its own NYXRuntime with its own model."""

    def test_three_instances_create_independently(self):
        from src.live.nyx_runtime import NYXRuntime
        runtimes = {}
        for sym in SYMBOLS:
            runtimes[sym] = NYXRuntime(
                symbol=sym,
                models_dir=HERE / 'models' / sym,
                state_dir=Path(tempfile.mkdtemp()) / sym,
            )
        assert len(runtimes) == 3
        for sym in SYMBOLS:
            assert runtimes[sym].symbol == sym

    def test_each_instance_loads_correct_model(self):
        from src.live.nyx_runtime import NYXRuntime
        for sym in SYMBOLS:
            rt = NYXRuntime(
                symbol=sym,
                models_dir=HERE / 'models' / sym,
                state_dir=Path(tempfile.mkdtemp()) / sym,
            )
            assert rt.decider.symbol == sym
            assert len(rt.decider.feature_names) == 128

    def test_feature_names_match_across_assets(self):
        """All 3 assets must have the same 128 feature names (canonical set)."""
        import json
        feature_sets = {}
        for sym in SYMBOLS:
            fn_path = HERE / 'models' / sym / 'feature_names.json'
            feature_sets[sym] = set(json.load(open(fn_path)))
        assert feature_sets['BTCUSDT'] == feature_sets['ETHUSDT']
        assert feature_sets['BTCUSDT'] == feature_sets['SOLUSDT']


class TestStateIsolation:
    """No cross-contamination between runtime instances."""

    def test_separate_oms_per_instance(self):
        from src.live.nyx_runtime import NYXRuntime
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt_eth = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=HERE / 'models' / 'ETHUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'ETHUSDT',
        )
        assert rt_btc.oms is not rt_eth.oms

    def test_separate_portfolio_per_instance(self):
        from src.live.nyx_runtime import NYXRuntime
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt_eth = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=HERE / 'models' / 'ETHUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'ETHUSDT',
        )
        assert rt_btc.portfolio is not rt_eth.portfolio
        assert rt_btc.portfolio.available_balance == rt_eth.portfolio.available_balance

    def test_separate_risk_engine_per_instance(self):
        from src.live.nyx_runtime import NYXRuntime
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt_sol = NYXRuntime(
            symbol='SOLUSDT',
            models_dir=HERE / 'models' / 'SOLUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'SOLUSDT',
        )
        rt_btc.risk_engine.activate_kill_switch(reason='test')
        assert rt_btc.risk_engine.is_killed is True
        assert rt_sol.risk_engine.is_killed is False

    def test_separate_metrics_per_instance(self):
        from src.live.nyx_runtime import NYXRuntime
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt_eth = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=HERE / 'models' / 'ETHUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'ETHUSDT',
        )
        assert rt_btc.metrics is not rt_eth.metrics

    def test_separate_state_store_dirs(self):
        from src.live.nyx_runtime import NYXRuntime
        import tempfile as tf
        base = Path(tf.mkdtemp())
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=base / 'BTCUSDT',
        )
        rt_eth = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=HERE / 'models' / 'ETHUSDT',
            state_dir=base / 'ETHUSDT',
        )
        assert rt_btc.state_store.state_dir != rt_eth.state_store.state_dir


class TestIndependentBarProcessing:
    """Each instance processes bars without affecting others."""

    def _make_bar(self, symbol: str, close: float, ts: str = '2023-06-15T12:00:00') -> dict:
        return {
            'timestamp': ts,
            'open': close * 0.999,
            'high': close * 1.001,
            'low': close * 0.998,
            'close': close,
            'volume': 100.0,
        }

    def test_bar_on_stopped_instance_returns_stopped(self):
        from src.live.nyx_runtime import NYXRuntime
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        result = rt.on_bar(self._make_bar('BTCUSDT', 30000.0))
        assert result['action'] == 'SYSTEM_STOPPED'

    def test_bar_on_started_instance_processes(self):
        from src.live.nyx_runtime import NYXRuntime
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt.start()
        result = rt.on_bar(self._make_bar('BTCUSDT', 30000.0))
        assert result['action'] in ('FLAT', 'ORDER_SUBMITTED', 'SKIP_QUALITY', 'BLOCKED_RISK')
        assert 'layers' in result

    def test_two_instances_process_bars_independently(self):
        from src.live.nyx_runtime import NYXRuntime
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt_eth = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=HERE / 'models' / 'ETHUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'ETHUSDT',
        )
        rt_btc.start()
        rt_eth.start()
        r_btc = rt_btc.on_bar(self._make_bar('BTCUSDT', 30000.0))
        r_eth = rt_eth.on_bar(self._make_bar('ETHUSDT', 2000.0))
        assert r_btc['action'] in ('FLAT', 'ORDER_SUBMITTED', 'SKIP_QUALITY', 'BLOCKED_RISK')
        assert r_eth['action'] in ('FLAT', 'ORDER_SUBMITTED', 'SKIP_QUALITY', 'BLOCKED_RISK')

    def test_kill_switch_one_does_not_affect_other(self):
        from src.live.nyx_runtime import NYXRuntime
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt_eth = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=HERE / 'models' / 'ETHUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'ETHUSDT',
        )
        rt_btc.start()
        rt_eth.start()
        rt_btc.emergency_stop(reason='test isolation')
        assert rt_btc.is_running is False
        assert rt_eth.is_running is True


class TestControlPlaneIsolation:
    """Operator commands on one instance don't bleed into others."""

    def test_pause_one_instance(self):
        from src.live.nyx_runtime import NYXRuntime
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt_eth = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=HERE / 'models' / 'ETHUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'ETHUSDT',
        )
        rt_btc.start()
        rt_eth.start()
        rt_btc.pause()
        assert rt_btc.is_paused is True
        assert rt_eth.is_paused is False

    def test_stop_one_instance(self):
        from src.live.nyx_runtime import NYXRuntime
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt_sol = NYXRuntime(
            symbol='SOLUSDT',
            models_dir=HERE / 'models' / 'SOLUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'SOLUSDT',
        )
        rt_btc.start()
        rt_sol.start()
        rt_sol.stop()
        assert rt_sol.is_running is False
        assert rt_btc.is_running is True
