"""
TDD Tests — Ticket 42 — Canonical OOS Engine.

One parameter-driven engine for all OOS requests.
No new scripts needed. Supports mono/multi-asset, idealized/realistic,
any calendar window, any capital.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent


class TestOOSConfig:

    def test_config_creation(self):
        from src.live.oos_engine import OOSConfig
        cfg = OOSConfig(
            assets=['BTCUSDT'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            initial_capital=10_000.0,
        )
        assert cfg.assets == ['BTCUSDT']
        assert cfg.initial_capital == 10_000.0

    def test_config_defaults(self):
        from src.live.oos_engine import OOSConfig
        cfg = OOSConfig(assets=['BTCUSDT'], start_date='2023-01-01', end_date='2023-12-31')
        assert cfg.mode == 'realistic'
        assert cfg.initial_capital == 10_000.0
        assert cfg.train_end == '2022-12-31'

    def test_config_serializable(self):
        from src.live.oos_engine import OOSConfig
        cfg = OOSConfig(
            assets=['BTCUSDT', 'ETHUSDT'],
            start_date='2023-01-01', end_date='2023-06-30',
            initial_capital=1_000_000.0, mode='idealized',
        )
        d = cfg.to_dict()
        s = json.dumps(d)
        loaded = json.loads(s)
        assert loaded['assets'] == ['BTCUSDT', 'ETHUSDT']
        assert loaded['mode'] == 'idealized'

    def test_config_validation_bad_capital(self):
        from src.live.oos_engine import OOSConfig
        with pytest.raises(ValueError):
            OOSConfig(assets=['BTCUSDT'], start_date='2023-01-01',
                      end_date='2023-12-31', initial_capital=-100.0)

    def test_config_validation_empty_assets(self):
        from src.live.oos_engine import OOSConfig
        with pytest.raises(ValueError):
            OOSConfig(assets=[], start_date='2023-01-01', end_date='2023-12-31')

    def test_config_validation_bad_mode(self):
        from src.live.oos_engine import OOSConfig
        with pytest.raises(ValueError):
            OOSConfig(assets=['BTCUSDT'], start_date='2023-01-01',
                      end_date='2023-12-31', mode='magic')

    def test_config_from_dict(self):
        from src.live.oos_engine import OOSConfig
        d = {'assets': ['SOLUSDT'], 'start_date': '2023-06-01',
             'end_date': '2023-12-31', 'initial_capital': 50_000.0}
        cfg = OOSConfig.from_dict(d)
        assert cfg.assets == ['SOLUSDT']
        assert cfg.initial_capital == 50_000.0


class TestCanonicalOOSEngine:

    def test_engine_instantiation(self):
        from src.live.oos_engine import CanonicalOOSEngine
        engine = CanonicalOOSEngine()
        assert engine is not None

    def test_run_oos_returns_result(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine, OOSResult
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31', initial_capital=10_000.0,
        )
        engine = CanonicalOOSEngine()
        result = engine.run_oos(cfg)
        assert isinstance(result, OOSResult)
        assert result.run_id is not None
        assert result.config is not None
        assert result.per_asset is not None

    def test_run_oos_mono_asset(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31', initial_capital=10_000.0,
        )
        engine = CanonicalOOSEngine()
        result = engine.run_oos(cfg)
        assert len(result.per_asset) == 1
        assert result.per_asset[0]['symbol'] == 'BTCUSDT'

    def test_run_oos_multi_asset(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        cfg = OOSConfig(
            assets=['BTCUSDT', 'ETHUSDT'], start_date='2023-01-01',
            end_date='2023-12-31', initial_capital=10_000.0,
        )
        engine = CanonicalOOSEngine()
        result = engine.run_oos(cfg)
        assert len(result.per_asset) == 2
        syms = {r['symbol'] for r in result.per_asset}
        assert syms == {'BTCUSDT', 'ETHUSDT'}
        assert result.portfolio is not None

    def test_idealized_mode_no_execution(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31', mode='idealized',
        )
        engine = CanonicalOOSEngine()
        result = engine.run_oos(cfg)
        r = result.per_asset[0]
        assert 'execution' not in r or r.get('mode') == 'idealized'

    def test_realistic_mode_has_execution(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31', mode='realistic',
        )
        engine = CanonicalOOSEngine()
        result = engine.run_oos(cfg)
        r = result.per_asset[0]
        assert 'execution' in r
        assert r['execution']['placed'] > 0


class TestCapitalScaling:

    def test_capital_100(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31', initial_capital=100.0,
        )
        engine = CanonicalOOSEngine()
        result = engine.run_oos(cfg)
        assert result.per_asset[0]['capital'] == 100.0

    def test_capital_10M(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31', initial_capital=10_000_000.0,
        )
        engine = CanonicalOOSEngine()
        result = engine.run_oos(cfg)
        assert result.per_asset[0]['capital'] == 10_000_000.0


class TestReportPersistence:

    def test_report_saved(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        reports_dir = Path(tempfile.mkdtemp())
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31',
        )
        engine = CanonicalOOSEngine(reports_dir=reports_dir)
        result = engine.run_oos(cfg)
        path = reports_dir / f'oos_{result.run_id}.json'
        assert path.exists()

    def test_load_report(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        reports_dir = Path(tempfile.mkdtemp())
        engine = CanonicalOOSEngine(reports_dir=reports_dir)
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31',
        )
        result = engine.run_oos(cfg)
        loaded = engine.load_oos_report(result.run_id)
        assert loaded.run_id == result.run_id

    def test_list_runs(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        reports_dir = Path(tempfile.mkdtemp())
        engine = CanonicalOOSEngine(reports_dir=reports_dir)
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31',
        )
        engine.run_oos(cfg)
        runs = engine.list_oos_runs()
        assert len(runs) >= 1


class TestDeterminism:

    def test_same_config_same_result(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        cfg = OOSConfig(
            assets=['BTCUSDT'], start_date='2023-01-01',
            end_date='2023-12-31', mode='idealized',
        )
        engine = CanonicalOOSEngine()
        r1 = engine.run_oos(cfg)
        r2 = engine.run_oos(cfg)
        p1 = r1.per_asset[0]['performance']
        p2 = r2.per_asset[0]['performance']
        assert p1['sharpe'] == p2['sharpe']
        assert p1['total_pnl'] == p2['total_pnl']
