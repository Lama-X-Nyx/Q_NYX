"""
TDD Tests — Ticket 45 — OOS Performance Optimization.

Caching, reuse, profiling. Same outputs, faster execution.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest


class TestCacheKeySystem:

    def test_cache_key_deterministic(self):
        from src.live.oos_cache import compute_cache_key
        k1 = compute_cache_key(assets=['BTCUSDT'], start='2023-01-01',
                               end='2023-12-31', capital=10_000.0, mode='idealized')
        k2 = compute_cache_key(assets=['BTCUSDT'], start='2023-01-01',
                               end='2023-12-31', capital=10_000.0, mode='idealized')
        assert k1 == k2

    def test_cache_key_differs_on_capital(self):
        from src.live.oos_cache import compute_cache_key
        k1 = compute_cache_key(assets=['BTCUSDT'], start='2023-01-01',
                               end='2023-12-31', capital=10_000.0, mode='idealized')
        k2 = compute_cache_key(assets=['BTCUSDT'], start='2023-01-01',
                               end='2023-12-31', capital=100_000.0, mode='idealized')
        assert k1 != k2

    def test_cache_key_differs_on_mode(self):
        from src.live.oos_cache import compute_cache_key
        k1 = compute_cache_key(assets=['BTCUSDT'], start='2023-01-01',
                               end='2023-12-31', capital=10_000.0, mode='idealized')
        k2 = compute_cache_key(assets=['BTCUSDT'], start='2023-01-01',
                               end='2023-12-31', capital=10_000.0, mode='realistic')
        assert k1 != k2

    def test_cache_key_differs_on_assets(self):
        from src.live.oos_cache import compute_cache_key
        k1 = compute_cache_key(assets=['BTCUSDT'], start='2023-01-01',
                               end='2023-12-31', capital=10_000.0, mode='idealized')
        k2 = compute_cache_key(assets=['BTCUSDT', 'ETHUSDT'], start='2023-01-01',
                               end='2023-12-31', capital=10_000.0, mode='idealized')
        assert k1 != k2


class TestOOSResultCache:

    def test_cache_miss_returns_none(self):
        from src.live.oos_cache import OOSResultCache
        cache = OOSResultCache(Path(tempfile.mkdtemp()))
        result = cache.get('nonexistent_key')
        assert result is None

    def test_cache_put_and_get(self):
        from src.live.oos_cache import OOSResultCache
        cache = OOSResultCache(Path(tempfile.mkdtemp()))
        data = {'run_id': 'test', 'per_asset': [{'symbol': 'BTCUSDT'}]}
        cache.put('key123', data)
        loaded = cache.get('key123')
        assert loaded is not None
        assert loaded['run_id'] == 'test'

    def test_cache_survives_reinstantiation(self):
        from src.live.oos_cache import OOSResultCache
        d = Path(tempfile.mkdtemp())
        cache1 = OOSResultCache(d)
        cache1.put('key456', {'value': 42})
        cache2 = OOSResultCache(d)
        assert cache2.get('key456')['value'] == 42

    def test_cache_list_keys(self):
        from src.live.oos_cache import OOSResultCache
        cache = OOSResultCache(Path(tempfile.mkdtemp()))
        cache.put('a', {'x': 1})
        cache.put('b', {'x': 2})
        keys = cache.list_keys()
        assert set(keys) == {'a', 'b'}


class TestDataCache:

    def test_data_cache_loads_once(self):
        from src.live.oos_cache import DataCache
        cache = DataCache()
        df1 = cache.load_ohlcv('BTCUSDT', '15m')
        df2 = cache.load_ohlcv('BTCUSDT', '15m')
        assert df1 is df2

    def test_data_cache_different_assets(self):
        from src.live.oos_cache import DataCache
        cache = DataCache()
        btc = cache.load_ohlcv('BTCUSDT', '15m')
        eth = cache.load_ohlcv('ETHUSDT', '15m')
        assert btc is not eth

    def test_feature_cache(self):
        from src.live.oos_cache import DataCache
        cache = DataCache()
        f1 = cache.load_features('BTCUSDT', '15m')
        f2 = cache.load_features('BTCUSDT', '15m')
        assert f1 is f2


class TestProfilingReport:

    def test_profiler_records_timing(self):
        from src.live.oos_cache import OOSProfiler
        profiler = OOSProfiler()
        profiler.start('data_loading')
        import time; time.sleep(0.01)
        profiler.stop('data_loading')
        report = profiler.report()
        assert 'data_loading' in report
        assert report['data_loading'] >= 0.01

    def test_profiler_multiple_sections(self):
        from src.live.oos_cache import OOSProfiler
        profiler = OOSProfiler()
        profiler.start('a')
        profiler.stop('a')
        profiler.start('b')
        profiler.stop('b')
        report = profiler.report()
        assert 'a' in report
        assert 'b' in report

    def test_profiler_serializable(self):
        from src.live.oos_cache import OOSProfiler
        profiler = OOSProfiler()
        profiler.start('x')
        profiler.stop('x')
        s = json.dumps(profiler.report())
        assert 'x' in s


class TestCachedEngineIntegration:

    def test_cached_engine_returns_same_result(self):
        from src.live.oos_engine import OOSConfig, CanonicalOOSEngine
        from src.live.oos_cache import OOSResultCache
        cache_dir = Path(tempfile.mkdtemp())
        cache = OOSResultCache(cache_dir)
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
