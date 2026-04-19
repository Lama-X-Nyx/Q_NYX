"""
TDD Tests — Ticket DATA-1 — Market Data Pipeline Core.

Bar store, gap detection, data quality, canonical data API.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestCanonicalBarStore:

    def test_append_and_read(self):
        from src.data_pipeline.bar_store import CanonicalBarStore
        store = CanonicalBarStore(Path(tempfile.mkdtemp()))
        bars = pd.DataFrame({
            'open': [100.0, 101.0], 'high': [102.0, 103.0],
            'low': [99.0, 100.0], 'close': [101.0, 102.0],
            'volume': [1000.0, 1100.0],
        }, index=pd.date_range('2023-01-01', periods=2, freq='15min'))
        store.append('BTCUSDT', '15m', bars)
        loaded = store.get_bars('BTCUSDT', '15m', '2023-01-01', '2023-01-02')
        assert len(loaded) == 2
        assert loaded.iloc[0]['close'] == 101.0

    def test_append_is_idempotent(self):
        from src.data_pipeline.bar_store import CanonicalBarStore
        store = CanonicalBarStore(Path(tempfile.mkdtemp()))
        bars = pd.DataFrame({
            'open': [100.0], 'high': [102.0], 'low': [99.0],
            'close': [101.0], 'volume': [1000.0],
        }, index=pd.DatetimeIndex(['2023-01-01 00:00:00']))
        store.append('BTCUSDT', '15m', bars)
        store.append('BTCUSDT', '15m', bars)
        loaded = store.get_bars('BTCUSDT', '15m', '2023-01-01', '2023-01-02')
        assert len(loaded) == 1

    def test_multi_symbol(self):
        from src.data_pipeline.bar_store import CanonicalBarStore
        store = CanonicalBarStore(Path(tempfile.mkdtemp()))
        bars = pd.DataFrame({
            'open': [100.0], 'high': [102.0], 'low': [99.0],
            'close': [101.0], 'volume': [1000.0],
        }, index=pd.DatetimeIndex(['2023-01-01']))
        store.append('BTCUSDT', '15m', bars)
        store.append('ETHUSDT', '15m', bars)
        assert len(store.get_bars('BTCUSDT', '15m')) == 1
        assert len(store.get_bars('ETHUSDT', '15m')) == 1

    def test_list_symbols(self):
        from src.data_pipeline.bar_store import CanonicalBarStore
        store = CanonicalBarStore(Path(tempfile.mkdtemp()))
        bars = pd.DataFrame({
            'open': [100.0], 'high': [102.0], 'low': [99.0],
            'close': [101.0], 'volume': [1000.0],
        }, index=pd.DatetimeIndex(['2023-01-01']))
        store.append('BTCUSDT', '15m', bars)
        store.append('ETHUSDT', '1h', bars)
        syms = store.list_symbols()
        assert 'BTCUSDT' in syms
        assert 'ETHUSDT' in syms

    def test_metadata(self):
        from src.data_pipeline.bar_store import CanonicalBarStore
        store = CanonicalBarStore(Path(tempfile.mkdtemp()))
        bars = pd.DataFrame({
            'open': [100.0, 101.0], 'high': [102.0, 103.0],
            'low': [99.0, 100.0], 'close': [101.0, 102.0],
            'volume': [1000.0, 1100.0],
        }, index=pd.date_range('2023-01-01', periods=2, freq='15min'))
        store.append('BTCUSDT', '15m', bars)
        meta = store.get_metadata('BTCUSDT', '15m')
        assert meta['n_bars'] == 2
        assert 'first_ts' in meta
        assert 'last_ts' in meta


class TestGapDetection:

    def test_no_gaps(self):
        from src.data_pipeline.bar_store import detect_gaps
        idx = pd.date_range('2023-01-01', periods=10, freq='15min')
        gaps = detect_gaps(idx, freq='15min')
        assert len(gaps) == 0

    def test_detects_gap(self):
        from src.data_pipeline.bar_store import detect_gaps
        idx = pd.DatetimeIndex([
            '2023-01-01 00:00', '2023-01-01 00:15',
            '2023-01-01 00:45',  # gap at 00:30
            '2023-01-01 01:00',
        ])
        gaps = detect_gaps(idx, freq='15min')
        assert len(gaps) >= 1

    def test_duplicate_detection(self):
        from src.data_pipeline.bar_store import detect_duplicates
        idx = pd.DatetimeIndex([
            '2023-01-01 00:00', '2023-01-01 00:15',
            '2023-01-01 00:15',  # duplicate
            '2023-01-01 00:30',
        ])
        dups = detect_duplicates(idx)
        assert len(dups) == 1


class TestDataQuality:

    def test_quality_report(self):
        from src.data_pipeline.bar_store import CanonicalBarStore, data_quality_report
        store = CanonicalBarStore(Path(tempfile.mkdtemp()))
        bars = pd.DataFrame({
            'open': [100.0, 101.0, 102.0], 'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0], 'close': [101.0, 102.0, 103.0],
            'volume': [1000.0, 1100.0, 1200.0],
        }, index=pd.date_range('2023-01-01', periods=3, freq='15min'))
        store.append('BTCUSDT', '15m', bars)
        report = data_quality_report(store, 'BTCUSDT', '15m')
        assert 'n_bars' in report
        assert 'n_gaps' in report
        assert 'n_duplicates' in report
        assert report['n_gaps'] == 0
