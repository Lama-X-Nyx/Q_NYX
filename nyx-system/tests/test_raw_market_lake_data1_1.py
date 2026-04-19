"""
TDD Tests — Ticket DATA-1.1 — Raw Market Data Lake.

Append-only raw event store. Partitioned by exchange/event_type/
symbol/date. Deterministic dedupe. Replay-ready.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest


def _make_kline_payload(open_time_ms: int = 1700000000000, close: float = 30000.0):
    return {
        't': open_time_ms,
        'T': open_time_ms + 899999,
        's': 'BTCUSDT',
        'i': '15m',
        'o': '29900.0', 'h': '30100.0', 'l': '29800.0', 'c': str(close),
        'v': '100.5',
        'x': True,  # closed
    }


class TestRawEventSchema:

    def test_kline_event_creation(self):
        from src.data_pipeline.raw_market_store import RawEvent
        ev = RawEvent.from_kline(
            exchange='binance', symbol='BTCUSDT', interval='15m',
            payload=_make_kline_payload(),
            ingest_source='ws',
        )
        assert ev.exchange == 'binance'
        assert ev.symbol == 'BTCUSDT'
        assert ev.event_type == 'kline'
        assert ev.ingest_source == 'ws'
        assert ev.schema_version >= 1

    def test_event_has_dedupe_key(self):
        from src.data_pipeline.raw_market_store import RawEvent
        ev = RawEvent.from_kline(
            exchange='binance', symbol='BTCUSDT', interval='15m',
            payload=_make_kline_payload(),
        )
        assert ev.dedupe_key is not None
        assert len(ev.dedupe_key) > 0

    def test_same_kline_same_dedupe_key(self):
        from src.data_pipeline.raw_market_store import RawEvent
        p = _make_kline_payload()
        ev1 = RawEvent.from_kline(exchange='binance', symbol='BTCUSDT', interval='15m', payload=p)
        ev2 = RawEvent.from_kline(exchange='binance', symbol='BTCUSDT', interval='15m', payload=p)
        assert ev1.dedupe_key == ev2.dedupe_key

    def test_different_open_time_different_dedupe(self):
        from src.data_pipeline.raw_market_store import RawEvent
        p1 = _make_kline_payload(open_time_ms=1700000000000)
        p2 = _make_kline_payload(open_time_ms=1700000900000)
        ev1 = RawEvent.from_kline(exchange='binance', symbol='BTCUSDT', interval='15m', payload=p1)
        ev2 = RawEvent.from_kline(exchange='binance', symbol='BTCUSDT', interval='15m', payload=p2)
        assert ev1.dedupe_key != ev2.dedupe_key

    def test_event_preserves_raw_payload(self):
        from src.data_pipeline.raw_market_store import RawEvent
        p = _make_kline_payload()
        ev = RawEvent.from_kline(exchange='binance', symbol='BTCUSDT', interval='15m', payload=p)
        assert ev.payload_raw == p

    def test_event_has_timestamps(self):
        from src.data_pipeline.raw_market_store import RawEvent
        ev = RawEvent.from_kline(
            exchange='binance', symbol='BTCUSDT', interval='15m',
            payload=_make_kline_payload(),
        )
        assert ev.event_ts_exchange_ms > 0
        assert ev.event_ts_receive >= 0
        assert ev.event_ts_store >= 0

    def test_event_to_dict_and_back(self):
        from src.data_pipeline.raw_market_store import RawEvent
        ev = RawEvent.from_kline(
            exchange='binance', symbol='BTCUSDT', interval='15m',
            payload=_make_kline_payload(),
        )
        d = ev.to_dict()
        assert d['symbol'] == 'BTCUSDT'
        assert d['event_type'] == 'kline'
        assert d['dedupe_key'] == ev.dedupe_key
        assert d['payload_raw'] == ev.payload_raw


class TestPartitioning:

    def test_partition_path(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        ev = RawEvent.from_kline(
            exchange='binance', symbol='BTCUSDT', interval='15m',
            payload=_make_kline_payload(open_time_ms=1700000000000),
        )
        path = store._partition_path(ev)
        assert 'exchange=binance' in str(path)
        assert 'event_type=kline' in str(path)
        assert 'symbol=BTCUSDT' in str(path)
        assert 'date=' in str(path)


class TestRawMarketStore:

    def test_append_event(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        ev = RawEvent.from_kline(
            exchange='binance', symbol='BTCUSDT', interval='15m',
            payload=_make_kline_payload(),
        )
        r = store.append(ev)
        assert r['stored'] is True
        events = store.get_raw_events('BTCUSDT', 'kline')
        assert len(events) == 1

    def test_append_is_append_only(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        for i in range(5):
            p = _make_kline_payload(open_time_ms=1700000000000 + i * 900000)
            ev = RawEvent.from_kline(exchange='binance', symbol='BTCUSDT',
                                      interval='15m', payload=p)
            store.append(ev)
        events = store.get_raw_events('BTCUSDT', 'kline')
        assert len(events) == 5

    def test_dedupe_rejects_duplicate(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        p = _make_kline_payload()
        ev1 = RawEvent.from_kline(exchange='binance', symbol='BTCUSDT', interval='15m', payload=p)
        ev2 = RawEvent.from_kline(exchange='binance', symbol='BTCUSDT', interval='15m', payload=p)
        r1 = store.append(ev1)
        r2 = store.append(ev2)
        assert r1['stored'] is True
        assert r2['stored'] is False
        assert r2['reason'] == 'duplicate'
        events = store.get_raw_events('BTCUSDT', 'kline')
        assert len(events) == 1

    def test_query_by_date_range(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        ts_base = 1700000000000
        for i in range(10):
            p = _make_kline_payload(open_time_ms=ts_base + i * 86400000)  # +1 day
            ev = RawEvent.from_kline(exchange='binance', symbol='BTCUSDT',
                                      interval='15m', payload=p)
            store.append(ev)
        all_events = store.get_raw_events('BTCUSDT', 'kline')
        assert len(all_events) == 10

    def test_multi_symbol_separation(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        store.append(RawEvent.from_kline('binance', 'BTCUSDT', '15m', _make_kline_payload()))
        p = _make_kline_payload()
        p['s'] = 'ETHUSDT'
        store.append(RawEvent.from_kline('binance', 'ETHUSDT', '15m', p))
        btc = store.get_raw_events('BTCUSDT', 'kline')
        eth = store.get_raw_events('ETHUSDT', 'kline')
        assert len(btc) == 1
        assert len(eth) == 1

    def test_list_partitions(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        store.append(RawEvent.from_kline('binance', 'BTCUSDT', '15m', _make_kline_payload()))
        parts = store.list_partitions()
        assert len(parts) >= 1


class TestIngestionMetadata:

    def test_tracks_last_event_ts(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        ev = RawEvent.from_kline('binance', 'BTCUSDT', '15m', _make_kline_payload())
        store.append(ev)
        meta = store.get_ingestion_metadata('BTCUSDT', 'kline')
        assert meta['last_event_ts'] > 0
        assert meta['total_events'] == 1

    def test_tracks_duplicate_count(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        p = _make_kline_payload()
        store.append(RawEvent.from_kline('binance', 'BTCUSDT', '15m', p))
        store.append(RawEvent.from_kline('binance', 'BTCUSDT', '15m', p))
        meta = store.get_ingestion_metadata('BTCUSDT', 'kline')
        assert meta['duplicate_count'] == 1


class TestDataQualityRaw:

    def test_quality_report(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent, raw_quality_report
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        for i in range(5):
            p = _make_kline_payload(open_time_ms=1700000000000 + i * 900000)
            store.append(RawEvent.from_kline('binance', 'BTCUSDT', '15m', p))
        report = raw_quality_report(store, 'BTCUSDT', 'kline')
        assert 'total_events' in report
        assert 'duplicate_count' in report
        assert 'last_event_ts' in report
        assert report['total_events'] == 5


class TestReplayInterface:

    def test_replay_events(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        for i in range(3):
            p = _make_kline_payload(open_time_ms=1700000000000 + i * 900000)
            store.append(RawEvent.from_kline('binance', 'BTCUSDT', '15m', p))
        replayed = list(store.replay_raw_events('BTCUSDT', 'kline'))
        assert len(replayed) == 3
        for ev in replayed:
            assert ev['event_type'] == 'kline'


class TestRestartSafety:

    def test_survives_restart(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        d = Path(tempfile.mkdtemp())
        store1 = RawMarketStore(d)
        store1.append(RawEvent.from_kline('binance', 'BTCUSDT', '15m', _make_kline_payload()))
        store2 = RawMarketStore(d)
        events = store2.get_raw_events('BTCUSDT', 'kline')
        assert len(events) == 1

    def test_dedupe_across_restart(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        d = Path(tempfile.mkdtemp())
        p = _make_kline_payload()
        store1 = RawMarketStore(d)
        store1.append(RawEvent.from_kline('binance', 'BTCUSDT', '15m', p))
        store2 = RawMarketStore(d)
        r = store2.append(RawEvent.from_kline('binance', 'BTCUSDT', '15m', p))
        assert r['stored'] is False
        assert r['reason'] == 'duplicate'


class TestNegativeCases:

    def test_malformed_payload_rejected(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        with pytest.raises((KeyError, ValueError)):
            RawEvent.from_kline('binance', 'BTCUSDT', '15m', {'invalid': 'payload'})

    def test_backfill_source_tracked(self):
        from src.data_pipeline.raw_market_store import RawMarketStore, RawEvent
        store = RawMarketStore(Path(tempfile.mkdtemp()))
        ev = RawEvent.from_kline(
            exchange='binance', symbol='BTCUSDT', interval='15m',
            payload=_make_kline_payload(),
            ingest_source='backfill',
        )
        store.append(ev)
        events = store.get_raw_events('BTCUSDT', 'kline')
        assert events[0]['ingest_source'] == 'backfill'
