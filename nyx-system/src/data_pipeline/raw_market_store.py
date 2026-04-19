"""
Raw Market Data Lake — Ticket DATA-1.1.

Append-only, partitioned, exchange-native raw event storage.
Canonical source for replay, bar rebuild, debug, retraining.

Partitioning:
  raw/
    exchange={exchange}/
      event_type={kline|trade}/
        symbol={symbol}/
          date={YYYY-MM-DD}/
            events.jsonl

Dedupe:
  kline: symbol + interval + start_time + close_flag
  trade: symbol + trade_id

Format: JSONL (append-only, inspectable, deterministic).
Metadata sidecar for fast status checks.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


SCHEMA_VERSION = 1


@dataclass
class RawEvent:
    """Canonical raw market event. Exchange-native fields preserved."""
    exchange: str
    symbol: str
    event_type: str
    event_ts_exchange_ms: int
    payload_raw: Dict[str, Any]
    dedupe_key: str
    ingest_source: str = 'ws'
    schema_version: int = SCHEMA_VERSION
    event_ts_receive: float = field(default_factory=time.time)
    event_ts_store: float = 0.0

    @classmethod
    def from_kline(
        cls,
        exchange: str,
        symbol: str,
        interval: str,
        payload: Dict[str, Any],
        ingest_source: str = 'ws',
    ) -> 'RawEvent':
        if 't' not in payload or 's' not in payload:
            raise KeyError('kline payload missing required fields: t, s')
        open_time_ms = int(payload['t'])
        closed = bool(payload.get('x', False))
        dedupe = f'{symbol}|{interval}|{open_time_ms}|{int(closed)}'
        dedupe_key = hashlib.sha256(dedupe.encode()).hexdigest()[:16]
        return cls(
            exchange=exchange,
            symbol=symbol,
            event_type='kline',
            event_ts_exchange_ms=open_time_ms,
            payload_raw=dict(payload),
            dedupe_key=dedupe_key,
            ingest_source=ingest_source,
        )

    @classmethod
    def from_trade(
        cls,
        exchange: str,
        symbol: str,
        payload: Dict[str, Any],
        ingest_source: str = 'ws',
    ) -> 'RawEvent':
        if 't' not in payload and 'T' not in payload:
            raise KeyError('trade payload missing timestamp')
        if 'a' not in payload and 'id' not in payload:
            raise KeyError('trade payload missing trade id')
        trade_id = payload.get('a', payload.get('id'))
        event_ts = int(payload.get('T', payload.get('t', 0)))
        dedupe = f'{symbol}|trade|{trade_id}'
        dedupe_key = hashlib.sha256(dedupe.encode()).hexdigest()[:16]
        return cls(
            exchange=exchange,
            symbol=symbol,
            event_type='trade',
            event_ts_exchange_ms=event_ts,
            payload_raw=dict(payload),
            dedupe_key=dedupe_key,
            ingest_source=ingest_source,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'exchange': self.exchange,
            'symbol': self.symbol,
            'event_type': self.event_type,
            'event_ts_exchange_ms': self.event_ts_exchange_ms,
            'event_ts_receive': self.event_ts_receive,
            'event_ts_store': self.event_ts_store,
            'payload_raw': self.payload_raw,
            'dedupe_key': self.dedupe_key,
            'ingest_source': self.ingest_source,
            'schema_version': self.schema_version,
        }


class RawMarketStore:
    """Append-only partitioned raw market event store."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._dedupe_cache: Dict[str, set] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def _partition_path(self, event: RawEvent) -> Path:
        date_utc = datetime.fromtimestamp(
            event.event_ts_exchange_ms / 1000, tz=timezone.utc
        ).strftime('%Y-%m-%d')
        return (
            self.base_dir
            / f'exchange={event.exchange}'
            / f'event_type={event.event_type}'
            / f'symbol={event.symbol}'
            / f'date={date_utc}'
        )

    def _events_file(self, event: RawEvent) -> Path:
        return self._partition_path(event) / 'events.jsonl'

    def _meta_file(self, symbol: str, event_type: str) -> Path:
        return self.base_dir / f'meta_{symbol}_{event_type}.json'

    def _meta_key(self, symbol: str, event_type: str) -> str:
        return f'{symbol}|{event_type}'

    def _load_dedupe_cache(self, symbol: str, event_type: str) -> set:
        k = self._meta_key(symbol, event_type)
        if k in self._dedupe_cache:
            return self._dedupe_cache[k]
        seen = set()
        prefix_parts = [
            d for d in self.base_dir.rglob('events.jsonl')
            if f'symbol={symbol}' in str(d) and f'event_type={event_type}' in str(d)
        ]
        for path in prefix_parts:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        seen.add(ev.get('dedupe_key'))
                    except Exception:
                        continue
        self._dedupe_cache[k] = seen
        return seen

    def _load_meta(self, symbol: str, event_type: str) -> Dict[str, Any]:
        k = self._meta_key(symbol, event_type)
        if k in self._meta:
            return self._meta[k]
        path = self._meta_file(symbol, event_type)
        if path.exists():
            self._meta[k] = json.loads(path.read_text())
        else:
            self._meta[k] = {
                'symbol': symbol,
                'event_type': event_type,
                'total_events': 0,
                'duplicate_count': 0,
                'last_event_ts': 0,
                'last_store_ts': 0,
                'write_failures': 0,
            }
        return self._meta[k]

    def _save_meta(self, symbol: str, event_type: str) -> None:
        k = self._meta_key(symbol, event_type)
        if k not in self._meta:
            return
        path = self._meta_file(symbol, event_type)
        path.write_text(json.dumps(self._meta[k], indent=2))

    def append(self, event: RawEvent) -> Dict[str, Any]:
        meta = self._load_meta(event.symbol, event.event_type)
        cache = self._load_dedupe_cache(event.symbol, event.event_type)

        if event.dedupe_key in cache:
            meta['duplicate_count'] += 1
            self._save_meta(event.symbol, event.event_type)
            return {'stored': False, 'reason': 'duplicate', 'dedupe_key': event.dedupe_key}

        event.event_ts_store = time.time()
        path = self._events_file(event)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'a') as f:
                f.write(json.dumps(event.to_dict(), default=str) + '\n')
        except Exception as exc:
            meta['write_failures'] += 1
            self._save_meta(event.symbol, event.event_type)
            return {'stored': False, 'reason': 'write_error', 'error': str(exc)}

        cache.add(event.dedupe_key)
        meta['total_events'] += 1
        meta['last_event_ts'] = max(meta['last_event_ts'], event.event_ts_exchange_ms)
        meta['last_store_ts'] = event.event_ts_store
        self._save_meta(event.symbol, event.event_type)
        return {'stored': True, 'dedupe_key': event.dedupe_key}

    def get_raw_events(
        self,
        symbol: str,
        event_type: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for path in self.base_dir.rglob('events.jsonl'):
            if f'symbol={symbol}' not in str(path):
                continue
            if f'event_type={event_type}' not in str(path):
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    ts = ev.get('event_ts_exchange_ms', 0)
                    if start_ms is not None and ts < start_ms:
                        continue
                    if end_ms is not None and ts > end_ms:
                        continue
                    events.append(ev)
        events.sort(key=lambda e: e.get('event_ts_exchange_ms', 0))
        return events

    def replay_raw_events(
        self,
        symbol: str,
        event_type: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        for ev in self.get_raw_events(symbol, event_type, start_ms, end_ms):
            yield ev

    def list_partitions(self) -> List[str]:
        parts = []
        for path in self.base_dir.rglob('events.jsonl'):
            rel = path.relative_to(self.base_dir).parent
            parts.append(str(rel))
        return sorted(set(parts))

    def get_ingestion_metadata(self, symbol: str, event_type: str) -> Dict[str, Any]:
        return dict(self._load_meta(symbol, event_type))


def raw_quality_report(
    store: RawMarketStore,
    symbol: str,
    event_type: str,
) -> Dict[str, Any]:
    meta = store.get_ingestion_metadata(symbol, event_type)
    events = store.get_raw_events(symbol, event_type)
    n = len(events)
    out_of_order = 0
    last_ts = 0
    for ev in events:
        ts = ev.get('event_ts_exchange_ms', 0)
        if ts < last_ts:
            out_of_order += 1
        last_ts = ts
    return {
        'symbol': symbol,
        'event_type': event_type,
        'total_events': n,
        'duplicate_count': meta.get('duplicate_count', 0),
        'write_failures': meta.get('write_failures', 0),
        'last_event_ts': meta.get('last_event_ts', 0),
        'out_of_order': out_of_order,
        'quality': 'good' if out_of_order == 0 and meta.get('write_failures', 0) == 0 else 'degraded',
    }
