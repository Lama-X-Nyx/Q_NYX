"""
OOS Performance Cache — Ticket 45.

Caching, reuse, profiling for the Canonical OOS Engine.
Same outputs, faster execution. Determinism preserved.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


_HERE = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _HERE / 'data' / 'raw'
_MTF_DIR = _DATA_DIR / 'mtf'
_FEAT_DIR = _HERE / 'data' / 'features'

_SOL_CSV = {
    '15m': 'SOLUSDT_15minutes', '1h': 'SOLUSDT_1hour',
    '4h': 'SOLUSDT_4hours', '1d': 'SOLUSDT_1day',
}


def compute_cache_key(
    assets: List[str],
    start: str,
    end: str,
    capital: float,
    mode: str,
) -> str:
    raw = f'{sorted(assets)}|{start}|{end}|{capital}|{mode}'
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class OOSResultCache:
    """Disk-persisted cache for OOS run results."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f'cache_{key}.json'

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def put(self, key: str, data: Dict[str, Any]) -> None:
        p = self._path(key)
        p.write_text(json.dumps(data, indent=2, default=str))

    def list_keys(self) -> List[str]:
        return [
            p.stem.replace('cache_', '')
            for p in self.cache_dir.glob('cache_*.json')
        ]


class DataCache:
    """In-memory cache for OHLCV and feature DataFrames."""

    def __init__(self) -> None:
        self._ohlcv: Dict[str, pd.DataFrame] = {}
        self._features: Dict[str, pd.DataFrame] = {}

    def load_ohlcv(self, symbol: str, tf: str) -> pd.DataFrame:
        key = f'{symbol}_{tf}'
        if key in self._ohlcv:
            return self._ohlcv[key]
        df = self._read_ohlcv(symbol, tf)
        self._ohlcv[key] = df
        return df

    def load_features(self, symbol: str, tf: str) -> pd.DataFrame:
        key = f'{symbol}_{tf}'
        if key in self._features:
            return self._features[key]
        df = pd.read_parquet(_FEAT_DIR / f'{symbol}_features_{tf}.parquet')
        self._features[key] = df
        return df

    @staticmethod
    def _read_ohlcv(symbol: str, tf: str) -> pd.DataFrame:
        if symbol == 'BTCUSDT':
            df = pd.read_csv(_MTF_DIR / f'BTCUSDT_{tf}.csv')
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns='Unnamed: 0')
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        elif symbol == 'ETHUSDT':
            df = pd.read_csv(_DATA_DIR / f'ETHUSDT_{tf}.csv')
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns='Unnamed: 0')
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        elif symbol == 'SOLUSDT':
            name = _SOL_CSV[tf]
            df = pd.read_csv(
                _DATA_DIR / f'{name}.csv',
                usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            )
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime').drop(columns='timestamp')
        else:
            raise ValueError(f'Unknown symbol: {symbol}')
        for c in ('open', 'high', 'low', 'close'):
            df = df[df[c] > 0]
        return df


class OOSProfiler:
    """Lightweight profiling for OOS runs."""

    def __init__(self) -> None:
        self._starts: Dict[str, float] = {}
        self._durations: Dict[str, float] = {}

    def start(self, section: str) -> None:
        self._starts[section] = time.monotonic()

    def stop(self, section: str) -> None:
        if section in self._starts:
            self._durations[section] = time.monotonic() - self._starts[section]
            del self._starts[section]

    def report(self) -> Dict[str, float]:
        return dict(self._durations)
