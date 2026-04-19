"""
Canonical Bar Store — Ticket DATA-1.

Append-only persistent storage for market bars. Supports multiple
symbols and timeframes. Deduplication, gap detection, quality checks.

Ready for backfill (historical) + live (WebSocket) ingestion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class CanonicalBarStore:
    """Persistent bar storage per symbol × timeframe."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, timeframe: str) -> Path:
        return self.base_dir / f'{symbol}_{timeframe}.parquet'

    def append(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> None:
        path = self._path(symbol, timeframe)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, bars])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
        else:
            combined = bars.copy()
        combined.to_parquet(path)

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        path = self._path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        return df

    def get_metadata(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        df = self.get_bars(symbol, timeframe)
        if df.empty:
            return {'n_bars': 0, 'first_ts': None, 'last_ts': None}
        return {
            'n_bars': len(df),
            'first_ts': str(df.index.min()),
            'last_ts': str(df.index.max()),
        }

    def list_symbols(self) -> List[str]:
        syms = set()
        for p in self.base_dir.glob('*.parquet'):
            parts = p.stem.rsplit('_', 1)
            if len(parts) == 2:
                syms.add(parts[0])
        return sorted(syms)


def detect_gaps(
    index: pd.DatetimeIndex,
    freq: str = '15min',
) -> List[Dict[str, Any]]:
    if len(index) < 2:
        return []
    expected = pd.date_range(index[0], index[-1], freq=freq)
    missing = expected.difference(index)
    gaps = []
    for ts in missing:
        gaps.append({'missing_ts': str(ts), 'freq': freq})
    return gaps


def detect_duplicates(index: pd.DatetimeIndex) -> List[str]:
    dups = index[index.duplicated()]
    return [str(ts) for ts in dups]


def data_quality_report(
    store: CanonicalBarStore,
    symbol: str,
    timeframe: str,
) -> Dict[str, Any]:
    df = store.get_bars(symbol, timeframe)
    if df.empty:
        return {'n_bars': 0, 'n_gaps': 0, 'n_duplicates': 0, 'quality': 'empty'}
    gaps = detect_gaps(df.index, freq=timeframe)
    dups = detect_duplicates(df.index)
    quality = 'good' if not gaps and not dups else 'degraded'
    return {
        'n_bars': len(df),
        'n_gaps': len(gaps),
        'n_duplicates': len(dups),
        'first_ts': str(df.index.min()),
        'last_ts': str(df.index.max()),
        'quality': quality,
    }
