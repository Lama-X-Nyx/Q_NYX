"""
Dataset Builder — Ticket ML-1.

Deterministic dataset snapshots from canonical bar store.
Versioned, reproducible, no leakage.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetSnapshot:
    dataset_id: str
    asset: str
    start_date: str
    end_date: str
    feature_version: str
    label_version: str
    n_samples: int
    feature_names: List[str]
    build_timestamp: float = field(default_factory=time.time)

    @property
    def fingerprint(self) -> str:
        raw = (f'{self.dataset_id}|{self.asset}|{self.start_date}|'
               f'{self.end_date}|{self.feature_version}|{self.label_version}|'
               f'{self.n_samples}|{sorted(self.feature_names)}')
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_id': self.dataset_id,
            'asset': self.asset,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'feature_version': self.feature_version,
            'label_version': self.label_version,
            'n_samples': self.n_samples,
            'feature_names': self.feature_names,
            'build_timestamp': self.build_timestamp,
            'fingerprint': self.fingerprint,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DatasetSnapshot':
        return cls(
            dataset_id=d['dataset_id'],
            asset=d['asset'],
            start_date=d['start_date'],
            end_date=d['end_date'],
            feature_version=d['feature_version'],
            label_version=d['label_version'],
            n_samples=d['n_samples'],
            feature_names=d.get('feature_names', []),
            build_timestamp=d.get('build_timestamp', time.time()),
        )
