"""
Model Registry V2 — Ticket ML-1.

Canonical registry for model versions with explicit promotion workflow:
  candidate → approved → production → retired

Only one production model per asset. Promotion audited. Rollback supported.
Persisted to disk (JSON).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_STATUSES = {'candidate', 'approved', 'production', 'retired'}
VALID_TRANSITIONS = {
    ('candidate', 'approved'),
    ('approved', 'production'),
    ('production', 'retired'),
}


class ModelRegistryV2:
    """Persistent model registry with promotion workflow."""

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self._load()

    def _path(self) -> Path:
        return self.registry_dir / 'model_registry.json'

    def _save(self) -> None:
        self._path().write_text(json.dumps(self._models, indent=2, default=str))

    def _load(self) -> None:
        p = self._path()
        if p.exists():
            self._models = json.loads(p.read_text())

    def register(
        self,
        model_id: str,
        asset: str,
        dataset_id: str,
        feature_version: str,
        artifact_path: str,
        validation_summary: Dict[str, Any],
    ) -> None:
        self._models[model_id] = {
            'model_id': model_id,
            'asset': asset,
            'dataset_id': dataset_id,
            'feature_version': feature_version,
            'artifact_path': artifact_path,
            'validation_summary': validation_summary,
            'status': 'candidate',
            'registered_at': time.time(),
            'promoted_at': None,
        }
        self._save()
        self._audit('register', model_id, asset, 'candidate')

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._models.get(model_id)

    def promote(self, model_id: str, target_status: str) -> None:
        if model_id not in self._models:
            raise KeyError(f'Model {model_id} not found')
        if target_status not in VALID_STATUSES:
            raise ValueError(f'Invalid status: {target_status}')

        current = self._models[model_id]['status']
        if (current, target_status) not in VALID_TRANSITIONS:
            raise ValueError(
                f'Invalid transition: {current} → {target_status}'
            )

        if target_status == 'production':
            asset = self._models[model_id]['asset']
            for mid, m in self._models.items():
                if m['asset'] == asset and m['status'] == 'production' and mid != model_id:
                    m['status'] = 'retired'
                    self._audit('auto_retire', mid, asset, 'retired')

        self._models[model_id]['status'] = target_status
        self._models[model_id]['promoted_at'] = time.time()
        self._save()
        self._audit('promote', model_id, self._models[model_id]['asset'], target_status)

    def rollback(self, asset: str, target_model_id: str) -> None:
        if target_model_id not in self._models:
            raise KeyError(f'Model {target_model_id} not found')

        for mid, m in self._models.items():
            if m['asset'] == asset and m['status'] == 'production':
                m['status'] = 'retired'

        self._models[target_model_id]['status'] = 'production'
        self._models[target_model_id]['promoted_at'] = time.time()
        self._save()
        self._audit('rollback', target_model_id, asset, 'production')

    def get_production(self, asset: str) -> Optional[Dict[str, Any]]:
        for m in self._models.values():
            if m['asset'] == asset and m['status'] == 'production':
                return m
        return None

    def list_models(self, asset: Optional[str] = None) -> List[Dict[str, Any]]:
        models = list(self._models.values())
        if asset:
            models = [m for m in models if m['asset'] == asset]
        return models

    def _audit(self, action: str, model_id: str, asset: str, status: str) -> None:
        self.audit_log.append({
            'timestamp': time.time(),
            'action': action,
            'model_id': model_id,
            'asset': asset,
            'status': status,
        })
