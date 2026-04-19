"""
TDD Tests — Ticket ML-1 — Training Pipeline + Model Registry Core.

Dataset snapshots, model versioning, promotion, rollback.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestDatasetSnapshot:

    def test_create_snapshot(self):
        from src.ml_pipeline.dataset_builder import DatasetSnapshot
        snap = DatasetSnapshot(
            dataset_id='ds-001', asset='BTCUSDT',
            start_date='2020-01-01', end_date='2022-12-31',
            feature_version='v1', label_version='triple_barrier',
            n_samples=1248, feature_names=['f1', 'f2'],
        )
        assert snap.dataset_id == 'ds-001'
        assert snap.n_samples == 1248

    def test_snapshot_serializable(self):
        from src.ml_pipeline.dataset_builder import DatasetSnapshot
        snap = DatasetSnapshot(
            dataset_id='ds-002', asset='ETHUSDT',
            start_date='2020-01-01', end_date='2022-12-31',
            feature_version='v1', label_version='triple_barrier',
            n_samples=500, feature_names=['a', 'b'],
        )
        d = snap.to_dict()
        s = json.dumps(d)
        assert 'ds-002' in s

    def test_snapshot_from_dict(self):
        from src.ml_pipeline.dataset_builder import DatasetSnapshot
        d = {
            'dataset_id': 'ds-003', 'asset': 'SOLUSDT',
            'start_date': '2020-08-01', 'end_date': '2022-12-31',
            'feature_version': 'v1', 'label_version': 'tb',
            'n_samples': 300, 'feature_names': ['x'],
        }
        snap = DatasetSnapshot.from_dict(d)
        assert snap.asset == 'SOLUSDT'

    def test_snapshot_has_fingerprint(self):
        from src.ml_pipeline.dataset_builder import DatasetSnapshot
        snap = DatasetSnapshot(
            dataset_id='ds-004', asset='BTCUSDT',
            start_date='2020-01-01', end_date='2022-12-31',
            feature_version='v1', label_version='tb',
            n_samples=100, feature_names=['a'],
        )
        assert snap.fingerprint is not None
        assert len(snap.fingerprint) > 0


class TestModelRegistry:

    def test_register_model(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register(
            model_id='m-001', asset='BTCUSDT',
            dataset_id='ds-001', feature_version='v1',
            artifact_path='/models/BTCUSDT',
            validation_summary={'sharpe': 7.0, 'accuracy': 0.9},
        )
        m = reg.get('m-001')
        assert m is not None
        assert m['asset'] == 'BTCUSDT'
        assert m['status'] == 'candidate'

    def test_promote_to_approved(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/path',
                      {'sharpe': 5.0})
        reg.promote('m-001', 'approved')
        assert reg.get('m-001')['status'] == 'approved'

    def test_promote_to_production(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/path',
                      {'sharpe': 5.0})
        reg.promote('m-001', 'approved')
        reg.promote('m-001', 'production')
        assert reg.get('m-001')['status'] == 'production'

    def test_only_one_production_per_asset(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/p1', {})
        reg.register('m-002', 'BTCUSDT', 'ds-002', 'v1', '/p2', {})
        reg.promote('m-001', 'approved')
        reg.promote('m-001', 'production')
        reg.promote('m-002', 'approved')
        reg.promote('m-002', 'production')
        assert reg.get('m-001')['status'] == 'retired'
        assert reg.get('m-002')['status'] == 'production'

    def test_get_production_model(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/p', {})
        reg.promote('m-001', 'approved')
        reg.promote('m-001', 'production')
        prod = reg.get_production('BTCUSDT')
        assert prod is not None
        assert prod['model_id'] == 'm-001'

    def test_no_production_returns_none(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        assert reg.get_production('BTCUSDT') is None

    def test_rollback(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/p1', {})
        reg.register('m-002', 'BTCUSDT', 'ds-002', 'v1', '/p2', {})
        reg.promote('m-001', 'approved')
        reg.promote('m-001', 'production')
        reg.promote('m-002', 'approved')
        reg.promote('m-002', 'production')
        reg.rollback('BTCUSDT', 'm-001')
        assert reg.get('m-001')['status'] == 'production'
        assert reg.get('m-002')['status'] == 'retired'

    def test_list_models(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/p', {})
        reg.register('m-002', 'ETHUSDT', 'ds-002', 'v1', '/p', {})
        models = reg.list_models()
        assert len(models) == 2

    def test_list_models_by_asset(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/p', {})
        reg.register('m-002', 'BTCUSDT', 'ds-002', 'v1', '/p', {})
        reg.register('m-003', 'ETHUSDT', 'ds-003', 'v1', '/p', {})
        btc = reg.list_models(asset='BTCUSDT')
        assert len(btc) == 2

    def test_registry_persists(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        d = Path(tempfile.mkdtemp())
        reg1 = ModelRegistryV2(d)
        reg1.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/p', {})
        reg2 = ModelRegistryV2(d)
        assert reg2.get('m-001') is not None


class TestPromotionValidation:

    def test_cannot_promote_nonexistent(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        with pytest.raises(KeyError):
            reg.promote('nonexistent', 'approved')

    def test_cannot_skip_to_production(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/p', {})
        with pytest.raises(ValueError):
            reg.promote('m-001', 'production')

    def test_cannot_promote_to_invalid_status(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/p', {})
        with pytest.raises(ValueError):
            reg.promote('m-001', 'magic')


class TestRegistryAudit:

    def test_actions_logged(self):
        from src.ml_pipeline.model_registry_v2 import ModelRegistryV2
        reg = ModelRegistryV2(Path(tempfile.mkdtemp()))
        reg.register('m-001', 'BTCUSDT', 'ds-001', 'v1', '/p', {})
        reg.promote('m-001', 'approved')
        assert len(reg.audit_log) >= 2
        assert reg.audit_log[0]['action'] == 'register'
        assert reg.audit_log[1]['action'] == 'promote'
