"""
TDD Tests — Ticket 28 — Model Governance.

ModelRegistry extends the EXISTING `train_asset_model` path (not a
parallel system). Every model version is stored with metadata.
Promotion requires validation. Rollback is instant.

Natural owner : `src/ml/train_asset_model.py` (already does
save/load). The registry WRAPS it, not replaces it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def registry_dir(tmp_path):
    return tmp_path / 'registry'


@pytest.fixture
def mock_artifacts(tmp_path):
    """Create minimal mock model artifacts."""
    d = tmp_path / 'mock_model'
    d.mkdir()
    (d / 'ml_filter_v1.pkl').write_bytes(b'mock-model-v1')
    (d / 'scaler.pkl').write_bytes(b'mock-scaler')
    (d / 'feature_names.json').write_text('["f1","f2"]')
    (d / 'training_metadata.json').write_text(json.dumps({
        'symbol': 'BTCUSDT', 'train_end': '2022-12-31',
        'n_features': 2, 'in_sample_accuracy': 0.90,
    }))
    return d


# ===========================================================================
class TestModelRegistry:

    def test_register_version(self, registry_dir, mock_artifacts):
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir)
        version = reg.register(
            symbol='BTCUSDT',
            artifact_dir=mock_artifacts,
            validation={'oos_sharpe': 9.96, 'oos_trades': 54},
        )
        assert version == 'v1'

    def test_second_register_increments(self, registry_dir, mock_artifacts):
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir)
        reg.register('BTCUSDT', mock_artifacts, {})
        v2 = reg.register('BTCUSDT', mock_artifacts, {})
        assert v2 == 'v2'

    def test_list_versions(self, registry_dir, mock_artifacts):
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir)
        reg.register('BTCUSDT', mock_artifacts, {})
        reg.register('BTCUSDT', mock_artifacts, {})
        versions = reg.list_versions('BTCUSDT')
        assert versions == ['v1', 'v2']

    def test_load_version(self, registry_dir, mock_artifacts):
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir)
        reg.register('BTCUSDT', mock_artifacts, {})
        loaded = reg.load('BTCUSDT', 'v1')
        assert loaded is not None
        assert (loaded / 'ml_filter_v1.pkl').exists()

    def test_metadata_stored(self, registry_dir, mock_artifacts):
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir)
        reg.register('BTCUSDT', mock_artifacts,
                     {'oos_sharpe': 9.96})
        meta = reg.get_metadata('BTCUSDT', 'v1')
        assert meta['validation']['oos_sharpe'] == 9.96
        assert meta['symbol'] == 'BTCUSDT'


# ===========================================================================
class TestPromotion:

    def test_promote_sets_active(self, registry_dir, mock_artifacts):
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir)
        reg.register('BTCUSDT', mock_artifacts, {})
        reg.promote('BTCUSDT', 'v1')
        assert reg.active_version('BTCUSDT') == 'v1'

    def test_promote_requires_validation(self, registry_dir, mock_artifacts):
        """Cannot promote a version that has no validation metadata."""
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir, require_validation=True)
        reg.register('BTCUSDT', mock_artifacts, validation={})
        with pytest.raises((ValueError, RuntimeError)):
            reg.promote('BTCUSDT', 'v1')

    def test_promote_with_validation_ok(self, registry_dir, mock_artifacts):
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir, require_validation=True)
        reg.register('BTCUSDT', mock_artifacts,
                     validation={'oos_sharpe': 5.0, 'oos_trades': 30})
        reg.promote('BTCUSDT', 'v1')
        assert reg.active_version('BTCUSDT') == 'v1'


# ===========================================================================
class TestRollback:

    def test_rollback_to_previous(self, registry_dir, mock_artifacts):
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir)
        reg.register('BTCUSDT', mock_artifacts, {})
        reg.register('BTCUSDT', mock_artifacts, {})
        reg.promote('BTCUSDT', 'v1')  # first promote
        reg.promote('BTCUSDT', 'v2')  # second — v1 goes to history
        assert reg.active_version('BTCUSDT') == 'v2'
        reg.rollback('BTCUSDT')
        assert reg.active_version('BTCUSDT') == 'v1'

    def test_rollback_no_previous_raises(self, registry_dir, mock_artifacts):
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir)
        reg.register('BTCUSDT', mock_artifacts, {})
        reg.promote('BTCUSDT', 'v1')
        with pytest.raises((ValueError, RuntimeError)):
            reg.rollback('BTCUSDT')

    def test_load_active_returns_promoted_version(self, registry_dir, mock_artifacts):
        from src.ml.model_registry import ModelRegistry
        reg = ModelRegistry(registry_dir)
        reg.register('BTCUSDT', mock_artifacts, {})
        reg.promote('BTCUSDT', 'v1')
        active_dir = reg.load_active('BTCUSDT')
        assert active_dir is not None
        assert (active_dir / 'ml_filter_v1.pkl').exists()
