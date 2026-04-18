"""
TDD Tests — Ticket 41 — Per-Asset ML Artifact Validation.

Ensures ETH and SOL have dedicated, current artifacts trained on
the canonical full 128-feature stack. No cross-asset contamination.
"""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
MODELS = HERE / 'models'
CANONICAL_FEATURE_COUNT = 128
REQUIRED_PREFIXES = ('h1_', 'h4_', 'd1_')


class TestArtifactExistence:

    @pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    def test_model_file_exists(self, symbol):
        assert (MODELS / symbol / 'ml_filter_v1.pkl').exists()

    @pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    def test_scaler_file_exists(self, symbol):
        assert (MODELS / symbol / 'scaler.pkl').exists()

    @pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    def test_feature_names_file_exists(self, symbol):
        assert (MODELS / symbol / 'feature_names.json').exists()

    @pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    def test_training_metadata_exists(self, symbol):
        assert (MODELS / symbol / 'training_metadata.json').exists()


class TestFeatureCoverage:

    @pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    def test_feature_count_is_canonical(self, symbol):
        fn = json.load(open(MODELS / symbol / 'feature_names.json'))
        assert len(fn) == CANONICAL_FEATURE_COUNT, (
            f'{symbol}: {len(fn)} features, expected {CANONICAL_FEATURE_COUNT}'
        )

    @pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    def test_required_prefixes_present(self, symbol):
        fn = json.load(open(MODELS / symbol / 'feature_names.json'))
        for prefix in REQUIRED_PREFIXES:
            assert any(f.startswith(prefix) for f in fn), (
                f'{symbol}: missing {prefix}* features'
            )

    def test_all_assets_same_feature_names(self):
        sets = {}
        for sym in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT'):
            sets[sym] = set(json.load(open(MODELS / sym / 'feature_names.json')))
        assert sets['BTCUSDT'] == sets['ETHUSDT'], 'BTC != ETH features'
        assert sets['BTCUSDT'] == sets['SOLUSDT'], 'BTC != SOL features'


class TestTrainingMetadata:

    @pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    def test_metadata_has_required_fields(self, symbol):
        meta = json.load(open(MODELS / symbol / 'training_metadata.json'))
        for field in ('symbol', 'train_end', 'n_train_candidates',
                      'n_features', 'in_sample_accuracy'):
            assert field in meta, f'{symbol}: missing {field}'

    @pytest.mark.parametrize('symbol', ['ETHUSDT', 'SOLUSDT'])
    def test_metadata_symbol_matches(self, symbol):
        meta = json.load(open(MODELS / symbol / 'training_metadata.json'))
        assert meta['symbol'] == symbol

    @pytest.mark.parametrize('symbol', ['ETHUSDT', 'SOLUSDT'])
    def test_metadata_feature_count_matches(self, symbol):
        meta = json.load(open(MODELS / symbol / 'training_metadata.json'))
        assert meta['n_features'] == CANONICAL_FEATURE_COUNT

    @pytest.mark.parametrize('symbol', ['ETHUSDT', 'SOLUSDT'])
    def test_sufficient_training_candidates(self, symbol):
        meta = json.load(open(MODELS / symbol / 'training_metadata.json'))
        assert meta['n_train_candidates'] >= 100, (
            f'{symbol}: only {meta["n_train_candidates"]} candidates'
        )


class TestRuntimeCompatibility:

    @pytest.mark.parametrize('symbol', ['ETHUSDT', 'SOLUSDT'])
    def test_runtime_loads_correct_artifact(self, symbol):
        from src.live.nyx_runtime import NYXRuntime
        rt = NYXRuntime(
            symbol=symbol,
            models_dir=MODELS / symbol,
            state_dir=Path(tempfile.mkdtemp()) / symbol,
        )
        assert rt.decider.symbol == symbol
        assert len(rt.decider.feature_names) == CANONICAL_FEATURE_COUNT

    def test_eth_does_not_load_btc_model(self):
        from src.live.nyx_runtime import NYXRuntime
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=MODELS / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt_eth = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=MODELS / 'ETHUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'ETHUSDT',
        )
        assert rt_btc.decider._meta is not rt_eth.decider._meta

    def test_sol_does_not_load_btc_model(self):
        from src.live.nyx_runtime import NYXRuntime
        rt_btc = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=MODELS / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        rt_sol = NYXRuntime(
            symbol='SOLUSDT',
            models_dir=MODELS / 'SOLUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'SOLUSDT',
        )
        assert rt_btc.decider._meta is not rt_sol.decider._meta


class TestOOSReportsExist:

    @pytest.mark.parametrize('symbol', ['ETHUSDT', 'SOLUSDT'])
    def test_full_stack_oos_report_exists(self, symbol):
        path = HERE / 'reports' / f'{symbol}_full_stack_oos.json'
        assert path.exists(), f'{symbol} full-stack OOS report missing'

    @pytest.mark.parametrize('symbol', ['ETHUSDT', 'SOLUSDT'])
    def test_oos_report_has_both_modes(self, symbol):
        path = HERE / 'reports' / f'{symbol}_full_stack_oos.json'
        data = json.loads(path.read_text())
        assert 'idealized_baseline' in data
        assert 'performance' in data
        assert 'execution' in data

    @pytest.mark.parametrize('symbol', ['ETHUSDT', 'SOLUSDT'])
    def test_oos_report_uses_correct_symbol(self, symbol):
        path = HERE / 'reports' / f'{symbol}_full_stack_oos.json'
        data = json.loads(path.read_text())
        assert data.get('symbol') == symbol
