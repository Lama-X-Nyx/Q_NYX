"""
TDD Tests — StateManager

Durable state for warm restart.

Contract:
  - Snapshot full runtime state (positions, capital, last_bar_ts, model version)
    atomically (tmp file → fsync → rename).
  - Load previous snapshot if exists.
  - Missing snapshot → fresh state (not an error).
  - Corrupted snapshot → raises a specific error (not silent loss).
"""
import json
from pathlib import Path

import pytest


def _state(**over) -> dict:
    base = {
        'capital': 10_000.0,
        'equity': 10_123.4,
        'positions': {
            'BTCUSDT': {
                'entry_price': 45000.0,
                'qty': 0.01,
                'stop': 44100.0,
                'tp': 46800.0,
                'direction': 1,
                'entry_time': '2025-01-01T00:00:00+00:00',
            }
        },
        'last_bar_ts': {
            'BTCUSDT_15m': '2025-01-01T00:00:00+00:00',
            'BTCUSDT_1h':  '2025-01-01T00:00:00+00:00',
            'BTCUSDT_1d':  '2024-12-31T00:00:00+00:00',
        },
        'model_version': 'v0.3.2',
        'schema_version': 1,
    }
    base.update(over)
    return base


# ===========================================================================
# TEST 1: save + load roundtrip
# ===========================================================================
class TestRoundtrip:

    def test_save_then_load(self, tmp_path):
        from src.paper_live.state_manager import StateManager
        sm = StateManager(tmp_path / "state.json")
        st = _state()
        sm.save(st)
        loaded = sm.load()
        assert loaded == st

    def test_load_missing_returns_none(self, tmp_path):
        from src.paper_live.state_manager import StateManager
        sm = StateManager(tmp_path / "state.json")
        assert sm.load() is None


# ===========================================================================
# TEST 2: atomic write
# ===========================================================================
class TestAtomicity:

    def test_no_partial_file_left_on_tmp(self, tmp_path):
        """After save, there must not be a leftover .tmp file."""
        from src.paper_live.state_manager import StateManager
        path = tmp_path / "state.json"
        sm = StateManager(path)
        sm.save(_state())
        leftovers = list(tmp_path.glob("state.json.tmp*"))
        assert leftovers == []

    def test_overwrite_preserves_latest(self, tmp_path):
        """Calling save twice must leave only the latest state."""
        from src.paper_live.state_manager import StateManager
        sm = StateManager(tmp_path / "state.json")
        sm.save(_state(capital=10_000.0))
        sm.save(_state(capital=12_345.0))
        st = sm.load()
        assert st is not None
        assert st['capital'] == 12_345.0


# ===========================================================================
# TEST 3: corruption detection
# ===========================================================================
class TestCorruption:

    def test_corrupted_json_raises(self, tmp_path):
        from src.paper_live.state_manager import StateManager, StateCorruptionError
        path = tmp_path / "state.json"
        path.write_text("{not valid json")
        sm = StateManager(path)
        with pytest.raises(StateCorruptionError):
            sm.load()

    def test_schema_version_mismatch_raises(self, tmp_path):
        from src.paper_live.state_manager import StateManager, StateSchemaError
        sm = StateManager(tmp_path / "state.json")
        sm.save(_state(schema_version=99))
        with pytest.raises(StateSchemaError):
            sm.load()


# ===========================================================================
# TEST 4: required fields
# ===========================================================================
class TestValidation:

    def test_save_strips_non_serializable(self, tmp_path):
        """Non-JSON-serializable fields should raise a TypeError at save-time."""
        from src.paper_live.state_manager import StateManager
        sm = StateManager(tmp_path / "state.json")
        with pytest.raises((TypeError, ValueError)):
            sm.save({'bad': object()})

    def test_last_ts_per_tf_preserved(self, tmp_path):
        from src.paper_live.state_manager import StateManager
        sm = StateManager(tmp_path / "state.json")
        sm.save(_state())
        st = sm.load()
        assert st is not None
        assert st['last_bar_ts']['BTCUSDT_15m'] == '2025-01-01T00:00:00+00:00'
        assert st['last_bar_ts']['BTCUSDT_1d'] == '2024-12-31T00:00:00+00:00'
