"""
StateManager — Atomic state snapshot for warm restart.

Writes go through a tmp file + fsync + rename so a crash cannot leave a
half-written state.json on disk. Corrupted or schema-incompatible files
raise explicit exceptions — never silently zero the strategy.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union


CURRENT_SCHEMA_VERSION = 1


class StateError(Exception):
    """Base class for state-manager errors."""


class StateCorruptionError(StateError):
    """Raised when state file exists but is unreadable / malformed."""


class StateSchemaError(StateError):
    """Raised when the state file has an unsupported schema_version."""


class StateManager:
    """Atomic JSON snapshot of runtime state."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    def save(self, state: Dict[str, Any]) -> None:
        """Serialize `state` to disk atomically."""
        # Fail fast on non-serializable content (TypeError) before touching disk.
        payload = json.dumps(state, default=None)

        # Write to tmp in same dir (same filesystem → atomic rename).
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.path)
        except Exception:
            # Clean up the tmp file if rename failed.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # -----------------------------------------------------------------
    def load(self) -> Optional[Dict[str, Any]]:
        """Return previously-saved state, or None if no snapshot exists."""
        if not self.path.exists():
            return None
        try:
            text = self.path.read_text()
            state = json.loads(text)
        except json.JSONDecodeError as e:
            raise StateCorruptionError(
                f"State file {self.path} is not valid JSON: {e}"
            ) from e
        except OSError as e:
            raise StateCorruptionError(
                f"State file {self.path} could not be read: {e}"
            ) from e

        schema = state.get("schema_version", None)
        if schema is not None and schema != CURRENT_SCHEMA_VERSION:
            raise StateSchemaError(
                f"schema_version={schema} in {self.path} incompatible with "
                f"current={CURRENT_SCHEMA_VERSION}"
            )
        return state
