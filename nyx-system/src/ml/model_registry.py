"""
Model Registry — versioned model lifecycle (Ticket 28).

Extends the EXISTING `train_asset_model` save/load path (Rule 5 —
minimal surface, natural owner). Every model version is stored with
full metadata. Promotion requires validation. Rollback is instant.

Layout :
  registry_dir/
    BTCUSDT/
      v1/
        ml_filter_v1.pkl
        scaler.pkl
        feature_names.json
        training_metadata.json
        registry_meta.json    ← version + validation + promotion status
      v2/
        ...
      active.json             ← {version: "v2", promoted_at: ...}
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """Versioned model store with promotion + rollback."""

    def __init__(
        self,
        registry_dir: Path,
        require_validation: bool = False,
    ) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.require_validation = require_validation

    def _sym_dir(self, symbol: str) -> Path:
        d = self.registry_dir / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _next_version(self, symbol: str) -> str:
        existing = self.list_versions(symbol)
        if not existing:
            return 'v1'
        last_num = max(int(v[1:]) for v in existing)
        return f'v{last_num + 1}'

    # ------------------------------------------------------------------
    def register(
        self,
        symbol: str,
        artifact_dir: Path,
        validation: Dict[str, Any],
    ) -> str:
        """Copy artifacts into the registry under a new version."""
        version = self._next_version(symbol)
        dest = self._sym_dir(symbol) / version
        dest.mkdir(parents=True, exist_ok=True)

        for f in Path(artifact_dir).iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(dest / f.name))

        meta = {
            'symbol': symbol,
            'version': version,
            'registered_at': time.time(),
            'validation': dict(validation),
        }
        (dest / 'registry_meta.json').write_text(
            json.dumps(meta, indent=2, default=str)
        )
        return version

    def list_versions(self, symbol: str) -> List[str]:
        sym = self._sym_dir(symbol)
        versions = []
        for d in sorted(sym.iterdir()):
            if d.is_dir() and d.name.startswith('v'):
                versions.append(d.name)
        return versions

    def load(self, symbol: str, version: str) -> Optional[Path]:
        d = self._sym_dir(symbol) / version
        if d.is_dir():
            return d
        return None

    def get_metadata(self, symbol: str, version: str) -> Dict[str, Any]:
        meta_path = self._sym_dir(symbol) / version / 'registry_meta.json'
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    # ------------------------------------------------------------------
    def promote(self, symbol: str, version: str) -> None:
        """Set a version as the active (production) model."""
        meta = self.get_metadata(symbol, version)
        if self.require_validation:
            val = meta.get('validation', {})
            if not val:
                raise ValueError(
                    f'Cannot promote {symbol}/{version} — no validation '
                    'metadata (require_validation=True)'
                )

        active_path = self._sym_dir(symbol) / 'active.json'
        history = []
        if active_path.exists():
            prev = json.loads(active_path.read_text())
            if 'history' in prev:
                history = prev['history']
            if 'version' in prev:
                history.append(prev['version'])

        payload = {
            'version': version,
            'promoted_at': time.time(),
            'history': history,
        }
        active_path.write_text(json.dumps(payload, indent=2))

    def active_version(self, symbol: str) -> Optional[str]:
        active_path = self._sym_dir(symbol) / 'active.json'
        if not active_path.exists():
            return None
        data = json.loads(active_path.read_text())
        return data.get('version')

    def load_active(self, symbol: str) -> Optional[Path]:
        version = self.active_version(symbol)
        if version is None:
            return None
        return self.load(symbol, version)

    def rollback(self, symbol: str) -> None:
        """Revert to the previous promoted version."""
        active_path = self._sym_dir(symbol) / 'active.json'
        if not active_path.exists():
            raise RuntimeError(f'No active version for {symbol}')
        data = json.loads(active_path.read_text())
        history = data.get('history', [])
        if not history:
            raise RuntimeError(
                f'No previous version to rollback to for {symbol}'
            )
        prev = history.pop()
        data['version'] = prev
        data['promoted_at'] = time.time()
        data['history'] = history
        active_path.write_text(json.dumps(data, indent=2))
