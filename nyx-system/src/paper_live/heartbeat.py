"""
Heartbeat — file-based liveness signal.

The strategy calls tick() each bar; Docker HEALTHCHECK or systemd watchdog
reads the file and considers the service dead if age > threshold.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


class Heartbeat:
    """Atomic JSON heartbeat file."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def tick(self, context: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()
        iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        payload: Dict[str, Any] = {"ts": now, "iso": iso}
        if context:
            payload.update(context)
        # Atomic write.
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(payload))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def age_seconds(self) -> float:
        if not self.path.exists():
            return math.inf
        try:
            data = json.loads(self.path.read_text())
            last = float(data.get("ts", 0.0))
        except (OSError, ValueError, json.JSONDecodeError):
            return math.inf
        return max(0.0, time.time() - last)

    def is_alive(self, max_age_s: float) -> bool:
        return self.age_seconds() < max_age_s
