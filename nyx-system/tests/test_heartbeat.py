"""
TDD Tests — Heartbeat

File-based liveness signal (can be monitored by docker HEALTHCHECK or systemd).

Contract:
  - tick(): atomically update heartbeat file with current UTC timestamp.
  - age_seconds(): how old is the last heartbeat.
  - is_alive(max_age_s): True if age < max_age_s.
  - Missing file → is_alive returns False, age_seconds returns infinity.
"""
import time
from pathlib import Path

import pytest


class TestHeartbeatBasic:

    def test_tick_creates_file(self, tmp_path):
        from src.paper_live.heartbeat import Heartbeat
        hb = Heartbeat(tmp_path / "hb.json")
        hb.tick()
        assert (tmp_path / "hb.json").exists()

    def test_age_is_small_after_tick(self, tmp_path):
        from src.paper_live.heartbeat import Heartbeat
        hb = Heartbeat(tmp_path / "hb.json")
        hb.tick()
        age = hb.age_seconds()
        assert 0 <= age < 2.0

    def test_is_alive_after_tick(self, tmp_path):
        from src.paper_live.heartbeat import Heartbeat
        hb = Heartbeat(tmp_path / "hb.json")
        hb.tick()
        assert hb.is_alive(max_age_s=60) is True

    def test_is_alive_false_when_stale(self, tmp_path):
        from src.paper_live.heartbeat import Heartbeat
        hb = Heartbeat(tmp_path / "hb.json")
        hb.tick()
        # Simulate stale: manually rewrite file with old timestamp.
        import json
        (tmp_path / "hb.json").write_text(
            json.dumps({"ts": 0.0, "iso": "1970-01-01T00:00:00Z"})
        )
        assert hb.is_alive(max_age_s=60) is False


class TestHeartbeatMissing:

    def test_is_alive_false_when_file_missing(self, tmp_path):
        from src.paper_live.heartbeat import Heartbeat
        hb = Heartbeat(tmp_path / "hb.json")
        assert hb.is_alive(max_age_s=60) is False

    def test_age_infinite_when_file_missing(self, tmp_path):
        from src.paper_live.heartbeat import Heartbeat
        hb = Heartbeat(tmp_path / "hb.json")
        import math
        assert math.isinf(hb.age_seconds())


class TestHeartbeatContext:

    def test_tick_includes_iso_timestamp(self, tmp_path):
        from src.paper_live.heartbeat import Heartbeat
        hb = Heartbeat(tmp_path / "hb.json")
        hb.tick(context={"bars_processed": 1234})
        import json
        data = json.loads((tmp_path / "hb.json").read_text())
        assert 'iso' in data
        assert 'ts' in data
        assert data['bars_processed'] == 1234
