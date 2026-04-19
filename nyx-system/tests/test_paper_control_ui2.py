"""
TDD Tests — Ticket UI-2 — Paper Trading Control Plane.

PaperControlService wraps NYXRuntime control methods with state
machine, audit logging, and validation. Backend is sovereign.
"""
from __future__ import annotations

import pytest


class TestPaperControlState:

    def test_initial_state_disabled(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        assert svc.current_state == 'paper_disabled'

    def test_enable(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        r = svc.execute('on')
        assert r['success'] is True
        assert svc.current_state == 'paper_enabled'

    def test_disable(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        svc.execute('on')
        r = svc.execute('off')
        assert r['success'] is True
        assert svc.current_state == 'paper_disabled'

    def test_pause(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        svc.execute('on')
        r = svc.execute('pause')
        assert r['success'] is True
        assert svc.current_state == 'paper_paused'

    def test_resume(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        svc.execute('on')
        svc.execute('pause')
        r = svc.execute('resume')
        assert r['success'] is True
        assert svc.current_state == 'paper_enabled'

    def test_pause_when_disabled_rejected(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        r = svc.execute('pause')
        assert r['success'] is False

    def test_resume_when_disabled_rejected(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        r = svc.execute('resume')
        assert r['success'] is False

    def test_critical_block(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        svc.execute('on')
        svc.set_critical_block('risk_engine')
        assert svc.current_state == 'paper_critical_blocked'

    def test_on_does_not_override_critical(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        svc.execute('on')
        svc.set_critical_block('test')
        r = svc.execute('on')
        assert r['success'] is False
        assert svc.current_state == 'paper_critical_blocked'


class TestPaperControlActions:

    def test_cancel_all(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        svc.execute('on')
        r = svc.execute('cancel_all')
        assert r['success'] is True

    def test_flatten(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        svc.execute('on')
        r = svc.execute('flatten')
        assert r['success'] is True

    def test_cancel_all_when_disabled_ok(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        r = svc.execute('cancel_all')
        assert r['success'] is True

    def test_invalid_action_rejected(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        r = svc.execute('invalid_action')
        assert r['success'] is False


class TestAuditLogging:

    def test_actions_logged(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        svc.execute('on')
        svc.execute('pause')
        svc.execute('resume')
        assert len(svc.audit_log) == 3
        assert svc.audit_log[0]['action'] == 'on'

    def test_audit_entry_structure(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        svc.execute('on')
        entry = svc.audit_log[0]
        assert 'timestamp' in entry
        assert 'action' in entry
        assert 'previous_state' in entry
        assert 'new_state' in entry
        assert 'success' in entry
        assert 'source' in entry


class TestStatusSnapshot:

    def test_status_snapshot(self):
        from cockpit.api.paper_control import PaperControlService
        svc = PaperControlService()
        svc.execute('on')
        status = svc.status()
        assert status['current_state'] == 'paper_enabled'
        assert 'last_updated_at' in status
        assert 'source' in status
        assert 'restrictions' in status


class TestRuntimeIntegration:

    def _make_runtime(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        HERE = Path(__file__).resolve().parent.parent
        return NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )

    def test_on_starts_runtime(self):
        from cockpit.api.paper_control import PaperControlService
        rt = self._make_runtime()
        svc = PaperControlService(runtimes={'BTCUSDT': rt})
        svc.execute('on')
        assert rt.is_running is True

    def test_off_stops_runtime(self):
        from cockpit.api.paper_control import PaperControlService
        rt = self._make_runtime()
        svc = PaperControlService(runtimes={'BTCUSDT': rt})
        svc.execute('on')
        svc.execute('off')
        assert rt.is_running is False

    def test_pause_pauses_runtime(self):
        from cockpit.api.paper_control import PaperControlService
        rt = self._make_runtime()
        svc = PaperControlService(runtimes={'BTCUSDT': rt})
        svc.execute('on')
        svc.execute('pause')
        assert rt.is_paused is True
