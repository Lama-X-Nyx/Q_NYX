"""
TDD Tests — Ticket SEC-1.1 — First-Time Admin Bootstrap.

One-shot initialization. Auto-disables after first use.
"""
from __future__ import annotations

import pytest


class TestUserCount:

    def test_empty_store_has_zero_users(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        assert store.user_count() == 0

    def test_store_with_users_counts(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        store.add_user('a', 'pass', 'operator')
        store.add_user('b', 'pass', 'viewer')
        assert store.user_count() == 2


class TestBootstrapLogic:

    def test_bootstrap_succeeds_when_empty(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        result = store.bootstrap('admin', 'securepass123')
        assert result['success'] is True
        assert store.user_count() == 1

    def test_bootstrap_creates_admin_role(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        store.bootstrap('admin', 'securepass123')
        user = store.authenticate('admin', 'securepass123')
        assert user is not None
        assert user['role'] == 'admin'

    def test_bootstrap_rejected_when_users_exist(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        store.bootstrap('admin', 'pass1')
        result = store.bootstrap('admin2', 'pass2')
        assert result['success'] is False
        assert store.user_count() == 1

    def test_bootstrap_rejected_twice(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        store.bootstrap('admin', 'pass1')
        r2 = store.bootstrap('admin', 'pass2')
        r3 = store.bootstrap('another', 'pass3')
        assert r2['success'] is False
        assert r3['success'] is False

    def test_login_works_after_bootstrap(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        store.bootstrap('operator1', 'mypass')
        user = store.authenticate('operator1', 'mypass')
        assert user is not None
        assert user['username'] == 'operator1'

    def test_bootstrap_does_not_log_password(self):
        from cockpit.api.security import UserStore, SecurityAuditLog
        store = UserStore()
        audit = SecurityAuditLog()
        store.bootstrap('admin', 'supersecret', audit_log=audit)
        for entry in audit.entries:
            assert 'supersecret' not in str(entry)

    def test_bootstrap_is_audited(self):
        from cockpit.api.security import UserStore, SecurityAuditLog
        store = UserStore()
        audit = SecurityAuditLog()
        store.bootstrap('admin', 'pass123', audit_log=audit)
        assert len(audit.entries) >= 1
        assert audit.entries[-1]['action'] == 'bootstrap_admin_created'

    def test_needs_bootstrap_flag(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        assert store.needs_bootstrap() is True
        store.bootstrap('admin', 'pass')
        assert store.needs_bootstrap() is False


class TestDefaultAdminRemoved:

    def test_no_default_admin_in_production_mode(self):
        from cockpit.api.security import UserStore
        store = UserStore(create_default_admin=False)
        assert store.user_count() == 0
        assert store.authenticate('admin', 'admin') is None


class TestBootstrapRateLimiting:

    def test_rate_limited(self):
        from cockpit.api.security import UserStore, RateLimiter
        store = UserStore()
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.allow('bootstrap')
        assert rl.allow('bootstrap') is False
