"""
TDD Tests — Ticket SEC-1 — Operational Security & Access Control.

Auth, RBAC, session management, secret protection, audit logging.
Backend-enforced, zero-trust.
"""
from __future__ import annotations

import time
import pytest


class TestPasswordHashing:

    def test_hash_password(self):
        from cockpit.api.security import hash_password, verify_password
        h = hash_password('secret123')
        assert h != 'secret123'
        assert verify_password('secret123', h) is True

    def test_wrong_password_rejected(self):
        from cockpit.api.security import hash_password, verify_password
        h = hash_password('correct')
        assert verify_password('wrong', h) is False

    def test_hash_is_unique_per_call(self):
        from cockpit.api.security import hash_password
        h1 = hash_password('same')
        h2 = hash_password('same')
        assert h1 != h2


class TestJWT:

    def test_create_token(self):
        from cockpit.api.security import create_token, decode_token
        token = create_token(user_id='admin', role='operator')
        assert isinstance(token, str)

    def test_decode_valid_token(self):
        from cockpit.api.security import create_token, decode_token
        token = create_token(user_id='admin', role='operator')
        payload = decode_token(token)
        assert payload is not None
        assert payload['user_id'] == 'admin'
        assert payload['role'] == 'operator'

    def test_expired_token_rejected(self):
        from cockpit.api.security import create_token, decode_token
        token = create_token(user_id='admin', role='viewer', expires_seconds=0)
        time.sleep(0.1)
        payload = decode_token(token)
        assert payload is None

    def test_tampered_token_rejected(self):
        from cockpit.api.security import create_token, decode_token
        token = create_token(user_id='admin', role='operator')
        tampered = token[:-5] + 'XXXXX'
        payload = decode_token(tampered)
        assert payload is None


class TestRBAC:

    def test_viewer_can_read(self):
        from cockpit.api.security import check_permission
        assert check_permission('viewer', 'read') is True

    def test_viewer_cannot_control(self):
        from cockpit.api.security import check_permission
        assert check_permission('viewer', 'control') is False

    def test_operator_can_control(self):
        from cockpit.api.security import check_permission
        assert check_permission('operator', 'control') is True

    def test_operator_can_read(self):
        from cockpit.api.security import check_permission
        assert check_permission('operator', 'read') is True

    def test_admin_can_everything(self):
        from cockpit.api.security import check_permission
        assert check_permission('admin', 'read') is True
        assert check_permission('admin', 'control') is True
        assert check_permission('admin', 'admin') is True

    def test_unknown_role_denied(self):
        from cockpit.api.security import check_permission
        assert check_permission('hacker', 'read') is False


class TestUserStore:

    def test_create_and_authenticate(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        store.add_user('operator1', 'pass123', 'operator')
        user = store.authenticate('operator1', 'pass123')
        assert user is not None
        assert user['role'] == 'operator'

    def test_bad_password_returns_none(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        store.add_user('user1', 'correct', 'viewer')
        assert store.authenticate('user1', 'wrong') is None

    def test_unknown_user_returns_none(self):
        from cockpit.api.security import UserStore
        store = UserStore()
        assert store.authenticate('nobody', 'x') is None

    def test_default_admin_exists(self):
        from cockpit.api.security import UserStore
        store = UserStore(create_default_admin=True)
        user = store.authenticate('admin', 'admin')
        assert user is not None
        assert user['role'] == 'admin'


class TestSecurityAudit:

    def test_log_login(self):
        from cockpit.api.security import SecurityAuditLog
        log = SecurityAuditLog()
        log.record('login', user_id='admin', success=True)
        assert len(log.entries) == 1
        assert log.entries[0]['action'] == 'login'

    def test_log_failed_login(self):
        from cockpit.api.security import SecurityAuditLog
        log = SecurityAuditLog()
        log.record('login', user_id='unknown', success=False)
        assert log.entries[0]['success'] is False

    def test_log_entry_has_timestamp(self):
        from cockpit.api.security import SecurityAuditLog
        log = SecurityAuditLog()
        log.record('control_action', user_id='op1', success=True, resource='paper/on')
        entry = log.entries[0]
        assert 'timestamp' in entry
        assert 'user_id' in entry
        assert 'action' in entry
        assert 'resource' in entry

    def test_secrets_never_logged(self):
        from cockpit.api.security import SecurityAuditLog
        log = SecurityAuditLog()
        log.record('login', user_id='admin', success=True, extra={'password': 'secret123'})
        for entry in log.entries:
            assert 'password' not in str(entry)


class TestEnvironmentIsolation:

    def test_environment_config(self):
        from cockpit.api.security import EnvironmentConfig
        cfg = EnvironmentConfig(env='paper')
        assert cfg.env == 'paper'

    def test_environments_are_distinct(self):
        from cockpit.api.security import EnvironmentConfig
        paper = EnvironmentConfig(env='paper')
        live = EnvironmentConfig(env='live')
        assert paper.env != live.env

    def test_live_requires_explicit_flag(self):
        from cockpit.api.security import EnvironmentConfig
        cfg = EnvironmentConfig(env='paper')
        assert cfg.is_live is False
        live = EnvironmentConfig(env='live')
        assert live.is_live is True


class TestRateLimiter:

    def test_allows_under_limit(self):
        from cockpit.api.security import RateLimiter
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.allow('user1') is True

    def test_blocks_over_limit(self):
        from cockpit.api.security import RateLimiter
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.allow('user1')
        assert rl.allow('user1') is False

    def test_different_users_independent(self):
        from cockpit.api.security import RateLimiter
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.allow('user1')
        rl.allow('user1')
        assert rl.allow('user1') is False
        assert rl.allow('user2') is True
