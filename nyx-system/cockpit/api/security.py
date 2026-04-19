"""
Operational Security & Access Control — Ticket SEC-1.

Backend-enforced, zero-trust: auth (JWT + bcrypt), RBAC,
session management, secret protection, rate limiting,
environment isolation, security audit logging.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import bcrypt
import jwt

_JWT_SECRET = os.environ.get('NYX_JWT_SECRET', 'nyx-dev-secret-change-in-production')
_JWT_ALGORITHM = 'HS256'
_SENSITIVE_KEYS = {'password', 'secret', 'api_key', 'token', 'private_key'}


# =========================================================================
# Password Hashing (bcrypt)
# =========================================================================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


# =========================================================================
# JWT Tokens
# =========================================================================

def create_token(
    user_id: str,
    role: str,
    expires_seconds: int = 3600,
) -> str:
    payload = {
        'user_id': user_id,
        'role': role,
        'iat': time.time(),
        'exp': time.time() + expires_seconds,
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# =========================================================================
# RBAC
# =========================================================================

ROLE_PERMISSIONS: Dict[str, set] = {
    'viewer': {'read'},
    'operator': {'read', 'control'},
    'admin': {'read', 'control', 'admin'},
}


def check_permission(role: str, action: str) -> bool:
    perms = ROLE_PERMISSIONS.get(role, set())
    return action in perms


# =========================================================================
# User Store (in-memory, backed by environment for production)
# =========================================================================

class UserStore:

    def __init__(self, create_default_admin: bool = False) -> None:
        self._users: Dict[str, Dict[str, str]] = {}
        if create_default_admin:
            self.add_user('admin', 'admin', 'admin')

    def user_count(self) -> int:
        return len(self._users)

    def needs_bootstrap(self) -> bool:
        return self.user_count() == 0

    def add_user(self, username: str, password: str, role: str) -> None:
        self._users[username] = {
            'username': username,
            'password_hash': hash_password(password),
            'role': role,
        }

    def bootstrap(
        self,
        username: str,
        password: str,
        audit_log: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if self.user_count() > 0:
            if audit_log:
                audit_log.record('bootstrap_rejected', user_id=username, success=False,
                                 resource='already_initialized')
            return {'success': False, 'reason': 'users_already_exist'}
        self.add_user(username, password, 'admin')
        if audit_log:
            audit_log.record('bootstrap_admin_created', user_id=username, success=True)
        return {'success': True, 'username': username, 'role': 'admin'}

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, str]]:
        user = self._users.get(username)
        if user is None:
            return None
        if not verify_password(password, user['password_hash']):
            return None
        return {'username': user['username'], 'role': user['role']}


# =========================================================================
# Security Audit Log
# =========================================================================

class SecurityAuditLog:

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def record(
        self,
        action: str,
        user_id: str = '',
        success: bool = True,
        resource: str = '',
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        sanitized = {}
        if extra:
            sanitized = {
                k: v for k, v in extra.items()
                if k.lower() not in _SENSITIVE_KEYS
            }
        self.entries.append({
            'timestamp': time.time(),
            'user_id': user_id,
            'action': action,
            'success': success,
            'resource': resource,
            **sanitized,
        })


# =========================================================================
# Environment Isolation
# =========================================================================

@dataclass
class EnvironmentConfig:
    env: str = 'paper'

    @property
    def is_live(self) -> bool:
        return self.env == 'live'

    @property
    def is_paper(self) -> bool:
        return self.env == 'paper'

    @property
    def is_testnet(self) -> bool:
        return self.env == 'testnet'


# =========================================================================
# Rate Limiter (sliding window per user/IP)
# =========================================================================

class RateLimiter:

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def allow(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True
