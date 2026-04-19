"""
Paper Trading Control Service — Ticket UI-2.

Backend-sovereign control plane for paper trading.
Wraps NYXRuntime control methods with state machine,
validation, and audit logging.

States: paper_disabled → paper_enabled ↔ paper_paused
        paper_enabled → paper_critical_blocked (system-triggered)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


VALID_STATES = {
    'paper_disabled',
    'paper_enabled',
    'paper_paused',
    'paper_critical_blocked',
}

VALID_TRANSITIONS = {
    ('paper_disabled', 'on'): 'paper_enabled',
    ('paper_enabled', 'off'): 'paper_disabled',
    ('paper_enabled', 'pause'): 'paper_paused',
    ('paper_paused', 'resume'): 'paper_enabled',
    ('paper_paused', 'off'): 'paper_disabled',
}

ALWAYS_ALLOWED = {'cancel_all', 'flatten'}


class PaperControlService:
    """Sovereign paper trading control. Backend is truth."""

    def __init__(
        self,
        runtimes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.current_state = 'paper_disabled'
        self._source = 'init'
        self._reason = ''
        self._last_updated = time.time()
        self._runtimes = runtimes or {}
        self.audit_log: List[Dict[str, Any]] = []

    def execute(self, action: str, source: str = 'operator', reason: str = '') -> Dict[str, Any]:
        prev = self.current_state

        if action in ALWAYS_ALLOWED:
            self._do_action(action)
            self._log(action, prev, self.current_state, True, source, reason)
            return {'success': True, 'state': self.current_state, 'action': action}

        if action == 'on' and self.current_state == 'paper_critical_blocked':
            self._log(action, prev, prev, False, source, 'cannot override critical block')
            return {'success': False, 'state': self.current_state, 'reason': 'critical_blocked'}

        transition_key = (self.current_state, action)
        new_state = VALID_TRANSITIONS.get(transition_key)

        if new_state is None:
            self._log(action, prev, prev, False, source,
                      f'invalid transition: {self.current_state} + {action}')
            return {
                'success': False,
                'state': self.current_state,
                'reason': f'invalid: {self.current_state} + {action}',
            }

        self.current_state = new_state
        self._source = source
        self._reason = reason
        self._last_updated = time.time()

        self._apply_to_runtimes(action)
        self._log(action, prev, new_state, True, source, reason)

        return {'success': True, 'state': self.current_state, 'action': action}

    def set_critical_block(self, reason: str = '') -> None:
        prev = self.current_state
        self.current_state = 'paper_critical_blocked'
        self._source = 'system'
        self._reason = reason
        self._last_updated = time.time()
        self._apply_to_runtimes('emergency_stop')
        self._log('critical_block', prev, self.current_state, True, 'system', reason)

    def clear_critical_block(self, source: str = 'operator') -> Dict[str, Any]:
        if self.current_state != 'paper_critical_blocked':
            return {'success': False, 'reason': 'not in critical state'}
        prev = self.current_state
        self.current_state = 'paper_disabled'
        self._source = source
        self._last_updated = time.time()
        self._log('clear_critical', prev, self.current_state, True, source, '')
        return {'success': True, 'state': self.current_state}

    def status(self) -> Dict[str, Any]:
        return {
            'current_state': self.current_state,
            'last_updated_at': self._last_updated,
            'source': self._source,
            'reason': self._reason,
            'restrictions': self._get_restrictions(),
        }

    def _get_restrictions(self) -> List[str]:
        r = []
        if self.current_state in ('paper_disabled', 'paper_paused', 'paper_critical_blocked'):
            r.append('no_new_orders')
        if self.current_state == 'paper_critical_blocked':
            r.append('system_blocked')
        return r

    def _apply_to_runtimes(self, action: str) -> None:
        for sym, rt in self._runtimes.items():
            if action == 'on':
                rt.start()
            elif action == 'off':
                rt.stop()
            elif action == 'pause':
                rt.pause()
            elif action == 'resume':
                rt.resume()
            elif action == 'emergency_stop':
                if hasattr(rt, 'emergency_stop'):
                    rt.emergency_stop(reason='critical_block')
                else:
                    rt.stop()
            elif action == 'cancel_all':
                if hasattr(rt, 'cancel_all_orders'):
                    rt.cancel_all_orders()
            elif action == 'flatten':
                if hasattr(rt, 'flatten_all'):
                    rt.flatten_all()

    def _do_action(self, action: str) -> None:
        for sym, rt in self._runtimes.items():
            if action == 'cancel_all' and hasattr(rt, 'cancel_all_orders'):
                rt.cancel_all_orders()
            elif action == 'flatten' and hasattr(rt, 'flatten_all'):
                rt.flatten_all()

    def _log(self, action: str, prev: str, new: str, success: bool,
             source: str, reason: str) -> None:
        self.audit_log.append({
            'timestamp': time.time(),
            'action': action,
            'previous_state': prev,
            'new_state': new,
            'success': success,
            'source': source,
            'reason': reason,
        })
