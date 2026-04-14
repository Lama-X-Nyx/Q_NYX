"""
TDD Tests — PaperLiveRunner

End-to-end wiring: DataValidator → StateManager → DecisionLogger →
MakerFirstBroker → Heartbeat → TelegramAlerter.

Contract:
  - on_bar(pair, bar): validate, advance broker, call strategy, log decision,
    update state, tick heartbeat.
  - Idempotent on duplicate bar (via DataValidator stale check).
  - Crash-safe: state saved on every bar.
  - start() warm-restarts from last state.
"""
from pathlib import Path

import pytest


def _bar(ts: str = "2025-01-01T00:00:00+00:00", price: float = 45000.0,
         vol: float = 100.0) -> dict:
    return {
        'timestamp': ts,
        'open': price,
        'high': price * 1.001,
        'low':  price * 0.999,
        'close': price * 1.0005,
        'volume': vol,
    }


class _StubStrategy:
    """Minimal strategy double: alternating BUY / WAIT."""
    def __init__(self):
        self.i = 0

    def decide(self, pair: str, bar: dict) -> dict:
        self.i += 1
        if self.i % 2 == 0:
            return {'action': 'BUY', 'orch_score': 0.72, 'size_factor': 1.0,
                    'blocked_by': [], 'features': {'x': 1.0},
                    'agent_results': {
                        'context': {'state': 'bullish', 'score': 0.8, 'passed': True},
                        'regime':  {'state': 'trend_plus', 'score': 0.7, 'passed': True},
                        'setup':   {'state': 'valid_setup', 'score': 0.65, 'passed': True},
                        'entry':   {'state': 'ready', 'score': 0.72, 'passed': True,
                                    'direction': 1, 'p_up': 0.6, 'p_down': 0.2},
                    }}
        return {'action': 'WAIT', 'orch_score': 0.3, 'size_factor': 0.0,
                'blocked_by': ['regime'], 'features': {'x': 0.0},
                'agent_results': {
                    'context': {'state': 'neutral', 'score': 0.4, 'passed': False},
                    'regime':  {'state': 'range', 'score': 0.3, 'passed': False},
                    'setup':   {'state': 'no_setup', 'score': 0.3, 'passed': False},
                    'entry':   {'state': 'not_ready', 'score': 0.3, 'passed': False,
                                'direction': 0, 'p_up': 0.4, 'p_down': 0.3},
                }}


@pytest.fixture
def runner(tmp_path):
    from src.paper_live.runner import PaperLiveRunner
    return PaperLiveRunner(
        strategy=_StubStrategy(),
        db_path=tmp_path / "decisions.db",
        state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        telegram_token=None, telegram_chat=None,
    )


# ===========================================================================
# TEST 1: single-bar wiring
# ===========================================================================
class TestSingleBar:

    def test_on_bar_logs_one_decision(self, runner, tmp_path):
        runner.on_bar('BTCUSDT', _bar())
        runner.shutdown()

        from src.paper_live.decision_logger import PersistentDecisionLogger
        with PersistentDecisionLogger(tmp_path / "decisions.db") as dl:
            assert dl.count() == 1

    def test_on_bar_ticks_heartbeat(self, runner, tmp_path):
        runner.on_bar('BTCUSDT', _bar())
        assert (tmp_path / "hb.json").exists()
        runner.shutdown()

    def test_on_bar_saves_state(self, runner, tmp_path):
        runner.on_bar('BTCUSDT', _bar())
        assert (tmp_path / "state.json").exists()
        runner.shutdown()


# ===========================================================================
# TEST 2: idempotence / duplicate rejection
# ===========================================================================
class TestIdempotence:

    def test_duplicate_bar_rejected(self, runner, tmp_path):
        runner.on_bar('BTCUSDT', _bar(ts='2025-01-01T00:00:00+00:00'))
        # Same timestamp again — should raise (validator) OR be no-op.
        from src.paper_live.data_validator import DataValidationError
        with pytest.raises(DataValidationError):
            runner.on_bar('BTCUSDT', _bar(ts='2025-01-01T00:00:00+00:00'))
        runner.shutdown()


# ===========================================================================
# TEST 3: warm restart
# ===========================================================================
class TestWarmRestart:

    def test_restart_resumes_from_last_ts(self, tmp_path):
        from src.paper_live.runner import PaperLiveRunner

        r1 = PaperLiveRunner(
            strategy=_StubStrategy(),
            db_path=tmp_path / "decisions.db",
            state_path=tmp_path / "state.json",
            heartbeat_path=tmp_path / "hb.json",
            telegram_token=None, telegram_chat=None,
        )
        r1.on_bar('BTCUSDT', _bar(ts="2025-01-01T00:00:00+00:00"))
        r1.shutdown()

        r2 = PaperLiveRunner(
            strategy=_StubStrategy(),
            db_path=tmp_path / "decisions.db",
            state_path=tmp_path / "state.json",
            heartbeat_path=tmp_path / "hb.json",
            telegram_token=None, telegram_chat=None,
        )
        # The runner should have restored last_bar_ts.
        assert r2.last_bar_ts.get('BTCUSDT') == "2025-01-01T00:00:00+00:00"
        r2.shutdown()


# ===========================================================================
# TEST 4: multi-bar end-to-end
# ===========================================================================
class TestMultiBar:

    def test_many_bars_all_logged(self, runner, tmp_path):
        for i in range(20):
            runner.on_bar('BTCUSDT', _bar(
                ts=f"2025-01-01T{i:02d}:00:00+00:00",
                price=45000.0 + i * 10,
            ))
        runner.shutdown()

        from src.paper_live.decision_logger import PersistentDecisionLogger
        with PersistentDecisionLogger(tmp_path / "decisions.db") as dl:
            assert dl.count() == 20
