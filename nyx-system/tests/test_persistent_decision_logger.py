"""
TDD Tests — PersistentDecisionLogger

Durable decision logger backed by SQLite.
Contract:
  - Append-only: every decision persisted immediately (no data loss on crash).
  - WAL mode: concurrent read while writing (for monitoring).
  - Schema: one table `decisions` with all fields the strategy produces.
  - Recovery: load all decisions from DB back into a DataFrame.
  - Idempotent writes: duplicate (timestamp, pair) → raises or no-op.
"""
import os
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "decisions.db"


def _sample_decision(ts: str = "2025-01-01T00:00:00+00:00", pair: str = "BTCUSDT") -> dict:
    return {
        'timestamp': ts,
        'pair': pair,
        'action': 'BUY',
        'orch_score': 0.78,
        'size_factor': 1.2,
        'blocked_by': '',
        'price': 45000.0,
        'atr': 450.0,
        'volume_ratio': 2.3,
        'model_version': 'v0.3.2',
        'context_state': 'bullish',
        'context_score': 0.82,
        'context_passed': True,
        'regime_state': 'trend_plus',
        'regime_score': 0.71,
        'regime_passed': True,
        'setup_state': 'valid_setup',
        'setup_score': 0.68,
        'setup_passed': True,
        'entry_state': 'ready',
        'entry_score': 0.75,
        'entry_passed': True,
        'features_json': '{"ema_ratio_9_21": 0.003}',
        'trade_entry_price': 45000.0,
        'trade_stop': 44100.0,
        'trade_tp': 46800.0,
        'trade_direction': 1,
        'trade_size': 0.01,
    }


# ===========================================================================
# TEST 1: schema + table creation
# ===========================================================================
class TestSchema:

    def test_creates_db_file(self, tmp_db):
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        assert tmp_db.exists()
        logger.close()

    def test_creates_decisions_table(self, tmp_db):
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        logger.close()
        con = sqlite3.connect(tmp_db)
        row = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='decisions'"
        ).fetchone()
        con.close()
        assert row is not None

    def test_uses_wal_mode(self, tmp_db):
        """WAL enables concurrent reads while writing."""
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        mode = logger._con.execute("PRAGMA journal_mode").fetchone()[0]
        logger.close()
        assert str(mode).lower() == 'wal'


# ===========================================================================
# TEST 2: log + query
# ===========================================================================
class TestLogAndQuery:

    def test_log_one_decision(self, tmp_db):
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        logger.log(_sample_decision())
        df = logger.load_all()
        logger.close()
        assert len(df) == 1
        assert df.iloc[0]['pair'] == 'BTCUSDT'
        assert df.iloc[0]['action'] == 'BUY'

    def test_log_persists_on_close_and_reopen(self, tmp_db):
        """After close+reopen, decisions must still be there."""
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        logger.log(_sample_decision())
        logger.close()

        logger2 = PersistentDecisionLogger(tmp_db)
        df = logger2.load_all()
        logger2.close()
        assert len(df) == 1

    def test_log_many(self, tmp_db):
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        for i in range(100):
            d = _sample_decision(ts=f"2025-01-01T{i:02d}:00:00+00:00")
            logger.log(d)
        df = logger.load_all()
        logger.close()
        assert len(df) == 100

    def test_query_by_date(self, tmp_db):
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        logger.log(_sample_decision(ts="2025-01-01T00:00:00+00:00"))
        logger.log(_sample_decision(ts="2025-01-02T00:00:00+00:00"))
        logger.log(_sample_decision(ts="2025-01-03T00:00:00+00:00"))
        df = logger.load_range(start="2025-01-02", end="2025-01-03")
        logger.close()
        assert len(df) == 2


# ===========================================================================
# TEST 3: durability under crash (append-only, immediate commit)
# ===========================================================================
class TestDurability:

    def test_no_data_loss_without_explicit_commit(self, tmp_db):
        """Even without close(), a raw sqlite connection must see the data."""
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        logger.log(_sample_decision())

        # Read from a *second* connection — must see the row.
        con = sqlite3.connect(tmp_db)
        n = con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        con.close()
        logger.close()
        assert n == 1

    def test_idempotent_duplicate_key(self, tmp_db):
        """Same (timestamp, pair) logged twice → second is rejected or no-op."""
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        logger.log(_sample_decision())
        # second write with SAME key should not create a duplicate
        logger.log(_sample_decision())
        df = logger.load_all()
        logger.close()
        assert len(df) == 1


# ===========================================================================
# TEST 4: stats helpers
# ===========================================================================
class TestStats:

    def test_count_decisions(self, tmp_db):
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        for i in range(5):
            logger.log(_sample_decision(ts=f"2025-01-01T{i:02d}:00:00+00:00"))
        assert logger.count() == 5
        logger.close()

    def test_last_decision_timestamp(self, tmp_db):
        from src.paper_live.decision_logger import PersistentDecisionLogger
        logger = PersistentDecisionLogger(tmp_db)
        logger.log(_sample_decision(ts="2025-01-01T00:00:00+00:00"))
        logger.log(_sample_decision(ts="2025-01-02T00:00:00+00:00"))
        last = logger.last_timestamp()
        logger.close()
        assert last is not None
        assert "2025-01-02" in str(last)
