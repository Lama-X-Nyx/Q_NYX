"""
PersistentDecisionLogger — Durable SQLite-backed decision log.

Contract:
- Append-only: every log() commits immediately.
- WAL mode: concurrent reads possible.
- UNIQUE(timestamp, pair): second write with same key is ignored (idempotent).
- Recovery: load_all() / load_range() rebuild a pandas DataFrame.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    pair                TEXT    NOT NULL,
    action              TEXT    NOT NULL,
    orch_score          REAL,
    size_factor         REAL,
    blocked_by          TEXT,
    price               REAL,
    atr                 REAL,
    volume_ratio        REAL,
    model_version       TEXT,
    context_state       TEXT,
    context_score       REAL,
    context_passed      INTEGER,
    regime_state        TEXT,
    regime_score        REAL,
    regime_passed       INTEGER,
    setup_state         TEXT,
    setup_score         REAL,
    setup_passed        INTEGER,
    entry_state         TEXT,
    entry_score         REAL,
    entry_passed        INTEGER,
    features_json       TEXT,
    trade_entry_price   REAL,
    trade_stop          REAL,
    trade_tp            REAL,
    trade_direction     INTEGER,
    trade_size          REAL,
    UNIQUE (timestamp, pair)
);
CREATE INDEX IF NOT EXISTS ix_decisions_ts   ON decisions(timestamp);
CREATE INDEX IF NOT EXISTS ix_decisions_pair ON decisions(pair);
"""


_COLUMNS = [
    'timestamp', 'pair', 'action', 'orch_score', 'size_factor', 'blocked_by',
    'price', 'atr', 'volume_ratio', 'model_version',
    'context_state', 'context_score', 'context_passed',
    'regime_state', 'regime_score', 'regime_passed',
    'setup_state', 'setup_score', 'setup_passed',
    'entry_state', 'entry_score', 'entry_passed',
    'features_json',
    'trade_entry_price', 'trade_stop', 'trade_tp',
    'trade_direction', 'trade_size',
]


class PersistentDecisionLogger:
    """SQLite-backed append-only decision log (WAL mode)."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path), isolation_level=None)
        # WAL: concurrent readers, durable writes
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.executescript(_SCHEMA)

    # ----- write -----------------------------------------------------------

    def log(self, decision: Dict[str, Any]) -> None:
        """Persist one decision. Idempotent on (timestamp, pair)."""
        row = {k: decision.get(k) for k in _COLUMNS}
        placeholders = ",".join(["?"] * len(_COLUMNS))
        cols = ",".join(_COLUMNS)
        self._con.execute(
            f"INSERT OR IGNORE INTO decisions ({cols}) VALUES ({placeholders})",
            tuple(row[c] for c in _COLUMNS),
        )

    # ----- read ------------------------------------------------------------

    def load_all(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM decisions ORDER BY timestamp ASC", self._con
        )

    def load_range(self, start: str, end: str) -> pd.DataFrame:
        """Load decisions with timestamp in [start, end]. Date-only end = inclusive day."""
        # If end is date-only (YYYY-MM-DD), expand to end-of-day so ISO timestamps
        # on that same date are included.
        if len(end) == 10 and end.count('-') == 2:
            end_cmp = end + "T99:99:99"
        else:
            end_cmp = end
        return pd.read_sql_query(
            "SELECT * FROM decisions WHERE timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp ASC",
            self._con, params=(start, end_cmp),
        )

    def count(self) -> int:
        cur = self._con.execute("SELECT COUNT(*) FROM decisions")
        return int(cur.fetchone()[0])

    def last_timestamp(self) -> Optional[str]:
        cur = self._con.execute("SELECT MAX(timestamp) FROM decisions")
        row = cur.fetchone()
        return row[0] if row else None

    # ----- lifecycle -------------------------------------------------------

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

    def __enter__(self) -> "PersistentDecisionLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
