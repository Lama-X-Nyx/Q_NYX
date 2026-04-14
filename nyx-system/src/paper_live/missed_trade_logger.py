"""
MissedTradeLogger — durable SQLite log of REJECTED / TIMED_OUT orders.

This is the honest ledger of "trades the strategy wanted but execution
reality denied". Required for:
  - paper → live parity check (miss rate should match)
  - alpha-after-execution analysis
  - detecting regimes where post-only is infeasible
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd


_SCHEMA = """
CREATE TABLE IF NOT EXISTS missed_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    oid             INTEGER NOT NULL UNIQUE,
    pair            TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    qty             REAL    NOT NULL,
    limit_price     REAL    NOT NULL,
    mark_price      REAL,
    placed_at       TEXT    NOT NULL,
    final_state     TEXT    NOT NULL,
    reason          TEXT,
    bars_waited     INTEGER,
    timed_out_at    TEXT
);
CREATE INDEX IF NOT EXISTS ix_missed_pair  ON missed_trades(pair);
CREATE INDEX IF NOT EXISTS ix_missed_state ON missed_trades(final_state);
"""


_COLUMNS = [
    'oid', 'pair', 'side', 'qty', 'limit_price', 'mark_price',
    'placed_at', 'final_state', 'reason', 'bars_waited', 'timed_out_at',
]


class MissedTradeLogger:
    """Append-only SQLite log of missed trades."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path), isolation_level=None)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.executescript(_SCHEMA)

    # -----------------------------------------------------------------
    def log(self, order: Dict[str, Any]) -> None:
        row = {k: order.get(k) for k in _COLUMNS}
        placeholders = ",".join(["?"] * len(_COLUMNS))
        cols = ",".join(_COLUMNS)
        self._con.execute(
            f"INSERT OR IGNORE INTO missed_trades ({cols}) VALUES ({placeholders})",
            tuple(row[c] for c in _COLUMNS),
        )

    # -----------------------------------------------------------------
    def count(self) -> int:
        return int(self._con.execute(
            "SELECT COUNT(*) FROM missed_trades"
        ).fetchone()[0])

    def count_by_pair(self, pair: str) -> int:
        return int(self._con.execute(
            "SELECT COUNT(*) FROM missed_trades WHERE pair=?", (pair,)
        ).fetchone()[0])

    def count_by_state(self, state: str) -> int:
        return int(self._con.execute(
            "SELECT COUNT(*) FROM missed_trades WHERE final_state=?", (state,)
        ).fetchone()[0])

    def load_all(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM missed_trades ORDER BY id ASC", self._con
        )

    # -----------------------------------------------------------------
    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

    def __enter__(self) -> "MissedTradeLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
