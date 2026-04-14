"""
TDD Tests — MissedTradeLogger

Persistent SQLite log of trades that were requested but NEVER filled
(REJECTED or TIMED_OUT). This is the ground truth of "alpha lost to
execution reality".

Contract:
  - log(order) persists one missed opportunity.
  - Idempotent on oid.
  - Queries: count(), by_pair(), by_reason(), since(ts).
  - miss_rate_over(window) compared to a count of intended+filled orders.
"""
import pytest


def _rejected(oid: int = 1, pair: str = 'BTCUSDT') -> dict:
    return {
        'oid': oid,
        'pair': pair,
        'side': 'buy',
        'qty': 0.01,
        'limit_price': 45000.0,
        'mark_price': 44999.0,
        'placed_at': '2025-01-01T00:00:00+00:00',
        'final_state': 'REJECTED',
        'reason': 'post_only would cross',
        'bars_waited': 0,
        'timed_out_at': None,
    }


def _timed_out(oid: int = 2, pair: str = 'BTCUSDT') -> dict:
    return {
        'oid': oid,
        'pair': pair,
        'side': 'buy',
        'qty': 0.01,
        'limit_price': 40000.0,
        'mark_price': 45000.0,
        'placed_at': '2025-01-01T00:00:00+00:00',
        'final_state': 'TIMED_OUT',
        'reason': 'no fill after max_wait=3 bars',
        'bars_waited': 3,
        'timed_out_at': '2025-01-01T00:45:00+00:00',
    }


class TestSchema:

    def test_creates_db(self, tmp_path):
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        mtl = MissedTradeLogger(tmp_path / "missed.db")
        assert (tmp_path / "missed.db").exists()
        mtl.close()

    def test_creates_missed_table(self, tmp_path):
        import sqlite3
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        mtl = MissedTradeLogger(tmp_path / "missed.db")
        mtl.close()
        con = sqlite3.connect(tmp_path / "missed.db")
        r = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='missed_trades'"
        ).fetchone()
        con.close()
        assert r is not None


class TestLogAndQuery:

    def test_log_timed_out(self, tmp_path):
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        with MissedTradeLogger(tmp_path / "m.db") as mtl:
            mtl.log(_timed_out())
            assert mtl.count() == 1

    def test_log_rejected(self, tmp_path):
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        with MissedTradeLogger(tmp_path / "m.db") as mtl:
            mtl.log(_rejected())
            assert mtl.count() == 1

    def test_log_idempotent_on_oid(self, tmp_path):
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        with MissedTradeLogger(tmp_path / "m.db") as mtl:
            mtl.log(_timed_out(oid=5))
            mtl.log(_timed_out(oid=5))  # same oid → no-op
            assert mtl.count() == 1

    def test_by_pair(self, tmp_path):
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        with MissedTradeLogger(tmp_path / "m.db") as mtl:
            mtl.log(_timed_out(oid=1, pair='BTCUSDT'))
            mtl.log(_timed_out(oid=2, pair='ETHUSDT'))
            mtl.log(_timed_out(oid=3, pair='BTCUSDT'))
            assert mtl.count_by_pair('BTCUSDT') == 2
            assert mtl.count_by_pair('ETHUSDT') == 1

    def test_by_reason_category(self, tmp_path):
        """Split REJECTED vs TIMED_OUT counts."""
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        with MissedTradeLogger(tmp_path / "m.db") as mtl:
            mtl.log(_rejected(oid=1))
            mtl.log(_timed_out(oid=2))
            mtl.log(_timed_out(oid=3))
            assert mtl.count_by_state('TIMED_OUT') == 2
            assert mtl.count_by_state('REJECTED') == 1


class TestReload:

    def test_persists_across_reopen(self, tmp_path):
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        with MissedTradeLogger(tmp_path / "m.db") as mtl:
            mtl.log(_timed_out())
        with MissedTradeLogger(tmp_path / "m.db") as mtl:
            assert mtl.count() == 1


class TestDataFrame:

    def test_load_all_returns_rows(self, tmp_path):
        from src.paper_live.missed_trade_logger import MissedTradeLogger
        with MissedTradeLogger(tmp_path / "m.db") as mtl:
            mtl.log(_rejected(oid=1))
            mtl.log(_timed_out(oid=2))
            df = mtl.load_all()
        assert len(df) == 2
        assert set(df['oid']) == {1, 2}
        assert 'final_state' in df.columns
