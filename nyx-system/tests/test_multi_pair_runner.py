"""
TDD Tests — MultiPairRunner

Runs N fully-isolated PaperLiveRunner instances in parallel (one per pair).
Storage is per-pair: each pair writes to its own decisions DB, missed DB,
state file, and heartbeat. Alerters are SHARED so ops see a single stream
with pair information in every message.

Contract:
  - Build from a config like {'ETHUSDT': {...}, 'XRPUSDT': {...}, 'SOLUSDT': {...}}.
  - on_bar(pair, bar) routes to the matching sub-runner.
  - on_bars({pair: bar, ...}) dispatches a batch (one bar per pair).
  - Unknown pair → raises UnknownPairError (fail loudly, don't silently drop).
  - Each sub-runner has isolated state: a crash in ETH must not corrupt XRP.
  - heartbeat_all() returns per-pair heartbeat status.
  - miss_rate_all() returns per-pair broker miss rate.
  - shutdown() closes every sub-runner.
"""
from pathlib import Path

import pytest


class _Strategy:
    """Minimal strategy: fires BUY for a specific pair on first bar."""
    def __init__(self, buy_on_pair: str = None):
        self.buy_on_pair = buy_on_pair
        self.fired_for: set = set()

    def decide(self, pair, bar):
        ar = {k: {'state': '', 'score': 0.0, 'passed': False}
              for k in ('context', 'regime', 'setup', 'entry')}
        if pair == self.buy_on_pair and pair not in self.fired_for:
            self.fired_for.add(pair)
            price = float(bar['close'])
            return {
                'action': 'BUY',
                'orch_score': 0.8,
                'size_factor': 1.0,
                'blocked_by': [],
                'features': {},
                'trade_size': 0.01,
                'trade_direction': 1,
                'trade_entry_price': price * 0.998,
                'trade_stop': price * 0.97,
                'trade_tp': price * 1.04,
                'order_intent': {
                    'side': 'buy', 'qty': 0.01,
                    'limit_price': price * 0.998,
                    'mark_price': price,
                    'max_wait_bars': 3,
                },
                'agent_results': ar,
            }
        return {
            'action': 'WAIT',
            'orch_score': 0.0,
            'size_factor': 0.0,
            'blocked_by': ['wait'],
            'features': {},
            'trade_size': 0.0,
            'agent_results': ar,
        }


def _bar(ts: str, price: float) -> dict:
    return {
        'timestamp': ts,
        'open':  price,
        'high':  price * 1.001,
        'low':   price * 0.999,
        'close': price * 1.0005,
        'volume': 100.0,
    }


PAIRS = ('ETHUSDT', 'XRPUSDT', 'SOLUSDT')


def _make_config(tmp_path, strategy_factory):
    """Config with isolated storage per pair."""
    return {
        pair: {
            'strategy': strategy_factory(pair),
            'storage_dir': tmp_path / pair,
            'max_wait_bars': 3,
        }
        for pair in PAIRS
    }


# ===========================================================================
# TEST 1: construction + isolation
# ===========================================================================
class TestConstruction:

    def test_builds_three_sub_runners(self, tmp_path):
        from src.paper_live.multi_pair_runner import MultiPairRunner
        cfg = _make_config(tmp_path, lambda p: _Strategy())
        mp = MultiPairRunner(cfg)
        try:
            assert set(mp.pairs()) == set(PAIRS)
            for pair in PAIRS:
                assert mp.runner_for(pair) is not None
        finally:
            mp.shutdown()

    def test_each_pair_has_isolated_storage(self, tmp_path):
        """ETH's DB files must be distinct from XRP's and SOL's."""
        from src.paper_live.multi_pair_runner import MultiPairRunner
        cfg = _make_config(tmp_path, lambda p: _Strategy())
        mp = MultiPairRunner(cfg)
        try:
            paths = {}
            for pair in PAIRS:
                r = mp.runner_for(pair)
                paths[pair] = {
                    'decisions': str(r.decision_logger.db_path),
                    'missed':    str(r.missed_logger.db_path),
                    'state':     str(r.state_manager.path),
                    'heartbeat': str(r.heartbeat.path),
                }
            # All paths distinct across pairs.
            all_files = [v for pair in PAIRS for v in paths[pair].values()]
            assert len(all_files) == len(set(all_files))
            # And each pair's files live under its own directory.
            for pair in PAIRS:
                for _kind, p in paths[pair].items():
                    assert pair in p
        finally:
            mp.shutdown()


# ===========================================================================
# TEST 2: bar routing
# ===========================================================================
class TestRouting:

    def test_on_bar_routes_to_right_pair(self, tmp_path):
        from src.paper_live.multi_pair_runner import MultiPairRunner
        cfg = _make_config(tmp_path, lambda p: _Strategy())
        mp = MultiPairRunner(cfg)
        try:
            mp.on_bar('ETHUSDT', _bar('2025-01-01T00:00:00+00:00', 3000.0))
            mp.on_bar('XRPUSDT', _bar('2025-01-01T00:00:00+00:00', 0.6))
            mp.on_bar('SOLUSDT', _bar('2025-01-01T00:00:00+00:00', 140.0))

            eth = mp.runner_for('ETHUSDT')
            xrp = mp.runner_for('XRPUSDT')
            sol = mp.runner_for('SOLUSDT')
            assert eth.bars_processed == 1
            assert xrp.bars_processed == 1
            assert sol.bars_processed == 1
        finally:
            mp.shutdown()

    def test_unknown_pair_raises(self, tmp_path):
        from src.paper_live.multi_pair_runner import MultiPairRunner, UnknownPairError
        cfg = _make_config(tmp_path, lambda p: _Strategy())
        mp = MultiPairRunner(cfg)
        try:
            with pytest.raises(UnknownPairError):
                mp.on_bar('BTCUSDT', _bar('2025-01-01T00:00:00+00:00', 45000.0))
        finally:
            mp.shutdown()

    def test_on_bars_batch(self, tmp_path):
        from src.paper_live.multi_pair_runner import MultiPairRunner
        cfg = _make_config(tmp_path, lambda p: _Strategy())
        mp = MultiPairRunner(cfg)
        try:
            mp.on_bars({
                'ETHUSDT': _bar('2025-01-01T00:00:00+00:00', 3000.0),
                'XRPUSDT': _bar('2025-01-01T00:00:00+00:00', 0.6),
                'SOLUSDT': _bar('2025-01-01T00:00:00+00:00', 140.0),
            })
            for pair in PAIRS:
                assert mp.runner_for(pair).bars_processed == 1
        finally:
            mp.shutdown()


# ===========================================================================
# TEST 3: isolation under failure
# ===========================================================================
class TestIsolation:

    def test_crash_in_one_pair_does_not_break_others(self, tmp_path):
        """If ETH strategy crashes, XRP and SOL must still process their bars."""
        from src.paper_live.multi_pair_runner import MultiPairRunner

        class _CrashETH(_Strategy):
            def decide(self, pair, bar):
                if pair == 'ETHUSDT':
                    raise RuntimeError("eth crash")
                return super().decide(pair, bar)

        cfg = {
            pair: {
                'strategy': _CrashETH(),
                'storage_dir': tmp_path / pair,
                'max_wait_bars': 3,
            }
            for pair in PAIRS
        }
        mp = MultiPairRunner(cfg)
        try:
            bars = {
                'ETHUSDT': _bar('2025-01-01T00:00:00+00:00', 3000.0),
                'XRPUSDT': _bar('2025-01-01T00:00:00+00:00', 0.6),
                'SOLUSDT': _bar('2025-01-01T00:00:00+00:00', 140.0),
            }
            # Batch must NOT abort on the first exception.
            errs = mp.on_bars(bars, raise_on_error=False)
            assert 'ETHUSDT' in errs
            # The two other pairs got their bars.
            assert mp.runner_for('XRPUSDT').bars_processed == 1
            assert mp.runner_for('SOLUSDT').bars_processed == 1
        finally:
            mp.shutdown()


# ===========================================================================
# TEST 4: aggregate stats
# ===========================================================================
class TestAggregates:

    def test_miss_rate_all_is_per_pair(self, tmp_path):
        from src.paper_live.multi_pair_runner import MultiPairRunner
        cfg = _make_config(tmp_path, lambda p: _Strategy())
        mp = MultiPairRunner(cfg)
        try:
            report = mp.miss_rate_all()
            assert set(report.keys()) == set(PAIRS)
            for rate in report.values():
                assert 0.0 <= rate <= 1.0
        finally:
            mp.shutdown()

    def test_heartbeat_status_per_pair(self, tmp_path):
        from src.paper_live.multi_pair_runner import MultiPairRunner
        cfg = _make_config(tmp_path, lambda p: _Strategy())
        mp = MultiPairRunner(cfg)
        try:
            # No ticks yet → all stale.
            status = mp.heartbeat_status(max_age_s=60.0)
            assert all(v is False for v in status.values())

            # After one bar on ETH, only ETH is alive.
            mp.on_bar('ETHUSDT', _bar('2025-01-01T00:00:00+00:00', 3000.0))
            status = mp.heartbeat_status(max_age_s=60.0)
            assert status['ETHUSDT'] is True
            assert status['XRPUSDT'] is False
            assert status['SOLUSDT'] is False
        finally:
            mp.shutdown()


# ===========================================================================
# TEST 5: shared alerter, pair-tagged messages
# ===========================================================================
class TestSharedAlerter:

    def test_trade_events_tagged_with_pair(self, tmp_path):
        """Event messages must include the pair so a single stream is
        enough for ops to see which pair did what."""
        from src.paper_live.multi_pair_runner import MultiPairRunner
        from src.paper_live.event_alerter import EventAlerter

        class _Recorder:
            def __init__(self):           self.calls = []
            def info(self, t):            self.calls.append(('info', t));     return True
            def warn(self, t):            self.calls.append(('warn', t));     return True
            def error(self, t):           self.calls.append(('error', t));    return True
            def critical(self, t):        self.calls.append(('critical', t)); return True
            def send(self, t):            self.calls.append(('send', t));     return True

        rec = _Recorder()
        cfg = _make_config(tmp_path, lambda p: _Strategy(buy_on_pair=p))
        mp = MultiPairRunner(cfg, shared_alerter=rec)
        try:
            # Fire one BUY on each pair.
            mp.on_bars({
                'ETHUSDT': _bar('2025-01-01T00:00:00+00:00', 3000.0),
                'XRPUSDT': _bar('2025-01-01T00:00:00+00:00', 0.6),
                'SOLUSDT': _bar('2025-01-01T00:00:00+00:00', 140.0),
            })
            trade_msgs = [t for lvl, t in rec.calls
                          if lvl == 'info' and 'trade_decision' in t.lower()]
            # Every message must mention its pair name.
            assert any('ETHUSDT' in t for t in trade_msgs)
            assert any('XRPUSDT' in t for t in trade_msgs)
            assert any('SOLUSDT' in t for t in trade_msgs)
        finally:
            mp.shutdown()
