"""
TDD Tests — HubSpokeRunner

Ties the hub-and-spoke architecture together:

  bars (per pair)
     │
     ▼
  SignalPod(asset)           ← per-asset logic, emits Signal
     │                       (one pod per enabled asset)
     ▼
  PortfolioAllocator         ← central hub with risk caps, clusters
     │
     ▼
  PostOnlyPaperBroker        ← shared execution layer
     │
     ▼
  DecisionLogger / MissedTradeLogger / Heartbeat / Events

Contract:
  - on_bars({symbol: bar, ...}) collects Signals from every pod, runs
    the allocator, and places post-only orders only for ApprovedTrades.
  - Pod crash for one asset does NOT prevent others from emitting signals.
  - FLAT pods still tick; their bars advance broker + missed sweep.
  - Each approved trade produces ONE trade_decision event.
  - Signals that are rejected by the allocator are logged but do NOT
    place an order.
"""
import pytest
from pathlib import Path


# --- test doubles ------------------------------------------------------------

class _StubPod:
    """Minimal SignalPod double: emits a canned Signal on the first bar,
    then FLAT forever."""
    def __init__(self, symbol, cluster='majors', edge=10.0, viability=0.8,
                 direction=1, conviction=0.7, size=0.3, crash=False):
        self.symbol = symbol
        self._cluster = cluster
        self._edge = edge
        self._viability = viability
        self._direction = direction
        self._conviction = conviction
        self._size = size
        self._crash = crash
        self._fired = False

    def on_bar(self, bar):
        from src.assets.signal import Signal
        if self._crash:
            raise RuntimeError(f"{self.symbol} pod crash")
        if not self._fired:
            self._fired = True
            return Signal(
                symbol=self.symbol,
                timestamp=bar['timestamp'],
                direction=self._direction,
                conviction=self._conviction,
                expected_edge_net=self._edge,
                maker_viability=self._viability,
                regime_tag='trend_plus',
                bull_bear_tag='bull',
                size_suggestion=self._size,
                cluster_group=self._cluster,
            )
        return Signal(
            symbol=self.symbol,
            timestamp=bar['timestamp'],
            direction=0,
            conviction=0.0,
            expected_edge_net=0.0,
            maker_viability=0.0,
            regime_tag='range',
            bull_bear_tag='range',
            size_suggestion=0.0,
            cluster_group=self._cluster,
        )


class _Recorder:
    def __init__(self):           self.calls = []
    def info(self, t):            self.calls.append(('info', t));     return True
    def warn(self, t):            self.calls.append(('warn', t));     return True
    def error(self, t):           self.calls.append(('error', t));    return True
    def critical(self, t):        self.calls.append(('critical', t)); return True
    def send(self, t):            self.calls.append(('send', t));     return True


def _bar(ts, price=100.0):
    return {'timestamp': ts,
            'open': price, 'high': price*1.001,
            'low':  price*0.999, 'close': price*1.0005,
            'volume': 100.0}


def _build(tmp_path, pods, alerter=None, max_open_positions=3,
           max_total_risk=0.05, max_asset_risk=0.015,
           max_cluster_risk=None):
    from src.assets.hub_spoke_runner import HubSpokeRunner
    return HubSpokeRunner(
        pods=pods,
        storage_root=tmp_path,
        shared_alerter=alerter,
        max_total_risk=max_total_risk,
        max_asset_risk=max_asset_risk,
        max_cluster_risk=max_cluster_risk,
        max_open_positions=max_open_positions,
    )


# ==============================================================================
# TEST 1: end-to-end happy path
# ==============================================================================
class TestEndToEnd:

    def test_signals_from_two_pods_both_approved(self, tmp_path):
        alerter = _Recorder()
        eth = _StubPod('ETHUSDT', cluster='majors', edge=12)
        sol = _StubPod('SOLUSDT', cluster='alts',   edge=8)
        runner = _build(tmp_path, [eth, sol], alerter=alerter)
        try:
            runner.on_bars({
                'ETHUSDT': _bar('2025-01-01T00:00:00+00:00', 3000.0),
                'SOLUSDT': _bar('2025-01-01T00:00:00+00:00',  140.0),
            })
        finally:
            runner.shutdown()

        trade_alerts = [t for lvl, t in alerter.calls
                        if lvl == 'info' and 'trade_decision' in t.lower()]
        assert len(trade_alerts) == 2

    def test_flat_signal_produces_no_trade(self, tmp_path):
        alerter = _Recorder()
        # Direction=0 → pod never places anything.
        eth = _StubPod('ETHUSDT', direction=0)
        runner = _build(tmp_path, [eth], alerter=alerter)
        try:
            runner.on_bars({
                'ETHUSDT': _bar('2025-01-01T00:00:00+00:00', 3000.0),
            })
        finally:
            runner.shutdown()
        trade_alerts = [t for lvl, t in alerter.calls
                        if 'trade_decision' in t.lower()]
        assert trade_alerts == []


# ==============================================================================
# TEST 2: allocator rejections
# ==============================================================================
class TestAllocatorRejection:

    def test_max_open_positions_limits_trades(self, tmp_path):
        """3 positive signals, but max_open_positions=1 → only 1 trade."""
        alerter = _Recorder()
        pods = [
            _StubPod('ETHUSDT', cluster='majors', edge=15),
            _StubPod('XRPUSDT', cluster='alts',   edge=12),
            _StubPod('SOLUSDT', cluster='alts',   edge=8),
        ]
        runner = _build(tmp_path, pods, alerter=alerter,
                        max_open_positions=1,
                        max_total_risk=0.015, max_asset_risk=0.015)
        try:
            runner.on_bars({
                'ETHUSDT': _bar('2025-01-01T00:00:00+00:00', 3000.0),
                'XRPUSDT': _bar('2025-01-01T00:00:00+00:00',    0.6),
                'SOLUSDT': _bar('2025-01-01T00:00:00+00:00',  140.0),
            })
        finally:
            runner.shutdown()
        trade_alerts = [t for lvl, t in alerter.calls
                        if lvl == 'info' and 'trade_decision' in t.lower()]
        assert len(trade_alerts) == 1
        # ETH has the highest edge → ETH wins.
        assert 'ETHUSDT' in trade_alerts[0]


# ==============================================================================
# TEST 3: pod crash isolation
# ==============================================================================
class TestPodIsolation:

    def test_crashed_pod_does_not_break_others(self, tmp_path):
        alerter = _Recorder()
        bad = _StubPod('XRPUSDT', crash=True)
        good = _StubPod('ETHUSDT', cluster='majors', edge=12)
        runner = _build(tmp_path, [bad, good], alerter=alerter)
        try:
            runner.on_bars({
                'ETHUSDT': _bar('2025-01-01T00:00:00+00:00', 3000.0),
                'XRPUSDT': _bar('2025-01-01T00:00:00+00:00',    0.6),
            })
        finally:
            runner.shutdown()
        # ETH went through.
        trade_alerts = [t for lvl, t in alerter.calls
                        if lvl == 'info' and 'trade_decision' in t.lower()]
        assert any('ETHUSDT' in t for t in trade_alerts)
        # The crash produced an api_error alert.
        api_errors = [t for lvl, t in alerter.calls if lvl == 'error']
        assert any('XRPUSDT' in t or 'pod' in t.lower() or 'api_error' in t.lower()
                   for t in api_errors)


# ==============================================================================
# TEST 4: introspection
# ==============================================================================
class TestIntrospection:

    def test_lists_pods(self, tmp_path):
        eth = _StubPod('ETHUSDT')
        xrp = _StubPod('XRPUSDT', cluster='alts')
        runner = _build(tmp_path, [eth, xrp])
        try:
            assert set(runner.pairs()) == {'ETHUSDT', 'XRPUSDT'}
        finally:
            runner.shutdown()

    def test_unknown_pair_raises(self, tmp_path):
        from src.paper_live.multi_pair_runner import UnknownPairError
        runner = _build(tmp_path, [_StubPod('ETHUSDT')])
        try:
            with pytest.raises(UnknownPairError):
                runner.on_bars({'DOGEUSDT': _bar('2025-01-01T00:00:00+00:00')})
        finally:
            runner.shutdown()
