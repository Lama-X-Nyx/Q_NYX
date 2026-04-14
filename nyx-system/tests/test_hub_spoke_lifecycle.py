"""
TDD Tests — HubSpokeRunner position lifecycle (step 2/4).

When a trade is approved, HubSpokeRunner MUST remember it in
`_open_positions[symbol]` together with the bar index at which it was
opened AND the `expected_hold_bars` from the Signal. On each
subsequent `on_bars()` the runner must age each open position by 1
and release those whose age exceeds their expected_hold_bars.

This closes the 98.5 % cut gap revealed in commit `be5450d` by
letting the same symbol accept a new signal once the previous trade
has "expired".
"""
from pathlib import Path
from typing import Any, Dict

import pytest


# ---- test doubles --------------------------------------------------------

class _FlipFlopPod:
    """Emits an actionable Signal on every bar we feed it. Used to
    stress position lifecycle (vs NYXPipelinePod which is sparse)."""

    def __init__(self, symbol: str, cluster: str = 'majors',
                 expected_hold_bars: int = 5):
        self.symbol = symbol
        self._cluster = cluster
        self._hold = expected_hold_bars

    def on_bar(self, bar: Dict[str, Any]):
        from src.assets.signal import Signal
        return Signal(
            symbol=self.symbol,
            timestamp=bar['timestamp'],
            direction=1,
            conviction=0.8,
            expected_edge_net=10.0,
            maker_viability=0.7,
            regime_tag='bull',
            bull_bear_tag='bull',
            size_suggestion=1.0,
            cluster_group=self._cluster,
            expected_hold_bars=self._hold,
        )


def _bar(ts: str, price: float = 1000.0) -> Dict[str, Any]:
    return {
        'timestamp': ts, 'open': price,
        'high': price * 1.002, 'low': price * 0.998,
        'close': price * 1.001, 'volume': 500.0,
    }


# ===========================================================================
class TestPositionRelease:
    """Once a position's expected_hold_bars have elapsed, it's released."""

    def test_second_signal_rejected_while_position_open(self, tmp_path):
        from src.assets.hub_spoke_runner import HubSpokeRunner
        pod = _FlipFlopPod('BTCUSDT', cluster='majors',
                            expected_hold_bars=5)
        runner = HubSpokeRunner(
            pods=[pod],
            storage_root=tmp_path,
            max_total_risk=0.04, max_asset_risk=0.015,
            max_open_positions=2,
        )
        try:
            # Bar 0: position opens.
            t0 = runner.on_bars({'BTCUSDT': _bar('2023-01-01T10:00:00')})
            assert len(t0) == 1, "first bar should open a position"

            # Bar 1-3: still within hold window → signal must be rejected.
            for i in range(1, 4):
                ts = f'2023-01-01T10:{i*15:02d}:00'
                ti = runner.on_bars({'BTCUSDT': _bar(ts)})
                assert len(ti) == 0, (
                    f"bar {i}: within hold window, should be rejected, "
                    f"got {len(ti)} approved"
                )
        finally:
            runner.shutdown()

    def test_signal_accepted_after_hold_expires(self, tmp_path):
        from src.assets.hub_spoke_runner import HubSpokeRunner
        pod = _FlipFlopPod('BTCUSDT', cluster='majors',
                            expected_hold_bars=3)
        runner = HubSpokeRunner(
            pods=[pod],
            storage_root=tmp_path,
            max_total_risk=0.04, max_asset_risk=0.015,
            max_open_positions=2,
        )
        try:
            # Bar 0: opens.
            runner.on_bars({'BTCUSDT': _bar('2023-01-01T10:00:00')})
            # Bars 1, 2, 3: within hold window (hold=3).
            for i in range(1, 4):
                runner.on_bars({'BTCUSDT': _bar(
                    f'2023-01-01T10:{i*15:02d}:00'
                )})
            # Bar 4: hold EXPIRED (age > 3). Expect new approval.
            ti = runner.on_bars({'BTCUSDT': _bar('2023-01-01T11:00:00')})
            assert len(ti) == 1, (
                f"bar 4 past expected_hold_bars=3, new signal should be "
                f"approved, got {len(ti)} approved"
            )
        finally:
            runner.shutdown()


class TestMultipleCycles:

    def test_many_cycles_produce_many_trades(self, tmp_path):
        """With a 5-bar hold, 60 bars should produce ~10-12 trades
        (60 / 5). Certainly more than the 1 that the old runner
        would have allowed."""
        from src.assets.hub_spoke_runner import HubSpokeRunner
        pod = _FlipFlopPod('BTCUSDT', cluster='majors',
                            expected_hold_bars=5)
        runner = HubSpokeRunner(
            pods=[pod],
            storage_root=tmp_path,
            max_total_risk=0.50, max_asset_risk=0.10,
            max_open_positions=2,
        )
        total = 0
        try:
            import pandas as pd
            ts = pd.Timestamp('2023-01-01T10:00:00')
            for _ in range(60):
                tlist = runner.on_bars({'BTCUSDT': _bar(ts.isoformat())})
                total += len(tlist)
                ts = ts + pd.Timedelta(minutes=15)
        finally:
            runner.shutdown()
        assert total >= 8, (
            f"60 bars × hold=5 should yield ≈ 10-12 trades, got {total}; "
            "lifecycle release didn't fire enough times"
        )


class TestGlobalMaxOpenPositions:
    """The `max_open_positions` cap still applies across symbols."""

    def test_max_open_respects_cap(self, tmp_path):
        from src.assets.hub_spoke_runner import HubSpokeRunner
        pods = [
            _FlipFlopPod('BTCUSDT', cluster='majors', expected_hold_bars=50),
            _FlipFlopPod('ETHUSDT', cluster='alts',   expected_hold_bars=50),
            _FlipFlopPod('SOLUSDT', cluster='alts',   expected_hold_bars=50),
        ]
        runner = HubSpokeRunner(
            pods=pods,
            storage_root=tmp_path,
            max_total_risk=0.05, max_asset_risk=0.015,
            max_open_positions=2,   # ← only 2 can be simultaneously open
        )
        try:
            t0 = runner.on_bars({
                'BTCUSDT': _bar('2023-01-01T10:00:00', 30000),
                'ETHUSDT': _bar('2023-01-01T10:00:00', 1800),
                'SOLUSDT': _bar('2023-01-01T10:00:00', 20),
            })
            # Three pods each emitted an actionable signal, but global
            # cap is 2.
            assert len(t0) == 2, (
                f"max_open_positions=2 but {len(t0)} approved"
            )
        finally:
            runner.shutdown()
