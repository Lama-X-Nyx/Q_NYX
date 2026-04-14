"""
TDD Tests — Signal.expected_hold_bars (position-lifecycle field).

A Signal now carries the pod's estimate of how long the position will
stay open, in 15m bars. Used by HubSpokeRunner to release
`_open_positions[symbol]` after expiry so next signals can be
approved.

Invariants :
  - field is a non-negative int
  - default = 50 (matches NYXPipeline.max_bars)
  - backwards-compatible: old call sites without the arg still build
  - validated in __post_init__: negative / non-int values raise
"""
import pytest


def _base_kwargs(**over):
    out = dict(
        symbol='ETHUSDT',
        timestamp='2023-01-01T10:00:00',
        direction=1,
        conviction=0.7,
        expected_edge_net=10.0,
        maker_viability=0.7,
        regime_tag='bull',
        bull_bear_tag='bull',
        size_suggestion=1.0,
        cluster_group='alts',
    )
    out.update(over)
    return out


class TestExpectedHoldBars:

    def test_default_is_50(self):
        from src.assets.signal import Signal
        s = Signal(**_base_kwargs())
        assert s.expected_hold_bars == 50, (
            f"default expected_hold_bars must be 50 (= NYXPipeline.max_bars), "
            f"got {s.expected_hold_bars}"
        )

    def test_custom_value_accepted(self):
        from src.assets.signal import Signal
        s = Signal(**_base_kwargs(expected_hold_bars=30))
        assert s.expected_hold_bars == 30

    def test_zero_accepted(self):
        """0 is a valid (instant-exit) hold time — used by FLAT signals."""
        from src.assets.signal import Signal
        s = Signal(**_base_kwargs(direction=0, conviction=0.0,
                                    expected_edge_net=0.0,
                                    maker_viability=0.0,
                                    bull_bear_tag='range',
                                    size_suggestion=0.0,
                                    expected_hold_bars=0))
        assert s.expected_hold_bars == 0

    def test_negative_raises(self):
        from src.assets.signal import Signal, SignalValidationError
        with pytest.raises(SignalValidationError):
            Signal(**_base_kwargs(expected_hold_bars=-1))

    def test_non_int_raises(self):
        from src.assets.signal import Signal, SignalValidationError
        # floats silently-truncated would be a bug; reject them.
        with pytest.raises((SignalValidationError, TypeError)):
            Signal(**_base_kwargs(expected_hold_bars=50.5))
