"""
TDD Tests — Signal dataclass

The contract between a per-asset SignalPod and the PortfolioAllocator.

Fields (from architecture spec):
  - symbol
  - timestamp
  - direction          int in {-1, 0, +1}
  - conviction         float in [0, 1]
  - expected_edge_net  float (bps after maker fee — can be negative)
  - maker_viability    float in [0, 1]
  - regime_tag         str
  - bull_bear_tag      str in {'bull', 'bear', 'range'}
  - size_suggestion    float — local size as fraction of asset_risk_cap
  - cluster_group      str  — 'majors' | 'alts' | ...

Invariants:
  - conviction in [0,1], maker_viability in [0,1], size_suggestion ≥ 0.
  - direction must be -1, 0, or +1.
  - Constructor rejects invalid values (fail loudly).
"""
import pytest


def _valid_kwargs(**over):
    base = dict(
        symbol='ETHUSDT',
        timestamp='2025-01-01T00:00:00+00:00',
        direction=1,
        conviction=0.72,
        expected_edge_net=8.5,
        maker_viability=0.80,
        regime_tag='trend_plus',
        bull_bear_tag='bull',
        size_suggestion=0.30,
        cluster_group='majors',
    )
    base.update(over)
    return base


class TestConstruction:

    def test_happy_path(self):
        from src.assets.signal import Signal
        s = Signal(**_valid_kwargs())
        assert s.symbol == 'ETHUSDT'
        assert s.direction == 1
        assert s.cluster_group == 'majors'

    def test_direction_zero_is_ok(self):
        from src.assets.signal import Signal
        s = Signal(**_valid_kwargs(direction=0))
        assert s.direction == 0


class TestValidation:

    def test_conviction_out_of_range_raises(self):
        from src.assets.signal import Signal, SignalValidationError
        with pytest.raises(SignalValidationError):
            Signal(**_valid_kwargs(conviction=1.5))
        with pytest.raises(SignalValidationError):
            Signal(**_valid_kwargs(conviction=-0.1))

    def test_maker_viability_out_of_range_raises(self):
        from src.assets.signal import Signal, SignalValidationError
        with pytest.raises(SignalValidationError):
            Signal(**_valid_kwargs(maker_viability=1.1))

    def test_bad_direction_raises(self):
        from src.assets.signal import Signal, SignalValidationError
        with pytest.raises(SignalValidationError):
            Signal(**_valid_kwargs(direction=2))
        with pytest.raises(SignalValidationError):
            Signal(**_valid_kwargs(direction=-2))

    def test_negative_size_raises(self):
        from src.assets.signal import Signal, SignalValidationError
        with pytest.raises(SignalValidationError):
            Signal(**_valid_kwargs(size_suggestion=-0.1))

    def test_bad_bull_bear_raises(self):
        from src.assets.signal import Signal, SignalValidationError
        with pytest.raises(SignalValidationError):
            Signal(**_valid_kwargs(bull_bear_tag='sideways_but_nope'))


class TestHelpers:

    def test_is_actionable_when_direction_nonzero(self):
        from src.assets.signal import Signal
        assert Signal(**_valid_kwargs(direction=1)).is_actionable() is True
        assert Signal(**_valid_kwargs(direction=-1)).is_actionable() is True
        assert Signal(**_valid_kwargs(direction=0)).is_actionable() is False

    def test_score_combines_edge_and_viability(self):
        from src.assets.signal import Signal
        a = Signal(**_valid_kwargs(expected_edge_net=10.0, maker_viability=1.0))
        b = Signal(**_valid_kwargs(expected_edge_net=10.0, maker_viability=0.1))
        # Same edge, lower viability → lower score.
        assert a.score() > b.score()

    def test_score_zero_when_edge_negative(self):
        """A signal with negative expected edge must never rank above flat."""
        from src.assets.signal import Signal
        a = Signal(**_valid_kwargs(expected_edge_net=-5.0))
        b = Signal(**_valid_kwargs(expected_edge_net=0.0))
        assert a.score() <= b.score()
