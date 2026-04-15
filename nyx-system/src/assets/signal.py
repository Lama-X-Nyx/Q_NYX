"""
Signal — typed contract between a per-asset SignalPod and the
PortfolioAllocator.

Every pod emits a Signal (possibly with direction=0 = FLAT) every bar.
The allocator ranks, filters, and sizes Signals before they reach the
execution engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field


_BULL_BEAR_TAGS = {'bull', 'bear', 'range'}
_VALID_DIRECTIONS = {-1, 0, 1}


class SignalValidationError(ValueError):
    """Raised when a Signal is constructed with out-of-range values."""


@dataclass(frozen=True)
class Signal:
    symbol: str
    timestamp: str
    direction: int                 # -1 / 0 / +1
    conviction: float              # [0, 1]
    expected_edge_net: float       # basis points after fees (may be negative)
    maker_viability: float         # [0, 1]
    regime_tag: str
    bull_bear_tag: str             # 'bull' | 'bear' | 'range'
    size_suggestion: float         # ≥ 0, fraction of per-asset risk_cap
    cluster_group: str             # e.g. 'majors' | 'alts'
    expected_hold_bars: int = 50   # 15m bars this position is expected
                                    # to stay open. HubSpokeRunner releases
                                    # `_open_positions[symbol]` after this
                                    # many bars so a subsequent signal on
                                    # the same symbol can be approved.
                                    # Default 50 = NYXPipeline.max_bars.

    def __post_init__(self) -> None:
        if self.direction not in _VALID_DIRECTIONS:
            raise SignalValidationError(
                f"{self.symbol}: direction={self.direction}, must be -1/0/+1"
            )
        if not 0.0 <= self.conviction <= 1.0:
            raise SignalValidationError(
                f"{self.symbol}: conviction={self.conviction} out of [0,1]"
            )
        if not 0.0 <= self.maker_viability <= 1.0:
            raise SignalValidationError(
                f"{self.symbol}: maker_viability={self.maker_viability} out of [0,1]"
            )
        if self.size_suggestion < 0:
            raise SignalValidationError(
                f"{self.symbol}: size_suggestion={self.size_suggestion} < 0"
            )
        if self.bull_bear_tag not in _BULL_BEAR_TAGS:
            raise SignalValidationError(
                f"{self.symbol}: bull_bear_tag={self.bull_bear_tag!r} "
                f"must be one of {sorted(_BULL_BEAR_TAGS)}"
            )
        # Position-lifecycle validation (bool inherits from int in Python;
        # explicit bool exclusion ensures integer-only).
        if not isinstance(self.expected_hold_bars, int) \
                or isinstance(self.expected_hold_bars, bool):
            raise SignalValidationError(
                f"{self.symbol}: expected_hold_bars must be int, got "
                f"{type(self.expected_hold_bars).__name__}"
            )
        if self.expected_hold_bars < 0:
            raise SignalValidationError(
                f"{self.symbol}: expected_hold_bars="
                f"{self.expected_hold_bars} < 0"
            )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def is_actionable(self) -> bool:
        """Signal calls for a trade (direction non-zero)."""
        return self.direction != 0

    def score(self) -> float:
        """Ranking score for the allocator.

        = max(0, expected_edge_net) × maker_viability × conviction.

        Negative edges score at zero (never rank above FLAT).
        """
        edge = max(0.0, self.expected_edge_net)
        return edge * self.maker_viability * self.conviction

    def to_meta_decision(
        self,
        threshold_used: float = 0.60,
        timeframe: str = '15m',
    ):
        """Adapter — convert pod `Signal` to canonical `MetaDecision`.

        The hub-spoke layer still produces `Signal` (compact contract
        for allocator ranking). This adapter lets any downstream
        component speak the unified vocabulary from Ticket 03.

        FLAT (direction=0) signals produce a passed=False decision
        with block_reasons=['signal flat'].
        """
        from src.agents.contracts import MetaDecision
        passed = self.direction != 0
        reasons = [] if passed else ['signal flat']
        return MetaDecision(
            asset=self.symbol,
            timestamp=self.timestamp,
            timeframe=timeframe,
            direction=int(self.direction),
            probability=float(self.conviction),
            threshold_used=float(threshold_used),
            passed=passed,
            block_reasons=reasons,
            expected_edge_net=float(self.expected_edge_net),
            candidate_quality=float(self.conviction),
            fractal_reports={},
            features_snapshot={
                'maker_viability':    float(self.maker_viability),
                'size_suggestion':    float(self.size_suggestion),
                'expected_hold_bars': float(self.expected_hold_bars),
            },
        )
