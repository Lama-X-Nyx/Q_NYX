"""
ExecutionProfile — per-asset broker/execution parameters.

Derives concrete paper-broker settings from the AssetRegistry's
`slippage_model` + `session_profile`:

  session_profile → max_wait_bars
    liquid          → 4   (tolerant, ETH-like)
    event_sensitive → 3   (strict, XRP-like)
    momentum_fast   → 2   (very strict, SOL-like)

  slippage_model → slippage_bps + maker_viability_threshold
    low            → 2  bps,  threshold 0.40
    medium         → 5  bps,  threshold 0.50
    medium_high    → 8  bps,  threshold 0.60
    high           → 12 bps,  threshold 0.65

These defaults encode the architectural choice:
  "SOL → timeout plus court, seuil maker plus dur"
  "ETH → plus tolérant, meilleur candidat"
  "XRP → filtre plus strict hors contexte"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import Asset


_MAX_WAIT_BY_SESSION = {
    'liquid':          4,
    'event_sensitive': 3,
    'momentum_fast':   2,
}

_SLIPPAGE_TABLE = {
    'low':         (2,  0.40),
    'medium':      (5,  0.50),
    'medium_high': (8,  0.60),
    'high':        (12, 0.65),
}


@dataclass(frozen=True)
class ExecutionProfile:
    symbol: str
    maker_fee: float
    taker_fee: float
    slippage_bps: int
    max_wait_bars: int
    maker_viability_threshold: float

    @classmethod
    def from_asset(cls, asset: "Asset") -> "ExecutionProfile":
        slip_bps, viability = _SLIPPAGE_TABLE[asset.slippage_model]
        max_wait = _MAX_WAIT_BY_SESSION[asset.session_profile]
        return cls(
            symbol=asset.symbol,
            maker_fee=asset.maker_fee,
            taker_fee=asset.taker_fee,
            slippage_bps=slip_bps,
            max_wait_bars=max_wait,
            maker_viability_threshold=viability,
        )
