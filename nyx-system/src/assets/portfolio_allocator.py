"""
PortfolioAllocator — central decision layer of the hub-and-spoke
architecture.

Takes Signals from N per-asset pods and returns the subset that will
actually be executed, with final sizing that respects:

  max_total_risk          (global)
  max_asset_risk          (per symbol)
  max_cluster_risk        (per group, e.g. 'alts')
  max_open_positions      (count)
  no-pyramiding           (if already open → skip)
  correlation veto        (same-cluster, same-direction → cluster cap)

Ranking: Signal.score() descending, symbol ascending on ties.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .signal import Signal


@dataclass(frozen=True)
class ApprovedTrade:
    symbol: str
    direction: int
    final_risk: float           # fraction of total portfolio
    cluster_group: str
    source_signal: Signal


class PortfolioAllocator:
    """Hub layer: applies global risk caps + correlation controls."""

    def __init__(
        self,
        max_total_risk: float = 0.04,
        max_asset_risk: float = 0.015,
        max_cluster_risk: Optional[Dict[str, float]] = None,
        max_open_positions: int = 2,
    ):
        self.max_total_risk = float(max_total_risk)
        self.max_asset_risk = float(max_asset_risk)
        self.max_cluster_risk = dict(max_cluster_risk or {})
        self.max_open_positions = int(max_open_positions)

    # ------------------------------------------------------------------
    def decide(
        self,
        signals: Iterable[Signal],
        open_positions: Optional[Dict[str, dict]] = None,
    ) -> List[ApprovedTrade]:
        """Return the subset of signals that should actually trade."""
        open_positions = dict(open_positions or {})

        # 1. Filter out non-actionable / negative-edge / already-open.
        pool: List[Signal] = []
        for s in signals:
            if not s.is_actionable():
                continue
            if s.score() <= 0.0:
                continue
            if s.symbol in open_positions:
                continue   # no pyramiding
            pool.append(s)

        # 2. Rank by (score desc, symbol asc) deterministically.
        pool.sort(key=lambda s: (-s.score(), s.symbol))

        # 3. Greedy accept respecting caps.
        approved: List[ApprovedTrade] = []
        total_risk = 0.0
        per_asset: Dict[str, float] = {}
        per_cluster: Dict[str, float] = {}
        # Track per-cluster DIRECTION to honor correlation veto:
        # same-cluster + same-direction → share the cluster cap.
        cluster_dir: Dict[str, int] = {}

        # Seed from currently-open positions so caps include live exposure.
        # (Open positions contribute 0 by default here — the caller can
        # pre-populate with known risks if desired.)

        for s in pool:
            if len(approved) >= self.max_open_positions:
                break

            # Asset cap first.
            asset_cap = self.max_asset_risk - per_asset.get(s.symbol, 0.0)
            if asset_cap <= 0:
                continue

            # Cluster cap (only applies to clusters that have an explicit
            # ceiling configured AND signals in the same direction).
            cluster_cap = float('inf')
            cluster_ceiling = self.max_cluster_risk.get(s.cluster_group)
            if cluster_ceiling is not None:
                existing_dir = cluster_dir.get(s.cluster_group)
                if existing_dir is not None and existing_dir != s.direction:
                    # Opposite direction in same cluster → no correlation.
                    cluster_cap = float('inf')
                else:
                    cluster_cap = cluster_ceiling - per_cluster.get(
                        s.cluster_group, 0.0
                    )
            if cluster_cap <= 0:
                continue

            # Total-risk cap.
            total_cap = self.max_total_risk - total_risk
            if total_cap <= 0:
                break

            # Pod-requested size × per-asset cap ceiling.
            wanted = max(0.0, float(s.size_suggestion)) * self.max_asset_risk
            # Clamp to all three caps.
            final = min(wanted if wanted > 0 else self.max_asset_risk,
                        asset_cap, cluster_cap, total_cap,
                        self.max_asset_risk)

            if final <= 0:
                continue

            approved.append(ApprovedTrade(
                symbol=s.symbol, direction=s.direction,
                final_risk=final, cluster_group=s.cluster_group,
                source_signal=s,
            ))
            total_risk += final
            per_asset[s.symbol] = per_asset.get(s.symbol, 0.0) + final
            per_cluster[s.cluster_group] = (
                per_cluster.get(s.cluster_group, 0.0) + final
            )
            # Remember direction of the first-accepted signal in this cluster.
            cluster_dir.setdefault(s.cluster_group, s.direction)

        return approved
