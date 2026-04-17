"""
Portfolio Allocator — Ticket 35.

Consumes per-asset decisions + inter-asset dependency outputs,
produces a coherent, capital-constrained, dependency-aware execution
plan.

Operates AFTER per-asset GBM + Jesse + Fractal Quality + Dependency,
BEFORE RiskEngine + OMS.

Alpha (GBM/Jesse) untouched. Dependency layer untouched.
Deterministic, reproducible, fully explainable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


_QUALITY_WEIGHT = {
    'high': 1.0,
    'medium': 0.7,
    'low': 0.4,
}


def compute_allocation_score(
    confidence: float,
    expected_edge_bps: float,
    quality_bucket: str,
    dependency_strength: float,
    contagion_risk: float,
    current_exposure_pct: float,
) -> float:
    """Compute a single ranking score for capital allocation.

    Higher = more deserving of capital.
    Deterministic, no randomness.
    """
    q_w = _QUALITY_WEIGHT.get(quality_bucket, 0.5)
    base = confidence * 0.35 + min(expected_edge_bps / 100.0, 1.0) * 0.25 + q_w * 0.20
    contagion_penalty = contagion_risk * 0.15
    exposure_penalty = current_exposure_pct * 0.15
    dep_penalty = dependency_strength * contagion_risk * 0.10
    score = base - contagion_penalty - exposure_penalty - dep_penalty
    return float(max(0.0, score))


class PortfolioAllocator:
    """Capital-constrained, dependency-aware trade arbiter."""

    def __init__(
        self,
        max_capital_per_trade_pct: float = 0.10,
        max_capital_per_asset_pct: float = 0.15,
        max_total_capital_pct: float = 0.30,
        max_directional_pct: float = 0.25,
        risk_per_trade_pct: float = 0.02,
    ) -> None:
        self.max_capital_per_trade_pct = max_capital_per_trade_pct
        self.max_capital_per_asset_pct = max_capital_per_asset_pct
        self.max_total_capital_pct = max_total_capital_pct
        self.max_directional_pct = max_directional_pct
        self.risk_per_trade_pct = risk_per_trade_pct

        self._last_cycle_results: List[Dict[str, Any]] = []
        self._total_cycles: int = 0

    def allocate(
        self,
        candidates: List[Dict[str, Any]],
        portfolio_ctx: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Allocate capital across candidate trades.

        Returns one allocation record per candidate with:
          decision, allocated_notional, allocated_size,
          allocation_reason, score.
        """
        if not candidates:
            self._last_cycle_results = []
            self._total_cycles += 1
            return []

        equity = float(portfolio_ctx.get('total_equity', 0.0))
        available = float(portfolio_ctx.get('available_capital', 0.0))
        exposure_by_asset = portfolio_ctx.get('exposure_by_asset', {})

        max_per_trade = equity * self.max_capital_per_trade_pct
        max_per_asset = equity * self.max_capital_per_asset_pct
        max_total = equity * self.max_total_capital_pct
        max_directional = equity * self.max_directional_pct

        scored = []
        for c in candidates:
            dep = c.get('dependency', {})
            existing_exp = float(exposure_by_asset.get(c['symbol'], 0.0))
            exp_pct = existing_exp / max(equity, 1.0)
            score = compute_allocation_score(
                confidence=c.get('confidence', 0.5),
                expected_edge_bps=c.get('expected_edge_bps', 0.0),
                quality_bucket=c.get('quality_bucket', 'medium'),
                dependency_strength=dep.get('dependency_strength', 0.0),
                contagion_risk=dep.get('contagion_risk', 0.0),
                current_exposure_pct=exp_pct,
            )
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        total_allocated = 0.0
        directional_allocated: Dict[int, float] = {1: 0.0, -1: 0.0}
        asset_allocated: Dict[str, float] = {}

        for score, c in scored:
            sym = c['symbol']
            direction = c.get('direction', 1)
            entry_price = float(c.get('entry_price', 1.0))
            size_hint = float(c.get('size_hint', 1.0))
            dep = c.get('dependency', {})
            dep_strength = dep.get('dependency_strength', 0.0)

            existing_exp = float(exposure_by_asset.get(sym, 0.0))
            already_alloc = asset_allocated.get(sym, 0.0)

            base_notional = equity * self.risk_per_trade_pct * size_hint
            base_notional = min(base_notional, max_per_trade)

            dep_factor = 1.0 - dep_strength * 0.4
            notional = base_notional * dep_factor

            room_asset = max(0.0, max_per_asset - existing_exp - already_alloc)
            notional = min(notional, room_asset)

            room_total = max(0.0, max_total - total_allocated)
            notional = min(notional, room_total)

            room_dir = max(0.0, max_directional - directional_allocated.get(direction, 0.0))
            notional = min(notional, room_dir)

            notional = min(notional, max(0.0, available))

            if notional < 1.0 or available < 1.0:
                results.append({
                    'symbol': sym,
                    'direction': direction,
                    'decision': 'REJECT' if available < 1.0 else 'DEFER',
                    'allocated_notional': 0.0,
                    'allocated_size': 0.0,
                    'allocation_reason': 'capital_exhausted' if available < 1.0 else 'insufficient_room',
                    'score': score,
                    'dependency_adjustment': dep_factor,
                })
                continue

            if notional < base_notional * 0.5:
                decision = 'APPROVE_REDUCED'
                reason = 'constraint_reduced'
            else:
                decision = 'APPROVE_FULL'
                reason = 'approved'

            size = notional / max(entry_price, 1e-12)

            results.append({
                'symbol': sym,
                'direction': direction,
                'decision': decision,
                'allocated_notional': float(notional),
                'allocated_size': float(size),
                'allocation_reason': reason,
                'score': score,
                'dependency_adjustment': dep_factor,
            })

            total_allocated += notional
            directional_allocated[direction] = directional_allocated.get(direction, 0.0) + notional
            asset_allocated[sym] = asset_allocated.get(sym, 0.0) + notional

        results.sort(key=lambda r: r['score'], reverse=True)

        self._last_cycle_results = results
        self._total_cycles += 1
        return results

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for persistence / monitoring."""
        return {
            'last_cycle_results': self._last_cycle_results,
            'total_cycles': self._total_cycles,
        }
