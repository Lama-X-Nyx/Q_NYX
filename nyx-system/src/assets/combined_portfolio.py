"""
Combined-portfolio helper — union of trades across multiple assets.

This is the simplest possible "portfolio" the PortfolioAllocator produces
when run mono-asset (one model per asset): we concatenate per-asset trades
chronologically and let the bootstrap / Monte Carlo / equity engine treat
them as if they came from a single account.

For the A/B study we only need:
  - per-asset trade lists (already produced by NYXPipeline.run)
  - combined timeline (sorted by timestamp)
  - tagged with asset + regime so regime_bootstrap still works

Real portfolio sizing (cluster caps, correlation veto) lives in
`PortfolioAllocator` and is already tested separately. Here we keep the
union simple so A/B numbers are interpretable.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


def combine_trades(trade_lists: Dict[str, Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge per-asset trade lists into one chronologically-sorted stream.

    Each returned trade carries an `asset` key for auditability.
    """
    out: List[Dict[str, Any]] = []
    for asset, trades in trade_lists.items():
        for t in trades:
            tt = dict(t)
            tt.setdefault('asset', asset)
            out.append(tt)

    def _key(t: Dict[str, Any]):
        ts = t.get('entry_time') or t.get('timestamp') or t.get('ts')
        # Sort by string; ISO-format timestamps are lexicographically ordered.
        return str(ts)

    out.sort(key=_key)
    return out


def aggregate_metrics(per_asset_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Sum/avg headline numbers across assets for a side-by-side table."""
    total_pnl = sum(r.get('total_pnl_dollars', 0.0)
                    for r in per_asset_results.values())
    total_trades = sum(r.get('n_trades', 0)
                       for r in per_asset_results.values())
    weights = []
    sharpes = []
    for r in per_asset_results.values():
        n = r.get('n_trades', 0) or 0
        if n > 0:
            weights.append(n)
            sharpes.append(r.get('sharpe', 0.0))
    weighted_sharpe = (
        sum(s * w for s, w in zip(sharpes, weights)) / sum(weights)
        if weights else 0.0
    )
    return {
        'total_pnl_dollars': float(total_pnl),
        'total_trades': int(total_trades),
        'weighted_avg_sharpe': float(weighted_sharpe),
        'per_asset': {k: {
            'n_trades':          v.get('n_trades', 0),
            'total_pnl_dollars': v.get('total_pnl_dollars', 0.0),
            'sharpe':            v.get('sharpe', 0.0),
            'max_drawdown_pct':  v.get('max_drawdown_pct', 0.0),
        } for k, v in per_asset_results.items()},
    }
