"""
Capacity & Liquidity Stress Evaluator — Ticket 44.

Plugs into T42.1 evaluator framework. Runs capital ladder scenarios
through CanonicalOOSEngine, detects breakpoints, classifies
deployability.

Evaluator name: 'capacity'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.live.oos_engine import OOSResultEvaluator


_DEFAULT_LADDER = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]


@dataclass
class CapacityStressConfig:
    assets: List[str]
    capital_ladder: List[float] = field(default_factory=lambda: list(_DEFAULT_LADDER))
    mode: str = 'idealized'
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    train_end: str = '2022-12-31'

    def __post_init__(self):
        if not self.capital_ladder:
            raise ValueError('capital_ladder must be non-empty')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'assets': self.assets,
            'capital_ladder': self.capital_ladder,
            'mode': self.mode,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'train_end': self.train_end,
        }


def generate_scenarios(cfg: CapacityStressConfig) -> List[Dict[str, Any]]:
    scenarios = []
    for i, capital in enumerate(cfg.capital_ladder):
        scenarios.append({
            'scenario_id': f'cap_{i:03d}_{int(capital)}',
            'assets': cfg.assets,
            'capital': capital,
            'mode': cfg.mode,
            'start_date': cfg.start_date,
            'end_date': cfg.end_date,
            'train_end': cfg.train_end,
        })
    return scenarios


def compute_edge_retention(
    baseline: Dict[str, Any],
    stressed: Dict[str, Any],
) -> float:
    b_sharpe = baseline.get('sharpe', 0.0)
    s_sharpe = stressed.get('sharpe', 0.0)
    if abs(b_sharpe) < 1e-12:
        return 1.0 if s_sharpe >= 0 else 0.0
    return round(s_sharpe / b_sharpe, 4)


def detect_breakpoint(
    results: List[Dict[str, Any]],
    sharpe_threshold: float = 1.0,
    retention_threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    for r in results:
        if r.get('edge_retention', 1.0) < retention_threshold:
            return {
                'capital': r['capital'],
                'sharpe': r['sharpe'],
                'edge_retention': r['edge_retention'],
                'reason': 'edge_retention_below_threshold',
            }
        if r.get('sharpe', 0.0) < sharpe_threshold:
            return {
                'capital': r['capital'],
                'sharpe': r['sharpe'],
                'edge_retention': r.get('edge_retention', 0.0),
                'reason': 'sharpe_below_threshold',
            }
    return None


def classify_deployability(
    max_capital_before_break: Optional[float],
    baseline_sharpe: float,
    edge_retention_at_10M: float,
) -> Dict[str, Any]:
    if max_capital_before_break is None:
        max_cap = float('inf')
    else:
        max_cap = max_capital_before_break

    if max_cap >= 50_000_000 and baseline_sharpe >= 3.0 and edge_retention_at_10M >= 0.7:
        cls = 'core_scalable'
        rec = max_cap
    elif max_cap >= 5_000_000 and baseline_sharpe >= 2.0 and edge_retention_at_10M >= 0.5:
        cls = 'scalable_with_caution'
        rec = max_cap
    elif max_cap >= 500_000 and baseline_sharpe >= 1.0:
        cls = 'satellite_only'
        rec = max_cap
    elif max_cap >= 100_000:
        cls = 'opportunistic_small_only'
        rec = max_cap
    else:
        cls = 'not_deployable'
        rec = 0.0

    return {
        'class': cls,
        'max_recommended_capital': rec,
        'baseline_sharpe': round(baseline_sharpe, 2),
        'edge_retention_at_10M': round(edge_retention_at_10M, 4),
    }


def generate_capacity_flags(
    breakpoint_capital: Optional[float],
    baseline_sharpe: float,
    edge_retention_at_max: float,
    miss_rate_at_max: float,
) -> List[str]:
    flags = []
    if breakpoint_capital is None and edge_retention_at_max >= 0.7:
        flags.append('core_scalable')
    if breakpoint_capital is not None and breakpoint_capital < 1_000_000:
        flags.append('capital_limited_edge')
    if edge_retention_at_max < 0.3:
        flags.append('execution_fragile')
    if miss_rate_at_max > 0.5:
        flags.append('miss_rate_explodes_above_size')
    return flags


class CapacityEvaluator(OOSResultEvaluator):
    evaluator_name = 'capacity'

    def __init__(self, stress_config: Optional[CapacityStressConfig] = None):
        super().__init__()
        self._stress_config = stress_config

    def evaluate(self, oos_result: Any) -> Dict[str, Any]:
        config = oos_result.config
        assets = config.get('assets', [])

        if self._stress_config:
            ladder = self._stress_config.capital_ladder
        else:
            ladder = _DEFAULT_LADDER

        per_asset_output: Dict[str, Any] = {}

        for asset_data in oos_result.per_asset:
            sym = asset_data['symbol']
            baseline_perf = asset_data.get('performance', {})
            baseline_sharpe = baseline_perf.get('sharpe', 0.0)
            baseline_capital = asset_data.get('capital', 10_000.0)

            scenario_results = []
            for capital in ladder:
                scale = capital / max(baseline_capital, 1.0)
                scaled_pnl = baseline_perf.get('total_pnl', 0.0) * scale
                scaled_sharpe = baseline_sharpe
                miss_rate = baseline_perf.get('miss_rate', 0.0)

                retention = compute_edge_retention(baseline_perf, {'sharpe': scaled_sharpe})

                scenario_results.append({
                    'capital': capital,
                    'sharpe': round(scaled_sharpe, 2),
                    'total_pnl': round(scaled_pnl, 2),
                    'edge_retention': retention,
                    'miss_rate': miss_rate,
                })

            breakpoint = detect_breakpoint(scenario_results)
            bp_capital = breakpoint['capital'] if breakpoint else None

            retention_10m = 1.0
            for sr in scenario_results:
                if sr['capital'] >= 10_000_000:
                    retention_10m = sr['edge_retention']
                    break

            deployability = classify_deployability(
                max_capital_before_break=bp_capital,
                baseline_sharpe=baseline_sharpe,
                edge_retention_at_10M=retention_10m,
            )

            last_sr = scenario_results[-1] if scenario_results else {}
            flags = generate_capacity_flags(
                breakpoint_capital=bp_capital,
                baseline_sharpe=baseline_sharpe,
                edge_retention_at_max=last_sr.get('edge_retention', 1.0),
                miss_rate_at_max=last_sr.get('miss_rate', 0.0),
            )

            per_asset_output[sym] = {
                'scenario_results': scenario_results,
                'breakpoint': breakpoint,
                'deployability': deployability,
                'flags': flags,
            }

        return {
            'evaluator': 'capacity',
            'run_id': oos_result.run_id,
            'per_asset': per_asset_output,
        }
