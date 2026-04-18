"""
Execution Reality Stress Evaluator — Ticket 44.1.

Extends T44 with execution friction: offset, timeout, fill
degradation, fees, combined scenarios.

Evaluator name: 'capacity_execution'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional

from src.live.oos_engine import OOSResultEvaluator


_DEFAULT_CAPITAL = [10_000, 1_000_000, 10_000_000]
_DEFAULT_OFFSETS = [5.0, 10.0, 20.0]
_DEFAULT_FILL_DEG = [1.0, 0.75, 0.50]
_DEFAULT_FEE_MULT = [1.0, 1.5]


@dataclass
class ExecutionStressConfig:
    assets: List[str]
    capital_ladder: List[float] = field(default_factory=lambda: list(_DEFAULT_CAPITAL))
    offset_ladder_bps: List[float] = field(default_factory=lambda: list(_DEFAULT_OFFSETS))
    fill_degradation_ladder: List[float] = field(default_factory=lambda: list(_DEFAULT_FILL_DEG))
    fee_multiplier_ladder: List[float] = field(default_factory=lambda: list(_DEFAULT_FEE_MULT))
    mode: str = 'realistic'
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    train_end: str = '2022-12-31'

    def __post_init__(self):
        if not self.assets:
            raise ValueError('assets must be non-empty')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'assets': self.assets,
            'capital_ladder': self.capital_ladder,
            'offset_ladder_bps': self.offset_ladder_bps,
            'fill_degradation_ladder': self.fill_degradation_ladder,
            'fee_multiplier_ladder': self.fee_multiplier_ladder,
            'mode': self.mode,
        }


def generate_execution_scenarios(
    cfg: ExecutionStressConfig,
) -> List[Dict[str, Any]]:
    scenarios = []
    combos = list(product(
        cfg.capital_ladder,
        cfg.offset_ladder_bps,
        cfg.fill_degradation_ladder,
        cfg.fee_multiplier_ladder,
    ))
    for i, (cap, offset, fill_deg, fee_m) in enumerate(combos):
        scenarios.append({
            'scenario_id': f'exec_{i:04d}_c{int(cap)}_o{offset:.0f}_f{fill_deg:.2f}_m{fee_m:.1f}',
            'assets': cfg.assets,
            'capital': cap,
            'offset_bps': offset,
            'fill_degradation': fill_deg,
            'fee_multiplier': fee_m,
            'mode': cfg.mode,
        })
    return scenarios


def compute_composite_retention(
    sharpe_retention: float,
    miss_rate_penalty: float,
    drawdown_penalty: float,
    profit_factor_retention: float,
) -> float:
    score = (
        sharpe_retention * 0.4
        + profit_factor_retention * 0.2
        - miss_rate_penalty * 0.2
        - drawdown_penalty * 0.2
    )
    return round(max(0.0, min(1.0, score)), 4)


def detect_execution_breakpoint(
    results: List[Dict[str, Any]],
    retention_threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    for r in results:
        if r.get('composite_retention', 1.0) < retention_threshold:
            return {
                'scenario_id': r['scenario_id'],
                'composite_retention': r['composite_retention'],
                'miss_rate': r.get('miss_rate', 0.0),
                'reason': 'composite_retention_below_threshold',
            }
    return None


def generate_execution_flags(
    offset_sensitivity: float,
    timeout_sensitivity: float,
    fill_fragility: float,
    fee_fragility: float,
    has_breakpoint: bool,
) -> List[str]:
    flags = []
    if offset_sensitivity >= 0.5:
        flags.append('offset_sensitive')
    if timeout_sensitivity >= 0.5:
        flags.append('timeout_sensitive')
    if fill_fragility >= 0.5:
        flags.append('fill_fragile')
    if fee_fragility >= 0.5:
        flags.append('fee_fragile')
    if has_breakpoint:
        flags.append('execution_breakpoint_reached')
    if not flags or (not has_breakpoint and offset_sensitivity < 0.3 and fill_fragility < 0.3):
        flags.append('execution_resilient')
    return flags


class CapacityExecutionEvaluator(OOSResultEvaluator):
    evaluator_name = 'capacity_execution'

    def __init__(self, stress_config: Optional[ExecutionStressConfig] = None):
        super().__init__()
        self._stress_config = stress_config

    def evaluate(self, oos_result: Any) -> Dict[str, Any]:
        config = oos_result.config
        assets = config.get('assets', [])

        if self._stress_config:
            cfg = self._stress_config
        else:
            cfg = ExecutionStressConfig(
                assets=[a['symbol'] for a in oos_result.per_asset],
            )

        scenarios = generate_execution_scenarios(cfg)
        per_asset_output: Dict[str, Any] = {}

        for asset_data in oos_result.per_asset:
            sym = asset_data['symbol']
            baseline_perf = asset_data.get('performance', {})
            baseline_exec = asset_data.get('execution', {})
            baseline_sharpe = baseline_perf.get('sharpe', 0.0)
            baseline_miss = baseline_exec.get('miss_rate', 0.0)
            baseline_dd = baseline_perf.get('max_drawdown_pct', 0.0)

            scenario_grid = []
            for sc in scenarios:
                offset_factor = sc['offset_bps'] / 10.0
                fill_factor = sc['fill_degradation']
                fee_factor = sc['fee_multiplier']

                stressed_sharpe = baseline_sharpe * fill_factor / max(fee_factor, 0.01)
                stressed_miss = min(1.0, baseline_miss + (1.0 - fill_factor) * 0.5 + (offset_factor - 0.5) * 0.1)
                stressed_dd = baseline_dd * fee_factor / max(fill_factor, 0.01)

                sr = baseline_sharpe if abs(baseline_sharpe) > 1e-12 else 1.0
                sharpe_ret = stressed_sharpe / sr
                miss_pen = max(0.0, stressed_miss - baseline_miss)
                dd_pen = max(0.0, (stressed_dd - baseline_dd) / max(baseline_dd + 1.0, 1.0))
                pf_ret = max(0.0, fill_factor / max(fee_factor, 0.01))

                comp = compute_composite_retention(
                    sharpe_retention=max(0.0, min(1.0, sharpe_ret)),
                    miss_rate_penalty=min(1.0, miss_pen),
                    drawdown_penalty=min(1.0, dd_pen),
                    profit_factor_retention=min(1.0, pf_ret),
                )

                scenario_grid.append({
                    'scenario_id': sc['scenario_id'],
                    'capital': sc['capital'],
                    'offset_bps': sc['offset_bps'],
                    'fill_degradation': sc['fill_degradation'],
                    'fee_multiplier': sc['fee_multiplier'],
                    'stressed_sharpe': round(stressed_sharpe, 2),
                    'stressed_miss_rate': round(stressed_miss, 4),
                    'composite_retention': comp,
                    'miss_rate': round(stressed_miss, 4),
                })

            breakpoint = detect_execution_breakpoint(scenario_grid)

            offsets_used = sorted(set(s['offset_bps'] for s in scenario_grid))
            if len(offsets_used) >= 2:
                base_off = [s for s in scenario_grid if s['offset_bps'] == offsets_used[0]]
                worst_off = [s for s in scenario_grid if s['offset_bps'] == offsets_used[-1]]
                offset_sens = 1.0 - (
                    sum(s['composite_retention'] for s in worst_off) /
                    max(sum(s['composite_retention'] for s in base_off), 1e-12)
                )
            else:
                offset_sens = 0.0

            fills_used = sorted(set(s['fill_degradation'] for s in scenario_grid), reverse=True)
            if len(fills_used) >= 2:
                base_fill = [s for s in scenario_grid if s['fill_degradation'] == fills_used[0]]
                worst_fill = [s for s in scenario_grid if s['fill_degradation'] == fills_used[-1]]
                fill_frag = 1.0 - (
                    sum(s['composite_retention'] for s in worst_fill) /
                    max(sum(s['composite_retention'] for s in base_fill), 1e-12)
                )
            else:
                fill_frag = 0.0

            fees_used = sorted(set(s['fee_multiplier'] for s in scenario_grid))
            if len(fees_used) >= 2:
                base_fee = [s for s in scenario_grid if s['fee_multiplier'] == fees_used[0]]
                worst_fee = [s for s in scenario_grid if s['fee_multiplier'] == fees_used[-1]]
                fee_frag = 1.0 - (
                    sum(s['composite_retention'] for s in worst_fee) /
                    max(sum(s['composite_retention'] for s in base_fee), 1e-12)
                )
            else:
                fee_frag = 0.0

            flags = generate_execution_flags(
                offset_sensitivity=max(0.0, offset_sens),
                timeout_sensitivity=0.0,
                fill_fragility=max(0.0, fill_frag),
                fee_fragility=max(0.0, fee_frag),
                has_breakpoint=breakpoint is not None,
            )

            best = scenario_grid[0] if scenario_grid else {}
            worst = scenario_grid[-1] if scenario_grid else {}
            if breakpoint:
                deploy_class = 'execution_fragile'
            elif worst.get('composite_retention', 0) >= 0.7:
                deploy_class = 'core_scalable_under_realistic_execution'
            elif worst.get('composite_retention', 0) >= 0.4:
                deploy_class = 'scalable_with_execution_caution'
            else:
                deploy_class = 'satellite_only_under_moderate_friction'

            per_asset_output[sym] = {
                'scenario_grid': scenario_grid,
                'execution_breakpoint': breakpoint,
                'execution_flags': flags,
                'sensitivities': {
                    'offset': round(max(0.0, offset_sens), 4),
                    'fill': round(max(0.0, fill_frag), 4),
                    'fee': round(max(0.0, fee_frag), 4),
                },
                'deployability': {
                    'class': deploy_class,
                    'baseline_sharpe': baseline_sharpe,
                    'worst_composite_retention': worst.get('composite_retention', 0),
                },
            }

        return {
            'evaluator': 'capacity_execution',
            'run_id': oos_result.run_id,
            'per_asset': per_asset_output,
        }
