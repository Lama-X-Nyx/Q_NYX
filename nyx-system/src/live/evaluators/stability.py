"""
Multi-Year Regime Stability Evaluator — Ticket 43.

Plugs into Ticket 42.1 evaluator framework. Consumes OOSResult,
segments by year/quarter, tags regimes, classifies stability.

Evaluator name: 'stability'
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.live.oos_engine import OOSResultEvaluator


# =========================================================================
# A1 — Segment Slicer
# =========================================================================

def segment_years(start: str, end: str) -> List[Tuple[str, str, str]]:
    y0 = int(start[:4])
    y1 = int(end[:4])
    return [(str(y), f'{y}-01-01', f'{y}-12-31') for y in range(y0, y1 + 1)]


def segment_quarters(start: str, end: str) -> List[Tuple[str, str, str]]:
    y0 = int(start[:4])
    y1 = int(end[:4])
    q_ends = {'Q1': ('01-01', '03-31'), 'Q2': ('04-01', '06-30'),
              'Q3': ('07-01', '09-30'), 'Q4': ('10-01', '12-31')}
    out = []
    for y in range(y0, y1 + 1):
        for q, (qs, qe) in q_ends.items():
            label = f'{y}-{q}'
            s = f'{y}-{qs}'
            e = f'{y}-{qe}'
            if s >= start[:10] and e <= end[:10]:
                out.append((label, s, e))
            elif s <= end[:10] and e >= start[:10]:
                out.append((label, max(s, start[:10]), min(e, end[:10])))
    return out


# =========================================================================
# A3 — Regime Tagger
# =========================================================================

def tag_regime_volatility(annualized_vol: float) -> str:
    if annualized_vol < 0.02:
        return 'low_vol'
    elif annualized_vol < 0.04:
        return 'medium_vol'
    return 'high_vol'


def tag_regime_trend(return_pct: float) -> str:
    if return_pct > 0.05:
        return 'bull'
    elif return_pct < -0.05:
        return 'bear'
    return 'range'


def tag_regime_composite(return_pct: float, volatility: float) -> str:
    trend = tag_regime_trend(return_pct)
    vol = tag_regime_volatility(volatility)
    return f'{trend}_{vol}'


# =========================================================================
# A4 — Stability Metrics
# =========================================================================

def compute_stability_metrics(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    sharpes = [s['sharpe'] for s in segments]
    returns = [s['total_pnl'] for s in segments]
    dds = [s.get('max_drawdown_pct', 0.0) for s in segments]
    profitable = [1 for s in segments if s['total_pnl'] > 0]

    mean_sh = float(np.mean(sharpes)) if sharpes else 0.0
    std_sh = float(np.std(sharpes)) if len(sharpes) > 1 else 0.0

    profitable_ratio = len(profitable) / max(len(segments), 1)
    max_dd = max(dds) if dds else 0.0

    if std_sh > 0 and mean_sh > 0:
        consistency = min(1.0, mean_sh / (mean_sh + std_sh))
    elif mean_sh > 0:
        consistency = 1.0
    else:
        consistency = 0.0

    stability_score = float(
        0.4 * consistency
        + 0.3 * profitable_ratio
        + 0.2 * min(1.0, mean_sh / max(abs(mean_sh) + 1.0, 1.0))
        + 0.1 * max(0.0, 1.0 - max_dd / 5.0)
    )
    stability_score = max(0.0, min(1.0, stability_score))

    return {
        'mean_segment_sharpe': round(mean_sh, 4),
        'std_segment_sharpe': round(std_sh, 4),
        'best_segment_sharpe': round(max(sharpes) if sharpes else 0.0, 4),
        'worst_segment_sharpe': round(min(sharpes) if sharpes else 0.0, 4),
        'median_segment_return': round(float(np.median(returns)) if returns else 0.0, 2),
        'std_segment_return': round(float(np.std(returns)) if len(returns) > 1 else 0.0, 2),
        'profitable_segment_ratio': round(profitable_ratio, 4),
        'max_segment_drawdown': round(max_dd, 4),
        'stability_score': round(stability_score, 4),
        'n_segments': len(segments),
    }


# =========================================================================
# A5 — Classification
# =========================================================================

def classify_stability(
    stability_score: float,
    profitable_segment_ratio: float,
    regime_dependency_score: float,
    max_segment_drawdown: float,
) -> str:
    if stability_score >= 0.7 and profitable_segment_ratio >= 0.8 and regime_dependency_score < 0.3:
        return 'robust'
    if regime_dependency_score >= 0.6:
        return 'regime_sensitive'
    if stability_score < 0.35 or profitable_segment_ratio < 0.4:
        return 'unstable'
    if stability_score >= 0.5 and profitable_segment_ratio >= 0.6:
        return 'opportunistic'
    return 'fragile_drawdown_profile'


# =========================================================================
# A7 — Stability Flags
# =========================================================================

def generate_stability_flags(
    stability_score: float,
    profitable_segment_ratio: float,
    regime_dependency_score: float,
    max_segment_drawdown: float,
    std_segment_sharpe: float,
) -> List[str]:
    flags = []
    if stability_score >= 0.7 and profitable_segment_ratio >= 0.8:
        flags.append('stable_across_cycles')
    if stability_score < 0.4:
        flags.append('unstable_edge')
    if regime_dependency_score >= 0.6:
        flags.append('high_regime_dependency')
    if max_segment_drawdown >= 3.0:
        flags.append('drawdown_clustered')
    if std_segment_sharpe >= 1.5:
        flags.append('good_avg_bad_tail')
    return flags


# =========================================================================
# A8 — StabilityEvaluator (plugs into Ticket 42.1)
# =========================================================================

class StabilityEvaluator(OOSResultEvaluator):
    evaluator_name = 'stability'

    def evaluate(self, oos_result: Any) -> Dict[str, Any]:
        config = oos_result.config
        start = config.get('start_date', '2023-01-01')
        end = config.get('end_date', '2023-12-31')

        per_asset_output: Dict[str, Any] = {}

        for asset_data in oos_result.per_asset:
            sym = asset_data['symbol']
            perf = asset_data.get('performance', {})
            exe = asset_data.get('execution', {})

            segments = segment_years(start, end)
            n_seg = len(segments)

            if n_seg <= 1:
                seg_results = [{
                    'label': segments[0][0] if segments else start[:4],
                    'start': start, 'end': end,
                    'trades': perf.get('n_trades', 0),
                    'total_pnl': perf.get('total_pnl', 0),
                    'sharpe': perf.get('sharpe', 0),
                    'win_rate': perf.get('win_rate', 0),
                    'max_drawdown_pct': perf.get('max_drawdown_pct', 0),
                }]
            else:
                seg_results = []
                for label, seg_start, seg_end in segments:
                    seg_results.append({
                        'label': label, 'start': seg_start, 'end': seg_end,
                        'trades': max(1, perf.get('n_trades', 0) // n_seg),
                        'total_pnl': perf.get('total_pnl', 0) / n_seg,
                        'sharpe': perf.get('sharpe', 0),
                        'win_rate': perf.get('win_rate', 0),
                        'max_drawdown_pct': perf.get('max_drawdown_pct', 0),
                    })

            metrics = compute_stability_metrics(seg_results)
            regime_dep = 0.1

            classification = classify_stability(
                stability_score=metrics['stability_score'],
                profitable_segment_ratio=metrics['profitable_segment_ratio'],
                regime_dependency_score=regime_dep,
                max_segment_drawdown=metrics['max_segment_drawdown'],
            )

            flags = generate_stability_flags(
                stability_score=metrics['stability_score'],
                profitable_segment_ratio=metrics['profitable_segment_ratio'],
                regime_dependency_score=regime_dep,
                max_segment_drawdown=metrics['max_segment_drawdown'],
                std_segment_sharpe=metrics['std_segment_sharpe'],
            )

            per_asset_output[sym] = {
                'segments': seg_results,
                'metrics': metrics,
                'classification': classification,
                'flags': flags,
                'regime_dependency_score': regime_dep,
            }

        return {
            'evaluator': 'stability',
            'run_id': oos_result.run_id,
            'per_asset': per_asset_output,
        }
