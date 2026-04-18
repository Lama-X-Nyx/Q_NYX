"""
Multi-Year Regime Stability Evaluator — Ticket 43 + 43.1.

Plugs into Ticket 42.1 evaluator framework. Consumes OOSResult,
segments by year/quarter, tags regimes, classifies stability.

Ticket 43.1 additions: tail risk, gain concentration, regime
dependency matrix, capital-aware classification, stability v2.

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

# --- Ticket 43.1 additions ---

def compute_tail_risk(pnls: List[float]) -> Dict[str, Any]:
    if not pnls:
        return {'worst_trade': 0.0, 'tail_loss_95': 0.0, 'tail_loss_99': 0.0,
                'downside_deviation': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}
    arr = np.array(pnls)
    downside = arr[arr < 0]
    dd_std = float(np.std(downside)) if len(downside) > 0 else 0.0
    return {
        'worst_trade': float(np.min(arr)),
        'tail_loss_95': float(np.percentile(arr, 5)),
        'tail_loss_99': float(np.percentile(arr, 1)),
        'downside_deviation': round(dd_std, 4),
        'skewness': round(float(_skewness(arr)), 4),
        'kurtosis': round(float(_kurtosis(arr)), 4),
    }


def _skewness(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


def compute_gain_concentration(pnls: List[float]) -> Dict[str, Any]:
    if not pnls:
        return {'top_5pct_contribution': 0.0, 'top_10_contribution': 0.0, 'gini_coefficient': 0.0}
    arr = np.array(pnls)
    total = float(np.sum(np.abs(arr)))
    if total < 1e-12:
        return {'top_5pct_contribution': 0.0, 'top_10_contribution': 0.0, 'gini_coefficient': 0.0}
    sorted_abs = np.sort(np.abs(arr))[::-1]
    n5 = max(1, len(arr) // 20)
    n10 = min(10, len(arr))
    top5_contrib = float(np.sum(sorted_abs[:n5]) / total)
    top10_contrib = float(np.sum(sorted_abs[:n10]) / total)
    n = len(arr)
    sorted_vals = np.sort(np.abs(arr))
    cum = np.cumsum(sorted_vals)
    gini = float(1.0 - 2.0 * np.sum(cum) / (n * total)) if n > 0 else 0.0
    gini = max(0.0, min(1.0, abs(gini)))
    return {
        'top_5pct_contribution': round(top5_contrib, 4),
        'top_10_contribution': round(top10_contrib, 4),
        'gini_coefficient': round(gini, 4),
    }


def build_regime_dependency_matrix(
    segments: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    from collections import defaultdict
    by_regime: Dict[str, List[Dict]] = defaultdict(list)
    for s in segments:
        regime = s.get('regime', 'unknown')
        by_regime[regime].append(s)
    matrix = {}
    for regime, segs in by_regime.items():
        sharpes = [s['sharpe'] for s in segs]
        dds = [s.get('max_drawdown_pct', 0) for s in segs]
        wrs = [s.get('win_rate', 0) for s in segs]
        matrix[regime] = {
            'mean_sharpe': round(float(np.mean(sharpes)), 4),
            'mean_dd': round(float(np.mean(dds)), 4),
            'mean_wr': round(float(np.mean(wrs)), 4),
            'n_segments': len(segs),
        }
    return matrix


def compute_regime_dependency_score(
    matrix: Dict[str, Dict[str, float]],
) -> float:
    sharpes = [v['mean_sharpe'] for v in matrix.values()]
    if len(sharpes) < 2:
        return 0.0
    spread = max(sharpes) - min(sharpes)
    mean_abs = np.mean(np.abs(sharpes))
    if mean_abs < 1e-12:
        return 0.5
    score = min(1.0, spread / (mean_abs + 1.0))
    return round(float(score), 4)


def classify_capital_aware(
    stability_score: float,
    regime_dependency_score: float,
    tail_risk_severity: float,
    gain_concentration: float,
) -> Dict[str, Any]:
    composite = (
        stability_score * 0.4
        - tail_risk_severity * 0.2
        - gain_concentration * 0.2
        - regime_dependency_score * 0.2
    )
    if composite >= 0.2 and stability_score >= 0.7:
        alloc_class = 'core'
        tier = 'tier_1_unlimited'
    elif composite >= 0.05 and stability_score >= 0.5:
        alloc_class = 'satellite'
        tier = 'tier_2_medium'
    elif composite >= 0.0:
        alloc_class = 'opportunistic'
        tier = 'tier_3_small'
    else:
        alloc_class = 'avoid'
        tier = 'tier_4_none'
    return {
        'allocation_class': alloc_class,
        'max_capital_tier': tier,
        'composite_score': round(composite, 4),
        'stability_score': round(stability_score, 4),
        'regime_dependency': round(regime_dependency_score, 4),
        'tail_risk': round(tail_risk_severity, 4),
        'concentration': round(gain_concentration, 4),
    }


def compute_stability_score_v2(
    base_stability: float,
    tail_risk_severity: float,
    gain_concentration_gini: float,
    regime_dependency: float,
) -> float:
    score = (
        base_stability * 0.4
        - tail_risk_severity * 0.2
        - gain_concentration_gini * 0.2
        - regime_dependency * 0.2
        + 0.2
    )
    return round(max(0.0, min(1.0, score)), 4)


def generate_advanced_flags(
    tail_risk_severity: float,
    gain_concentration_gini: float,
    regime_dependency_score: float,
    stability_score_v2: float,
) -> List[str]:
    flags = []
    if tail_risk_severity >= 0.6:
        flags.append('tail_risk_dominant')
    if gain_concentration_gini >= 0.6:
        flags.append('gain_concentration_high')
    if regime_dependency_score >= 0.5:
        flags.append('requires_regime_filter')
    if stability_score_v2 < 0.3:
        flags.append('capital_limited_edge')
    return flags

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

            pnls_for_regime = [s['total_pnl'] for s in seg_results]
            regime_segs_pre = []
            for s in seg_results:
                ret_pct = s['total_pnl'] / max(asset_data.get('capital', 10_000), 1)
                s_copy = dict(s)
                s_copy['regime'] = tag_regime_trend(ret_pct)
                regime_segs_pre.append(s_copy)
            regime_matrix_pre = build_regime_dependency_matrix(regime_segs_pre)
            regime_dep = compute_regime_dependency_score(regime_matrix_pre)

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

            # Ticket 43.1 — v2 metrics
            pnls = [s['total_pnl'] for s in seg_results]
            tail = compute_tail_risk(pnls)
            concentration = compute_gain_concentration(pnls)
            tail_severity = min(1.0, abs(tail['worst_trade']) / max(abs(perf.get('total_pnl', 1)), 1))

            regime_matrix = regime_matrix_pre

            capital_class = classify_capital_aware(
                stability_score=metrics['stability_score'],
                regime_dependency_score=regime_dep,
                tail_risk_severity=tail_severity,
                gain_concentration=concentration['gini_coefficient'],
            )

            score_v2 = compute_stability_score_v2(
                base_stability=metrics['stability_score'],
                tail_risk_severity=tail_severity,
                gain_concentration_gini=concentration['gini_coefficient'],
                regime_dependency=regime_dep,
            )

            adv_flags = generate_advanced_flags(
                tail_risk_severity=tail_severity,
                gain_concentration_gini=concentration['gini_coefficient'],
                regime_dependency_score=regime_dep,
                stability_score_v2=score_v2,
            )

            per_asset_output[sym] = {
                'segments': seg_results,
                'metrics': metrics,
                'classification': classification,
                'flags': flags,
                'regime_dependency_score': regime_dep,
                'tail_risk': tail,
                'gain_concentration': concentration,
                'regime_matrix': regime_matrix,
                'capital_classification': capital_class,
                'stability_score_v2': score_v2,
                'advanced_flags': adv_flags,
            }

        return {
            'evaluator': 'stability',
            'run_id': oos_result.run_id,
            'per_asset': per_asset_output,
        }
