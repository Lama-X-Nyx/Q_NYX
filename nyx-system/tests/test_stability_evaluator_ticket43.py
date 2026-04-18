"""
TDD Tests — Ticket 43 — Multi-Year Regime Stability Evaluator.

Plugs into Ticket 42.1 evaluator framework. Consumes OOSResult,
segments by year/quarter, tags regimes, classifies stability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent


def _make_segment_trades(n_segments: int = 4):
    """Create synthetic per-segment trade data for testing."""
    segments = []
    rng = np.random.RandomState(42)
    for i in range(n_segments):
        year = 2020 + i
        n_trades = 30 + rng.randint(0, 20)
        pnls = rng.randn(n_trades) * 100 + 20
        segments.append({
            'label': str(year),
            'start': f'{year}-01-01',
            'end': f'{year}-12-31',
            'trades': n_trades,
            'pnls': pnls.tolist(),
            'total_pnl': float(pnls.sum()),
            'sharpe': float(np.mean(pnls) / max(np.std(pnls), 1e-12) * np.sqrt(min(n_trades, 252))),
            'win_rate': float(np.mean(pnls > 0)),
            'max_drawdown_pct': float(rng.uniform(0.5, 3.0)),
        })
    return segments


class TestSegmentSlicer:

    def test_yearly_segments(self):
        from src.live.evaluators.stability import segment_years
        result = segment_years('2020-01-01', '2023-12-31')
        assert len(result) == 4
        assert result[0] == ('2020', '2020-01-01', '2020-12-31')
        assert result[3] == ('2023', '2023-01-01', '2023-12-31')

    def test_quarterly_segments(self):
        from src.live.evaluators.stability import segment_quarters
        result = segment_quarters('2023-01-01', '2023-12-31')
        assert len(result) == 4
        assert result[0][0] == '2023-Q1'
        assert result[3][0] == '2023-Q4'

    def test_single_year(self):
        from src.live.evaluators.stability import segment_years
        result = segment_years('2023-01-01', '2023-12-31')
        assert len(result) == 1


class TestRegimeTagger:

    def test_volatility_regime(self):
        from src.live.evaluators.stability import tag_regime_volatility
        assert tag_regime_volatility(0.01) == 'low_vol'
        assert tag_regime_volatility(0.03) == 'medium_vol'
        assert tag_regime_volatility(0.06) == 'high_vol'

    def test_trend_regime(self):
        from src.live.evaluators.stability import tag_regime_trend
        assert tag_regime_trend(0.15) == 'bull'
        assert tag_regime_trend(-0.15) == 'bear'
        assert tag_regime_trend(0.02) == 'range'

    def test_composite_regime(self):
        from src.live.evaluators.stability import tag_regime_composite
        label = tag_regime_composite(return_pct=0.15, volatility=0.05)
        assert label == 'bull_high_vol'


class TestStabilityMetrics:

    def test_compute_stability_metrics(self):
        from src.live.evaluators.stability import compute_stability_metrics
        segments = _make_segment_trades(4)
        metrics = compute_stability_metrics(segments)
        assert 'mean_segment_sharpe' in metrics
        assert 'std_segment_sharpe' in metrics
        assert 'best_segment_sharpe' in metrics
        assert 'worst_segment_sharpe' in metrics
        assert 'profitable_segment_ratio' in metrics
        assert 'stability_score' in metrics

    def test_stability_score_range(self):
        from src.live.evaluators.stability import compute_stability_metrics
        segments = _make_segment_trades(4)
        metrics = compute_stability_metrics(segments)
        assert 0.0 <= metrics['stability_score'] <= 1.0


class TestClassification:

    def test_classify_robust(self):
        from src.live.evaluators.stability import classify_stability
        result = classify_stability(
            stability_score=0.85,
            profitable_segment_ratio=1.0,
            regime_dependency_score=0.1,
            max_segment_drawdown=1.0,
        )
        assert result == 'robust'

    def test_classify_unstable(self):
        from src.live.evaluators.stability import classify_stability
        result = classify_stability(
            stability_score=0.2,
            profitable_segment_ratio=0.3,
            regime_dependency_score=0.3,
            max_segment_drawdown=5.0,
        )
        assert result == 'unstable'

    def test_classify_regime_sensitive(self):
        from src.live.evaluators.stability import classify_stability
        result = classify_stability(
            stability_score=0.5,
            profitable_segment_ratio=0.7,
            regime_dependency_score=0.8,
            max_segment_drawdown=2.0,
        )
        assert result == 'regime_sensitive'


class TestStabilityFlags:

    def test_flags_generated(self):
        from src.live.evaluators.stability import generate_stability_flags
        flags = generate_stability_flags(
            stability_score=0.3,
            profitable_segment_ratio=0.5,
            regime_dependency_score=0.8,
            max_segment_drawdown=4.0,
            std_segment_sharpe=2.0,
        )
        assert isinstance(flags, list)
        assert 'unstable_edge' in flags
        assert 'high_regime_dependency' in flags

    def test_no_flags_when_stable(self):
        from src.live.evaluators.stability import generate_stability_flags
        flags = generate_stability_flags(
            stability_score=0.9,
            profitable_segment_ratio=1.0,
            regime_dependency_score=0.1,
            max_segment_drawdown=0.5,
            std_segment_sharpe=0.5,
        )
        assert 'stable_across_cycles' in flags
        assert 'unstable_edge' not in flags


class TestStabilityEvaluatorIntegration:

    def test_evaluator_has_correct_name(self):
        from src.live.evaluators.stability import StabilityEvaluator
        ev = StabilityEvaluator()
        assert ev.evaluator_name == 'stability'

    def test_evaluator_produces_output(self):
        from src.live.evaluators.stability import StabilityEvaluator
        from src.live.oos_engine import OOSResult
        data = {
            'run_id': 'test-stability-001',
            'config': {'assets': ['BTCUSDT'], 'start_date': '2023-01-01',
                       'end_date': '2023-12-31', 'mode': 'realistic',
                       'initial_capital': 10_000.0, 'train_end': '2022-12-31'},
            'per_asset': [{
                'symbol': 'BTCUSDT', 'capital': 10_000.0, 'mode': 'realistic',
                'performance': {'n_trades': 39, 'win_rate': 0.769, 'sharpe': 4.2,
                                'total_pnl': 1150.0, 'max_drawdown_pct': 0.93},
                'execution': {'idealized_signals': 63, 'placed': 59, 'filled': 39,
                              'timed_out': 20, 'fill_rate': 0.66, 'miss_rate': 0.34,
                              'skipped_quality': 0, 'blocked_risk': 4},
            }],
            'portfolio': {'total_pnl': 1150.0, 'total_trades': 39},
        }
        ev = StabilityEvaluator()
        result = OOSResult(data)
        output = ev.evaluate(result)
        assert isinstance(output, dict)
        assert 'per_asset' in output
        assert 'BTCUSDT' in output['per_asset']
        btc = output['per_asset']['BTCUSDT']
        assert 'classification' in btc
        assert 'flags' in btc
        assert 'stability_score' in btc.get('metrics', btc)

    def test_evaluator_output_serializable(self):
        from src.live.evaluators.stability import StabilityEvaluator
        from src.live.oos_engine import OOSResult
        data = {
            'run_id': 'test-ser',
            'config': {'assets': ['BTCUSDT'], 'start_date': '2023-01-01',
                       'end_date': '2023-12-31', 'mode': 'realistic',
                       'initial_capital': 10_000.0, 'train_end': '2022-12-31'},
            'per_asset': [{
                'symbol': 'BTCUSDT', 'capital': 10_000.0, 'mode': 'realistic',
                'performance': {'n_trades': 39, 'win_rate': 0.769, 'sharpe': 4.2,
                                'total_pnl': 1150.0, 'max_drawdown_pct': 0.93},
                'execution': {'idealized_signals': 63, 'placed': 59, 'filled': 39,
                              'timed_out': 20, 'fill_rate': 0.66, 'miss_rate': 0.34,
                              'skipped_quality': 0, 'blocked_risk': 4},
            }],
            'portfolio': {'total_pnl': 1150.0, 'total_trades': 39},
        }
        ev = StabilityEvaluator()
        output = ev.evaluate(OOSResult(data))
        s = json.dumps(output, default=str)
        assert len(s) > 10

    def test_evaluator_deterministic(self):
        from src.live.evaluators.stability import StabilityEvaluator
        from src.live.oos_engine import OOSResult
        data = {
            'run_id': 'test-det',
            'config': {'assets': ['BTCUSDT'], 'start_date': '2023-01-01',
                       'end_date': '2023-12-31', 'mode': 'realistic',
                       'initial_capital': 10_000.0, 'train_end': '2022-12-31'},
            'per_asset': [{
                'symbol': 'BTCUSDT', 'capital': 10_000.0, 'mode': 'realistic',
                'performance': {'n_trades': 39, 'win_rate': 0.769, 'sharpe': 4.2,
                                'total_pnl': 1150.0, 'max_drawdown_pct': 0.93},
                'execution': {'idealized_signals': 63, 'placed': 59, 'filled': 39,
                              'timed_out': 20, 'fill_rate': 0.66, 'miss_rate': 0.34,
                              'skipped_quality': 0, 'blocked_risk': 4},
            }],
            'portfolio': {'total_pnl': 1150.0, 'total_trades': 39},
        }
        ev = StabilityEvaluator()
        r1 = ev.evaluate(OOSResult(data))
        r2 = ev.evaluate(OOSResult(data))
        assert json.dumps(r1, default=str) == json.dumps(r2, default=str)
