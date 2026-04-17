"""
TDD Tests — Ticket 34 — Dynamic Inter-Asset Dependency Layer.

Lead/lag relationships, non-stationary correlation, contagion,
regime conditioning, and decision modulation.

Operates AFTER per-asset GBM + Jesse + Fractal Quality,
BEFORE RiskEngine. Alpha untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# =========================================================================
# A1 — Lagged Feature Matrix
# =========================================================================
class TestLaggedFeatureMatrix:

    def _make_returns(self, n: int = 200, seed: int = 42) -> dict:
        rng = np.random.RandomState(seed)
        idx = pd.date_range('2023-01-01', periods=n, freq='1h')
        btc = pd.Series(rng.randn(n) * 0.01, index=idx, name='BTCUSDT')
        eth = pd.Series(
            0.6 * np.roll(btc.values, 4) + rng.randn(n) * 0.005,
            index=idx, name='ETHUSDT',
        )
        return {'BTCUSDT': btc, 'ETHUSDT': eth}

    def test_compute_lagged_returns(self):
        from src.live.inter_asset_dependency import compute_lagged_features
        rets = self._make_returns()
        result = compute_lagged_features(rets, 'BTCUSDT', 'ETHUSDT', lags=[1, 4, 8, 24])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        for lag in [1, 4, 8, 24]:
            assert f'ret_BTCUSDT_lag{lag}' in result.columns
            assert f'ret_ETHUSDT_lag{lag}' in result.columns

    def test_lagged_features_include_vol_ratio(self):
        from src.live.inter_asset_dependency import compute_lagged_features
        rets = self._make_returns()
        result = compute_lagged_features(rets, 'BTCUSDT', 'ETHUSDT', lags=[4])
        assert 'vol_ratio' in result.columns

    def test_lagged_features_include_momentum_spread(self):
        from src.live.inter_asset_dependency import compute_lagged_features
        rets = self._make_returns()
        result = compute_lagged_features(rets, 'BTCUSDT', 'ETHUSDT', lags=[4])
        assert 'momentum_spread' in result.columns

    def test_lagged_features_no_future_leakage(self):
        from src.live.inter_asset_dependency import compute_lagged_features
        rets = self._make_returns()
        result = compute_lagged_features(rets, 'BTCUSDT', 'ETHUSDT', lags=[4])
        assert not result.isna().all().any()
        assert result.index[-1] <= rets['BTCUSDT'].index[-1]


# =========================================================================
# A2 — Lead/Lag Detection
# =========================================================================
class TestLeadLagDetection:

    def _make_leader_follower(self, n: int = 500, lag: int = 4, seed: int = 42):
        rng = np.random.RandomState(seed)
        idx = pd.date_range('2023-01-01', periods=n, freq='1h')
        leader = pd.Series(rng.randn(n) * 0.02, index=idx)
        follower = pd.Series(
            0.7 * np.roll(leader.values, lag) + rng.randn(n) * 0.005,
            index=idx,
        )
        return leader, follower

    def test_detect_lead_lag_returns_dict(self):
        from src.live.inter_asset_dependency import detect_lead_lag
        leader, follower = self._make_leader_follower()
        result = detect_lead_lag(
            {'BTCUSDT': leader, 'ETHUSDT': follower},
            'BTCUSDT', 'ETHUSDT',
            max_lag=24,
        )
        assert isinstance(result, dict)
        assert 'leader_asset' in result
        assert 'follower_asset' in result
        assert 'lag_bars' in result
        assert 'confidence' in result

    def test_btc_leads_eth_detected(self):
        from src.live.inter_asset_dependency import detect_lead_lag
        leader, follower = self._make_leader_follower(lag=4)
        result = detect_lead_lag(
            {'BTCUSDT': leader, 'ETHUSDT': follower},
            'BTCUSDT', 'ETHUSDT',
            max_lag=24,
        )
        assert result['leader_asset'] == 'BTCUSDT'
        assert result['follower_asset'] == 'ETHUSDT'
        assert 2 <= result['lag_bars'] <= 8

    def test_confidence_in_range(self):
        from src.live.inter_asset_dependency import detect_lead_lag
        leader, follower = self._make_leader_follower()
        result = detect_lead_lag(
            {'BTCUSDT': leader, 'ETHUSDT': follower},
            'BTCUSDT', 'ETHUSDT',
            max_lag=24,
        )
        assert 0.0 <= result['confidence'] <= 1.0

    def test_symmetric_pair_low_confidence(self):
        from src.live.inter_asset_dependency import detect_lead_lag
        rng = np.random.RandomState(99)
        n = 500
        idx = pd.date_range('2023-01-01', periods=n, freq='1h')
        a = pd.Series(rng.randn(n) * 0.01, index=idx)
        b = pd.Series(rng.randn(n) * 0.01, index=idx)
        result = detect_lead_lag(
            {'A': a, 'B': b}, 'A', 'B', max_lag=24,
        )
        assert result['confidence'] < 0.5


# =========================================================================
# A3 — Regime Conditioning
# =========================================================================
class TestRegimeConditioning:

    def test_dependency_score_varies_by_regime(self):
        from src.live.inter_asset_dependency import compute_dependency_score
        score_trend = compute_dependency_score(
            lead_lag_confidence=0.8,
            correlation=0.7,
            regime='trending',
            contagion_risk=0.2,
        )
        score_range = compute_dependency_score(
            lead_lag_confidence=0.8,
            correlation=0.7,
            regime='ranging',
            contagion_risk=0.2,
        )
        assert score_trend != score_range

    def test_trending_regime_amplifies_dependency(self):
        from src.live.inter_asset_dependency import compute_dependency_score
        score_trend = compute_dependency_score(
            lead_lag_confidence=0.7,
            correlation=0.6,
            regime='trending',
            contagion_risk=0.1,
        )
        score_range = compute_dependency_score(
            lead_lag_confidence=0.7,
            correlation=0.6,
            regime='ranging',
            contagion_risk=0.1,
        )
        assert score_trend > score_range

    def test_dependency_score_in_range(self):
        from src.live.inter_asset_dependency import compute_dependency_score
        for regime in ('trending', 'ranging', 'unknown'):
            score = compute_dependency_score(
                lead_lag_confidence=0.5,
                correlation=0.5,
                regime=regime,
                contagion_risk=0.3,
            )
            assert 0.0 <= score <= 1.0


# =========================================================================
# A4 — Contagion Modeling
# =========================================================================
class TestContagionModeling:

    def _make_shock(self, n: int = 200, seed: int = 42):
        rng = np.random.RandomState(seed)
        idx = pd.date_range('2023-01-01', periods=n, freq='1h')
        btc = pd.Series(rng.randn(n) * 0.01, index=idx)
        btc.iloc[100] = -0.08  # shock event
        eth = pd.Series(rng.randn(n) * 0.008, index=idx)
        eth.iloc[102] = -0.06  # propagated shock
        return {'BTCUSDT': btc, 'ETHUSDT': eth}

    def test_detect_shock_events(self):
        from src.live.inter_asset_dependency import detect_contagion
        rets = self._make_shock()
        result = detect_contagion(rets, 'BTCUSDT', 'ETHUSDT', threshold_std=3.0)
        assert isinstance(result, dict)
        assert 'contagion_risk' in result
        assert 'spillover_probability' in result
        assert 'shock_events_leader' in result

    def test_shock_increases_contagion_risk(self):
        from src.live.inter_asset_dependency import detect_contagion
        rets_shock = self._make_shock()
        rets_calm = {
            'BTCUSDT': pd.Series(
                np.random.RandomState(42).randn(200) * 0.005,
                index=pd.date_range('2023-01-01', periods=200, freq='1h'),
            ),
            'ETHUSDT': pd.Series(
                np.random.RandomState(43).randn(200) * 0.005,
                index=pd.date_range('2023-01-01', periods=200, freq='1h'),
            ),
        }
        r_shock = detect_contagion(rets_shock, 'BTCUSDT', 'ETHUSDT')
        r_calm = detect_contagion(rets_calm, 'BTCUSDT', 'ETHUSDT')
        assert r_shock['contagion_risk'] >= r_calm['contagion_risk']

    def test_contagion_values_in_range(self):
        from src.live.inter_asset_dependency import detect_contagion
        rets = self._make_shock()
        result = detect_contagion(rets, 'BTCUSDT', 'ETHUSDT')
        assert 0.0 <= result['contagion_risk'] <= 1.0
        assert 0.0 <= result['spillover_probability'] <= 1.0


# =========================================================================
# A6 — Dependency Scoring (aggregated)
# =========================================================================
class TestDependencyScoring:

    def test_aggregate_dependency_output(self):
        from src.live.inter_asset_dependency import aggregate_dependency
        result = aggregate_dependency(
            lead_lag={'leader_asset': 'BTCUSDT', 'follower_asset': 'ETHUSDT',
                      'lag_bars': 4, 'confidence': 0.8},
            contagion={'contagion_risk': 0.3, 'spillover_probability': 0.4,
                       'shock_events_leader': 2},
            regime='trending',
            rolling_correlation=0.65,
        )
        assert 'dependency_strength' in result
        assert 'lag_alignment_score' in result
        assert 'divergence_score' in result
        assert 'contagion_risk' in result

    def test_dependency_strength_in_range(self):
        from src.live.inter_asset_dependency import aggregate_dependency
        result = aggregate_dependency(
            lead_lag={'leader_asset': 'BTCUSDT', 'follower_asset': 'ETHUSDT',
                      'lag_bars': 4, 'confidence': 0.7},
            contagion={'contagion_risk': 0.2, 'spillover_probability': 0.3,
                       'shock_events_leader': 1},
            regime='trending',
            rolling_correlation=0.5,
        )
        assert 0.0 <= result['dependency_strength'] <= 1.0
        assert 0.0 <= result['lag_alignment_score'] <= 1.0
        assert 0.0 <= result['divergence_score'] <= 1.0


# =========================================================================
# A7 — Decision Modulation
# =========================================================================
class TestDecisionModulation:

    def test_modulate_returns_dict(self):
        from src.live.inter_asset_dependency import modulate_decision
        result = modulate_decision(
            symbol='ETHUSDT',
            direction=1,
            size_multiplier=1.0,
            dependency={
                'dependency_strength': 0.7,
                'lag_alignment_score': 0.8,
                'divergence_score': 0.1,
                'contagion_risk': 0.2,
            },
            leader_direction=1,
        )
        assert 'adjusted_size_multiplier' in result
        assert 'confidence_adjustment' in result
        assert 'suppress_trade' in result

    def test_aligned_with_leader_increases_size(self):
        from src.live.inter_asset_dependency import modulate_decision
        result = modulate_decision(
            symbol='ETHUSDT',
            direction=1,
            size_multiplier=1.0,
            dependency={
                'dependency_strength': 0.8,
                'lag_alignment_score': 0.9,
                'divergence_score': 0.05,
                'contagion_risk': 0.1,
            },
            leader_direction=1,
        )
        assert result['adjusted_size_multiplier'] >= 1.0

    def test_misaligned_with_leader_reduces_size(self):
        from src.live.inter_asset_dependency import modulate_decision
        result = modulate_decision(
            symbol='ETHUSDT',
            direction=1,
            size_multiplier=1.0,
            dependency={
                'dependency_strength': 0.8,
                'lag_alignment_score': 0.9,
                'divergence_score': 0.05,
                'contagion_risk': 0.1,
            },
            leader_direction=-1,
        )
        assert result['adjusted_size_multiplier'] < 1.0

    def test_high_contagion_suppresses_trade(self):
        from src.live.inter_asset_dependency import modulate_decision
        result = modulate_decision(
            symbol='ETHUSDT',
            direction=1,
            size_multiplier=1.0,
            dependency={
                'dependency_strength': 0.9,
                'lag_alignment_score': 0.5,
                'divergence_score': 0.5,
                'contagion_risk': 0.9,
            },
            leader_direction=-1,
        )
        assert result['suppress_trade'] is True

    def test_direction_never_changed(self):
        from src.live.inter_asset_dependency import modulate_decision
        for ld in (-1, 0, 1):
            result = modulate_decision(
                symbol='ETHUSDT',
                direction=1,
                size_multiplier=1.0,
                dependency={
                    'dependency_strength': 0.8,
                    'lag_alignment_score': 0.5,
                    'divergence_score': 0.3,
                    'contagion_risk': 0.5,
                },
                leader_direction=ld,
            )
            assert 'direction' not in result or result.get('direction') == 1

    def test_btc_as_leader_not_modulated(self):
        """BTC is the leader — dependency layer should not reduce its own signal."""
        from src.live.inter_asset_dependency import modulate_decision
        result = modulate_decision(
            symbol='BTCUSDT',
            direction=1,
            size_multiplier=1.0,
            dependency={
                'dependency_strength': 0.8,
                'lag_alignment_score': 0.5,
                'divergence_score': 0.3,
                'contagion_risk': 0.5,
            },
            leader_direction=1,
        )
        assert result['adjusted_size_multiplier'] == 1.0
        assert result['suppress_trade'] is False


# =========================================================================
# A8 — InterAssetDependencyLayer (the orchestrator class)
# =========================================================================
class TestInterAssetDependencyLayer:

    def test_layer_instantiation(self):
        from src.live.inter_asset_dependency import InterAssetDependencyLayer
        layer = InterAssetDependencyLayer(symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        assert layer.symbols == ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

    def test_update_returns_with_single_bar(self):
        from src.live.inter_asset_dependency import InterAssetDependencyLayer
        layer = InterAssetDependencyLayer(symbols=['BTCUSDT', 'ETHUSDT'])
        layer.update_return('BTCUSDT', '2023-01-01T00:00', 0.01)
        layer.update_return('ETHUSDT', '2023-01-01T00:00', 0.008)
        assert len(layer._return_buffer['BTCUSDT']) == 1
        assert len(layer._return_buffer['ETHUSDT']) == 1

    def test_evaluate_requires_minimum_data(self):
        from src.live.inter_asset_dependency import InterAssetDependencyLayer
        layer = InterAssetDependencyLayer(
            symbols=['BTCUSDT', 'ETHUSDT'], min_bars=50,
        )
        layer.update_return('BTCUSDT', '2023-01-01T00:00', 0.01)
        layer.update_return('ETHUSDT', '2023-01-01T00:00', 0.008)
        result = layer.evaluate('ETHUSDT', direction=1, size_multiplier=1.0)
        assert result['adjusted_size_multiplier'] == 1.0
        assert result['suppress_trade'] is False

    def test_evaluate_deterministic(self):
        from src.live.inter_asset_dependency import InterAssetDependencyLayer
        layer = InterAssetDependencyLayer(symbols=['BTCUSDT', 'ETHUSDT'])
        rng = np.random.RandomState(42)
        idx = pd.date_range('2023-01-01', periods=200, freq='1h')
        for i, ts in enumerate(idx):
            layer.update_return('BTCUSDT', str(ts), float(rng.randn() * 0.01))
            layer.update_return('ETHUSDT', str(ts), float(rng.randn() * 0.008))
        r1 = layer.evaluate('ETHUSDT', direction=1, size_multiplier=1.0)
        r2 = layer.evaluate('ETHUSDT', direction=1, size_multiplier=1.0)
        assert r1 == r2


# =========================================================================
# A8 — NYXRuntime Integration
# =========================================================================
class TestRuntimeIntegration:

    def test_runtime_accepts_dependency_layer(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        from src.live.inter_asset_dependency import InterAssetDependencyLayer
        HERE = Path(__file__).resolve().parent.parent
        layer = InterAssetDependencyLayer(symbols=['BTCUSDT', 'ETHUSDT'])
        rt = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=HERE / 'models' / 'ETHUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'ETHUSDT',
            dependency_layer=layer,
        )
        assert rt.dependency_layer is layer

    def test_runtime_without_dependency_layer_still_works(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        HERE = Path(__file__).resolve().parent.parent
        rt = NYXRuntime(
            symbol='BTCUSDT',
            models_dir=HERE / 'models' / 'BTCUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'BTCUSDT',
        )
        assert rt.dependency_layer is None
        rt.start()
        bar = {
            'timestamp': '2023-06-15T12:00:00',
            'open': 30000.0, 'high': 30100.0,
            'low': 29900.0, 'close': 30050.0, 'volume': 100.0,
        }
        result = rt.on_bar(bar)
        assert result['action'] in ('FLAT', 'ORDER_SUBMITTED', 'SKIP_QUALITY', 'BLOCKED_RISK')

    def test_on_bar_includes_dependency_layer_in_result(self):
        import tempfile
        from pathlib import Path
        from src.live.nyx_runtime import NYXRuntime
        from src.live.inter_asset_dependency import InterAssetDependencyLayer
        HERE = Path(__file__).resolve().parent.parent
        layer = InterAssetDependencyLayer(symbols=['BTCUSDT', 'ETHUSDT'], min_bars=5)
        rng = np.random.RandomState(42)
        for i in range(50):
            ts = f'2023-06-15T{i:02d}:00:00'
            layer.update_return('BTCUSDT', ts, float(rng.randn() * 0.01))
            layer.update_return('ETHUSDT', ts, float(rng.randn() * 0.008))
        layer.set_leader_direction(1)
        rt = NYXRuntime(
            symbol='ETHUSDT',
            models_dir=HERE / 'models' / 'ETHUSDT',
            state_dir=Path(tempfile.mkdtemp()) / 'ETHUSDT',
            dependency_layer=layer,
        )
        rt.start()
        bar = {
            'timestamp': '2023-06-15T12:00:00',
            'open': 2000.0, 'high': 2010.0,
            'low': 1990.0, 'close': 2005.0, 'volume': 50.0,
        }
        result = rt.on_bar(bar)
        # If signal was FLAT, dependency layer won't appear
        # If signal passed, dependency layer should appear
        if result['action'] not in ('FLAT', 'SYSTEM_STOPPED', 'PAUSED'):
            assert 'dependency' in result['layers']
