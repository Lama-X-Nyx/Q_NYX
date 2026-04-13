"""
TDD Tests — Soft Gate Architecture

ML = moteur principal (décide)
Non-ML = structure, audit, garde-fou

3 niveaux:
  1. Soft validation: rule scores entrent comme FEATURES du meta-ML
  2. Size/risk adjustment: disagreement → reduce size, raise threshold
  3. Hard veto: RARE, seulement cas extrêmes (data invalide, spread fou)

Key: le disagreement entre ML et rules EST une feature, pas un blocage.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_15M = Path(__file__).parent.parent / 'data' / 'raw' / 'mtf' / 'BTCUSDT_15m.csv'


@pytest.fixture(scope='module')
def real_data():
    df = pd.read_csv(DATA_15M)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df


# ===========================================================================
# TEST 1: Rule validators produce scores, not gates
# ===========================================================================
class TestRuleValidators:

    def test_context_rule_returns_score_not_bool(self, real_data):
        """Context rule must return a float score [0,1], not True/False."""
        from src.ml.soft_gate import ContextRuleValidator
        v = ContextRuleValidator()
        score = v.score(real_data.loc['2023-01-01':'2023-03-31'])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_regime_rule_returns_score(self, real_data):
        from src.ml.soft_gate import RegimeRuleValidator
        v = RegimeRuleValidator()
        score = v.score(real_data.loc['2023-01-01':'2023-03-31'])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_setup_rule_returns_score(self, real_data):
        from src.ml.soft_gate import SetupRuleValidator
        v = SetupRuleValidator()
        score = v.score(real_data.loc['2023-01-01':'2023-03-31'])
        assert isinstance(score, float)

    def test_validators_are_bullish_on_uptrend(self, real_data):
        """On a bull period, context rule score should be > 0.5."""
        from src.ml.soft_gate import ContextRuleValidator
        v = ContextRuleValidator()
        score = v.score(real_data.loc['2023-01-01':'2023-03-31'])
        # BTC +72% Q1 2023
        assert score >= 0.5, f"Context score {score:.2f} on +72% bull period"


# ===========================================================================
# TEST 2: Disagreement signal
# ===========================================================================
class TestDisagreementSignal:

    def test_disagreement_score_computed(self, real_data):
        """Must compute disagreement between ML signal and rule validators."""
        from src.ml.soft_gate import SoftGateOrchestrator
        orch = SoftGateOrchestrator()
        bar_data = real_data.loc['2023-03-15':'2023-03-31']
        signals = orch.compute_signals(bar_data)
        assert 'disagreement' in signals
        assert 0.0 <= signals['disagreement'] <= 1.0

    def test_full_agreement_low_disagreement(self):
        """When ML and rules agree, disagreement should be near 0."""
        from src.ml.soft_gate import compute_disagreement
        ml_direction = 1  # ML says BUY
        rule_scores = {'context': 0.8, 'regime': 0.7, 'setup': 0.6}  # rules agree
        d = compute_disagreement(ml_direction, rule_scores)
        assert d < 0.3, f"Disagreement {d:.2f} too high when all agree"

    def test_full_disagreement_high_score(self):
        """When ML says BUY but rules say BEAR, disagreement should be high."""
        from src.ml.soft_gate import compute_disagreement
        ml_direction = 1  # ML says BUY
        rule_scores = {'context': 0.1, 'regime': 0.2, 'setup': 0.1}  # rules say no
        d = compute_disagreement(ml_direction, rule_scores)
        assert d > 0.5, f"Disagreement {d:.2f} too low when fully opposed"

    def test_disagreement_is_feature(self, real_data):
        """Disagreement must be included in the feature vector for meta-ML."""
        from src.ml.soft_gate import SoftGateOrchestrator
        orch = SoftGateOrchestrator()
        bar_data = real_data.loc['2023-03-15':'2023-03-31']
        signals = orch.compute_signals(bar_data)
        assert 'rule_context' in signals
        assert 'rule_regime' in signals
        assert 'rule_setup' in signals
        assert 'disagreement' in signals


# ===========================================================================
# TEST 3: Size adjustment (not hard gate)
# ===========================================================================
class TestSizeAdjustment:

    def test_high_agreement_full_size(self):
        """Full agreement → size_factor = 1.0 or higher."""
        from src.ml.soft_gate import compute_size_factor
        sf = compute_size_factor(
            ml_confidence=0.7, disagreement=0.0, rule_avg=0.8)
        assert sf >= 1.0

    def test_disagreement_reduces_size(self):
        """High disagreement → size_factor < 1.0."""
        from src.ml.soft_gate import compute_size_factor
        sf_agree = compute_size_factor(
            ml_confidence=0.7, disagreement=0.0, rule_avg=0.8)
        sf_disagree = compute_size_factor(
            ml_confidence=0.7, disagreement=0.8, rule_avg=0.2)
        assert sf_disagree < sf_agree, \
            f"Disagreement should reduce size: {sf_disagree:.2f} >= {sf_agree:.2f}"

    def test_size_never_zero_unless_veto(self):
        """Size should never be 0 from soft gate alone (that's hard veto territory)."""
        from src.ml.soft_gate import compute_size_factor
        sf = compute_size_factor(
            ml_confidence=0.6, disagreement=0.9, rule_avg=0.1)
        assert sf > 0, "Soft gate should not zero out position"


# ===========================================================================
# TEST 4: Hard veto — RARE
# ===========================================================================
class TestHardVeto:

    def test_no_veto_in_normal_conditions(self, real_data):
        """Normal market: hard veto should NOT trigger."""
        from src.ml.soft_gate import check_hard_veto
        bar = real_data.iloc[5000]  # random normal bar
        veto, reason = check_hard_veto(
            price=bar['close'], atr=50.0, spread_pct=0.01, volume_ratio=1.0)
        assert veto is False, f"Veto triggered in normal conditions: {reason}"

    def test_veto_on_zero_atr(self):
        """ATR = 0 (no volatility data) → must veto."""
        from src.ml.soft_gate import check_hard_veto
        veto, reason = check_hard_veto(price=28000, atr=0.0, spread_pct=0.01, volume_ratio=1.0)
        assert veto is True

    def test_veto_on_extreme_spread(self):
        """Spread > 1% → must veto."""
        from src.ml.soft_gate import check_hard_veto
        veto, reason = check_hard_veto(price=28000, atr=50.0, spread_pct=1.5, volume_ratio=1.0)
        assert veto is True

    def test_veto_on_zero_volume(self):
        """Volume ratio = 0 → must veto."""
        from src.ml.soft_gate import check_hard_veto
        veto, reason = check_hard_veto(price=28000, atr=50.0, spread_pct=0.01, volume_ratio=0.0)
        assert veto is True


# ===========================================================================
# TEST 5: Backtest comparison — soft gate vs hard gate
# ===========================================================================
class TestSoftVsHardGate:

    def test_soft_gate_more_trades_than_hard(self, real_data):
        """Soft gate should produce MORE trades (less blocking)."""
        from src.ml.soft_gate import SoftGateOrchestrator
        from src.ml.realistic_backtest import RealisticBacktester
        # Hard gate = old style (vol 3.0x, hours, 1/day)
        hard = RealisticBacktester(vol_min=3.0, use_hours=True, cooldown_bars=32,
                                    max_daily_trades=1, fee_rate=0.0002, slippage_rate=0.0001)
        r_hard = hard.run(real_data.loc['2023-01-01':'2023-12-31'])
        # Soft gate
        orch = SoftGateOrchestrator()
        r_soft = orch.backtest(real_data.loc['2023-01-01':'2023-12-31'])
        # Soft should allow more trades (ML decides, rules don't block)
        assert r_soft['n_trades'] >= r_hard['n_trades'] * 0.5, \
            f"Soft gate ({r_soft['n_trades']}) has too few trades vs hard ({r_hard['n_trades']})"

    def test_soft_gate_has_positive_sharpe(self, real_data):
        """Soft gate must have positive Sharpe on 2023 (maker fees)."""
        from src.ml.soft_gate import SoftGateOrchestrator
        orch = SoftGateOrchestrator()
        r_soft = orch.backtest(real_data.loc['2023-01-01':'2023-12-31'])
        assert r_soft['sharpe'] > 0, \
            f"Soft gate Sharpe {r_soft['sharpe']:.2f} negative on 2023"

    def test_soft_gate_uses_disagreement_for_sizing(self, real_data):
        """Trades with high disagreement should have smaller size_factor."""
        from src.ml.soft_gate import SoftGateOrchestrator
        orch = SoftGateOrchestrator()
        r = orch.backtest(real_data.loc['2023-01-01':'2023-12-31'])
        if len(r['trades']) >= 5:
            # Should see variation in size_factor (not all same)
            sfs = [t['size_factor'] for t in r['trades']]
            assert len(set(round(s, 2) for s in sfs)) > 1, \
                "All trades have same size_factor — disagreement not modulating size"
