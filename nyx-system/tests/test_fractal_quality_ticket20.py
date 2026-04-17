"""
TDD Tests — Ticket 20.

Jesse agents as POST-DECISION modulators. The GBM (173 features)
decides; the agents contextualize via a fractal quality score that
modulates position sizing. Multi-timeframe: Context(1D) + Regime(4H)
+ Setup(1H) + Entry(15M).

Key invariants :
  - GBM p_trade is NOT modified by agents
  - Trade direction is NOT changed by agents
  - Agents are called ONLY on GBM-approved candidates
  - Fractal quality uses ALL 4 TF
  - Size multiplier is deterministic from quality score
"""
from __future__ import annotations

import pytest


def _fr(agent, tf, passed, score=0.5):
    """Quick FractalReport builder."""
    from src.agents.contracts import FractalReport
    return FractalReport(
        asset='BTCUSDT', agent=agent, timeframe=tf,
        state='active' if passed else 'inactive',
        score=score, passed=passed,
        block_reasons=[] if passed else ['test'],
        timestamp='2023-06-15T10:00:00',
    )


# ===========================================================================
class TestComputeFractalQuality:

    def test_function_exists(self):
        from src.core.fractal_quality import compute_fractal_quality
        assert callable(compute_fractal_quality)

    def test_all_four_pass_high_quality(self):
        from src.core.fractal_quality import compute_fractal_quality
        reports = {
            'context': _fr('context', '1d', True, 0.8),
            'regime':  _fr('regime',  '4h', True, 0.7),
            'setup':   _fr('setup',   '1h', True, 0.6),
            'entry':   _fr('entry',   '15m', True, 0.7),
        }
        fq = compute_fractal_quality(reports)
        assert fq['fractal_quality'] == pytest.approx(1.0)
        assert fq['quality_bucket'] == 'high'
        assert fq['size_multiplier'] >= 1.0
        assert fq['skip_trade'] is False

    def test_zero_pass_skip_trade(self):
        from src.core.fractal_quality import compute_fractal_quality
        reports = {
            'context': _fr('context', '1d', False),
            'regime':  _fr('regime',  '4h', False),
            'setup':   _fr('setup',   '1h', False),
            'entry':   _fr('entry',   '15m', False),
        }
        fq = compute_fractal_quality(reports)
        assert fq['fractal_quality'] == pytest.approx(0.0)
        assert fq['skip_trade'] is True

    def test_two_pass_normal_size(self):
        from src.core.fractal_quality import compute_fractal_quality
        reports = {
            'context': _fr('context', '1d', True, 0.7),
            'regime':  _fr('regime',  '4h', True, 0.6),
            'setup':   _fr('setup',   '1h', False),
            'entry':   _fr('entry',   '15m', False),
        }
        fq = compute_fractal_quality(reports)
        assert fq['fractal_quality'] == pytest.approx(0.5)
        assert fq['size_multiplier'] == pytest.approx(1.0)
        assert fq['skip_trade'] is False

    def test_one_pass_cautious(self):
        from src.core.fractal_quality import compute_fractal_quality
        reports = {
            'context': _fr('context', '1d', True, 0.6),
            'regime':  _fr('regime',  '4h', False),
            'setup':   _fr('setup',   '1h', False),
            'entry':   _fr('entry',   '15m', False),
        }
        fq = compute_fractal_quality(reports)
        assert fq['fractal_quality'] == pytest.approx(0.25)
        assert fq['size_multiplier'] == pytest.approx(0.5)

    def test_three_pass_conviction(self):
        from src.core.fractal_quality import compute_fractal_quality
        reports = {
            'context': _fr('context', '1d', True, 0.8),
            'regime':  _fr('regime',  '4h', True, 0.7),
            'setup':   _fr('setup',   '1h', True, 0.6),
            'entry':   _fr('entry',   '15m', False),
        }
        fq = compute_fractal_quality(reports)
        assert fq['fractal_quality'] == pytest.approx(0.75)
        assert fq['size_multiplier'] >= 1.0
        assert fq['quality_bucket'] == 'high'


# ===========================================================================
class TestMTFAlignmentPreserved:
    """All 4 TF must be represented in the quality computation."""

    def test_requires_four_agents(self):
        from src.core.fractal_quality import compute_fractal_quality
        reports = {
            'context': _fr('context', '1d', True),
            'regime':  _fr('regime',  '4h', True),
        }
        fq = compute_fractal_quality(reports)
        # Missing setup + entry → quality ≤ 0.5 (only 2/4 pass)
        assert fq['fractal_quality'] <= 0.5

    def test_quality_reflects_all_four_tf(self):
        """Changing any single agent's passed status must change
        fractal_quality — proof that all 4 TF contribute."""
        from src.core.fractal_quality import compute_fractal_quality
        base = {
            'context': _fr('context', '1d', True),
            'regime':  _fr('regime',  '4h', True),
            'setup':   _fr('setup',   '1h', True),
            'entry':   _fr('entry',   '15m', True),
        }
        fq_all = compute_fractal_quality(base)
        for agent in ('context', 'regime', 'setup', 'entry'):
            modified = dict(base)
            modified[agent] = _fr(
                agent,
                base[agent].timeframe,
                False,
            )
            fq_mod = compute_fractal_quality(modified)
            assert fq_mod['fractal_quality'] < fq_all['fractal_quality'], (
                f'Flipping {agent} to passed=False must reduce quality'
            )


# ===========================================================================
class TestGBMUnchanged:
    """Modulation must NOT alter the GBM's decision."""

    def test_size_multiplier_does_not_affect_threshold(self):
        """The GBM threshold (0.60) is checked BEFORE agents are
        called. This test verifies the contract by checking that
        compute_fractal_quality does NOT return a threshold field."""
        from src.core.fractal_quality import compute_fractal_quality
        reports = {
            'context': _fr('context', '1d', False),
            'regime':  _fr('regime',  '4h', False),
            'setup':   _fr('setup',   '1h', False),
            'entry':   _fr('entry',   '15m', False),
        }
        fq = compute_fractal_quality(reports)
        assert 'threshold' not in fq, (
            'compute_fractal_quality must NOT return a threshold — '
            'agents modulate SIZE, not the GBM decision threshold'
        )
