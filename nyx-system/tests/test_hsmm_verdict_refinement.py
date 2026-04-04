"""
Test HSMM Verdict Refinement

Tests that the refined verdict logic correctly identifies:
- Successful feature alignment fix
- Partial improvements
- Range dominance
- Mixed results
"""

import sys
sys.path.insert(0, '.')

from src.validation.hsmm_deep_dive import _determine_verdict


def test_verdict_detects_successful_fix():
    """Test that verdict correctly identifies successful fix"""
    trend_plus_probs = [0.99, 0.81, 0.43, 0.63]
    range_probs = [0.00, 0.19, 0.57, 0.33]
    returns_means = [0.002, 0.003, 0.0001, 0.001]
    returns_stds = [0.009, 0.012, 0.002, 0.005]
    selected_states = ['trend_plus', 'trend_plus', 'range', 'trend_plus']
    
    issue, verdict = _determine_verdict(
        trend_plus_probs, range_probs, returns_means, returns_stds, selected_states
    )
    
    assert issue == "feature_alignment_fixed"
    assert "successfully unlocked" in verdict.lower()
    print("✅ Verdict detects successful fix")


def test_verdict_detects_partial_improvement():
    """Test partial improvement detection"""
    trend_plus_probs = [0.50, 0.40, 0.45, 0.45]
    range_probs = [0.50, 0.60, 0.55, 0.55]
    returns_means = [0.001, 0.002, 0.001, 0.001]
    returns_stds = [0.008, 0.009, 0.007, 0.008]
    selected_states = ['range', 'range', 'trend_plus', 'range']
    
    issue, verdict = _determine_verdict(
        trend_plus_probs, range_probs, returns_means, returns_stds, selected_states
    )
    
    assert issue == "partial_improvement"
    print("✅ Verdict detects partial improvement")


def test_verdict_no_longer_unclear():
    """Strong improvement should not return unclear"""
    trend_plus_probs = [0.75, 0.80, 0.70, 0.65]
    range_probs = [0.25, 0.20, 0.30, 0.35]
    returns_means = [0.002, 0.003, 0.002, 0.002]
    returns_stds = [0.010, 0.012, 0.009, 0.010]
    selected_states = ['trend_plus', 'trend_plus', 'trend_plus', 'trend_plus']
    
    issue, verdict = _determine_verdict(
        trend_plus_probs, range_probs, returns_means, returns_stds, selected_states
    )
    
    assert issue == "feature_alignment_fixed"
    print("✅ Strong improvement never unclear")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING HSMM VERDICT REFINEMENT")
    print("="*60 + "\n")
    
    test_verdict_detects_successful_fix()
    test_verdict_detects_partial_improvement()
    test_verdict_no_longer_unclear()
    
    print("\n✅ ALL HSMM VERDICT TESTS PASSED\n")
