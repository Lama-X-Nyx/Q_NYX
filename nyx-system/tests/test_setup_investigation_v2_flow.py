"""
Test Setup Investigation V2 Flow Conservation

Tests that candidate flow respects strict conservation rules
"""

import sys
sys.path.insert(0, '.')

from scripts.setup_investigation_v2 import (
    _estimate_candidate_flow_corrected,
    _estimate_rejection_ladder_corrected,
    _check_flow_conservation,
    _determine_root_cause_v2_corrected
)


def test_flow_conservation_basic():
    """Test that flow conservation rules are enforced"""
    print("Testing flow conservation...")
    
    # Mock period data
    period_data = {
        'patterns_observed': {
            'bullish_fvg': 2,
            'bearish_fvg': 1,
            'bullish_ob': 0,
            'bearish_ob': 1
        },
        'bars_ready_for_decision': 50,
        'setup_passed': 1
    }
    
    flow = _estimate_candidate_flow_corrected(period_data)
    ladder = _estimate_rejection_ladder_corrected(flow, period_data)
    check = _check_flow_conservation(flow, ladder)
    
    # Rule 1: total_candidates = fvg + ob
    assert flow['total_candidates_seen'] == (
        flow['fvg_candidates_seen'] + flow['ob_candidates_seen']
    ), "Total candidates != fvg + ob"
    
    # Rule 2: valid_patterns = candidates - pre_failures
    assert flow['valid_patterns_formed'] == (
        flow['total_candidates_seen'] - flow['pre_pattern_failures']
    ), "Valid patterns != candidates - pre_failures"
    
    # Rule 3: setup_pass = valid_patterns - post_failures
    assert flow['setup_pass_count'] == (
        flow['valid_patterns_formed'] - flow['post_pattern_failures']
    ), "Setup pass != valid_patterns - post_failures"
    
    # Rule 4: No negative values
    for key, val in flow.items():
        assert val >= 0, f"Negative value: {key} = {val}"
    
    # Rule 5: failed_alignment <= valid_patterns
    assert ladder['failed_alignment'] <= flow['valid_patterns_formed'], \
        f"Alignment failures ({ladder['failed_alignment']}) > valid patterns ({flow['valid_patterns_formed']})"
    
    # Conservation check should pass
    assert check['flow_conservation_ok'], f"Flow conservation failed: {check['notes']}"
    
    print("✅ Flow conservation rules enforced")


def test_no_negative_pre_pattern_failures():
    """Test that pre_pattern_failures is never negative"""
    print("Testing no negative pre_pattern failures...")
    
    period_data = {
        'patterns_observed': {'bullish_fvg': 5, 'bearish_fvg': 0, 'bullish_ob': 0, 'bearish_ob': 0},
        'bars_ready_for_decision': 50,
        'setup_passed': 2
    }
    
    flow = _estimate_candidate_flow_corrected(period_data)
    
    assert flow['pre_pattern_failures'] >= 0, \
        f"Pre-pattern failures is negative: {flow['pre_pattern_failures']}"
    
    print("✅ No negative pre_pattern_failures")


def test_alignment_never_exceeds_patterns():
    """Test that alignment failures never exceed valid patterns"""
    print("Testing alignment failures <= valid patterns...")
    
    period_data = {
        'patterns_observed': {'bullish_fvg': 3, 'bearish_fvg': 2, 'bullish_ob': 0, 'bearish_ob': 0},
        'bars_ready_for_decision': 50,
        'setup_passed': 1
    }
    
    flow = _estimate_candidate_flow_corrected(period_data)
    ladder = _estimate_rejection_ladder_corrected(flow, period_data)
    
    assert ladder['failed_alignment'] <= flow['valid_patterns_formed'], \
        f"Alignment failures ({ladder['failed_alignment']}) > valid patterns ({flow['valid_patterns_formed']})"
    
    print("✅ Alignment failures never exceed valid patterns")


def test_rejection_ladder_sums_correctly():
    """Test that rejection ladder components sum to totals"""
    print("Testing rejection ladder sums...")
    
    period_data = {
        'patterns_observed': {'bullish_fvg': 2, 'bearish_fvg': 3, 'bullish_ob': 1, 'bearish_ob': 0},
        'bars_ready_for_decision': 50,
        'setup_passed': 2
    }
    
    flow = _estimate_candidate_flow_corrected(period_data)
    ladder = _estimate_rejection_ladder_corrected(flow, period_data)
    
    # Pre-pattern sum
    pre_sum = (
        ladder['failed_structure_rules'] +
        ladder['failed_gap_threshold'] +
        ladder['failed_range_threshold'] +
        ladder['failed_followthrough'] +
        ladder['failed_other_pre_pattern']
    )
    assert pre_sum == flow['pre_pattern_failures'], \
        f"Pre-pattern sum ({pre_sum}) != pre_pattern_failures ({flow['pre_pattern_failures']})"
    
    # Post-pattern sum
    post_sum = ladder['failed_alignment'] + ladder['failed_other_post_pattern']
    assert post_sum == flow['post_pattern_failures'], \
        f"Post-pattern sum ({post_sum}) != post_pattern_failures ({flow['post_pattern_failures']})"
    
    print("✅ Rejection ladder sums correctly")


def test_verdict_requires_valid_flow():
    """Test that verdict is invalid if flow conservation fails"""
    print("Testing verdict requires valid flow...")
    
    # Create artificially broken flow (should not happen with corrected code)
    broken_flow = {
        'total_candidates_seen': 10,
        'fvg_candidates_seen': 5,
        'ob_candidates_seen': 5,
        'pre_pattern_failures': -5,  # INVALID!
        'valid_patterns_formed': 15,  # More than candidates!
        'post_pattern_failures': 3,
        'setup_pass_count': 12
    }
    
    broken_ladder = {
        'failed_structure_rules': 0,
        'failed_gap_threshold': 0,
        'failed_range_threshold': 0,
        'failed_followthrough': 0,
        'failed_other_pre_pattern': -5,  # INVALID!
        'failed_alignment': 3,
        'failed_other_post_pattern': 0
    }
    
    check = _check_flow_conservation(broken_flow, broken_ladder)
    assert not check['flow_conservation_ok'], "Should detect broken flow"
    assert len(check['notes']) > 0, "Should have error notes"
    
    root_cause, verdict = _determine_root_cause_v2_corrected(broken_flow, broken_ladder, check)
    assert root_cause == "diagnostic_invalid", \
        f"Should return diagnostic_invalid for broken flow, got {root_cause}"
    assert "inconsistent" in verdict.lower(), "Verdict should mention inconsistency"
    
    print("✅ Verdict correctly invalidated for broken flow")


def test_zero_patterns_case():
    """Test handling of zero patterns formed"""
    print("Testing zero patterns case...")
    
    period_data = {
        'patterns_observed': {'bullish_fvg': 0, 'bearish_fvg': 0, 'bullish_ob': 0, 'bearish_ob': 0},
        'bars_ready_for_decision': 50,
        'setup_passed': 0
    }
    
    flow = _estimate_candidate_flow_corrected(period_data)
    ladder = _estimate_rejection_ladder_corrected(flow, period_data)
    check = _check_flow_conservation(flow, ladder)
    
    assert flow['valid_patterns_formed'] == 0
    assert flow['setup_pass_count'] == 0
    assert check['flow_conservation_ok'], f"Flow should be valid: {check['notes']}"
    
    root_cause, verdict = _determine_root_cause_v2_corrected(flow, ladder, check)
    assert root_cause in ["pattern_scarcity", "pattern_filtering_too_severe"], \
        f"Root cause should indicate scarcity or filtering, got {root_cause}"
    
    print("✅ Zero patterns case handled correctly")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING SETUP INVESTIGATION V2 FLOW CONSERVATION")
    print("="*60 + "\n")
    
    test_flow_conservation_basic()
    test_no_negative_pre_pattern_failures()
    test_alignment_never_exceeds_patterns()
    test_rejection_ladder_sums_correctly()
    test_verdict_requires_valid_flow()
    test_zero_patterns_case()
    
    print("\n✅ ALL FLOW CONSERVATION TESTS PASSED\n")
