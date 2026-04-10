"""
Test Setup Investigation V2

Tests candidate flow and rejection ladder analysis
"""

import sys
sys.path.insert(0, '.')

from scripts.setup_investigation_v2 import (
    _estimate_candidate_flow,
    _estimate_rejection_ladder,
    _determine_root_cause_v2
)


def test_candidate_flow_estimation():
    """Test that candidate flow is estimated reasonably"""
    print("Testing candidate flow estimation...")
    
    # Case: Some patterns observed
    period_data = {
        'patterns_observed': {
            'bullish_fvg': 2,
            'bearish_fvg': 3,
            'bullish_ob': 1,
            'bearish_ob': 0
        },
        'block_reasons': {},
        'bars_ready_for_decision': 50
    }
    
    flow = _estimate_candidate_flow(period_data)
    
    assert flow['patterns_kept'] == 6  # 2+3+1+0
    assert flow['fvg_candidates_seen'] > flow['patterns_kept']
    assert flow['ob_candidates_seen'] >= 0
    assert flow['patterns_rejected'] >= 0
    
    print("✅ Candidate flow estimation works")


def test_rejection_ladder():
    """Test rejection ladder breakdown"""
    print("Testing rejection ladder...")
    
    period_data = {
        'block_reasons': {
            'alignment_score_too_low': 10
        }
    }
    
    candidate_flow = {
        'patterns_rejected': 25
    }
    
    ladder = _estimate_rejection_ladder(period_data, candidate_flow)
    
    assert 'failed_structure_rules' in ladder
    assert 'failed_gap_threshold' in ladder
    assert 'failed_alignment' in ladder
    assert ladder['failed_alignment'] == 10
    
    print("✅ Rejection ladder works")


def test_root_cause_determination():
    """Test root cause logic"""
    print("Testing root cause determination...")
    
    # Case A: Pattern scarcity
    flow = {'fvg_candidates_seen': 2, 'ob_candidates_seen': 3, 'patterns_kept': 0, 'patterns_rejected': 5}
    ladder = {'failed_structure_rules': 3, 'failed_gap_threshold': 1, 'failed_range_threshold': 0, 
              'failed_followthrough': 0, 'failed_alignment': 1, 'failed_other': 0}
    
    root_cause, verdict = _determine_root_cause_v2(flow, ladder, 0)
    assert root_cause == "pattern_scarcity"
    print(f"  Case A: {root_cause} ✅")
    
    # Case B: Pattern filtering too severe
    flow = {'fvg_candidates_seen': 30, 'ob_candidates_seen': 20, 'patterns_kept': 5, 'patterns_rejected': 45}
    ladder = {'failed_structure_rules': 25, 'failed_gap_threshold': 10, 'failed_range_threshold': 5,
              'failed_followthrough': 2, 'failed_alignment': 3, 'failed_other': 0}
    
    root_cause, verdict = _determine_root_cause_v2(flow, ladder, 5)
    assert root_cause == "pattern_filtering_too_severe"
    print(f"  Case B: {root_cause} ✅")
    
    # Case C: Alignment filtering
    flow = {'fvg_candidates_seen': 20, 'ob_candidates_seen': 15, 'patterns_kept': 10, 'patterns_rejected': 25}
    ladder = {'failed_structure_rules': 5, 'failed_gap_threshold': 2, 'failed_range_threshold': 1,
              'failed_followthrough': 0, 'failed_alignment': 17, 'failed_other': 0}
    
    root_cause, verdict = _determine_root_cause_v2(flow, ladder, 10)
    assert root_cause == "alignment_filtering"
    print(f"  Case C: {root_cause} ✅")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING SETUP INVESTIGATION V2")
    print("="*60 + "\n")
    
    test_candidate_flow_estimation()
    test_rejection_ladder()
    test_root_cause_determination()
    
    print("\n✅ ALL SETUP INVESTIGATION V2 TESTS PASSED\n")
