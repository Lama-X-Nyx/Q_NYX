"""
Test Setup Investigation V3 — Pattern-centric conservation

Tests:
1. Separation bar_level_summary / pattern_level_summary
2. Conservation correcte au niveau pattern
3. setup_pass_count_bars non utilisé dans la conservation
4. alignment_rejections <= valid_patterns_formed
5. Verdict interdit si flow_conservation_ok = false
6. JSON valide
7. Level linking detection
8. All verdict cases
"""

import sys
sys.path.insert(0, '.')

import json
from scripts.setup_investigation_v3 import (
    _reconstruct_pattern_level,
    _build_bar_level_summary,
    _build_rejection_summary,
    check_pattern_conservation,
    check_level_linking,
    determine_verdict,
    VERDICTS,
)


# ─── Helpers ───────────────────────────────────────────────────────────────

def _make_period(
    bars_ready=43,
    setup_passed=1,
    setup_blocked=42,
    setup_pass_rate=2.3,
    no_pattern=0,
    misaligned_pattern=0,
    alignment_score_too_low=42,
    unknown=0,
    bullish_fvg=0,
    bearish_fvg=0,
    bullish_ob=0,
    bearish_ob=0,
):
    """Build a mock period_data dict matching setup_deep_dive output."""
    return {
        'bars_ready_for_decision': bars_ready,
        'setup_passed': setup_passed,
        'setup_blocked': setup_blocked,
        'setup_pass_rate': setup_pass_rate,
        'block_reasons': {
            'no_pattern': no_pattern,
            'misaligned_pattern': misaligned_pattern,
            'alignment_score_too_low': alignment_score_too_low,
            'direction_conflict': 0,
            'setup_not_ready': 0,
            'unknown': unknown,
        },
        'patterns_observed': {
            'bullish_fvg': bullish_fvg,
            'bearish_fvg': bearish_fvg,
            'bullish_ob': bullish_ob,
            'bearish_ob': bearish_ob,
            'no_pattern': 0,
        },
        'alignment_summary': {
            'mean': 0.31,
            'median': 0.3,
            'min': 0.3,
            'max': 0.75,
            'threshold': 0.6,
        },
    }


def _full_pipeline(period_data):
    """Run the full V3 pipeline on a single period, return all outputs."""
    bar = _build_bar_level_summary(period_data)
    pat = _reconstruct_pattern_level(period_data)
    rej = _build_rejection_summary(pat)
    con = check_pattern_conservation(pat, rej)
    lnk = check_level_linking(bar, pat)
    root, verdict = determine_verdict(pat, con)
    return bar, pat, rej, con, lnk, root, verdict


# ─── Test 1: Separation ───────────────────────────────────────────────────

def test_separation_bar_vs_pattern():
    """bar_level_summary and pattern_level_summary are distinct dicts."""
    print("Test 1: Separation bar vs pattern...")
    period = _make_period()
    bar, pat, *_ = _full_pipeline(period)

    # Bar level must have its own keys
    assert 'bars_ready_for_decision' in bar
    assert 'setup_pass_count_bars' in bar
    assert 'block_reasons' in bar

    # Pattern level must NOT contain any bar-level key
    assert 'bars_ready_for_decision' not in pat
    assert 'setup_pass_count_bars' not in pat
    assert 'setup_pass_count' not in pat
    assert 'block_reasons' not in pat

    # Pattern level must have its own keys
    for k in [
        'fvg_candidates_seen', 'ob_candidates_seen',
        'total_candidates_seen', 'pre_pattern_failures',
        'valid_patterns_formed', 'alignment_rejections',
        'final_valid_patterns',
    ]:
        assert k in pat, f"Missing key in pattern_level: {k}"

    print("  ✅ Separation ok")


# ─── Test 2: Pattern-level conservation ────────────────────────────────────

def test_pattern_conservation_basic():
    """Conservation rules hold on the real-world period."""
    print("Test 2: Pattern conservation (real-world case)...")
    period = _make_period()  # mirrors BTCUSDT 2023-12-15
    _, pat, rej, con, *_ = _full_pipeline(period)

    assert con['flow_conservation_ok'], f"Conservation failed: {con['notes']}"

    # Verify each equation explicitly
    assert pat['total_candidates_seen'] == (
        pat['fvg_candidates_seen'] + pat['ob_candidates_seen']
    )
    assert pat['valid_patterns_formed'] == (
        pat['total_candidates_seen'] - pat['pre_pattern_failures']
    )
    assert pat['final_valid_patterns'] == (
        pat['valid_patterns_formed'] - pat['alignment_rejections']
    )

    print("  ✅ Conservation holds")


# ─── Test 3: setup_pass_count_bars NOT in conservation ─────────────────────

def test_setup_pass_count_bars_not_in_conservation():
    """setup_pass_count_bars must not appear in pattern_level or conservation."""
    print("Test 3: setup_pass_count_bars excluded from conservation...")

    # Deliberately create a case where bar-level and pattern-level differ
    period = _make_period(setup_passed=2, alignment_score_too_low=41)
    bar, pat, rej, con, *_ = _full_pipeline(period)

    # Conservation must still hold regardless of bar value
    assert con['flow_conservation_ok'], f"Conservation failed: {con['notes']}"

    # The bar-level value must not contaminate pattern conservation
    # (i.e., setup_pass_count_bars ≠ final_valid_patterns is allowed)
    assert bar['setup_pass_count_bars'] == 2
    # Pattern conservation is self-contained — it doesn't reference bar counts.

    print("  ✅ setup_pass_count_bars isolated from conservation")


# ─── Test 4: alignment_rejections <= valid_patterns_formed ─────────────────

def test_alignment_bounded_by_valid():
    """alignment_rejections can never exceed valid_patterns_formed."""
    print("Test 4: alignment_rejections <= valid_patterns_formed...")

    for passed in [0, 1, 5]:
        for align_low in [0, 10, 50]:
            period = _make_period(
                bars_ready=60,
                setup_passed=passed,
                setup_blocked=60 - passed,
                alignment_score_too_low=align_low,
                no_pattern=max(0, 60 - passed - align_low),
            )
            _, pat, *_ = _full_pipeline(period)
            assert pat['alignment_rejections'] <= pat['valid_patterns_formed'], (
                f"alignment({pat['alignment_rejections']}) > "
                f"valid({pat['valid_patterns_formed']}) "
                f"for passed={passed}, align_low={align_low}"
            )

    print("  ✅ Bound holds for all test combos")


# ─── Test 5: Verdict forbidden when conservation fails ─────────────────────

def test_verdict_forbidden_if_conservation_fails():
    """If flow_conservation_ok is False, verdict must be diagnostic_invalid."""
    print("Test 5: Verdict requires valid conservation...")

    broken_pattern = {
        'fvg_candidates_seen': 5,
        'ob_candidates_seen': 5,
        'total_candidates_seen': 10,
        'pre_pattern_failures': -3,   # IMPOSSIBLE
        'valid_patterns_formed': 13,  # > total candidates
        'alignment_rejections': 2,
        'final_valid_patterns': 11,
    }
    broken_rejection = _build_rejection_summary(broken_pattern)
    con = check_pattern_conservation(broken_pattern, broken_rejection)

    assert not con['flow_conservation_ok']

    root, _ = determine_verdict(broken_pattern, con)
    assert root == 'diagnostic_invalid', f"Expected diagnostic_invalid, got {root}"

    print("  ✅ Verdict correctly blocked")


# ─── Test 6: JSON structure valid ──────────────────────────────────────────

def test_json_structure():
    """Output structure matches the V3 spec."""
    print("Test 6: JSON structure...")
    period = _make_period()
    bar, pat, rej, con, lnk, root, verdict = _full_pipeline(period)

    record = {
        'bar_level_summary': bar,
        'pattern_level_summary': pat,
        'rejection_summary': rej,
        'consistency_checks': con,
        'level_linking': lnk,
        'primary_root_cause': root,
    }

    # Must be JSON-serializable
    s = json.dumps(record, indent=2)
    parsed = json.loads(s)

    # Top-level keys
    for k in [
        'bar_level_summary', 'pattern_level_summary', 'rejection_summary',
        'consistency_checks', 'level_linking', 'primary_root_cause',
    ]:
        assert k in parsed, f"Missing top key: {k}"

    # consistency_checks shape
    assert 'flow_conservation_ok' in parsed['consistency_checks']
    assert 'notes' in parsed['consistency_checks']

    # level_linking shape
    assert 'pattern_to_bar_link_consistent' in parsed['level_linking']
    assert 'notes' in parsed['level_linking']

    print("  ✅ JSON valid")


# ─── Test 7: Level linking detects divergence ──────────────────────────────

def test_level_linking_divergence():
    """level_linking flags when bar and pattern levels disagree."""
    print("Test 7: Level linking divergence...")

    # Case: bar says 2 passed, pattern says 1 final
    period = _make_period(
        bars_ready=50,
        setup_passed=2,
        setup_blocked=48,
        alignment_score_too_low=48,
    )
    bar, pat, _, _, lnk, *_ = _full_pipeline(period)

    if bar['setup_pass_count_bars'] != pat['final_valid_patterns']:
        assert not lnk['pattern_to_bar_link_consistent']
        assert len(lnk['notes']) > 0
        print("  ✅ Divergence detected and flagged")
    else:
        # If they happen to match, linking should be consistent
        assert lnk['pattern_to_bar_link_consistent']
        print("  ✅ Linking consistent (values matched)")


# ─── Test 8: All verdict cases reachable ───────────────────────────────────

def test_verdict_case_a():
    """Case A: very few candidates."""
    print("Test 8a: Verdict case A (scarcity)...")
    period = _make_period(
        bars_ready=5, setup_passed=0, setup_blocked=5,
        no_pattern=5, alignment_score_too_low=0,
    )
    *_, root, _ = _full_pipeline(period)
    assert root == 'candidate_scarcity', f"Got {root}"
    print("  ✅")


def test_verdict_case_b():
    """Case B: candidates but most die pre-pattern."""
    print("Test 8b: Verdict case B (pattern filtering)...")
    period = _make_period(
        bars_ready=50, setup_passed=0, setup_blocked=50,
        no_pattern=50, alignment_score_too_low=0,
    )
    *_, root, _ = _full_pipeline(period)
    # With zero valid patterns, should be B
    assert root in ('pattern_filtering_too_severe', 'candidate_scarcity'), f"Got {root}"
    print("  ✅")


def test_verdict_case_c():
    """Case C: alignment kills most patterns."""
    print("Test 8c: Verdict case C (alignment)...")
    period = _make_period(
        bars_ready=50, setup_passed=1, setup_blocked=49,
        alignment_score_too_low=49, no_pattern=0,
    )
    *_, root, _ = _full_pipeline(period)
    assert root == 'alignment_filtering', f"Got {root}"
    print("  ✅")


def test_verdict_case_e():
    """Case E: invalid conservation → diagnostic_invalid."""
    print("Test 8e: Verdict case E (invalid)...")
    broken = {
        'total_candidates_seen': 5,
        'fvg_candidates_seen': 5,
        'ob_candidates_seen': 5,  # sum ≠ total → broken
        'pre_pattern_failures': 0,
        'valid_patterns_formed': 5,
        'alignment_rejections': 0,
        'final_valid_patterns': 5,
    }
    rej = _build_rejection_summary(broken)
    con = check_pattern_conservation(broken, rej)
    root, _ = determine_verdict(broken, con)
    assert root == 'diagnostic_invalid', f"Got {root}"
    print("  ✅")


# ─── Test 9: Zero patterns zero passed ─────────────────────────────────────

def test_zero_everything():
    """Edge case: bars ready but nothing passed, no patterns."""
    print("Test 9: Zero patterns, zero passed...")
    period = _make_period(
        bars_ready=40, setup_passed=0, setup_blocked=40,
        no_pattern=40, alignment_score_too_low=0,
    )
    _, pat, rej, con, *_ = _full_pipeline(period)

    assert pat['valid_patterns_formed'] == 0
    assert pat['final_valid_patterns'] == 0
    assert pat['alignment_rejections'] == 0
    assert con['flow_conservation_ok'], f"Conservation failed: {con['notes']}"
    print("  ✅")


# ─── Test 10: Non-negative invariants ──────────────────────────────────────

def test_no_negatives_anywhere():
    """No field in pattern_level can be negative, across various inputs."""
    print("Test 10: Non-negative invariants...")
    cases = [
        _make_period(bars_ready=0, setup_passed=0, setup_blocked=0,
                     no_pattern=0, alignment_score_too_low=0),
        _make_period(bars_ready=100, setup_passed=100, setup_blocked=0,
                     no_pattern=0, alignment_score_too_low=0),
        _make_period(bars_ready=1, setup_passed=0, setup_blocked=1,
                     no_pattern=0, alignment_score_too_low=1),
    ]
    for i, period in enumerate(cases):
        _, pat, *_ = _full_pipeline(period)
        for k, v in pat.items():
            assert v >= 0, f"Case {i}: {k}={v} is negative"
    print("  ✅")


# ─── Runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING SETUP INVESTIGATION V3")
    print("=" * 60 + "\n")

    test_separation_bar_vs_pattern()
    test_pattern_conservation_basic()
    test_setup_pass_count_bars_not_in_conservation()
    test_alignment_bounded_by_valid()
    test_verdict_forbidden_if_conservation_fails()
    test_json_structure()
    test_level_linking_divergence()
    test_verdict_case_a()
    test_verdict_case_b()
    test_verdict_case_c()
    test_verdict_case_e()
    test_zero_everything()
    test_no_negatives_anywhere()

    print("\n" + "=" * 60)
    print("✅ ALL SETUP INVESTIGATION V3 TESTS PASSED")
    print("=" * 60 + "\n")
