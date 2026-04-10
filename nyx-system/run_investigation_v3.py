"""Run V3 investigation from existing deep dive JSON (no trading system needed)."""
import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from scripts.setup_investigation_v3 import (
    _build_bar_level_summary,
    _reconstruct_pattern_level,
    _build_rejection_summary,
    check_pattern_conservation,
    check_level_linking,
    determine_verdict,
)


def run_from_deep_dive_json(pair: str, output_dir: str):
    json_path = Path(output_dir) / 'fractal' / f'{pair}_setup_deep_dive.json'
    with open(json_path) as f:
        deep_dive = json.load(f)

    print(f"\n{'='*70}")
    print(f"SETUP INVESTIGATION V3 - {pair}")
    print(f"{'='*70}\n")

    periods_out = {}
    for date_str, period_data in deep_dive['periods'].items():
        label = date_str[:10]
        print(f"--- {label} ---")

        bar = _build_bar_level_summary(period_data)
        pat = _reconstruct_pattern_level(period_data)
        rej = _build_rejection_summary(pat)
        con = check_pattern_conservation(pat, rej)
        lnk = check_level_linking(bar, pat)
        root, verdict = determine_verdict(pat, con)

        print(f"  BAR LEVEL")
        print(f"    Ready bars:               {bar['bars_ready_for_decision']}")
        print(f"    Setup pass bars:          {bar['setup_pass_count_bars']}")
        print(f"    Setup pass rate:          {bar['setup_pass_rate']:.1f}%")
        print()
        print(f"  PATTERN LEVEL")
        print(f"    FVG candidates:           {pat['fvg_candidates_seen']}")
        print(f"    OB candidates:            {pat['ob_candidates_seen']}")
        print(f"    Total candidates:         {pat['total_candidates_seen']}")
        print(f"    Pre-pattern failures:     {pat['pre_pattern_failures']}")
        print(f"    Valid patterns formed:    {pat['valid_patterns_formed']}")
        print(f"    Alignment rejections:     {pat['alignment_rejections']}")
        print(f"    Final valid patterns:     {pat['final_valid_patterns']}")
        print()
        ok = '✅' if con['flow_conservation_ok'] else '❌'
        lk = '✅' if lnk['pattern_to_bar_link_consistent'] else '⚠️'
        print(f"  Flow conservation:          {ok}")
        if not con['flow_conservation_ok']:
            for n in con['notes']:
                print(f"    - {n}")
        print(f"  Pattern/bar linking:        {lk}")
        if not lnk['pattern_to_bar_link_consistent']:
            for n in lnk['notes']:
                print(f"    - {n}")
        print(f"  Root cause: {root}")
        print()

        periods_out[date_str] = {
            'bar_level_summary': bar,
            'pattern_level_summary': pat,
            'rejection_summary': rej,
            'consistency_checks': con,
            'level_linking': lnk,
            'primary_root_cause': root,
        }

    # Aggregate
    valid_periods = [
        p for p in periods_out.values()
        if p['consistency_checks']['flow_conservation_ok']
    ]
    if valid_periods:
        agg = {
            k: sum(p['pattern_level_summary'][k] for p in valid_periods)
            for k in [
                'fvg_candidates_seen', 'ob_candidates_seen',
                'total_candidates_seen', 'pre_pattern_failures',
                'valid_patterns_formed', 'alignment_rejections',
                'final_valid_patterns',
            ]
        }
        agg_rej = _build_rejection_summary(agg)
        agg_con = check_pattern_conservation(agg, agg_rej)
        main_root, main_verdict = determine_verdict(agg, agg_con)
    else:
        main_root = 'diagnostic_invalid'
        main_verdict = 'No periods with valid conservation'

    print(f"{'='*70}")
    print(f"VERDICT: {main_root}")
    print(f"{main_verdict}")
    print(f"{'='*70}\n")

    # Save
    out = Path(output_dir) / 'fractal' / f'{pair}_setup_investigation.json'
    report = {
        'pair': pair,
        'version': 'v3',
        'periods': periods_out,
        'cross_period_summary': {
            'main_root_cause': main_root,
            'verdict': main_verdict,
        },
    }
    with open(out, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✓ Saved: {out}")


if __name__ == '__main__':
    run_from_deep_dive_json('BTCUSDT', 'reports/validation')
