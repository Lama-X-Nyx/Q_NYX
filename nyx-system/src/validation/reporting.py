"""
Aggregated Temporal Report Generator

Combines all validation results into a single summary report.

Aggregates:
- Benchmarks
- Walk-forward windows
- Out-of-sample results

Usage:
    from src.validation.reporting import generate_summary_report
    
    generate_summary_report(pair='BTCUSDT', output_dir='reports/validation')
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def generate_summary_report(pair: str, validation_dir: str = 'reports/validation') -> Dict:
    """
    Generate aggregated summary report
    
    Args:
        pair: Trading pair
        validation_dir: Directory containing validation results
    
    Returns:
        Summary dict
    """
    
    print(f"\n{'='*80}")
    print(f"GENERATING VALIDATION SUMMARY - {pair}")
    print(f"{'='*80}")
    
    val_path = Path(validation_dir)
    
    summary = {
        'pair': pair,
        'benchmarks': None,
        'walk_forward': None,
        'out_of_sample': None,
        'final_verdict': None
    }
    
    # Load benchmarks
    benchmark_file = val_path / 'benchmarks' / f"{pair}_benchmarks.json"
    if benchmark_file.exists():
        with open(benchmark_file, 'r') as f:
            summary['benchmarks'] = json.load(f)
        print("✓ Loaded benchmarks")
    
    # Load walk-forward
    wf_file = val_path / 'walkforward' / f"{pair}_walkforward.json"
    if wf_file.exists():
        with open(wf_file, 'r') as f:
            summary['walk_forward'] = json.load(f)
        print("✓ Loaded walk-forward")
    
    # Load OOS
    oos_file = val_path / 'oos' / f"{pair}_oos.json"
    if oos_file.exists():
        with open(oos_file, 'r') as f:
            summary['out_of_sample'] = json.load(f)
        print("✓ Loaded out-of-sample")
    
    # Generate final verdict
    summary['final_verdict'] = _generate_final_verdict(summary)
    
    # Save summary
    summary_file = val_path / 'summary' / f"{pair}_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Summary saved: {summary_file}")
    
    # Generate markdown report
    md_report = _generate_markdown_report(summary)
    md_file = val_path / 'summary' / f"{pair}_summary.md"
    
    with open(md_file, 'w') as f:
        f.write(md_report)
    
    print(f"✓ Markdown report saved: {md_file}")
    
    return summary


def _generate_final_verdict(summary: Dict) -> Dict:
    """
    Generate final validation verdict
    
    Answers the 6 key questions from Phase 1 objectives
    
    Args:
        summary: Summary dict
    
    Returns:
        Verdict dict
    """
    
    verdict = {
        'questions': {},
        'overall_status': None,
        'recommendation': None
    }
    
    # Question 1: Does NYX beat simple benchmarks?
    if summary['benchmarks']:
        benchmark_results = summary['benchmarks'].get('results', {})
        
        # Extract benchmark metrics
        buy_hold_return = None
        sma_return = None
        random_return = None
        
        if 'buy_hold' in benchmark_results:
            buy_hold_return = benchmark_results['buy_hold'].get('metrics', {}).get('total_return')
        
        if 'sma_crossover' in benchmark_results:
            sma_return = benchmark_results['sma_crossover'].get('metrics', {}).get('total_return')
        
        if 'random_entry' in benchmark_results:
            random_return = benchmark_results['random_entry'].get('metrics', {}).get('total_return')
        
        # Get NYX return (would come from actual NYX backtest)
        # For now, we use walk-forward or OOS as proxy
        nyx_return = None
        
        if summary['out_of_sample']:
            oos_is_metrics = summary['out_of_sample'].get('in_sample', {}).get('metrics', {})
            nyx_return = oos_is_metrics.get('total_return')
        
        # Compare
        if nyx_return is not None:
            beats_count = 0
            total_benchmarks = 0
            
            for bench_name, bench_return in [
                ('buy_hold', buy_hold_return),
                ('sma_crossover', sma_return),
                ('random_entry', random_return)
            ]:
                if bench_return is not None:
                    total_benchmarks += 1
                    if nyx_return > bench_return:
                        beats_count += 1
            
            if total_benchmarks > 0:
                beat_pct = (beats_count / total_benchmarks) * 100
                
                if beat_pct >= 66:  # Beats 2/3 or more
                    verdict['questions']['beats_benchmarks'] = {
                        'answer': "YES",
                        'detail': f"NYX beats {beats_count}/{total_benchmarks} benchmarks ({beat_pct:.0f}%)",
                        'nyx_return': nyx_return,
                        'benchmarks': {
                            'buy_hold': buy_hold_return,
                            'sma_crossover': sma_return,
                            'random_entry': random_return
                        }
                    }
                else:
                    verdict['questions']['beats_benchmarks'] = {
                        'answer': "NO",
                        'detail': f"NYX only beats {beats_count}/{total_benchmarks} benchmarks ({beat_pct:.0f}%)",
                        'nyx_return': nyx_return,
                        'benchmarks': {
                            'buy_hold': buy_hold_return,
                            'sma_crossover': sma_return,
                            'random_entry': random_return
                        }
                    }
            else:
                verdict['questions']['beats_benchmarks'] = {
                    'answer': "INSUFFICIENT_DATA",
                    'detail': "Benchmarks executed but no valid returns to compare"
                }
        else:
            verdict['questions']['beats_benchmarks'] = {
                'answer': "PENDING",
                'detail': "Benchmarks available, NYX results needed for comparison"
            }
    else:
        verdict['questions']['beats_benchmarks'] = {
            'answer': "UNKNOWN",
            'detail': "Benchmark data not available"
        }
    
    # Question 2: Does NYX hold in out-of-sample?
    if summary['out_of_sample']:
        oos_verdict = summary['out_of_sample'].get('verdict', {})
        status = oos_verdict.get('status', 'UNKNOWN')
        
        verdict['questions']['holds_oos'] = {
            'answer': "YES" if status in ['STABLE', 'WARNING'] else "NO",
            'detail': oos_verdict.get('reason', 'No reason provided'),
            'status': status
        }
    else:
        verdict['questions']['holds_oos'] = {
            'answer': "UNKNOWN",
            'detail': "OOS data not available"
        }
    
    # Question 3: Does NYX survive Monte Carlo?
    verdict['questions']['survives_monte_carlo'] = {
        'answer': "PENDING",
        'detail': "Monte Carlo analysis pending (Sprint 3)"
    }
    
    # Question 4: Which parameters are sensitive?
    verdict['questions']['parameter_sensitivity'] = {
        'answer': "PENDING",
        'detail': "Sensitivity analysis pending (Sprint 3)"
    }
    
    # Question 5: When does NYX have edge (regimes)?
    verdict['questions']['regime_edge'] = {
        'answer': "PENDING",
        'detail': "Regime analysis pending (Sprint 3)"
    }
    
    # Question 6: When should NYX be stopped?
    verdict['questions']['stop_criteria'] = {
        'answer': "PENDING",
        'detail': "Risk engine v1 pending (Sprint 4)"
    }
    
    # Overall status
    oos_status = verdict['questions']['holds_oos'].get('answer', 'UNKNOWN')
    
    if oos_status == 'YES':
        verdict['overall_status'] = "PROMISING"
        verdict['recommendation'] = "Continue to Sprint 3 (Monte Carlo & Sensitivity)"
    elif oos_status == 'NO':
        verdict['overall_status'] = "CONCERNING"
        verdict['recommendation'] = "Review strategy before Sprint 3"
    else:
        verdict['overall_status'] = "INCOMPLETE"
        verdict['recommendation'] = "Complete validation pipeline"
    
    return verdict


def _generate_markdown_report(summary: Dict) -> str:
    """
    Generate markdown report
    
    Args:
        summary: Summary dict
    
    Returns:
        Markdown string
    """
    
    pair = summary['pair']
    verdict = summary['final_verdict']
    
    md = f"""# NYX Phase 1 Validation Summary - {pair}

## Overall Status: {verdict.get('overall_status', 'UNKNOWN')}

**Recommendation:** {verdict.get('recommendation', 'N/A')}

---

## Validation Questions

### 1. Does NYX beat simple benchmarks?
**Answer:** {verdict['questions']['beats_benchmarks']['answer']}  
**Detail:** {verdict['questions']['beats_benchmarks']['detail']}

### 2. Does NYX hold in out-of-sample?
**Answer:** {verdict['questions']['holds_oos']['answer']}  
**Detail:** {verdict['questions']['holds_oos']['detail']}

### 3. Does NYX survive Monte Carlo?
**Answer:** {verdict['questions']['survives_monte_carlo']['answer']}  
**Detail:** {verdict['questions']['survives_monte_carlo']['detail']}

### 4. Which parameters are sensitive?
**Answer:** {verdict['questions']['parameter_sensitivity']['answer']}  
**Detail:** {verdict['questions']['parameter_sensitivity']['detail']}

### 5. When does NYX have edge (regimes)?
**Answer:** {verdict['questions']['regime_edge']['answer']}  
**Detail:** {verdict['questions']['regime_edge']['detail']}

### 6. When should NYX be stopped?
**Answer:** {verdict['questions']['stop_criteria']['answer']}  
**Detail:** {verdict['questions']['stop_criteria']['detail']}

---

## Benchmarks

"""
    
    if summary['benchmarks']:
        md += "✓ Benchmarks executed\n\n"
        # Could add benchmark summary here
    else:
        md += "❌ No benchmark data available\n\n"
    
    md += "## Walk-Forward Analysis\n\n"
    
    if summary['walk_forward']:
        md += "✓ Walk-forward executed\n\n"
        # Could add WF summary here
    else:
        md += "❌ No walk-forward data available\n\n"
    
    md += "## Out-of-Sample\n\n"
    
    if summary['out_of_sample']:
        oos = summary['out_of_sample']
        verdict = oos.get('verdict', {})
        
        md += f"**Status:** {verdict.get('status', 'UNKNOWN')}\n\n"
        md += f"**Reason:** {verdict.get('reason', 'N/A')}\n\n"
    else:
        md += "❌ No OOS data available\n\n"
    
    md += "---\n\n"
    md += "*Report generated by NYX Phase 1 validation pipeline*\n"
    
    return md


if __name__ == "__main__":
    print("Aggregated Report - Example")
    
    # Generate summary
    summary = generate_summary_report(pair='BTCUSDT')
    
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")
    print(f"Status: {summary['final_verdict']['overall_status']}")
    print(f"Recommendation: {summary['final_verdict']['recommendation']}")
