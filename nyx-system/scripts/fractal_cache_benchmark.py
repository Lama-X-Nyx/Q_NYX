"""
Fractal Cache Benchmark

Measures performance improvement from fractal state caching.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, '.')

from src.data.mtf_loader import load_fractal_context
from src.core.fractal_cached_runner import FractalCachedRunner
from src.agents.orchestrator import Orchestrator


def run_fractal_cache_benchmark(pair: str, config: dict, output_dir: Path, steps: int = 100):
    """
    Benchmark cached vs uncached fractal pipeline
    
    Args:
        pair: Trading pair
        config: System configuration
        output_dir: Output directory
        steps: Number of steps to benchmark
    """
    
    print(f"\n{'='*80}")
    print(f"FRACTAL CACHE BENCHMARK - {pair}")
    print(f"{'='*80}")
    
    reference_date = datetime(2023, 12, 15)
    
    print(f"\nReference date: {reference_date.strftime('%Y-%m-%d')}")
    print(f"Steps to benchmark: {steps}")
    
    # Prepare data for all steps
    print(f"\n📊 Loading data...")
    mtf_data_steps = []
    
    for step in range(steps):
        step_date = reference_date + timedelta(minutes=15 * step)
        try:
            mtf_data = load_fractal_context(pair, step_date, config)
            mtf_data_steps.append(mtf_data)
        except Exception as e:
            print(f"Failed to load data for step {step}: {e}")
            break
    
    actual_steps = len(mtf_data_steps)
    print(f"✓ Loaded {actual_steps} steps")
    
    if actual_steps == 0:
        print("❌ No data loaded, cannot benchmark")
        return {'status': 'failed', 'reason': 'no_data'}
    
    # Benchmark 1: Uncached (standard orchestrator)
    print(f"\n🔬 Benchmark 1: Uncached Pipeline")
    print(f"  Running {actual_steps} steps with standard orchestrator...")
    
    orchestrator = Orchestrator(config)
    
    start_time = time.time()
    uncached_decisions = []
    
    for mtf_data in mtf_data_steps:
        decision = orchestrator.decide(mtf_data)
        uncached_decisions.append(decision)
    
    uncached_time = time.time() - start_time
    uncached_avg = uncached_time / actual_steps
    
    print(f"  Total time:    {uncached_time:.2f}s")
    print(f"  Avg per step:  {uncached_avg*1000:.1f}ms")
    
    # Benchmark 2: Cached
    print(f"\n🔬 Benchmark 2: Cached Pipeline")
    print(f"  Running {actual_steps} steps with cached runner...")
    
    cached_runner = FractalCachedRunner(config)
    
    start_time = time.time()
    cached_decisions = []
    
    for mtf_data in mtf_data_steps:
        decision = cached_runner.decide(mtf_data)
        cached_decisions.append(decision)
    
    cached_time = time.time() - start_time
    cached_avg = cached_time / actual_steps
    
    print(f"  Total time:    {cached_time:.2f}s")
    print(f"  Avg per step:  {cached_avg*1000:.1f}ms")
    
    # Get cache stats
    cache_stats = cached_runner.get_cache_stats()
    
    # Compute speedup
    speedup = uncached_time / cached_time if cached_time > 0 else 0
    time_saved = uncached_time - cached_time
    time_saved_pct = (time_saved / uncached_time * 100) if uncached_time > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Uncached time:     {uncached_time:.2f}s ({uncached_avg*1000:.1f}ms/step)")
    print(f"Cached time:       {cached_time:.2f}s ({cached_avg*1000:.1f}ms/step)")
    print(f"Time saved:        {time_saved:.2f}s ({time_saved_pct:.1f}%)")
    print(f"Speedup:           {speedup:.2f}x")
    
    print(f"\n📊 Cache Statistics:")
    print(f"  Context hits:    {cache_stats['context']['stats']['cache_hit_count']}")
    print(f"  Regime hits:     {cache_stats['regime']['stats']['cache_hit_count']}")
    print(f"  Setup hits:      {cache_stats['setup']['stats']['cache_hit_count']}")
    
    # Parity check
    print(f"\n🔍 Parity Check:")
    parity_issues = 0
    
    for i, (uncached, cached) in enumerate(zip(uncached_decisions, cached_decisions)):
        if uncached.action != cached.action:
            parity_issues += 1
            if parity_issues <= 3:  # Show first 3 issues
                print(f"  Step {i}: ACTION MISMATCH (uncached={uncached.action}, cached={cached.action})")
    
    if parity_issues == 0:
        print(f"  ✅ Perfect parity - all decisions match")
    else:
        print(f"  ❌ {parity_issues} parity issues found")
    
    # Save report
    output_dir_fractal = Path(output_dir) / 'fractal'
    output_dir_fractal.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir_fractal / f"{pair}_fractal_cache_benchmark.json"
    
    report = {
        'pair': pair,
        'steps': actual_steps,
        'uncached': {
            'total_time_seconds': uncached_time,
            'avg_time_ms': uncached_avg * 1000
        },
        'cached': {
            'total_time_seconds': cached_time,
            'avg_time_ms': cached_avg * 1000
        },
        'performance': {
            'time_saved_seconds': time_saved,
            'time_saved_percent': time_saved_pct,
            'speedup': speedup
        },
        'cache_stats': cache_stats,
        'parity': {
            'total_steps': actual_steps,
            'mismatches': parity_issues,
            'parity_rate': (actual_steps - parity_issues) / actual_steps * 100
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Benchmark report saved: {report_path}\n")
    
    return {
        'status': 'completed',
        'pair': pair,
        'steps': actual_steps,
        'speedup': speedup,
        'parity_issues': parity_issues,
        'output': str(report_path)
    }


if __name__ == "__main__":
    import yaml
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_fractal_cache_benchmark('BTCUSDT', config, Path('reports/validation'))
