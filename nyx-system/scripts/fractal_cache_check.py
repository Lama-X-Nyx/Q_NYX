"""
Fractal Cache Check

Validates that the fractal state cache correctly reuses
higher-timeframe states and only recomputes when necessary.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, '.')

from src.data.mtf_loader import load_fractal_context
from src.core.fractal_cached_runner import FractalCachedRunner


def run_fractal_cache_check(pair: str, config: dict, output_dir: Path, sample_date: str = None, steps: int = 20):
    """
    Check fractal cache behavior over multiple sequential steps
    
    Args:
        pair: Trading pair
        config: System configuration
        output_dir: Output directory
        sample_date: Reference date
        steps: Number of 15m steps to simulate
    """
    
    print(f"\n{'='*80}")
    print(f"FRACTAL CACHE CHECK - {pair}")
    print(f"{'='*80}")
    
    # Parse sample date
    if sample_date:
        current_date = datetime.strptime(sample_date, '%Y-%m-%d')
    else:
        current_date = datetime(2023, 12, 15)
    
    print(f"\nReference date: {current_date.strftime('%Y-%m-%d')}")
    print(f"Steps to simulate: {steps}")
    
    # Initialize cached runner
    print(f"\n🚀 Initializing Fractal Cached Runner...")
    runner = FractalCachedRunner(config)
    
    # Track decisions
    decisions = []
    step_details = []
    
    print(f"\n🔬 Running {steps} sequential steps...")
    
    for step in range(steps):
        # Load fractal context for this step
        # In a real scenario, each step would be 15m apart
        # For testing, we'll load the same data but the cache
        # should recognize unchanged higher timeframes
        
        step_date = current_date + timedelta(minutes=15 * step)
        
        try:
            # Load MTF data
            mtf_data = load_fractal_context(pair, step_date, config)
            
            # Get decision
            decision = runner.decide(mtf_data)
            decisions.append(decision)
            
            # Get cache stats at this step
            stats = runner.get_cache_stats()
            
            step_details.append({
                'step': step,
                'timestamp': step_date.isoformat(),
                'action': decision.action,
                'score': round(decision.score, 2),
                'cache_stats': stats
            })
            
            if step % 5 == 0:
                print(f"  Step {step:3d}: {decision.action:5s} (score: {decision.score:.2f})")
        
        except Exception as e:
            print(f"  Step {step:3d}: Failed - {e}")
            break
    
    # Get final cache stats
    final_stats = runner.get_cache_stats()
    
    print(f"\n{'='*80}")
    print(f"CACHE STATISTICS")
    print(f"{'='*80}")
    
    # Context stats
    context_stats = final_stats['context']['stats']
    print(f"\n📊 Context ({runner.context_tf}):")
    print(f"  Recomputes:    {context_stats['recompute_count']}")
    print(f"  Cache hits:    {context_stats['cache_hit_count']}")
    print(f"  Cache misses:  {context_stats['cache_miss_count']}")
    print(f"  Hit rate:      {context_stats['hit_rate']:.1f}%")
    
    # Regime stats
    regime_stats = final_stats['regime']['stats']
    print(f"\n📊 Regime ({runner.regime_tf}):")
    print(f"  Recomputes:    {regime_stats['recompute_count']}")
    print(f"  Cache hits:    {regime_stats['cache_hit_count']}")
    print(f"  Cache misses:  {regime_stats['cache_miss_count']}")
    print(f"  Hit rate:      {regime_stats['hit_rate']:.1f}%")
    
    # Setup stats
    setup_stats = final_stats['setup']['stats']
    print(f"\n📊 Setup ({runner.setup_tf}):")
    print(f"  Recomputes:    {setup_stats['recompute_count']}")
    print(f"  Cache hits:    {setup_stats['cache_hit_count']}")
    print(f"  Cache misses:  {setup_stats['cache_miss_count']}")
    print(f"  Hit rate:      {setup_stats['hit_rate']:.1f}%")
    
    # Conclusion
    print(f"\n{'='*80}")
    print(f"CONCLUSION")
    print(f"{'='*80}")
    
    total_steps = len(step_details)
    
    # Expected behavior:
    # - Context should recompute rarely (only when 1d changes)
    # - Regime should recompute occasionally (when 1h changes)
    # - Setup should recompute every step (15m always new)
    
    context_recomputes = context_stats['recompute_count']
    regime_recomputes = regime_stats['recompute_count']
    setup_recomputes = setup_stats['recompute_count']
    
    # Context: should recompute rarely (1D changes ~1x per day)
    # Allow up to 2 recomputes or 20% of steps, whichever is higher
    context_threshold = max(2, total_steps * 0.20)
    if context_recomputes <= context_threshold:
        context_verdict = "✅ Context cache working (rare recomputes)"
    else:
        context_verdict = "❌ Context cache not working (too many recomputes)"
    
    # Regime: should recompute occasionally (1H changes multiple times per day)
    # Allow up to 50% of steps (reasonable for 1H on 15m steps)
    regime_threshold = total_steps * 0.50
    if regime_recomputes <= regime_threshold:
        regime_verdict = "✅ Regime cache working (occasional recomputes)"
    else:
        regime_verdict = "❌ Regime cache not working (too many recomputes)"
    
    if setup_recomputes == total_steps:
        setup_verdict = "✅ Setup always fresh (recomputes every step)"
    else:
        setup_verdict = "⚠️ Setup not recomputing every step"
    
    print(context_verdict)
    print(regime_verdict)
    print(setup_verdict)
    
    # Overall verdict
    print(f"\n{'-'*80}")
    if (context_recomputes <= context_threshold and 
        regime_recomputes <= regime_threshold and
        setup_recomputes == total_steps):
        print("✅ Higher-timeframe states are being reused correctly.")
    else:
        print("❌ Cache behavior needs investigation.")
    print(f"{'-'*80}\n")
    
    # Save report
    output_dir_fractal = Path(output_dir) / 'fractal'
    output_dir_fractal.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir_fractal / f"{pair}_fractal_cache_check.json"
    
    report = {
        'pair': pair,
        'reference_date': current_date.isoformat(),
        'steps_processed': total_steps,
        'cache_stats': final_stats,
        'step_details': step_details[:10],  # First 10 for brevity
        'verdicts': {
            'context': context_verdict,
            'regime': regime_verdict,
            'setup': setup_verdict
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Cache check report saved: {report_path}\n")
    
    return {
        'status': 'completed',
        'pair': pair,
        'steps_processed': total_steps,
        'context_recomputes': context_recomputes,
        'regime_recomputes': regime_recomputes,
        'setup_recomputes': setup_recomputes,
        'output': str(report_path)
    }


if __name__ == "__main__":
    import yaml
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_fractal_cache_check('BTCUSDT', config, Path('reports/validation'))
