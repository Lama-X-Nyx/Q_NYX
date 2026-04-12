#!/usr/bin/env python3
"""
NYX Validation CLI

Single entry point for all Phase 1 validation tasks.

Usage:
    python scripts/run_validation.py --mode benchmark --pair BTCUSDT
    python scripts/run_validation.py --mode walkforward --pair ETHUSDT
    python scripts/run_validation.py --mode all --output reports/validation
"""

import argparse
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_baseline_config():
    """Load frozen baseline configuration"""
    config_path = Path('config/validation_baseline.yaml')
    
    if not config_path.exists():
        print("❌ Baseline config not found: config/validation_baseline.yaml")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_benchmark(pair: str, config: dict, output_dir: Path):
    """Run benchmark comparisons"""
    print(f"\n{'='*80}")
    print(f"BENCHMARK MODE - {pair}")
    print(f"{'='*80}")
    
    # Load data
    data_file = f"data/raw/{pair}_1h.csv"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return {
            'status': 'error',
            'pair': pair,
            'message': f'Data file not found: {data_file}'
        }
    
    # Import benchmarks module
    from src.validation.benchmarks import run_all_benchmarks
    from src.validation.metrics import calculate_metrics, compare_metrics
    
    # Load data
    import pandas as pd
    data = pd.read_csv(data_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    print(f"✓ Loaded data: {len(data)} rows")
    
    # Run benchmarks
    print(f"\nRunning benchmarks for {pair}...")
    results = run_all_benchmarks(data, config)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    output_file = output_dir / f"{pair}_benchmarks.json"
    
    # Convert to serializable format
    def make_serializable(obj):
        """Convert non-serializable objects to serializable format"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (pd.Series, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved: {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    # Helper functions for safe formatting
    def fmt_pct(v):
        """Format percentage, handle None"""
        return "N/A" if v is None else f"{v:.2f}%"
    
    def fmt_num(v):
        """Format number, handle None"""
        return "N/A" if v is None else f"{v:.2f}"
    
    def fmt_int(v):
        """Format integer, handle None"""
        return "N/A" if v is None else str(int(v))
    
    for name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            blown_up_flag = " [BLOWN UP]" if metrics.get('is_blown_up', False) else ""
            
            print(f"\n{name.upper()}{blown_up_flag}:")
            print(f"  Total Return: {fmt_pct(metrics['total_return'])}")
            print(f"  CAGR:         {fmt_pct(metrics['cagr'])}")
            print(f"  Sharpe:       {fmt_num(metrics['sharpe_ratio'])}")
            print(f"  Max DD:       {fmt_pct(metrics['max_drawdown'])}")
            print(f"  Win Rate:     {fmt_pct(metrics['win_rate'])}")
            print(f"  Trades:       {fmt_int(metrics['total_trades'])}")
    
    return {
        'status': 'completed',
        'pair': pair,
        'results': serializable_results,
        'output_file': str(output_file)
    }


def run_walkforward(pair: str, config: dict, output_dir: Path):
    """Run walk-forward analysis"""
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD MODE - {pair}")
    print(f"{'='*80}")
    
    # Load data
    data_file = f"data/raw/{pair}_1h.csv"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return {
            'status': 'error',
            'pair': pair,
            'message': f'Data file not found: {data_file}'
        }
    
    # Import walk-forward module
    from src.validation.walk_forward import WalkForward
    
    # Load data
    import pandas as pd
    data = pd.read_csv(data_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    print(f"✓ Loaded data: {len(data)} rows")
    
    # Run walk-forward
    wf = WalkForward(config)
    results = wf.run(data, pair)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    output_file = output_dir / f"{pair}_walkforward.json"
    
    # Convert to serializable format
    def make_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved: {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*80}")
    print(f"Windows: {results['num_windows']}")
    
    agg = results['aggregated']
    print(f"\nAggregated Metrics:")
    print(f"  Mean Return: {agg['total_return']['mean']:.2f}% (±{agg['total_return']['std']:.2f}%)")
    print(f"  Mean Sharpe: {agg['sharpe_ratio']['mean']:.2f}" if agg['sharpe_ratio']['mean'] else "  Mean Sharpe: N/A")
    print(f"  Worst DD:    {agg['max_drawdown']['worst']:.2f}%")
    print(f"  Blown Up:    {agg['windows_blown_up']}/{results['num_windows']} windows")
    
    return {
        'status': 'completed',
        'pair': pair,
        'results': serializable_results,
        'output_file': str(output_file)
    }


def run_oos(pair: str, config: dict, output_dir: Path):
    """Run out-of-sample analysis"""
    print(f"\n{'='*80}")
    print(f"OUT-OF-SAMPLE MODE - {pair}")
    print(f"{'='*80}")
    
    # Load data
    data_file = f"data/raw/{pair}_1h.csv"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return {
            'status': 'error',
            'pair': pair,
            'message': f'Data file not found: {data_file}'
        }
    
    # Import OOS module
    from src.validation.oos_report import OOSReport, save_oos_report
    
    # Load data
    import pandas as pd
    data = pd.read_csv(data_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    print(f"✓ Loaded data: {len(data)} rows")
    
    # Run OOS
    oos = OOSReport(config)
    results = oos.run(data, pair)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    save_oos_report(results, str(output_dir))
    
    # Print verdict
    print(f"\n{'='*80}")
    print("OOS VERDICT")
    print(f"{'='*80}")
    print(f"Status: {results['verdict']['status']}")
    print(f"Reason: {results['verdict']['reason']}")
    
    return {
        'status': 'completed',
        'pair': pair,
        'verdict': results['verdict']
    }


def run_montecarlo(pair: str, config: dict, output_dir: Path):
    """Run Monte Carlo simulation"""
    print(f"\n{'='*80}")
    print(f"MONTE CARLO MODE - {pair}")
    print(f"{'='*80}")
    
    # TODO: Implement in P1-08 (Sprint 3)
    print("⏳ Monte Carlo module to be implemented in Sprint 3 (P1-08)")
    
    return {
        'status': 'pending',
        'pair': pair,
        'message': 'Monte Carlo implementation pending (Sprint 3)'
    }


def run_sensitivity(pair: str, config: dict, output_dir: Path):
    """Run sensitivity analysis"""
    print(f"\n{'='*80}")
    print(f"SENSITIVITY MODE - {pair}")
    print(f"{'='*80}")
    
    # TODO: Implement in P1-09 (Sprint 3)
    print("⏳ Sensitivity module to be implemented in Sprint 3 (P1-09)")
    
    return {
        'status': 'pending',
        'pair': pair,
        'message': 'Sensitivity implementation pending (Sprint 3)'
    }


def run_all(pair: str, config: dict, output_dir: Path):
    """Run all validation modes"""
    print(f"\n{'='*80}")
    print(f"ALL MODES - {pair}")
    print(f"{'='*80}")
    
    results = {
        'pair': pair,
        'timestamp': datetime.now().isoformat(),
        'modes': {}
    }
    
    # Run each mode
    print("\n[1/4] BENCHMARKS")
    results['modes']['benchmark'] = run_benchmark(pair, config, output_dir / 'benchmarks')
    
    print("\n[2/4] WALK-FORWARD")
    results['modes']['walkforward'] = run_walkforward(pair, config, output_dir / 'walkforward')
    
    print("\n[3/4] OUT-OF-SAMPLE")
    results['modes']['oos'] = run_oos(pair, config, output_dir / 'oos')
    
    print("\n[4/4] AGGREGATED SUMMARY")
    # Generate final summary report
    from src.validation.reporting import generate_summary_report
    
    try:
        summary = generate_summary_report(pair, validation_dir=str(output_dir))
        results['summary'] = summary
        print("✓ Summary report generated")
    except Exception as e:
        print(f"⚠ Summary report failed: {e}")
        results['summary'] = {'error': str(e)}
    
    return results


def run_funnel(pair: str, config: dict, output_dir: Path):
    """Run signal funnel diagnostics"""
    print(f"\n{'='*80}")
    print(f"SIGNAL FUNNEL MODE - {pair}")
    print(f"{'='*80}")
    
    # Load data
    data_file = f"data/raw/{pair}_1h.csv"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return {
            'status': 'error',
            'pair': pair,
            'message': f'Data file not found: {data_file}'
        }
    
    # Import funnel module
    from src.validation.signal_funnel import SignalFunnel, save_funnel_report
    
    # Load data
    import pandas as pd
    data = pd.read_csv(data_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    print(f"✓ Loaded data: {len(data)} rows")
    
    # Run funnel
    funnel = SignalFunnel(config)
    report = funnel.run(data, pair)
    
    # Save report
    funnel_dir = output_dir / 'funnel'
    save_funnel_report(report, str(funnel_dir))
    
    # Print summary
    print(report['summary'])
    
    return {
        'status': 'completed',
        'pair': pair,
        'report': report
    }


def run_funnel_compare(pair: str, config: dict, output_dir: Path, sample_bars: "int | None" = None):
    """Run funnel comparison between min_required_bars=50 and 100"""
    print(f"\n{'='*80}")
    print(f"FUNNEL COMPARE MODE - {pair}")
    print(f"{'='*80}")
    
    # Load data
    data_file = f"data/raw/{pair}_1h.csv"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return {
            'status': 'error',
            'pair': pair,
            'message': f'Data file not found: {data_file}'
        }
    
    # Import funnel module
    from src.validation.signal_funnel import SignalFunnel
    import json
    
    # Load data
    import pandas as pd
    data = pd.read_csv(data_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    # Apply sample if requested
    if sample_bars:
        data = data.tail(sample_bars)
        print(f"✓ Using sample: last {sample_bars} bars")
    else:
        print(f"✓ Loaded data: {len(data)} bars (full dataset)")
    
    # Run comparison
    comparison = SignalFunnel.compare_min_bars(
        data=data,
        pair=pair,
        base_config=config,
        min_bars_list=[50, 100]
    )
    
    # Save comparison
    funnel_dir = output_dir / 'funnel'
    funnel_dir.mkdir(parents=True, exist_ok=True)
    json_file = funnel_dir / f"{pair}_funnel_compare.json"
    
    with open(json_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"\n✓ Comparison saved: {json_file}")
    
    # Print summary
    print(comparison['summary'])
    
    return {
        'status': 'completed',
        'pair': pair,
        'comparison': comparison
    }


def run_smc_diagnostics(pair: str, config: dict, output_dir: Path, sample_bars: "int | None" = None):
    """Run SMC pattern coverage diagnostics"""
    print(f"\n{'='*80}")
    print(f"SMC DIAGNOSTICS MODE - {pair}")
    print(f"{'='*80}")
    
    # Load data
    data_file = f"data/raw/{pair}_1h.csv"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return {
            'status': 'error',
            'pair': pair,
            'message': f'Data file not found: {data_file}'
        }
    
    # Import module
    from src.validation.smc_diagnostics import SMCDiagnostics, save_smc_diagnostics
    
    # Load data
    import pandas as pd
    data = pd.read_csv(data_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    # Apply sample if requested
    if sample_bars:
        data = data.tail(sample_bars)
        print(f"✓ Using sample: last {sample_bars} bars")
    else:
        print(f"✓ Loaded data: {len(data)} bars (full dataset)")
    
    # Ensure min_required_bars = 50 for this diagnostic
    if 'strategy' not in config:
        config['strategy'] = {}
    config['strategy']['min_required_bars'] = 50
    print(f"✓ Set min_required_bars = 50 (for HSMM access)")
    
    # Run diagnostics
    diag = SMCDiagnostics(config)
    report = diag.run(data, pair)
    
    # Save report
    smc_dir = output_dir / 'smc'
    save_smc_diagnostics(report, str(smc_dir))
    
    # Print summary
    print(report['summary'])
    
    return {
        'status': 'completed',
        'pair': pair,
        'report': report
    }


def run_ob_audit(pair: str, config: dict, output_dir: Path, sample_bars: "int | None" = None, sensitivity: bool = False):
    """Run Order Block coverage audit"""
    print(f"\n{'='*80}")
    print(f"ORDER BLOCK AUDIT MODE - {pair}")
    print(f"{'='*80}")
    
    # Load data
    data_file = f"data/raw/{pair}_1h.csv"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return {
            'status': 'error',
            'pair': pair,
            'message': f'Data file not found: {data_file}'
        }
    
    # Import module
    from src.validation.order_block_audit import OrderBlockAudit, save_ob_audit
    
    # Load data
    import pandas as pd
    data = pd.read_csv(data_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    # Apply sample if requested
    if sample_bars:
        data = data.tail(sample_bars)
        print(f"✓ Using sample: last {sample_bars} bars")
    else:
        print(f"✓ Loaded data: {len(data)} bars (full dataset)")
    
    # Ensure SMC config exists
    if 'smc' not in config.get('strategy', {}):
        config['strategy']['smc'] = {}
    if 'ob_lookback' not in config['strategy']['smc']:
        config['strategy']['smc']['ob_lookback'] = 5
    if 'ob_range_threshold' not in config['strategy']['smc']:
        config['strategy']['smc']['ob_range_threshold'] = 0.015
    
    # Run audit
    audit = OrderBlockAudit(config)
    report = audit.run(data, pair, sensitivity=sensitivity)
    
    # Save report
    smc_dir = output_dir / 'smc'
    save_ob_audit(report, str(smc_dir))
    
    # Print summary
    print(report['summary'])
    
    # Print sensitivity if run
    if sensitivity and 'sensitivity' in report:
        print(f"\n{'='*80}")
        print("SENSITIVITY RESULTS")
        print(f"{'='*80}\n")
        
        print(f"{'Lookback':<10} {'Threshold':<12} {'Bullish':<10} {'Bearish':<10} {'Total':<10}")
        print("-" * 52)
        
        for result in report['sensitivity']:
            print(f"{result['ob_lookback']:<10} "
                  f"{result['ob_range_threshold']*100:<11.2f}% "
                  f"{result['bullish_ob_detected']:<10} "
                  f"{result['bearish_ob_detected']:<10} "
                  f"{result['total_detected']:<10}")
    
    return {
        'status': 'completed',
        'pair': pair,
        'report': report
    }


def run_pattern_quality(pair: str, config: dict, output_dir: Path, sample_bars: "int | None" = None):
    """Run pattern quality attribution analysis"""
    print(f"\n{'='*80}")
    print(f"PATTERN QUALITY MODE - {pair}")
    print(f"{'='*80}")
    
    # Load data
    data_file = f"data/raw/{pair}_1h.csv"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return {
            'status': 'error',
            'pair': pair,
            'message': f'Data file not found: {data_file}'
        }
    
    # Import module
    from src.validation.pattern_quality import PatternQuality, save_pattern_quality
    
    # Load data
    import pandas as pd
    data = pd.read_csv(data_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    # Apply sample if requested
    if sample_bars:
        data = data.tail(sample_bars)
        print(f"✓ Using sample: last {sample_bars} bars")
    else:
        print(f"✓ Loaded data: {len(data)} bars (full dataset)")
    
    # Set min_required_bars to 50 for HSMM access
    if 'strategy' not in config:
        config['strategy'] = {}
    config['strategy']['min_required_bars'] = 50
    print(f"✓ Set min_required_bars = 50 (for HSMM access)")
    
    # Run analysis
    pq = PatternQuality(config)
    report = pq.run(data, pair)
    
    # Save report
    smc_dir = output_dir / 'smc'
    save_pattern_quality(report, str(smc_dir))
    
    # Print summary
    print(report['summary'])
    
    return {
        'status': 'completed',
        'pair': pair,
        'report': report
    }


def run_mtf_baseline_check(pair: str, config: dict, output_dir: Path, sample_bars: "int | None" = None):
    """Run MTF baseline validation check"""
    print(f"\n{'='*80}")
    print(f"MTF BASELINE CHECK - {pair}")
    print(f"{'='*80}")
    
    # Import modules
    from src.data.mtf_loader import MTFLoader
    from src.core.nyx_engine_mtf import NYXEngineMTF
    import json
    
    # Load MTF data
    loader = MTFLoader("data/raw/mtf")
    
    print(f"\n📊 Loading multi-timeframe data...")
    mtf_data = loader.load(pair)
    
    if not mtf_data:
        print(f"❌ No MTF data found for {pair}")
        return {
            'status': 'error',
            'pair': pair,
            'message': f'No MTF data found for {pair}'
        }
    
    print(f"✓ Loaded {len(mtf_data)} timeframes:")
    for tf, df in mtf_data.items():
        print(f"  {tf}: {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
    
    # Apply sample if requested
    if sample_bars:
        from src.data.mtf_loader import load_mtf_sample
        mtf_data = load_mtf_sample(pair, sample_bars, "data/raw/mtf")
        print(f"\n✓ Applied sample: last {sample_bars} bars from lowest TF")
        for tf, df in mtf_data.items():
            print(f"  {tf}: {len(df)} bars")
    
    # Validate alignment
    print(f"\n🔍 Validating alignment...")
    validation = loader.validate_alignment(mtf_data)
    
    if validation['valid']:
        print(f"✅ Alignment valid")
    else:
        print(f"❌ Alignment issues:")
        for issue in validation['issues']:
            print(f"  ⚠️  {issue}")
    
    # Test alignment at specific timestamp
    if '15m' in mtf_data and len(mtf_data['15m']) > 100:
        test_time = pd.Timestamp(mtf_data['15m'].index[-100])

        print(f"\n🔗 Testing closed-candle alignment at {test_time}:")
        aligned = loader.align_at_timestamp(mtf_data, test_time, '15m')

        for tf, df in aligned.items():
            last_time = pd.Timestamp(df.index[-1]) if len(df) > 0 else None
            delta = (test_time - last_time).total_seconds() / 60 if last_time else None
            print(f"  {tf}: last candle at {last_time} (Δ={delta:.0f}m)")
    
    # Initialize MTF engine
    print(f"\n🚀 Testing MTF engine...")
    engine = NYXEngineMTF(config)
    
    # Generate sample signal
    current_time = pd.Timestamp(mtf_data['15m'].index[-1]) if '15m' in mtf_data else pd.Timestamp(mtf_data['1h'].index[-1])

    signal = engine.generate_signal_mtf(
        pair=pair,
        mtf_data=mtf_data,
        current_date=current_time.isoformat()
    )
    
    print(f"\n📈 Sample signal generated:")
    print(f"  Action: {signal['action']}")
    print(f"  Intent Daily: {signal.get('intent_daily', 'N/A')}")
    print(f"  Stability 4H: {signal.get('stability_4h', 0):.3f}")
    print(f"  Alignment 15M: {signal.get('alignment_15m', 0):.3f}")
    print(f"  SdC: {signal.get('sdc', 0):.1f}")
    
    # Build report
    report = {
        'pair': pair,
        'timeframes_loaded': list(mtf_data.keys()),
        'bars_per_tf': {tf: len(df) for tf, df in mtf_data.items()},
        'validation': validation,
        'sample_signal': {
            'timestamp': current_time.isoformat(),
            'action': signal['action'],
            'intent_daily': signal.get('intent_daily'),
            'stability_4h': signal.get('stability_4h', 0),
            'alignment_15m': signal.get('alignment_15m', 0),
            'sdc': signal.get('sdc', 0),
            'conditions_met': signal.get('conditions_met', {})
        }
    }
    
    # Save report
    mtf_dir = output_dir / 'mtf'
    mtf_dir.mkdir(parents=True, exist_ok=True)
    json_file = mtf_dir / f"{pair}_mtf_baseline_check.json"
    
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Report saved: {json_file}")
    
    return {
        'status': 'completed',
        'pair': pair,
        'report': report
    }


def main():
    parser = argparse.ArgumentParser(
        description='NYX Phase 1 Validation CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  benchmark    - Compare NYX vs simple strategies
  walkforward  - Walk-forward validation
  montecarlo   - Monte Carlo simulation
  sensitivity  - Parameter sensitivity analysis
  all          - Run all modes

Examples:
  python scripts/run_validation.py --mode benchmark --pair BTCUSDT
  python scripts/run_validation.py --mode all --pair ETHUSDT --output reports/validation
        """
    )
    
    parser.add_argument(
        '--mode',
        required=True,
        choices=['benchmark', 'walkforward', 'oos', 'funnel', 'funnel_compare', 'smc_diagnostics', 'ob_audit', 'pattern_quality', 'mtf_baseline_check', 'fractal_check', 'fractal_bottleneck', 'fractal_geometry', 'setup_bottleneck', 'setup_coverage_multi', 'smc_autopsy', 'setup_detector_check', 'bullish_audit', 'fractal_cache_check', 'fractal_cache_benchmark', 'regime_tuning', 'hsmm_deep_dive', 'regime_feature_check', 'phase1_rerun', 'setup_deep_dive', 'setup_investigation', 'setup_investigation_v2', 'montecarlo', 'sensitivity', 'all'],
        help='Validation mode to run'
    )
    
    parser.add_argument(
        '--pair',
        required=True,
        help='Trading pair (e.g., BTCUSDT, ETHUSDT)'
    )
    
    parser.add_argument(
        '--sample-bars',
        type=int,
        default=None,
        help='Use only the last N bars for funnel_compare (speeds up testing)'
    )
    
    parser.add_argument(
        '--sample-date',
        type=str,
        default='2024-01-01',
        help='Reference date for fractal_geometry and regime_feature_check modes (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=20,
        help='Number of steps for fractal_cache_check or fractal_cache_benchmark'
    )
    
    parser.add_argument(
        '--dates',
        type=str,
        default=None,
        help='Comma-separated dates for regime_tuning (YYYY-MM-DD,YYYY-MM-DD,...)'
    )
    
    parser.add_argument(
        '--sensitivity',
        action='store_true',
        help='Run sensitivity analysis for ob_audit mode'
    )
    
    parser.add_argument(
        '--use-mtf',
        action='store_true',
        help='Use multi-timeframe engine instead of single-TF'
    )
    
    parser.add_argument(
        '--config',
        default='config/validation_baseline.yaml',
        help='Baseline configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        default='reports/validation',
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "█" * 80)
    print("NYX PHASE 1 - VALIDATION CLI")
    print("█" * 80)
    print(f"Mode:       {args.mode}")
    print(f"Pair:       {args.pair}")
    print(f"Config:     {args.config}")
    print(f"Output:     {args.output_dir}")
    print("█" * 80)
    
    # Load config
    config = load_baseline_config()
    
    # Validate pair
    if args.pair not in config['pairs']:
        print(f"\n❌ Invalid pair: {args.pair}")
        print(f"   Allowed pairs: {', '.join(config['pairs'])}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    results: dict = {}
    if args.mode == 'benchmark':
        results = run_benchmark(args.pair, config, output_dir)
    elif args.mode == 'walkforward':
        results = run_walkforward(args.pair, config, output_dir)
    elif args.mode == 'oos':
        results = run_oos(args.pair, config, output_dir)
    elif args.mode == 'funnel':
        results = run_funnel(args.pair, config, output_dir)
    elif args.mode == 'funnel_compare':
        sample_bars = getattr(args, 'sample_bars', None)
        results = run_funnel_compare(args.pair, config, output_dir, sample_bars)
    elif args.mode == 'smc_diagnostics':
        sample_bars = getattr(args, 'sample_bars', None)
        results = run_smc_diagnostics(args.pair, config, output_dir, sample_bars)
    elif args.mode == 'ob_audit':
        sample_bars = getattr(args, 'sample_bars', None)
        sensitivity = getattr(args, 'sensitivity', False)
        results = run_ob_audit(args.pair, config, output_dir, sample_bars, sensitivity)
    elif args.mode == 'pattern_quality':
        sample_bars = getattr(args, 'sample_bars', None)
        results = run_pattern_quality(args.pair, config, output_dir, sample_bars)
    elif args.mode == 'mtf_baseline_check':
        sample_bars = getattr(args, 'sample_bars', None)
        results = run_mtf_baseline_check(args.pair, config, output_dir, sample_bars)
    elif args.mode == 'fractal_check':
        from scripts.fractal_check import run_fractal_check
        sample_date = getattr(args, 'sample_date', '2024-01-01')
        results = run_fractal_check(args.pair, config, output_dir, sample_date)
    elif args.mode == 'fractal_bottleneck':
        from scripts.fractal_bottleneck import run_fractal_bottleneck
        sample_date = getattr(args, 'sample_date', '2024-01-01')
        results = run_fractal_bottleneck(args.pair, config, output_dir, sample_date)
    elif args.mode == 'fractal_geometry':
        from scripts.fractal_geometry import run_fractal_geometry
        sample_date = getattr(args, 'sample_date', '2024-01-01')
        results = run_fractal_geometry(args.pair, config, output_dir, sample_date)
    elif args.mode == 'setup_bottleneck':
        from scripts.setup_bottleneck import run_setup_bottleneck
        sample_date = getattr(args, 'sample_date', '2024-01-01')
        results = run_setup_bottleneck(args.pair, config, output_dir, sample_date)
    elif args.mode == 'setup_coverage_multi':
        from scripts.setup_coverage_multi import run_setup_coverage_multi
        # Check if custom dates provided
        dates_arg = getattr(args, 'dates', None)
        if dates_arg:
            custom_dates = [d.strip() for d in dates_arg.split(',')]
            results = run_setup_coverage_multi(args.pair, config, output_dir, dates=custom_dates)
        else:
            results = run_setup_coverage_multi(args.pair, config, output_dir)
    elif args.mode == 'smc_autopsy':
        from scripts.smc_autopsy import run_smc_autopsy
        sample_date = getattr(args, 'sample_date', '2024-01-01')
        sensitivity_mode = getattr(args, 'sensitivity', False)
        results = run_smc_autopsy(args.pair, config, output_dir, sample_date, sensitivity=sensitivity_mode)
    elif args.mode == 'setup_detector_check':
        from scripts.setup_detector_check import run_setup_detector_check
        sample_date = getattr(args, 'sample_date', '2024-01-01')
        results = run_setup_detector_check(args.pair, config, output_dir, sample_date)
    elif args.mode == 'bullish_audit':
        from scripts.bullish_audit import run_bullish_audit
        # Check if custom dates provided
        dates_arg = getattr(args, 'dates', None)
        if dates_arg:
            custom_dates = [d.strip() for d in dates_arg.split(',')]
            results = run_bullish_audit(args.pair, config, output_dir, dates=custom_dates)
        else:
            results = run_bullish_audit(args.pair, config, output_dir)
    elif args.mode == 'fractal_cache_check':
        from scripts.fractal_cache_check import run_fractal_cache_check
        results = run_fractal_cache_check(args.pair, config, output_dir, args.sample_date, args.steps)
    elif args.mode == 'fractal_cache_benchmark':
        from scripts.fractal_cache_benchmark import run_fractal_cache_benchmark
        results = run_fractal_cache_benchmark(args.pair, config, output_dir, args.steps)
    elif args.mode == 'regime_tuning':
        from scripts.regime_tuning import run_regime_tuning
        from datetime import datetime
        
        # Parse dates if provided
        dates = None
        if args.dates:
            try:
                dates = [datetime.strptime(d.strip(), '%Y-%m-%d') for d in args.dates.split(',')]
            except ValueError as e:
                print(f"\n❌ Invalid date format: {e}")
                print("   Expected format: YYYY-MM-DD,YYYY-MM-DD,...")
                sys.exit(1)
        
        results = run_regime_tuning(args.pair, config, str(output_dir), dates=dates)
    elif args.mode == 'hsmm_deep_dive':
        from src.validation.hsmm_deep_dive import run_hsmm_deep_dive
        from datetime import datetime

        # Parse dates if provided
        dates = None
        if args.dates:
            try:
                dates = [datetime.strptime(d.strip(), '%Y-%m-%d') for d in args.dates.split(',')]
            except ValueError as e:
                print(f"\n❌ Invalid date format: {e}")
                print("   Expected format: YYYY-MM-DD,YYYY-MM-DD,...")
                sys.exit(1)

        results = run_hsmm_deep_dive(args.pair, config, str(output_dir), dates=dates)
    elif args.mode == 'regime_feature_check':
        from scripts.regime_feature_check import run_regime_feature_check
        from datetime import datetime

        # Parse sample date if provided
        sample_date = None
        if hasattr(args, 'sample_date') and args.sample_date:
            try:
                sample_date = datetime.strptime(args.sample_date, '%Y-%m-%d')
            except ValueError as e:
                print(f"\n❌ Invalid sample-date format: {e}")
                print("   Expected format: YYYY-MM-DD")
                sys.exit(1)

        results = run_regime_feature_check(args.pair, config, str(output_dir), sample_date=sample_date)
    elif args.mode == 'phase1_rerun':
        from scripts.phase1_rerun import run_phase1_rerun

        results = run_phase1_rerun(args.pair, config, output_dir)
    elif args.mode == 'setup_deep_dive':
        from scripts.setup_deep_dive import run_setup_deep_dive
        from datetime import datetime

        # Parse dates if provided
        dates = None
        if hasattr(args, 'dates') and args.dates:
            dates = [datetime.strptime(d, '%Y-%m-%d') for d in args.dates.split(',')]

        results = run_setup_deep_dive(args.pair, config, str(output_dir), dates=dates)
    elif args.mode == 'setup_investigation':
        from scripts.setup_investigation import run_setup_investigation
        from datetime import datetime

        # Parse dates if provided
        dates = None
        if hasattr(args, 'dates') and args.dates:
            dates = [datetime.strptime(d, '%Y-%m-%d') for d in args.dates.split(',')]

        results = run_setup_investigation(args.pair, config, str(output_dir), dates=dates)
    elif args.mode == 'setup_investigation_v2':
        from scripts.setup_investigation_v2 import run_setup_investigation_v2
        from datetime import datetime

        # Parse dates if provided
        dates = None
        if hasattr(args, 'dates') and args.dates:
            dates = [datetime.strptime(d, '%Y-%m-%d') for d in args.dates.split(',')]

        results = run_setup_investigation_v2(args.pair, config, str(output_dir), dates=dates)
    elif args.mode == 'montecarlo':
        results = run_montecarlo(args.pair, config, output_dir)
    elif args.mode == 'sensitivity':
        results = run_sensitivity(args.pair, config, output_dir)
    elif args.mode == 'all':
        results = run_all(args.pair, config, output_dir)
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results: {results.get('status', 'completed')}")
    print(f"Output:  {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

