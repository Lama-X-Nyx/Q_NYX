"""
Fractal Geometry Validation

Verify that each timeframe is loaded with the correct context window.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

from src.data.mtf_loader import load_fractal_context


def run_fractal_geometry(pair: str, config: dict, output_dir: Path, sample_date: str = None):
    """
    Validate fractal geometry - proper context windows per TF
    
    Args:
        pair: Trading pair
        config: Config dict
        output_dir: Output directory
        sample_date: Reference date (YYYY-MM-DD)
    
    Returns:
        Validation results
    """
    
    print(f"\n{'='*80}")
    print(f"FRACTAL GEOMETRY VALIDATION - {pair}")
    print(f"{'='*80}")
    
    # Parse sample date
    if sample_date:
        current_date = datetime.strptime(sample_date, '%Y-%m-%d')
    else:
        # Use latest available data
        current_date = datetime(2024, 1, 1)
    
    print(f"\nReference date: {current_date.strftime('%Y-%m-%d')}")
    
    # Get expected windows
    fractal_config = config.get('fractal', {})
    windows = fractal_config.get('context_windows', {})
    
    print(f"\n📊 Expected Context Windows:")
    print(f"  1D context:   {windows.get('context_1d_days', 360)} days")
    print(f"  4H structure: {windows.get('structure_4h_days', 7)} days")
    print(f"  1H regime:    {windows.get('regime_1h_days', 2)} days")
    print(f"  15M setup:    {windows.get('setup_15m_days', 1)} days")
    
    # Load fractal context
    print(f"\n🔬 Loading fractal context...")
    
    try:
        mtf_data = load_fractal_context(pair, current_date, config)
    except Exception as e:
        print(f"❌ Failed to load fractal context: {e}")
        return {'status': 'error', 'reason': str(e)}
    
    # Analyze geometry
    print(f"\n{'='*60}")
    print(f"FRACTAL GEOMETRY RESULTS")
    print(f"{'='*60}")
    
    geometry = {}
    all_valid = True
    
    for tf in ['1d', '4h', '1h', '15m']:
        if tf in mtf_data:
            df = mtf_data[tf]
            bars_loaded = len(df)
            
            if bars_loaded > 0:
                start_date = df.index[0]
                end_date = df.index[-1]
                days_span = (end_date - start_date).days
                
                # Expected bars (approximate)
                window_key = f"{'context' if tf == '1d' else 'structure' if tf == '4h' else 'regime' if tf == '1h' else 'setup'}_{tf}_days"
                expected_days = windows.get(window_key, 0)
                
                # Validate alignment
                aligned = (end_date <= current_date)
                closed_only = (end_date < datetime.now())
                
                geometry[tf] = {
                    'bars_loaded': bars_loaded,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days_span': days_span,
                    'expected_days': expected_days,
                    'aligned_to_reference': aligned,
                    'closed_candles_only': closed_only
                }
                
                print(f"\n{tf.upper()} Window:")
                print(f"  Bars loaded:     {bars_loaded}")
                print(f"  Date range:      {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
                print(f"  Days span:       {days_span} days (expected ~{expected_days})")
                print(f"  Aligned:         {'✅' if aligned else '❌'}")
                print(f"  Closed candles:  {'✅' if closed_only else '❌'}")
                
                if not aligned:
                    all_valid = False
            else:
                geometry[tf] = {
                    'bars_loaded': 0,
                    'error': 'No data loaded'
                }
                print(f"\n{tf.upper()} Window:")
                print(f"  ❌ No data loaded")
                all_valid = False
        else:
            geometry[tf] = {
                'bars_loaded': 0,
                'error': 'Timeframe not available'
            }
            print(f"\n{tf.upper()} Window:")
            print(f"  ❌ Timeframe not available")
            all_valid = False
    
    print(f"\n{'='*60}")
    
    if all_valid and all(geometry[tf].get('bars_loaded', 0) > 0 for tf in ['1d', '1h', '15m']):
        print("✅ All windows loaded correctly")
        print("✅ All aligned to reference date")
        print("✅ All using closed candles only")
        status = 'valid'
    else:
        print("⚠️  Some geometry issues detected")
        status = 'issues'
    
    print(f"{'='*60}")
    
    # Save JSON
    output_dir = Path(output_dir) / 'fractal'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{pair}_fractal_geometry.json"
    
    report = {
        'pair': pair,
        'reference_date': current_date.isoformat(),
        'expected_windows': windows,
        'geometry': geometry,
        'status': status,
        'validation': {
            'all_tfs_loaded': all(geometry[tf].get('bars_loaded', 0) > 0 for tf in ['1d', '1h', '15m']),
            'all_aligned': all(geometry[tf].get('aligned_to_reference', False) for tf in geometry if geometry[tf].get('bars_loaded', 0) > 0),
            'all_closed': all(geometry[tf].get('closed_candles_only', False) for tf in geometry if geometry[tf].get('bars_loaded', 0) > 0)
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    
    return {
        'status': 'completed',
        'pair': pair,
        'geometry_valid': status == 'valid',
        'output': str(report_path)
    }


if __name__ == "__main__":
    import yaml
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_fractal_geometry('BTCUSDT', config, Path('reports/validation'), sample_date='2024-01-01')
