"""
Setup Detector Check

Verify SMCDetector is correctly instantiated and wired to SetupAgent.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, '.')

from src.data.mtf_loader import load_fractal_context
from src.agents.setup_agent import SetupAgent


def run_setup_detector_check(pair: str, config: dict, output_dir: Path, sample_date: str = None):
    """
    Check SetupAgent → SMCDetector wiring
    
    Args:
        pair: Trading pair
        config: System configuration
        output_dir: Output directory
        sample_date: Reference date
    """
    
    print(f"\n{'='*80}")
    print(f"SETUP DETECTOR CHECK - {pair}")
    print(f"{'='*80}")
    
    # Parse sample date
    if sample_date:
        current_date = datetime.strptime(sample_date, '%Y-%m-%d')
    else:
        current_date = datetime(2024, 1, 1)
    
    print(f"\nReference date: {current_date.strftime('%Y-%m-%d')}")
    
    # Step 1: Try to instantiate SetupAgent
    print(f"\n🔬 Step 1: Instantiating SetupAgent...")
    
    detector_initialized = False
    parameters_used = None
    detector_error = False
    error_type = None
    error_message = None
    
    try:
        agent = SetupAgent(config)
        detector_initialized = True
        
        # Extract parameters that were used
        parameters_used = {
            'ob_range_threshold': agent.smc.ob_range_threshold,
            'fvg_min_gap': agent.smc.fvg_min_gap,
            'liquidity_lookback': agent.smc.liquidity_lookback
        }
        
        print(f"✓ SetupAgent instantiated successfully")
        print(f"✓ SMCDetector initialized with:")
        print(f"    ob_range_threshold:  {parameters_used['ob_range_threshold']}")
        print(f"    fvg_min_gap:         {parameters_used['fvg_min_gap']}")
        print(f"    liquidity_lookback:  {parameters_used['liquidity_lookback']}")
        
    except Exception as e:
        detector_error = True
        error_type = type(e).__name__
        error_message = str(e)
        
        print(f"❌ Failed to instantiate SetupAgent")
        print(f"    Error: {error_type}: {error_message}")
    
    # Step 2: Try to use detector with real data
    print(f"\n🔬 Step 2: Testing detector with real data...")
    
    patterns_detected = None
    test_error = False
    test_error_type = None
    test_error_message = None
    
    if detector_initialized:
        try:
            # Load data
            print(f"Loading fractal context...")
            mtf_data = load_fractal_context(pair, current_date, config)
            setup_data = mtf_data.get('15m')
            
            if setup_data is not None and len(setup_data) > 50:
                print(f"✓ Loaded {len(setup_data)} bars")
                
                # Run analysis
                print(f"Running SetupAgent.analyze()...")
                result = agent.analyze(setup_data, context_state='bullish')
                
                print(f"\nResult:")
                print(f"  State:  {result.state}")
                print(f"  Passed: {result.passed}")
                print(f"  Ready:  {result.ready}")
                print(f"  Reason: {result.reason}")
                
                # Check if detector error
                if result.state == 'detector_error':
                    test_error = True
                    test_error_type = result.metadata.get('exception_type')
                    test_error_message = result.metadata.get('exception_message')
                    
                    print(f"\n❌ Detector Error Detected:")
                    print(f"    Type: {test_error_type}")
                    print(f"    Message: {test_error_message}")
                else:
                    patterns_detected = result.metadata.get('patterns', {})
                    print(f"\n✓ Detector executed successfully")
                    print(f"  Patterns: {patterns_detected}")
            
            else:
                print(f"❌ Insufficient data")
        
        except Exception as e:
            test_error = True
            test_error_type = type(e).__name__
            test_error_message = str(e)
            
            print(f"❌ Test failed: {test_error_type}: {test_error_message}")
    
    # Save report
    output_dir_fractal = Path(output_dir) / 'fractal'
    output_dir_fractal.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir_fractal / f"{pair}_setup_detector_check.json"
    
    report = {
        'pair': pair,
        'reference_date': current_date.isoformat(),
        'detector_initialized': detector_initialized,
        'parameters_used': parameters_used,
        'detector_error': detector_error or test_error,
        'error_type': error_type or test_error_type,
        'error_message': error_message or test_error_message,
        'patterns_detected': patterns_detected,
        'verdict': _determine_verdict(
            detector_initialized,
            detector_error,
            test_error,
            error_type,
            test_error_type
        )
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    
    # Print verdict
    print(f"\n{'='*80}")
    print(f"VERDICT")
    print(f"{'='*80}")
    print(report['verdict'])
    print(f"{'='*80}\n")
    
    return {
        'status': 'completed',
        'pair': pair,
        'detector_initialized': detector_initialized,
        'detector_error': detector_error or test_error,
        'output': str(report_path)
    }


def _determine_verdict(detector_initialized, detector_error, test_error, 
                       error_type, test_error_type):
    """Determine check verdict"""
    
    if not detector_initialized:
        if error_type == 'TypeError' and 'must be numeric' in str(error_type):
            return "WIRING BUG CONFIRMED: SetupAgent passing wrong types to SMCDetector"
        else:
            return f"SetupAgent instantiation failed: {error_type}"
    
    elif test_error:
        if test_error_type == 'TypeError':
            return "SMCDetector crashes during execution with TypeError - likely wiring issue"
        else:
            return f"Detector execution error: {test_error_type}"
    
    else:
        return "✓ SMCDetector correctly wired and functioning"


if __name__ == "__main__":
    import yaml
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_setup_detector_check('BTCUSDT', config, Path('reports/validation'), sample_date='2023-12-15')
