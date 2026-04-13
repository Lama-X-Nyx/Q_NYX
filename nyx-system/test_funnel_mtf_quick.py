"""
Quick Funnel MTF Test

Tests signal generation with MTF engine on sample data
"""

import sys
sys.path.insert(0, '/home/claude/nyx-phase1/nyx-system')

import pandas as pd
import yaml
from src.data.mtf_loader import load_mtf_sample
from src.core.nyx_engine_mtf import NYXEngineMTF

# Load config
with open('config/validation_baseline.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load MTF data (larger sample for meaningful results)
print("Loading MTF data...")
mtf_data = load_mtf_sample('BTCUSDT', sample_bars=10000)

print(f"Loaded {len(mtf_data)} timeframes:")
for tf, df in mtf_data.items():
    print(f"  {tf}: {len(df)} bars")

# Initialize MTF engine
print("\nInitializing MTF engine...")
engine = NYXEngineMTF(config)

# Test on last 100 bars of 15M
print("\nGenerating signals...")

warmup = 100
signals_generated = 0
hsmm_passed = 0
smc_passed = 0
stability_passed = 0
alignment_passed = 0
trades = 0

for i in range(warmup, min(len(mtf_data['15m']), 200)):  # Test 100 bars
    current_time = pd.Timestamp(mtf_data['15m'].index[i])

    # Get aligned data
    from src.data.mtf_loader import MTFLoader
    loader = MTFLoader('data/raw/mtf')
    aligned_data = loader.align_at_timestamp(mtf_data, current_time, '15m')

    # Generate signal
    signal = engine.generate_signal_mtf('BTCUSDT', aligned_data, current_time.isoformat())
    
    signals_generated += 1
    
    # Track funnel
    sdc = signal.get('sdc', 0)
    stability = signal.get('stability_4h', 0)
    alignment = signal.get('alignment_15m', 0)
    smc_patterns = signal.get('smc_patterns', {})
    
    # Debug first few
    if signals_generated <= 3:
        print(f"\nSignal {signals_generated}:")
        print(f"  Intent: {signal.get('intent_daily')}")
        print(f"  SdC: {sdc:.4f}")
        print(f"  Stability: {stability:.3f}")
        print(f"  Alignment: {alignment:.3f}")
        print(f"  Patterns: {smc_patterns}")
    
    if sdc > 5.0:
        hsmm_passed += 1
        
    if stability >= 0.60:
        stability_passed += 1
        
    if alignment >= 0.60:
        alignment_passed += 1
        
    has_pattern = any(smc_patterns.values()) if smc_patterns else False
    if has_pattern:
        smc_passed += 1
    
    if signal['action'] in ['BUY', 'SELL']:
        trades += 1

# Report
print(f"\n{'='*60}")
print(f"FUNNEL MTF TEST RESULTS")
print(f"{'='*60}")
print(f"Signals evaluated:     {signals_generated}")
print(f"SdC > 5:               {hsmm_passed:6d} ({hsmm_passed/signals_generated*100:5.1f}%)")
print(f"Stability ≥ 0.60:      {stability_passed:6d} ({stability_passed/signals_generated*100:5.1f}%)")
print(f"Alignment ≥ 0.60:      {alignment_passed:6d} ({alignment_passed/signals_generated*100:5.1f}%)")
print(f"SMC patterns found:    {smc_passed:6d} ({smc_passed/signals_generated*100:5.1f}%)")
print(f"Trades executed:       {trades:6d} ({trades/signals_generated*100:5.1f}%)")
print(f"{'='*60}")

# Interpretation
print(f"\nBottleneck analysis:")
print(f"  Main filter: ", end='')
if hsmm_passed < signals_generated * 0.5:
    print("SdC (HSMM confidence)")
elif stability_passed < signals_generated * 0.5:
    print("Stability_4H")
elif alignment_passed < signals_generated * 0.5:
    print("Alignment_15M (SMC patterns)")
else:
    print("Other conditions (RR, risk, macro)")
