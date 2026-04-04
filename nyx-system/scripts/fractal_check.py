"""
Fractal Check Mode

Test fractal agent architecture on sample data.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, '.')

from src.data.mtf_loader import load_fractal_context
from src.agents.orchestrator import Orchestrator


def run_fractal_check(pair: str, config: dict, output_dir: Path, sample_date: str = None):
    """
    Run fractal architecture validation
    
    Tests 3-agent baseline (Context, Regime, Setup)
    """
    
    print(f"\n{'='*80}")
    print(f"FRACTAL CHECK - {pair}")
    print(f"{'='*80}")
    
    # Check fractal enabled
    fractal_config = config.get('fractal', {})
    if not fractal_config.get('enabled', False):
        print("❌ Fractal architecture not enabled in config")
        return {'status': 'error', 'reason': 'fractal not enabled'}
    
    # Get active TFs
    timeframes_config = fractal_config.get('timeframes', {})
    context_tf = timeframes_config.get('context_tf', '1d')
    regime_tf = timeframes_config.get('regime_tf', '1h')
    setup_tf = timeframes_config.get('setup_tf', '15m')
    
    print(f"\n📊 Active Fractal Baseline:")
    print(f"  Context TF: {context_tf}")
    print(f"  Regime TF:  {regime_tf}")
    print(f"  Setup TF:   {setup_tf}")
    print(f"  Entry TF:   DEFERRED (5M not yet active)")
    
    # Parse sample date
    if sample_date:
        current_date = datetime.strptime(sample_date, '%Y-%m-%d')
    else:
        # Use latest available data
        current_date = datetime(2024, 1, 1)
    
    print(f"\n📊 Loading fractal context (reference: {current_date.strftime('%Y-%m-%d')})...")
    
    try:
        mtf_data = load_fractal_context(pair, current_date, config)
    except Exception as e:
        print(f"❌ Failed to load fractal context: {e}")
        return {'status': 'error', 'reason': str(e)}
    
    print(f"✓ Loaded {len(mtf_data)} timeframes:")
    for tf, df in mtf_data.items():
        print(f"  {tf}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Verify required TFs present
    required_tfs = [context_tf, regime_tf, setup_tf]
    missing = [tf for tf in required_tfs if tf not in mtf_data]
    
    if missing:
        print(f"❌ Missing required timeframes: {missing}")
        return {'status': 'error', 'reason': f'missing TFs: {missing}'}
    
    # Initialize Orchestrator
    print(f"\n🚀 Initializing Fractal Orchestrator...")
    orchestrator = Orchestrator(config)
    
    # Run through sample
    print(f"\n🔬 Testing fractal pipeline...")
    
    decisions = []
    warmup = 50  # Reduced from 100 to work with 1-day 15M window
    
    # Track diagnostics
    bars_ready_for_decision = 0
    bars_skipped_due_to_readiness = 0
    
    # Readiness blocks
    blocked_by_context_readiness = 0
    blocked_by_regime_readiness = 0
    blocked_by_setup_readiness = 0
    
    # Logic blocks
    blocked_by_context_logic = 0
    blocked_by_regime_logic = 0
    blocked_by_setup_logic = 0
    blocked_by_risk = 0
    
    approved_buy = 0
    approved_sell = 0
    wait_neutral = 0
    
    # Get lowest TF for iteration
    lowest_tf = min(mtf_data.keys(), key=lambda x: _tf_to_minutes(x))
    available_bars = len(mtf_data[lowest_tf])
    
    # Use all available bars (up to reasonable limit for performance)
    max_signals = min(100, available_bars - warmup)
    
    print(f"  Available bars ({lowest_tf}): {available_bars}")
    print(f"  Evaluating {max_signals} signals (after {warmup} bar warmup)...")
    
    for i in range(warmup, warmup + max_signals):
        try:
            current_time = mtf_data[lowest_tf].index[i]
            
            # Align data at this timestamp
            from src.data.mtf_loader import MTFLoader
            loader = MTFLoader('data/raw/mtf')
            aligned_data = loader.align_at_timestamp(mtf_data, current_time, lowest_tf)
            
            # Get decision
            decision = orchestrator.decide(aligned_data)
            decisions.append(decision)
            
            # Check if pipeline ready (all agents have enough data)
            all_agents_ready = all(
                comp.ready for comp in decision.components.values()
            )
            
            if not all_agents_ready:
                # Not ready - track readiness blocks
                bars_skipped_due_to_readiness += 1
                
                for name, comp in decision.components.items():
                    if not comp.ready:
                        if name == 'context':
                            blocked_by_context_readiness += 1
                        elif name == 'regime':
                            blocked_by_regime_readiness += 1
                        elif name == 'setup':
                            blocked_by_setup_readiness += 1
            else:
                # Pipeline ready - count as valid decision
                bars_ready_for_decision += 1
                
                # Track logic blocks or approvals
                if decision.action == 'WAIT':
                    # Check if blocked by logic or just neutral
                    if 'context' in decision.blocked_by:
                        blocked_by_context_logic += 1
                    if 'regime' in decision.blocked_by:
                        blocked_by_regime_logic += 1
                    if 'setup' in decision.blocked_by:
                        blocked_by_setup_logic += 1
                    if 'risk' in decision.blocked_by:
                        blocked_by_risk += 1
                    if not decision.blocked_by:
                        wait_neutral += 1
                elif decision.action == 'BUY':
                    approved_buy += 1
                elif decision.action == 'SELL':
                    approved_sell += 1
                
        except Exception as e:
            # Skip errors
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"FRACTAL DIAGNOSTIC RESULTS")
    print(f"{'='*60}")
    print(f"Bars analyzed:               {len(decisions)}")
    print(f"Bars ready for decision:     {bars_ready_for_decision}")
    print(f"Bars skipped (readiness):    {bars_skipped_due_to_readiness}")
    
    print(f"\nBlocked by readiness:        {bars_skipped_due_to_readiness}")
    if bars_skipped_due_to_readiness > 0:
        print(f"  Context readiness:         {blocked_by_context_readiness:6d}")
        print(f"  Regime readiness:          {blocked_by_regime_readiness:6d}")
        print(f"  Setup readiness:           {blocked_by_setup_readiness:6d}")
    
    if bars_ready_for_decision > 0:
        print(f"\nBlocked by logic:            {blocked_by_context_logic + blocked_by_regime_logic + blocked_by_setup_logic}")
        print(f"  Context logic:             {blocked_by_context_logic:6d}")
        print(f"  Regime logic:              {blocked_by_regime_logic:6d}")
        print(f"  Setup logic:               {blocked_by_setup_logic:6d}")
        print(f"  Risk:                      {blocked_by_risk:6d}")
    
    print(f"\nActions:")
    print(f"  Approved BUY:              {approved_buy:6d}")
    print(f"  Approved SELL:             {approved_sell:6d}")
    print(f"  Wait (neutral):            {wait_neutral:6d}")
    print(f"{'='*60}")
    
    # Bottleneck analysis
    if bars_skipped_due_to_readiness > len(decisions) * 0.5:
        print(f"\n⚠️  WARNING: {bars_skipped_due_to_readiness/len(decisions)*100:.0f}% of bars skipped due to readiness.")
        print(f"   Main issue: Not enough data loaded.")
        if blocked_by_context_readiness > 0:
            print(f"   → Context needs more {context_tf} data (need {fractal_config.get('fractal_readiness', {}).get('context_min_bars', 50)} bars)")
        if blocked_by_regime_readiness > 0:
            print(f"   → Regime needs more {regime_tf} data (need {fractal_config.get('fractal_readiness', {}).get('regime_min_bars', 100)} bars)")
        if blocked_by_setup_readiness > 0:
            print(f"   → Setup needs more {setup_tf} data (need {fractal_config.get('fractal_readiness', {}).get('setup_min_bars', 50)} bars)")
        print(f"   Action: Increase --sample-bars or use longer period")
    elif bars_ready_for_decision > 0:
        print(f"\nBottleneck analysis (on ready bars):")
        logic_blocks = [
            ('Context', blocked_by_context_logic),
            ('Regime', blocked_by_regime_logic),
            ('Setup', blocked_by_setup_logic),
            ('Risk', blocked_by_risk)
        ]
        max_block = max(logic_blocks, key=lambda x: x[1])
        if max_block[1] > 0:
            print(f"  Main bottleneck: {max_block[0].upper()} ({max_block[1]} blocks)")
        else:
            print(f"  No major bottlenecks detected")
    
    # Sample decisions
    print(f"\nSample decisions (first 3):")
    for i, decision in enumerate(decisions[:3]):
        print(f"\n  Decision {i+1}:")
        print(f"    Action: {decision.action}")
        print(f"    Blocked by: {decision.blocked_by if decision.blocked_by else 'None'}")
        for name, result in decision.components.items():
            print(f"    {name}: {result.state} (passed={result.passed}, score={result.score:.2f})")
    
    # Save JSON
    output_dir = Path(output_dir) / 'fractal'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{pair}_fractal_check.json"
    
    report = {
        'pair': pair,
        'reference_date': current_date.isoformat(),
        'signals_evaluated': len(decisions),
        'bars_ready_for_decision': bars_ready_for_decision,
        'bars_skipped_due_to_readiness': bars_skipped_due_to_readiness,
        'active_tfs': {
            'context': context_tf,
            'regime': regime_tf,
            'setup': setup_tf,
            'entry': 'DEFERRED'
        },
        'readiness_requirements': config.get('fractal_readiness', {}),
        'diagnostics': {
            'readiness_blocks': {
                'context': blocked_by_context_readiness,
                'regime': blocked_by_regime_readiness,
                'setup': blocked_by_setup_readiness
            },
            'logic_blocks': {
                'context': blocked_by_context_logic,
                'regime': blocked_by_regime_logic,
                'setup': blocked_by_setup_logic,
                'risk': blocked_by_risk
            },
            'actions': {
                'approved_buy': approved_buy,
                'approved_sell': approved_sell,
                'wait_neutral': wait_neutral
            }
        },
        'sample_decisions': [d.to_dict() for d in decisions[:5]]
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    
    return {
        'status': 'completed',
        'pair': pair,
        'signals': len(decisions),
        'output': str(report_path)
    }


def _tf_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes"""
    mapping = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    return mapping.get(tf, 1440)


if __name__ == "__main__":
    import yaml
    
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    run_fractal_check('BTCUSDT', config, Path('reports/validation'), sample_bars=1000)
