"""
Fractal Bottleneck Discovery

Identify the real business bottleneck in the fractal pipeline after warmup.
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


def run_fractal_bottleneck(pair: str, config: dict, output_dir: Path, sample_date: str = ""):
    """
    Run fractal bottleneck analysis
    
    Identifies which agent is the primary blocker after warmup phase.
    """
    
    print(f"\n{'='*80}")
    print(f"FRACTAL BOTTLENECK DISCOVERY - {pair}")
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
    
    # Initialize Orchestrator
    print(f"\n🚀 Initializing Fractal Orchestrator...")
    orchestrator = Orchestrator(config)
    
    # Run through sample
    print(f"\n🔬 Analyzing fractal bottleneck...")
    
    decisions = []
    warmup = 50  # Reduced to work with 1-day 15M window
    
    # Track overall
    bars_ready_for_decision = 0
    bars_skipped_due_to_readiness = 0
    
    # Track readiness blocks
    readiness_blocks_context = 0
    readiness_blocks_regime = 0
    readiness_blocks_setup = 0
    
    # Track logic blocks (on ready bars only)
    logic_blocks_context = 0
    logic_blocks_regime = 0
    logic_blocks_setup = 0
    logic_blocks_risk = 0
    
    # Track agent passes (on ready bars only)
    context_passed_count = 0
    regime_passed_count = 0
    setup_passed_count = 0
    
    # Track actions
    approved_buy = 0
    approved_sell = 0
    wait_neutral = 0
    
    # Sample blocked decisions for audit
    sample_blocked = []
    
    # Get lowest TF for iteration
    lowest_tf = min(mtf_data.keys(), key=lambda x: _tf_to_minutes(x))
    n_signals = len(mtf_data[lowest_tf]) - warmup
    
    print(f"  Evaluating {n_signals} signals (after {warmup} bar warmup)...")
    
    for i in range(warmup, warmup + n_signals):
        try:
            current_time = mtf_data[lowest_tf].index[i]
            
            # Align data at this timestamp
            from src.data.mtf_loader import MTFLoader
            loader = MTFLoader('data/raw/mtf')
            aligned_data = loader.align_at_timestamp(mtf_data, current_time, lowest_tf)
            
            # Get decision
            decision = orchestrator.decide(aligned_data)
            decisions.append(decision)
            
            # Check if all agents ready
            all_agents_ready = all(
                comp.ready for comp in decision.components.values()
            )
            
            if not all_agents_ready:
                # Readiness block
                bars_skipped_due_to_readiness += 1
                
                for name, comp in decision.components.items():
                    if not comp.ready:
                        if name == 'context':
                            readiness_blocks_context += 1
                        elif name == 'regime':
                            readiness_blocks_regime += 1
                        elif name == 'setup':
                            readiness_blocks_setup += 1
            else:
                # Pipeline ready - count this as evaluable
                bars_ready_for_decision += 1
                
                # Track agent passes
                if decision.components['context'].passed:
                    context_passed_count += 1
                if decision.components['regime'].passed:
                    regime_passed_count += 1
                if decision.components['setup'].passed:
                    setup_passed_count += 1
                
                # Track logic blocks
                if not decision.components['context'].passed:
                    logic_blocks_context += 1
                if not decision.components['regime'].passed:
                    logic_blocks_regime += 1
                if not decision.components['setup'].passed:
                    logic_blocks_setup += 1
                if 'risk' in decision.blocked_by:
                    logic_blocks_risk += 1
                
                # Track actions
                if decision.action == 'BUY':
                    approved_buy += 1
                elif decision.action == 'SELL':
                    approved_sell += 1
                elif decision.action == 'WAIT' and not decision.blocked_by:
                    wait_neutral += 1
                
                # Sample blocked decisions
                if decision.action == 'WAIT' and decision.blocked_by and len(sample_blocked) < 10:
                    sample_blocked.append({
                        'timestamp': str(current_time),
                        'blocked_by': decision.blocked_by,
                        'reason': decision.reason,
                        'components': {
                            name: {
                                'state': comp.state,
                                'passed': comp.passed,
                                'score': comp.score
                            }
                            for name, comp in decision.components.items()
                        }
                    })
                    
        except Exception as e:
            # Skip errors
            continue
    
    # Calculate ready-only metrics
    if bars_ready_for_decision > 0:
        context_pass_rate = (context_passed_count / bars_ready_for_decision) * 100
        regime_pass_rate = (regime_passed_count / bars_ready_for_decision) * 100
        setup_pass_rate = (setup_passed_count / bars_ready_for_decision) * 100
        final_trade_rate = ((approved_buy + approved_sell) / bars_ready_for_decision) * 100
    else:
        context_pass_rate = 0.0
        regime_pass_rate = 0.0
        setup_pass_rate = 0.0
        final_trade_rate = 0.0
    
    # Bottleneck ranking
    bottleneck_ranking = sorted([
        {'agent': 'context', 'blocks': logic_blocks_context},
        {'agent': 'regime', 'blocks': logic_blocks_regime},
        {'agent': 'setup', 'blocks': logic_blocks_setup},
        {'agent': 'risk', 'blocks': logic_blocks_risk}
    ], key=lambda x: x['blocks'], reverse=True)
    
    primary_bottleneck = str(bottleneck_ranking[0]['agent']) if int(bottleneck_ranking[0]['blocks']) > 0 else 'none'
    
    # Summary
    print(f"\n{'='*60}")
    print(f"FRACTAL BOTTLENECK RESULTS")
    print(f"{'='*60}")
    print(f"Bars analyzed:               {len(decisions)}")
    print(f"Bars ready for decision:     {bars_ready_for_decision}")
    print(f"Bars skipped (readiness):    {bars_skipped_due_to_readiness}")
    
    if bars_ready_for_decision > 0:
        print(f"\nREADY-ONLY PASS RATES")
        print(f"Context pass rate:           {context_pass_rate:5.1f}%")
        print(f"Regime pass rate:            {regime_pass_rate:5.1f}%")
        print(f"Setup pass rate:             {setup_pass_rate:5.1f}%")
        print(f"Final trade rate:            {final_trade_rate:5.1f}%")
        
        print(f"\nLOGIC BLOCKS (Ready Bars Only)")
        print(f"Context blocked:             {logic_blocks_context:6d}")
        print(f"Regime blocked:              {logic_blocks_regime:6d}")
        print(f"Setup blocked:               {logic_blocks_setup:6d}")
        print(f"Risk blocked:                {logic_blocks_risk:6d}")
    
    print(f"\nACTIONS")
    print(f"BUY approved:                {approved_buy:6d}")
    print(f"SELL approved:               {approved_sell:6d}")
    print(f"WAIT neutral:                {wait_neutral:6d}")
    
    if bars_ready_for_decision > 0:
        print(f"\nPRIMARY BOTTLENECK: {primary_bottleneck.upper()}")
    else:
        print(f"\n⚠️  WARNING: No bars ready for decision")
        print(f"   Increase --sample-bars or use longer period")
    
    print(f"{'='*60}")
    
    # Save JSON
    output_dir = Path(output_dir) / 'fractal'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{pair}_fractal_bottleneck.json"
    
    report = {
        'pair': pair,
        'reference_date': current_date.isoformat(),
        'bars_analyzed': len(decisions),
        'bars_ready_for_decision': bars_ready_for_decision,
        'bars_skipped_due_to_readiness': bars_skipped_due_to_readiness,
        'active_agents': ['context', 'regime', 'setup'],
        'diagnostics': {
            'readiness_blocks': {
                'context': readiness_blocks_context,
                'regime': readiness_blocks_regime,
                'setup': readiness_blocks_setup
            },
            'logic_blocks': {
                'context': logic_blocks_context,
                'regime': logic_blocks_regime,
                'setup': logic_blocks_setup,
                'risk': logic_blocks_risk
            },
            'actions': {
                'approved_buy': approved_buy,
                'approved_sell': approved_sell,
                'wait_neutral': wait_neutral
            }
        },
        'ready_only_summary': {
            'context_pass_rate': round(context_pass_rate, 1),
            'regime_pass_rate': round(regime_pass_rate, 1),
            'setup_pass_rate': round(setup_pass_rate, 1),
            'final_trade_rate': round(final_trade_rate, 1),
            'primary_bottleneck': primary_bottleneck
        },
        'bottleneck_ranking': bottleneck_ranking,
        'sample_blocked_decisions': sample_blocked
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    
    return {
        'status': 'completed',
        'pair': pair,
        'bars_ready': bars_ready_for_decision,
        'primary_bottleneck': primary_bottleneck,
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
    
    run_fractal_bottleneck('BTCUSDT', config, Path('reports/validation'))
