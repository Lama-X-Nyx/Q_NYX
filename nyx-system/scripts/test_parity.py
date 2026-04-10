"""
Simple Parity Validation

Tests that cached runner produces identical decisions to uncached orchestrator.
"""

import sys
sys.path.insert(0, '.')

import yaml
from datetime import datetime
from pathlib import Path

from src.data.mtf_loader import load_fractal_context
from src.core.fractal_cached_runner import FractalCachedRunner
from src.agents.orchestrator import Orchestrator


def test_parity():
    """Test parity between cached and uncached pipelines"""
    
    print("="*80)
    print("PARITY VALIDATION TEST")
    print("="*80)
    
    # Load config
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load real data
    print("\n1. Loading real data...")
    current_date = datetime(2023, 12, 15)
    mtf_data = load_fractal_context('BTCUSDT', current_date, config)
    print(f"   ✓ Loaded {len(mtf_data)} timeframes")
    
    # Test uncached
    print("\n2. Testing uncached orchestrator...")
    orchestrator = Orchestrator(config)
    uncached = orchestrator.decide(mtf_data)
    
    print(f"   Action:      {uncached.action}")
    print(f"   Score:       {uncached.score:.4f}")
    print(f"   Reason:      {uncached.reason}")
    print(f"   Blocked by:  {uncached.blocked_by}")
    
    # Test cached
    print("\n3. Testing cached runner...")
    cached_runner = FractalCachedRunner(config)
    cached = cached_runner.decide(mtf_data)
    
    print(f"   Action:      {cached.action}")
    print(f"   Score:       {cached.score:.4f}")
    print(f"   Reason:      {cached.reason}")
    print(f"   Blocked by:  {cached.blocked_by}")
    
    # Compare
    print("\n" + "="*80)
    print("PARITY CHECK")
    print("="*80)
    
    action_match = uncached.action == cached.action
    score_diff = abs(uncached.score - cached.score)
    score_match = score_diff < 0.001
    reason_match = uncached.reason == cached.reason
    blocked_match = uncached.blocked_by == cached.blocked_by
    
    print(f"\nAction match:      {'✅' if action_match else '❌'}")
    if not action_match:
        print(f"  Uncached: {uncached.action}")
        print(f"  Cached:   {cached.action}")
    
    print(f"\nScore match:       {'✅' if score_match else '❌'}")
    if not score_match:
        print(f"  Uncached: {uncached.score:.6f}")
        print(f"  Cached:   {cached.score:.6f}")
        print(f"  Diff:     {score_diff:.6f}")
    
    print(f"\nReason match:      {'✅' if reason_match else '❌'}")
    if not reason_match:
        print(f"  Uncached: {uncached.reason}")
        print(f"  Cached:   {cached.reason}")
    
    print(f"\nBlocked_by match:  {'✅' if blocked_match else '❌'}")
    if not blocked_match:
        print(f"  Uncached: {uncached.blocked_by}")
        print(f"  Cached:   {cached.blocked_by}")
    
    # Overall verdict
    print("\n" + "="*80)
    if action_match and score_match and reason_match and blocked_match:
        print("✅ PERFECT PARITY - All fields match!")
        print("="*80)
        return True
    else:
        print("❌ PARITY MISMATCH - Some fields differ")
        print("="*80)
        return False


def test_multiple_calls():
    """Test parity maintained across multiple calls"""
    
    print("\n" + "="*80)
    print("MULTIPLE CALLS PARITY TEST")
    print("="*80)
    
    # Load config and data
    with open('config/validation_baseline.yaml') as f:
        config = yaml.safe_load(f)
    
    current_date = datetime(2023, 12, 15)
    mtf_data = load_fractal_context('BTCUSDT', current_date, config)
    
    orchestrator = Orchestrator(config)
    cached_runner = FractalCachedRunner(config)
    
    print("\nTesting 3 sequential calls...")
    
    all_match = True
    
    for i in range(3):
        uncached = orchestrator.decide(mtf_data)
        cached = cached_runner.decide(mtf_data)
        
        action_match = uncached.action == cached.action
        score_match = abs(uncached.score - cached.score) < 0.001
        reason_match = uncached.reason == cached.reason
        blocked_match = uncached.blocked_by == cached.blocked_by
        
        call_match = action_match and score_match and reason_match and blocked_match
        
        print(f"\nCall {i+1}: {'✅' if call_match else '❌'}")
        if not call_match:
            print(f"  Action:     {uncached.action} vs {cached.action} {'✅' if action_match else '❌'}")
            print(f"  Score:      {uncached.score:.4f} vs {cached.score:.4f} {'✅' if score_match else '❌'}")
            print(f"  Reason:     {'✅' if reason_match else '❌'}")
            print(f"  Blocked_by: {'✅' if blocked_match else '❌'}")
        
        all_match = all_match and call_match
    
    print("\n" + "="*80)
    if all_match:
        print("✅ ALL CALLS MATCHED - Parity maintained across calls")
    else:
        print("❌ SOME CALLS MISMATCHED")
    print("="*80)
    
    return all_match


if __name__ == "__main__":
    # Run both tests
    test1_pass = test_parity()
    test2_pass = test_multiple_calls()
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Single call parity:   {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Multiple call parity: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print("="*80)
    
    if test1_pass and test2_pass:
        print("\n✅ ALL PARITY TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ PARITY TESTS FAILED")
        sys.exit(1)
