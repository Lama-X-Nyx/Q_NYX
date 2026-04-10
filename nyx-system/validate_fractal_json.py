"""
Validate fractal_check JSON schema
"""
import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
import tempfile
import yaml

# Load config
with open('config/validation_baseline.yaml') as f:
    config = yaml.safe_load(f)

# Run fractal_check
from scripts.fractal_check import run_fractal_check

with tempfile.TemporaryDirectory() as tmpdir:
    temp_dir = Path(tmpdir)
    
    print("Running fractal_check...")
    result = run_fractal_check('BTCUSDT', config, temp_dir, sample_bars=500)
    
    # Load JSON
    json_path = temp_dir / 'fractal' / 'BTCUSDT_fractal_check.json'
    with open(json_path) as f:
        report = json.load(f)
    
    print("\n✅ Validating JSON schema...")
    
    # Check top-level fields
    assert 'bars_ready_for_decision' in report, "Missing bars_ready_for_decision"
    assert 'bars_skipped_due_to_readiness' in report, "Missing bars_skipped_due_to_readiness"
    assert 'diagnostics' in report, "Missing diagnostics"
    print("✅ Top-level fields OK")
    
    # Check diagnostics structure
    diag = report['diagnostics']
    
    assert 'readiness_blocks' in diag, "Missing readiness_blocks"
    readiness = diag['readiness_blocks']
    assert 'context' in readiness, "Missing readiness_blocks.context"
    assert 'regime' in readiness, "Missing readiness_blocks.regime"
    assert 'setup' in readiness, "Missing readiness_blocks.setup"
    print("✅ readiness_blocks structure OK")
    
    assert 'logic_blocks' in diag, "Missing logic_blocks"
    logic = diag['logic_blocks']
    assert 'context' in logic, "Missing logic_blocks.context"
    assert 'regime' in logic, "Missing logic_blocks.regime"
    assert 'setup' in logic, "Missing logic_blocks.setup"
    assert 'risk' in logic, "Missing logic_blocks.risk"
    print("✅ logic_blocks structure OK")
    
    assert 'actions' in diag, "Missing actions"
    actions = diag['actions']
    assert 'approved_buy' in actions, "Missing actions.approved_buy"
    assert 'approved_sell' in actions, "Missing actions.approved_sell"
    assert 'wait_neutral' in actions, "Missing actions.wait_neutral"
    print("✅ actions structure OK")
    
    # Check sample decisions have readiness fields
    assert 'sample_decisions' in report, "Missing sample_decisions"
    if len(report['sample_decisions']) > 0:
        decision = report['sample_decisions'][0]
        assert 'components' in decision, "Missing components in decision"
        
        for agent_name, agent_result in decision['components'].items():
            assert 'ready' in agent_result, f"Missing ready in {agent_name}"
            assert 'blocked_by_readiness' in agent_result, f"Missing blocked_by_readiness in {agent_name}"
        
        print("✅ sample_decisions have readiness fields")
    
    print("\n" + "="*60)
    print("✅ ALL VALIDATIONS PASSED")
    print("="*60)
    print(f"\nJSON Schema Summary:")
    print(f"  bars_ready_for_decision: {report['bars_ready_for_decision']}")
    print(f"  bars_skipped_due_to_readiness: {report['bars_skipped_due_to_readiness']}")
    print(f"  readiness_blocks: {diag['readiness_blocks']}")
    print(f"  logic_blocks: {diag['logic_blocks']}")
    print(f"  actions: {diag['actions']}")
