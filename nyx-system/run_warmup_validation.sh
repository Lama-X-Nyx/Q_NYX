#!/bin/bash
# Final warmup validation

echo "=========================================="
echo "WARMUP TICKET - FINAL VALIDATION"
echo "=========================================="
echo ""

# Run fractal_check with small sample (readiness test)
echo "1. Testing readiness blocks (small sample)..."
python scripts/run_validation.py --mode fractal_check --pair BTCUSDT --sample-bars 1000 > /tmp/warmup_small.log 2>&1
if grep -q "Bars skipped (readiness):" /tmp/warmup_small.log; then
    echo "   ✅ Readiness blocks detected"
else
    echo "   ❌ Readiness blocks NOT detected"
    exit 1
fi

# Validate JSON schema
echo ""
echo "2. Validating JSON schema..."
python validate_fractal_json.py > /tmp/warmup_json.log 2>&1
if grep -q "ALL VALIDATIONS PASSED" /tmp/warmup_json.log; then
    echo "   ✅ JSON schema valid"
else
    echo "   ❌ JSON schema invalid"
    cat /tmp/warmup_json.log | tail -20
    exit 1
fi

# Check actual JSON file
echo ""
echo "3. Checking real JSON structure..."
JSON_FILE="reports/validation/fractal/BTCUSDT_fractal_check.json"
if [ -f "$JSON_FILE" ]; then
    # Check for new structure
    if grep -q "readiness_blocks" "$JSON_FILE" && \
       grep -q "logic_blocks" "$JSON_FILE" && \
       grep -q "bars_ready_for_decision" "$JSON_FILE"; then
        echo "   ✅ New JSON structure present"
    else
        echo "   ❌ New JSON structure missing"
        exit 1
    fi
    
    # Check old structure is gone
    if grep -q '"blocked_by_context"' "$JSON_FILE"; then
        echo "   ❌ Old flat structure still present"
        exit 1
    else
        echo "   ✅ Old flat structure removed"
    fi
else
    echo "   ❌ JSON file not found: $JSON_FILE"
    exit 1
fi

# Extract key metrics
echo ""
echo "4. Sample metrics from real run:"
python -c "
import json
with open('$JSON_FILE') as f:
    data = json.load(f)
print(f\"   bars_ready_for_decision: {data['bars_ready_for_decision']}\")
print(f\"   bars_skipped_due_to_readiness: {data['bars_skipped_due_to_readiness']}\")
print(f\"   readiness_blocks: {data['diagnostics']['readiness_blocks']}\")
print(f\"   logic_blocks: {data['diagnostics']['logic_blocks']}\")
"

echo ""
echo "=========================================="
echo "✅ ALL WARMUP VALIDATIONS PASSED"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✅ Readiness tracking works"
echo "  ✅ JSON schema updated"
echo "  ✅ Old structure removed"
echo "  ✅ New structure validated"
echo ""
echo "WARMUP TICKET: COMPLETE ✅"
