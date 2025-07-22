#!/bin/bash

# MemMimic Complexity Check Script
# Runs complexity analysis and generates reports

set -e

echo "🔍 Running MemMimic Code Complexity Analysis"
echo "============================================="

# Configuration
CONFIG_FILE="scripts/complexity_config.json"
TARGET_DIR="src/memmimic/memory/"
OUTPUT_DIR="reports/complexity"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ Target directory not found: $TARGET_DIR"
    exit 1
fi

echo "📁 Analyzing directory: $TARGET_DIR"
echo "⚙️  Using configuration: $CONFIG_FILE"
echo ""

# Generate JSON report for automation
echo "📊 Generating JSON report..."
python scripts/complexity_monitor.py "$TARGET_DIR" \
    --config "$CONFIG_FILE" \
    --recursive \
    --format json \
    --output "$OUTPUT_DIR/complexity_report_${TIMESTAMP}.json"

# Generate human-readable report
echo "📋 Generating text report..."
python scripts/complexity_monitor.py "$TARGET_DIR" \
    --config "$CONFIG_FILE" \
    --recursive \
    --format text \
    --output "$OUTPUT_DIR/complexity_report_${TIMESTAMP}.txt"

# Create latest symlinks
ln -sf "complexity_report_${TIMESTAMP}.json" "$OUTPUT_DIR/latest.json"
ln -sf "complexity_report_${TIMESTAMP}.txt" "$OUTPUT_DIR/latest.txt"

echo ""
echo "✅ Complexity analysis completed!"
echo "📄 Reports generated:"
echo "   JSON: $OUTPUT_DIR/complexity_report_${TIMESTAMP}.json"
echo "   Text: $OUTPUT_DIR/complexity_report_${TIMESTAMP}.txt"
echo "   Latest: $OUTPUT_DIR/latest.{json,txt}"

# Display summary
echo ""
echo "📈 COMPLEXITY SUMMARY:"
echo "====================="

# Extract summary from JSON report using Python
python3 -c "
import json
import sys

try:
    with open('$OUTPUT_DIR/complexity_report_${TIMESTAMP}.json', 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    print(f'Total files analyzed: {summary[\"files_analyzed\"]}')
    print(f'Total functions: {summary[\"total_functions\"]}')
    print(f'Total classes: {summary[\"total_classes\"]}')
    print(f'High complexity items: {summary[\"high_complexity_count\"]}')
    
    if summary['high_complexity_count'] > 0:
        print('')
        print('⚠️  HIGH COMPLEXITY ITEMS DETECTED!')
        print('Consider refactoring these functions/classes:')
        for item in data['high_complexity_items'][:5]:  # Show first 5
            print(f'  - {item[\"name\"]} ({item[\"type\"]}) - Score: {item[\"complexity_score\"]:.1f}')
        
        if len(data['high_complexity_items']) > 5:
            print(f'  ... and {len(data[\"high_complexity_items\"]) - 5} more')
        
        sys.exit(1)  # Exit with error code
    else:
        print('')
        print('✅ All complexity within acceptable limits!')
        
except Exception as e:
    print(f'Error reading report: {e}')
    sys.exit(1)
"

echo ""
echo "🔄 To run this check regularly, add to your crontab:"
echo "    0 9 * * * cd /path/to/memmimic && ./scripts/run_complexity_check.sh"