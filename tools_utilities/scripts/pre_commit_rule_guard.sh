#!/bin/bash
# Pre-commit hook to check for overconfident language in staged files

echo "🔍 Running anti-overconfidence validation..."

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|md|txt)$')

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

# Check each file for overconfident patterns
VALIDATION_FAILED=0

for FILE in $STAGED_FILES; do
    # Check for forbidden patterns
    if grep -qE '(100%|absolutely certain|definitely works|guaranteed to|never fails|always works)' "$FILE"; then
        echo "❌ Overconfident language detected in $FILE"
        echo "   Please revise to include uncertainty markers and confidence levels ≤90%"
        VALIDATION_FAILED=1
    fi
    
    # Check for unsourced claims in markdown files
    if [[ "$FILE" == *.md ]]; then
        if grep -qE '(increases? performance|proven to|studies show|research indicates)' "$FILE"; then
            if ! grep -qE '\[.*\]|\(https?://|Source:|Reference:' "$FILE"; then
                echo "⚠️  Unsourced claims detected in $FILE"
                echo "   Please add citations for performance or research claims"
            fi
        fi
    fi
done

if [ $VALIDATION_FAILED -eq 1 ]; then
    echo ""
    echo "📋 Required elements for all technical content:"
    echo "  • Explicit confidence percentage (≤90%)"
    echo "  • Source citations for claims"
    echo "  • Uncertainty acknowledgment where appropriate"
    echo "  • No absolute language (always/never/definitely/guaranteed)"
    echo ""
    echo "Run 'python tools_utilities/confidence_validator.py --check <file>' to validate"
    exit 1
fi

echo "✅ Anti-overconfidence validation passed"
exit 0