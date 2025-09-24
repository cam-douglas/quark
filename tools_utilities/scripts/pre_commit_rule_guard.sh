#!/bin/bash
# Quark Validation Pre-commit Hook

echo "🔍 Running pre-commit validation..."

# Run quick validation
python state/todo/core/validate_launcher.py validate

if [ $? -ne 0 ]; then
    echo "❌ Validation failed. Please fix issues before committing."
    exit 1
fi

# Run rules validation
python state/todo/core/validate_launcher.py rules

if [ $? -ne 0 ]; then
    echo "❌ Rules validation failed. Please fix issues before committing."
    echo "Hint: Run 'make validate-sync' to sync rules between cursor and quark"
    exit 1
fi

echo "✅ Pre-commit validation passed"
exit 0