#!/usr/bin/env bash
set -euo pipefail

# Check commit size and warn if too large
STAGED_FILES=$(git diff --cached --name-only | wc -l)
STAGED_LINES=$(git diff --cached --numstat | awk '{added+=$1; deleted+=$2} END {print added+deleted}')

echo "📊 Commit size: $STAGED_FILES files, $STAGED_LINES lines changed"

# Warn if commit is very large
if [ "$STAGED_FILES" -gt 50 ]; then
    echo "⚠️  WARNING: Large commit ($STAGED_FILES files)"
    echo "💡 Consider breaking into smaller logical commits"
    echo "Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Commit cancelled. Use 'git reset' to unstage files."
        exit 1
    fi
fi

if [ "$STAGED_LINES" -gt 1000 ]; then
    echo "⚠️  WARNING: Large change ($STAGED_LINES lines)"
    echo "💡 Consider smaller, focused commits"
fi

echo "✅ Proceeding with commit"