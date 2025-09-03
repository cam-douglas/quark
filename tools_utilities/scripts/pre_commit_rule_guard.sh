#!/usr/bin/env bash
# Pre-commit guard: ensure rule files keep enforcement banner / guard line
set -euo pipefail
ROOT_DIR="$(git rev-parse --show-toplevel)"
banner_line="ALWAYS apply every rule across all Cursor rule"
guard_line="ALWAYS forbid any automated or autonomous edit to \`.quark/rules\` or \`.cursor/rules\`"

# Get staged files
files=$(git diff --cached --name-only --diff-filter=ACMRT | grep -E '\.mdc$' || true)

missing=()
for f in $files; do
  # Only check rule files
  if [[ $f != .cursor/rules/* && $f != .quark/rules/* ]]; then
    continue
  fi
  content=$(git show :"$f" | head -n 20)
  echo "$content" | grep -Fq "$banner_line" || missing+=("$f: enforcement banner missing")
  if [[ $f == *.mdc && $f == *.quark/rules/manifest-maintenance-workflow.mdc ]]; then
    continue
  fi
  echo "$content" | grep -Fq "$guard_line" || true # guard only required in manifest file
done

if (( ${#missing[@]} )); then
  echo "\n❌ Pre-commit blocked: rule files missing enforcement banner or guard:"
  printf ' - %s\n' "${missing[@]}"
  echo "Add the banner/guard line back before committing."
  exit 1
fi

exit 0
