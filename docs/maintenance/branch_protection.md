# Branch-Protection Rules

The `main` branch must have the following protections enabled:

1. **Required status check**: `reorg-safety-check` GitHub Action must pass.
2. **Required reviews**: At least **one** code-owner review before merge.
3. **Linear history**: prevent merge commits (squash or rebase only).
4. **No force-pushes**.
5. **Dismiss stale approvals on new commits**.
