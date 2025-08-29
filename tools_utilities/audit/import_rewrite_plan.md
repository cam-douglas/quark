# Phase-2 — Import-Rewrite & Move Strategy

This document explains **how we will automatically update Python imports** and move files according to the path-mapping draft, while guaranteeing zero-breakage.

---
## 1.  Move Model

We will execute moves in **three buckets** so that test failures are easier to locate:

| Bucket | Current roots                                   | New root               |
|--------|-------------------------------------------------|------------------------|
| A      | brain_architecture/**, brain_modules/**          | /brain/**              |
| B      | ml_architecture/**, integration/**, training/** | /ml/**                 |
| C      | Remaining: tools_utilities/, testing/, etc.     | /utilities/, /tests/…  |

Each bucket will be rewritten and tested separately with a `--dry-run` flag before committing.

---
## 2.  Import Update Tooling

• **rope-based batch refactor** for simple `import xxx` → `import brain.xxx` changes.  
• **bowler** for pattern-based rewrites when aliasing (`from brain_architecture.xxx import Y`).  
• **Fallback regex** for unconventional dynamic imports (will be flagged for manual review).

Scripts to be generated:

| Script | Purpose |
|--------|---------|
| `tools_utilities/audit/rewrite_bucket.py` | Run rope & bowler on a single bucket (args: mapping CSV subset) |
| `tools_utilities/audit/move_bucket.py`    | Physically move files after imports are rewritten |
| `tools_utilities/audit/dry_run.py`        | Execute rewrite + move into a temp dir and run tests |

---
## 2.  Bucket-by-Bucket Workflow Checklist

| Step | Command | Expected Output |
|------|---------|-----------------|
|1|`git checkout -b reorg_phase2`| new branch created |
|2|`python tools_utilities/audit/dry_run.py A`| temp clone, tests green |
|3|`python tools_utilities/audit/rewrite_bucket.py audit_outputs/mapping_bucket_A.csv --confirm`| import rewrite summary |
|4|`python tools_utilities/audit/move_bucket.py audit_outputs/mapping_bucket_A.csv --confirm`| file moves logged |
|5|`pytest -q`| all tests pass |
|6|`git add -A && git commit -m "feat(reorg): apply bucket A moves"`| commit |
|7|Repeat steps 2-6 for buckets B then C| |
|8|`git push origin reorg_phase2 && open PR`| ready for review |

---
## 3.  Safety Nets
1. **Clean-tree guard** – scripts abort if `git status --porcelain` isn’t empty.
2. **Automatic backups** – `move_bucket.py` leaves a copy under `.backup_reorg/<bucket>/` when `--confirm` is used.
3. **Failure gate** – if `pytest` fails, scripts print red banner and exit, no further buckets run until issue fixed.
4. **Log files** – every rewrite/move writes CSV + human log to `audit_outputs/phase2_logs/`.

---
## 4.  Roll-back Procedure
If any stage fails after commits:
```bash
git checkout reorg_phase2
git reset --hard HEAD~1   # or earlier good commit
git clean -fd
git push --force-with-lease origin reorg_phase2  # if branch already pushed
```
If failure occurs during dry-run before commit just delete temp dir `/tmp/quark_reorg_*`.

---
## 5.  Human-in-the-Loop Gates
• Scripts require `--confirm` flag to write changes – prevents accidental edits.
• After each dry-run, developer reviews `phase2_logs/<bucket>_report.txt` and explicitly reruns with `--confirm`.
• Final PR requires approval from **Project Orchestrator** and **QA & Reproducibility Engineer** roles before merge.

---
## 4.  Link Rewriter (Phase-3 preview)

A separate `link_updater.py` will walk *.md / *.rst / *.ipynb metadata and update relative links based on the final move map.

---
## 5.  Next Steps

1. Generate `rewrite_bucket.py` skeleton with mapping-loader.  
2. Produce per-bucket mapping CSVs (`mapping_bucket_A.csv`, …).  
3. Dry-run Bucket A and collect test report.  
4. Iterate through remaining buckets.

### Rope command template
```
rope -pr <repo_root> --rename brain_architecture brain.architecture
```

### Bowler example
```
python -m bowler -m ql -f 'from brain_architecture.' -f 'import brain_architecture.' \
    -q 'merge' --touch --write-mapping mapping.txt -w
```

Rollback: `git reset --hard && git clean -fd`
