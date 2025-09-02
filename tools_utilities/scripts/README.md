# scripts

Path: `tools_utilities/scripts`

Purpose: Helper utilities that support Quark development: training/finetuning entry-points, S3 sync & resource-management tooling, roadmap maintenance, and CI helpers.

## Subdirectories
- __pycache__/

## Key Scripts
| File | Category | Purpose |
|------|----------|---------|
| `aws_utils.py` | Cloud | Bootstrap SageMaker IAM role & ECR image |
| `check_quark_state.py` | Info | Quick status snapshot from state docs |
| `dataset_discovery.py` | Data | Locate datasets / shard groups in S3 |
| `eval_streaming.py` | Training | Lightweight evaluation pass on a val split |
| `finetune_streaming.py` | Training | Continue training from a checkpoint |
| `generate_repo_index.py` | Indexing | Build/refresh top-level `repo_index.json` |
| `generate_roadmap_index.py` | Indexing | (Manual) build a comprehensive roadmap index |
| `integrate_cli.py` | Dev UX | Helper for integrating Quark into other CLIs |
| `link_validate.py` | Docs | Fail-fast check for broken relative links |
| `nightly_repo_index.py` | CI | Cron-style index generation (invoked manually now) |
| `pipeline_orchestrator.py` | Training | Policy-aware dispatcher for train/finetune jobs |
| `pre_push_update.py` | Maintenance | Rebuild `master_roadmap.md`, validate links, sync YAML tasks, refresh README |
| `quark_cli.py` | Dev UX | Natural-language command wrapper ("train quark" etc.) |
| `quark_rm_cli.py` | Data | Approve/blacklist resources via ResourceManager |
| `s3_sync.py` | Data | Sync local `data/` with S3 (called by pre-push hook) |
| `serial_pytest_runner.py` | CI | Per-file timeout runner for flaky tests |
| `train_streaming.py` | Training | Main streaming-dataset training stub |
| `update_master_roadmap.py` | Docs | Snapshot copy helper used by hook |
| `Dockerfile` | Cloud | Base image for SageMaker/cloud jobs |

## Related Links
* [Root README](../../README.md)
* [Repository Index](../../repo_index.json)

---

## Training & Fine-tuning Scripts

| Script | Purpose | Typical Local Run | Cloud Run with Auto-AWS |
|--------|---------|-------------------|-------------------------|
| `train_streaming.py` | Main training entry-point; streams dataset shards or local files. | `python train_streaming.py --config <cfg>.yaml` | `python train_streaming.py --config <cfg>.yaml --backend cloud --deploy` |
| `finetune_streaming.py` | Continues training from a checkpoint with lower LR. | `python finetune_streaming.py --config <cfg>.yaml --checkpoint ckpt.pt` | `python finetune_streaming.py --config <cfg>.yaml --backend cloud --deployment` |
| `pipeline_orchestrator.py` | High-level entry that reads `training_pipeline.yaml` and then spawns `train_streaming.py` or `finetune_streaming.py`. Natural-language backends/deploy flags still work. | `python pipeline_orchestrator.py train cloud` | `python pipeline_orchestrator.py finetune local --checkpoint ckpt.pt` |

### Natural-language Backend Selector
* Local synonyms: `local`, `locally`, `localhost`  → runs inside Cursor.
* Cloud synonyms: `cloud`, `stream`, `streaming`, `sagemaker`, `remote` → dispatches to AWS SageMaker.

### Deploy Flag
Use `--deploy`, `--deployment`, or `--deploy-endpoint` (boolean) to automatically stand up a SageMaker inference endpoint after the training job completes.

### Auto AWS Setup (`aws_utils.py`)
On first cloud invocation the helper will:
1. Create/attach IAM role `quark-sagemaker-role` with required policies.
2. Build & push the local `Dockerfile` to ECR repository `quark-brain-train` (if present). Otherwise it defaults to SageMaker’s PyTorch image.
3. Launch the training or fine-tuning job and print the job name as JSON.

No manual console work is required after AWS credentials are configured (`aws configure`).
