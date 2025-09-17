# project

Path: `management/configurations/project`

Purpose: Directory providing source files and/or submodules relevant to its path in the Quark repository.

## Subdirectories
- (none)

## Files
- 1.bug_report.yml
- 2.feature_request.yml
- 3.other-issues.yml
- FUNDING.yml
- FUNDING_1.yml
- FUNDING_2.yml
- MANIFEST.in
- METADATA_1.toml
- METADATA_10.toml
- METADATA_100.toml
- METADATA_101.toml
- METADATA_102.toml
- METADATA_103.toml
- METADATA_104.toml
- METADATA_105.toml
- METADATA_106.toml
- METADATA_107.toml
- METADATA_108.toml
- METADATA_109.toml
- METADATA_11.toml
- METADATA_110.toml
- METADATA_111.toml
- METADATA_112.toml
- METADATA_113.toml
- METADATA_114.toml
- METADATA_115.toml
- METADATA_116.toml
- METADATA_117.toml
- METADATA_118.toml
- METADATA_119.toml
- (+393 more)

## Links
- [Root README](../../README.md)
- [Repo Index](../../repo_index.json)

---

## ML Experiment Configs

YAML files in this directory define **all** hyper-parameters and dataset pointers for the scripts in `tools_utilities/scripts/`.

Key fields used by the automation layer:

| Key | Example | Description |
|-----|---------|-------------|
| `bucket` | `s3://my-dataset-bucket` | Root path for streaming datasets. |
| `train_prefix` | `datasets/cortex/train-` | S3 prefix or local dir for train split. |
| `data_mode` | `streaming` or `local` | Chooses `S3 streaming` vs local FS loader. |
| `model_name` | `gpt2-small` | Baseline model architecture. |
| `lr`, `batch_size`, `epochs` | various | Standard optimisation knobs. |
| `sm_instances` | `2` | SageMaker instance count when `--backend cloud`. |
| `sm_instance_type` | `ml.g5.2xlarge` | GPU type for SageMaker training job. |
| `image_uri` | custom ECR URI | Override Docker image; otherwise auto-built or stock. |

All fields can be overridden on the CLI:

```bash
python train_streaming.py --config training_config.yaml \
       --override lr=1e-4 batch_size=32
```

Commit a new YAML for every experiment you want to reproduce; store checkpoints with the same filename prefix under `runs/`.

# Project Configurations

This directory holds deployment and monitoring configuration assets.

## Grafana Dashboards

See `grafana_dashboards/` for dashboards relevant to brainstem segmentation monitoring.
