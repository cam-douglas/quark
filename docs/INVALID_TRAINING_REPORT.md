# Invalid Training Run Report for Task 2.3.3

## 1. Executive Summary

This report concludes that the training artifacts provided for the validation of **Task 2.3.3: Hybrid model training** are **invalid**. The training run did not meet the requirements outlined in the `foundation_layer_tasks.md` document and does not represent a meaningful step towards completing the task.

The key findings are:
- The trained model was a **simplified placeholder**, not the required `GNNViTHybrid` architecture.
- The model was trained on **randomly generated noise**, not the biologically-relevant `SyntheticEmbryoDataGenerator`.
- **No meaningful evaluation metrics** (e.g., accuracy, Dice score) were recorded. The training logs are empty.

**Conclusion:** Task 2.3.3 remains **incomplete**. A new, valid training run must be performed.

## 2. Evidence

### 2.1. Static Code Analysis

- Analysis of `data/experiments/brainstem_training/simple_vm_training.py` revealed that the script explicitly defines and trains a `SimpleGNNViTHybrid`.
- The script's `generate_synthetic_data` function uses `torch.randn` and `torch.randint`, generating random data that has no connection to the project's validated data simulators.
- The training loop only calculates and attempts to log `BCELoss`. No validation loop or accuracy/Dice metrics are present.

### 2.2. Log File Analysis

- The script `tools_utilities/log_validator.py` was executed on the provided event file (`data/experiments/brainstem_training/events.out.tfevents`) and all files in the actual output directory (`local_logs/`).
- **Result:** The script found **zero** scalar summaries. No metrics of any kind were successfully logged during the training run.

### 2.3. Model Checkpoint Analysis

- The script `tools_utilities/model_validator.py` was executed on the model artifact (`data/models/gnn_vit_hybrid/gnn_vit_hybrid_final.pth`).
- **Result:** The script successfully loaded the checkpoint into the `SimpleGNNViTHybrid` architecture.
- **Confirmation:** This proves that the artifact is the simplified placeholder model, not the full, required architecture. The total parameter count of **69,317,760** corresponds to this simple model.

## 3. Required Next Steps

1.  **Perform a valid training run** using the correct `GNNViTHybrid` model architecture.
2.  The training must use the **`SyntheticEmbryoDataGenerator`** to ensure biological relevance.
3.  The training script must include a **validation loop** and log appropriate metrics, including **Dice score**, to allow for proper evaluation.
4.  Provide the new, valid artifacts for another round of validation.