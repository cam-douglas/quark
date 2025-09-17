import math
import time

import numpy as np

from brain.modules.brainstem_segmentation.metrics import record_inference, record_dice
from brain.modules.brainstem_segmentation.inference_algorithms import compute_overall_dice


def test_record_inference_runs_without_error():
    start = time.time()
    time.sleep(0.01)
    latency = time.time() - start
    # Should not raise
    record_inference(latency, True)
    record_inference(latency, False)


def test_record_dice_runs_without_error():
    record_dice(measured_overall=0.91, target_overall=0.87)
    record_dice(measured_overall=0.80, target_overall=0.87)


def test_compute_overall_dice_simple_cases():
    pred = np.zeros((4, 4, 4), dtype=np.int32)
    tgt = np.zeros_like(pred)

    # All background → treated as perfect per-class → mean 1.0
    dice = compute_overall_dice(pred, tgt, num_classes=3)
    assert math.isfinite(dice)
    assert 0.99 <= dice <= 1.0

    # Single class match
    pred[:, :, :] = 1
    tgt[:, :, :] = 1
    dice = compute_overall_dice(pred, tgt, num_classes=3)
    assert 0.99 <= dice <= 1.0

    # Partial overlap
    pred[:2, :, :] = 1
    pred[2:, :, :] = 2
    tgt[:3, :, :] = 1
    tgt[3:, :, :] = 2
    dice = compute_overall_dice(pred, tgt, num_classes=3)
    assert 0.0 <= dice <= 1.0
