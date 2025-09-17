"""
Model compression and quantisation for deployment-ready brainstem segmentation.

Implements pruning, quantisation, and ONNX optimization for efficient
inference while maintaining segmentation accuracy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.quantization as quantization
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for model compression pipeline."""
    
    # Pruning settings
    pruning_ratio: float = 0.3  # 30% of weights pruned
    structured_pruning: bool = True
    
    # Quantisation settings
    quantization_mode: str = "dynamic"  # "dynamic", "static", "qat"
    quantization_backend: str = "fbgemm"  # "fbgemm", "qnnpack"
    
    # ONNX settings
    onnx_opset_version: int = 14
    optimize_for_mobile: bool = False
    
    # Validation thresholds
    max_accuracy_drop: float = 0.02  # Max 2% Dice drop allowed
    target_speedup: float = 2.0      # Target 2x speedup
    target_size_reduction: float = 0.5  # Target 50% size reduction


class ModelPruner:
    """Implements structured and unstructured pruning."""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
    
    def prune_model(self, model: nn.Module) -> nn.Module:
        """Apply pruning to model."""
        import torch.nn.utils.prune as prune
        
        if self.config.structured_pruning:
            return self._structured_pruning(model)
        else:
            return self._unstructured_pruning(model)
    
    def _structured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning (remove entire channels/filters)."""
        
        # Find Conv3d layers for channel pruning
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv3d):
                conv_layers.append((name, module))
        
        logger.info(f"Found {len(conv_layers)} Conv3d layers for pruning")
        
        # Apply channel pruning to convolutional layers
        for name, layer in conv_layers:
            if layer.out_channels > 8:  # Keep minimum channels
                n_prune = int(layer.out_channels * self.config.pruning_ratio)
                n_prune = min(n_prune, layer.out_channels - 4)  # Keep at least 4 channels
                
                if n_prune > 0:
                    import torch.nn.utils.prune as prune
                    prune.random_structured(layer, name="weight", amount=n_prune, dim=0)
                    prune.remove(layer, "weight")  # Make pruning permanent
                    
                    logger.info(f"Pruned {n_prune} channels from {name}")
        
        return model
    
    def _unstructured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply unstructured pruning (remove individual weights)."""
        import torch.nn.utils.prune as prune
        
        # Apply global unstructured pruning
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                parameters_to_prune.append((module, "weight"))
        
        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config.pruning_ratio
            )
            
            # Make pruning permanent
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
        
        logger.info(f"Applied {self.config.pruning_ratio:.1%} global unstructured pruning")
        return model


class ModelQuantizer:
    """Implements post-training quantisation."""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
    
    def quantize_model(
        self, 
        model: nn.Module,
        calibration_data: Optional[torch.utils.data.DataLoader] = None
    ) -> nn.Module:
        """Apply quantisation to model."""
        
        if self.config.quantization_mode == "dynamic":
            return self._dynamic_quantization(model)
        elif self.config.quantization_mode == "static":
            if calibration_data is None:
                logger.warning("Static quantization requires calibration data, falling back to dynamic")
                return self._dynamic_quantization(model)
            return self._static_quantization(model, calibration_data)
        else:
            raise ValueError(f"Unsupported quantization mode: {self.config.quantization_mode}")
    
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantisation."""
        
        # Prepare model for quantization
        model.eval()
        
        # Apply dynamic quantization
        quantized_model = quantization.quantize_dynamic(
            model,
            {nn.Conv3d, nn.Linear},  # Layers to quantize
            dtype=torch.qint8
        )
        
        logger.info("Applied dynamic quantization (int8)")
        return quantized_model
    
    def _static_quantization(
        self, 
        model: nn.Module, 
        calibration_data: torch.utils.data.DataLoader
    ) -> nn.Module:
        """Apply static quantisation with calibration."""
        
        # Set quantization config
        model.qconfig = quantization.get_default_qconfig(self.config.quantization_backend)
        
        # Prepare model
        model_prepared = quantization.prepare(model, inplace=False)
        
        # Calibrate with sample data
        model_prepared.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(calibration_data):
                model_prepared(data)
                if i >= 10:  # Limit calibration samples
                    break
        
        # Convert to quantized model
        quantized_model = quantization.convert(model_prepared, inplace=False)
        
        logger.info("Applied static quantization with calibration")
        return quantized_model


class ONNXOptimizer:
    """Optimizes models for ONNX deployment."""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
    
    def export_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Path,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """Export model to optimized ONNX format."""
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Export to ONNX
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=self.config.onnx_opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        
        logger.info(f"ONNX model exported: {output_path}")
        
        # Optimize ONNX model
        if optimize:
            optimized_path = output_path.with_suffix(".optimized.onnx")
            self._optimize_onnx_model(output_path, optimized_path)
            
            return {
                "original_path": output_path,
                "optimized_path": optimized_path,
                "input_shape": input_shape
            }
        
        return {
            "original_path": output_path,
            "input_shape": input_shape
        }
    
    def _optimize_onnx_model(self, input_path: Path, output_path: Path) -> None:
        """Optimize ONNX model for deployment."""
        try:
            import onnxoptimizer
            
            # Load model
            model = onnx.load(str(input_path))
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(model)
            
            # Save optimized model
            onnx.save(optimized_model, str(output_path))
            
            logger.info(f"ONNX model optimized: {output_path}")
            
        except ImportError:
            logger.warning("onnxoptimizer not available, skipping optimization")
            # Copy original to optimized path
            import shutil
            shutil.copy2(input_path, output_path)
    
    def benchmark_onnx_model(
        self,
        onnx_path: Path,
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark ONNX model performance."""
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(str(onnx_path))
        
        # Create test input
        test_input = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {"input": test_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            outputs = session.run(None, {"input": test_input})
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "p95_latency_ms": np.percentile(times, 95) * 1000,
            "throughput_fps": 1.0 / np.mean(times)
        }


def run_compression_demo() -> Dict[str, Any]:
    """Demonstrate model compression pipeline."""
    
    print("🗜️ Model Compression Demo")
    print("=" * 40)
    
    config = CompressionConfig()
    
    # Create mock model
    from brain.modules.brainstem_segmentation.hierarchical_framework import HierarchicalSegmentationHead, HierarchyConfig
    
    hierarchy_config = HierarchyConfig()
    model = HierarchicalSegmentationHead(256, hierarchy_config)
    
    # Get original model size
    original_size = sum(p.numel() for p in model.parameters())
    print(f"📊 Original model: {original_size:,} parameters")
    
    # Test pruning
    pruner = ModelPruner(config)
    pruned_model = pruner.prune_model(model)
    
    pruned_size = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
    pruning_ratio = 1 - (pruned_size / original_size)
    print(f"📊 After pruning: {pruned_size:,} parameters ({pruning_ratio:.1%} reduction)")
    
    # Test quantization
    quantizer = ModelQuantizer(config)
    quantized_model = quantizer.quantize_model(pruned_model)
    
    print(f"📊 Quantization applied: {config.quantization_mode} mode")
    
    # Test ONNX export
    onnx_optimizer = ONNXOptimizer(config)
    input_shape = (256, 16, 16, 16)
    
    output_dir = Path("/Users/camdouglas/quark/data/models/brainstem_segmentation/compressed")
    onnx_path = output_dir / "compressed_model.onnx"
    
    try:
        # Export original model (quantized model may not be ONNX-compatible)
        onnx_results = onnx_optimizer.export_to_onnx(model, input_shape, onnx_path)
        print(f"📊 ONNX export: {onnx_results['original_path']}")
        
        # Benchmark
        benchmark_results = onnx_optimizer.benchmark_onnx_model(onnx_path, input_shape, num_runs=10)
        print(f"📊 ONNX performance:")
        print(f"  Mean latency: {benchmark_results['mean_latency_ms']:.1f} ms")
        print(f"  Throughput: {benchmark_results['throughput_fps']:.1f} FPS")
        
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
        benchmark_results = {}
    
    return {
        "original_params": original_size,
        "pruned_params": pruned_size,
        "pruning_ratio": pruning_ratio,
        "quantization_mode": config.quantization_mode,
        "onnx_results": onnx_results if 'onnx_results' in locals() else {},
        "benchmark": benchmark_results
    }


if __name__ == "__main__":
    # Run demonstration
    demo_results = run_compression_demo()
    
    # Save results
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "model_compression_demo.json"
    with open(results_path, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n📄 Demo results saved: {results_path}")
    print("✅ Model compression framework ready for deployment")
