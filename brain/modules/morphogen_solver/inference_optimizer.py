#!/usr/bin/env python3
"""Inference Pipeline Optimizer.

Optimizes diffusion model inference for real-time performance including
GPU acceleration, model compilation, and performance benchmarking.

Integration: Performance optimization component for inference pipeline
Rationale: Focused optimization logic separated from main inference coordinator
"""

from typing import Dict, Any, List
import torch
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

class InferenceOptimizer:
    """Optimizer for diffusion model inference pipeline.
    
    Provides performance optimization including GPU acceleration,
    model compilation, memory optimization, and real-time benchmarking
    for morphogen gradient prediction.
    """
    
    def __init__(self, enable_gpu_acceleration: bool = True):
        """Initialize inference optimizer.
        
        Args:
            enable_gpu_acceleration: Whether to enable GPU acceleration
        """
        self.enable_gpu = enable_gpu_acceleration and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.enable_gpu else "cpu")
        
        logger.info("Initialized InferenceOptimizer")
        logger.info(f"GPU acceleration: {self.enable_gpu}")
    
    def optimize_model_for_inference(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Optimize model for inference performance.
        
        Args:
            model: Model to optimize
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Optimizing model for inference")
        
        optimization_results = {}
        
        # Set model to evaluation mode
        model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            optimized_model = torch.compile(model, mode="reduce-overhead")
            optimization_results["model_compiled"] = True
            logger.info("Model compiled for optimization")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
            optimized_model = model
            optimization_results["model_compiled"] = False
        
        # GPU optimizations
        if self.enable_gpu:
            # Set optimal GPU settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            optimization_results["gpu_optimizations_enabled"] = True
            
            # Move model to GPU if not already
            optimized_model = optimized_model.to(self.device)
        
        # Disable gradient computation for inference
        torch.set_grad_enabled(False)
        
        optimization_results["optimized_model"] = optimized_model
        
        return optimization_results
    
    def benchmark_inference_performance(self, model: torch.nn.Module,
                                       input_shape: tuple, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        logger.info(f"Benchmarking inference performance ({num_runs} runs)")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # Warm up
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Benchmark runs
        inference_times = []
        memory_usage = []
        
        for run in range(num_runs):
            torch.cuda.empty_cache() if self.enable_gpu else None
            
            start_time = time.time()
            
            if self.enable_gpu:
                start_memory = torch.cuda.memory_allocated()
            
            # Run inference
            with torch.no_grad():
                prediction = model(dummy_input)
                if self.enable_gpu:
                    torch.cuda.synchronize()
            
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            
            if self.enable_gpu:
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append(peak_memory - start_memory)
        
        # Calculate statistics
        performance_metrics = {
            "mean_inference_time_s": float(np.mean(inference_times)),
            "std_inference_time_s": float(np.std(inference_times)),
            "min_inference_time_s": float(np.min(inference_times)),
            "max_inference_time_s": float(np.max(inference_times)),
            "throughput_samples_per_s": float(1.0 / np.mean(inference_times)),
        }
        
        if self.enable_gpu and memory_usage:
            performance_metrics.update({
                "mean_memory_usage_mb": float(np.mean(memory_usage) / 1024**2),
                "peak_memory_usage_mb": float(np.max(memory_usage) / 1024**2)
            })
        
        # Check real-time capability
        real_time_capable = performance_metrics["mean_inference_time_s"] < 0.1
        performance_metrics["real_time_capable"] = real_time_capable
        
        logger.info(f"Benchmark results: {performance_metrics['mean_inference_time_s']:.3f}s avg, "
                   f"{performance_metrics['throughput_samples_per_s']:.1f} samples/s")
        
        if real_time_capable:
            logger.info("✅ Model optimized for real-time performance")
        else:
            logger.warning(f"⚠️ Model not real-time capable: {performance_metrics['mean_inference_time_s']:.3f}s")
        
        return performance_metrics
    
    def optimize_memory_usage(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Optimize memory usage for inference.
        
        Args:
            model: Model to optimize
            
        Returns:
            Memory optimization results
        """
        memory_results = {}
        
        if self.enable_gpu:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Get initial memory usage
            initial_memory = torch.cuda.memory_allocated()
            
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                memory_results["flash_attention_enabled"] = True
            except:
                memory_results["flash_attention_enabled"] = False
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
                memory_results["memory_fraction_set"] = True
            
            memory_results["initial_memory_mb"] = initial_memory / 1024**2
        else:
            memory_results["gpu_memory_optimization"] = "not_available"
        
        return memory_results
    
    def create_inference_cache(self, cache_size: int = 10) -> Dict[str, Any]:
        """Create optimized inference cache.
        
        Args:
            cache_size: Maximum number of cached predictions
            
        Returns:
            Cache configuration
        """
        cache_config = {
            "enabled": True,
            "max_size": cache_size,
            "current_size": 0,
            "cache_storage": {},
            "hit_rate": 0.0,
            "total_requests": 0,
            "cache_hits": 0
        }
        
        logger.info(f"Created inference cache: max size {cache_size}")
        
        return cache_config
    
    def update_cache_statistics(self, cache_config: Dict[str, Any], 
                               cache_hit: bool) -> None:
        """Update cache hit/miss statistics.
        
        Args:
            cache_config: Cache configuration dictionary
            cache_hit: Whether this was a cache hit
        """
        cache_config["total_requests"] += 1
        
        if cache_hit:
            cache_config["cache_hits"] += 1
        
        # Update hit rate
        cache_config["hit_rate"] = cache_config["cache_hits"] / cache_config["total_requests"]
