#!/usr/bin/env python3
"""Benchmark vLLM performance against existing LocalLLMWrapper.

This script compares performance between the old LocalLLMWrapper and new VLLMBrainWrapper
for Quark brain simulation tasks.
"""

import time
import statistics
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def benchmark_vllm_vs_local():
    """Compare vLLM performance against LocalLLMWrapper."""
    
    print("🏁 vLLM vs LocalLLMWrapper Performance Benchmark")
    print("=" * 60)
    
    # Test prompts for brain simulation
    test_prompts = [
        "The human brain processes information by",
        "Neural networks learn through",
        "Consciousness emerges when neurons",
        "Memory formation occurs in the hippocampus via",
        "Attention mechanisms in the brain work by"
    ]
    
    model_path = project_root / "data/models/test_models/gpt2-small"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    results = {}
    
    # Benchmark vLLM
    print("\n🚀 Benchmarking vLLM...")
    try:
        from brain.externals.vllm_brain_wrapper import VLLMBrainWrapper
        
        vllm_wrapper = VLLMBrainWrapper(
            model_path,
            max_model_len=512,
            gpu_memory_utilization=0.3,
            enforce_eager=True
        )
        
        # Single generation benchmark
        vllm_times = []
        for prompt in test_prompts:
            start = time.time()
            response = vllm_wrapper.generate(prompt, max_tokens=50)
            end = time.time()
            vllm_times.append(end - start)
        
        # Batch generation benchmark
        batch_start = time.time()
        batch_responses = vllm_wrapper.generate_batch(test_prompts, max_tokens=50)
        batch_end = time.time()
        
        results['vllm'] = {
            'single_avg': statistics.mean(vllm_times),
            'single_min': min(vllm_times),
            'single_max': max(vllm_times),
            'batch_total': batch_end - batch_start,
            'batch_per_item': (batch_end - batch_start) / len(test_prompts),
            'throughput': len(test_prompts) / (batch_end - batch_start)
        }
        
        print(f"✅ vLLM benchmark completed")
        
    except Exception as e:
        print(f"❌ vLLM benchmark failed: {e}")
        results['vllm'] = None
    
    # Benchmark LocalLLMWrapper (if available)
    print("\n🐌 Benchmarking LocalLLMWrapper...")
    try:
        from brain.architecture.neural_core.cognitive_systems.local_llm_wrapper import LocalLLMWrapper
        
        local_wrapper = LocalLLMWrapper(model_path)
        
        local_times = []
        for prompt in test_prompts:
            start = time.time()
            response = local_wrapper.generate(prompt, max_new_tokens=50)
            end = time.time()
            local_times.append(end - start)
        
        results['local'] = {
            'single_avg': statistics.mean(local_times),
            'single_min': min(local_times),
            'single_max': max(local_times),
            'batch_total': sum(local_times),  # No batch support
            'batch_per_item': statistics.mean(local_times),
            'throughput': len(test_prompts) / sum(local_times)
        }
        
        print(f"✅ LocalLLMWrapper benchmark completed")
        
    except Exception as e:
        print(f"❌ LocalLLMWrapper benchmark failed: {e}")
        results['local'] = None
    
    # Display results
    print("\n📊 Performance Results")
    print("=" * 60)
    
    if results.get('vllm') and results.get('local'):
        vllm = results['vllm']
        local = results['local']
        
        print(f"{'Metric':<25} {'vLLM':<15} {'LocalLLM':<15} {'Speedup':<10}")
        print("-" * 65)
        
        # Single generation comparison
        speedup = local['single_avg'] / vllm['single_avg']
        print(f"{'Avg Generation (s)':<25} {vllm['single_avg']:<15.3f} {local['single_avg']:<15.3f} {speedup:<10.2f}x")
        
        # Throughput comparison
        throughput_speedup = vllm['throughput'] / local['throughput']
        print(f"{'Throughput (req/s)':<25} {vllm['throughput']:<15.2f} {local['throughput']:<15.2f} {throughput_speedup:<10.2f}x")
        
        # Batch vs sequential
        batch_speedup = local['batch_total'] / vllm['batch_total']
        print(f"{'Batch Processing (s)':<25} {vllm['batch_total']:<15.3f} {local['batch_total']:<15.3f} {batch_speedup:<10.2f}x")
        
        print(f"\n🎯 Summary:")
        print(f"   • vLLM is {speedup:.1f}x faster for single generations")
        print(f"   • vLLM is {throughput_speedup:.1f}x faster for throughput")
        print(f"   • vLLM batch processing is {batch_speedup:.1f}x faster")
        
    elif results.get('vllm'):
        vllm = results['vllm']
        print(f"vLLM Performance:")
        print(f"  • Average generation: {vllm['single_avg']:.3f}s")
        print(f"  • Throughput: {vllm['throughput']:.2f} req/s")
        print(f"  • Batch processing: {vllm['batch_total']:.3f}s total")
        
    print(f"\n💡 Recommendations:")
    if results.get('vllm') and results.get('local'):
        if speedup > 2:
            print(f"   ✅ Migrate to vLLM immediately - significant performance gains!")
        elif speedup > 1.5:
            print(f"   ✅ vLLM provides good performance improvements")
        else:
            print(f"   ⚠️ Performance gains are modest - consider other factors")
    
    print(f"   • Use vLLM batch processing for multiple brain simulations")
    print(f"   • Deploy vLLM on Google Cloud for scalability")
    print(f"   • Consider GPU instances for even better performance")

if __name__ == "__main__":
    benchmark_vllm_vs_local()
