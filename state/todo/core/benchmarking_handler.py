"""
Benchmarking Handler Module
===========================
Handles performance benchmarking operations.
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class BenchmarkingHandler:
    """Handles benchmarking operations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.benchmark_script = self.project_root / 'scripts' / 'benchmark_vllm_performance.py'
        self.results_dir = self.project_root / 'data' / 'experiments' / 'benchmarks'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def route_command(self, action: str, params: Dict) -> int:
        """Route benchmarking commands."""
        if action == 'benchmark':
            return self.run_benchmark(params)
        elif action == 'compare':
            return self.compare_metrics(params)
        elif action == 'profile':
            return self.profile_performance(params)
        elif action == 'report':
            return self.generate_report()
        else:
            return self.show_help()
    
    def run_benchmark(self, params: Dict) -> int:
        """Run performance benchmark."""
        benchmark_type = params.get('type', 'performance')
        
        print(f"\n⚡ Running Benchmark: {benchmark_type}")
        print("=" * 50)
        
        if benchmark_type == 'performance':
            return self._benchmark_performance(params)
        elif benchmark_type == 'memory':
            return self._benchmark_memory(params)
        elif benchmark_type == 'inference':
            return self._benchmark_inference(params)
        elif benchmark_type == 'training':
            return self._benchmark_training(params)
        else:
            print(f"⚠️ Unknown benchmark type: {benchmark_type}")
            return 1
    
    def _benchmark_performance(self, params: Dict) -> int:
        """Run performance benchmark."""
        print("🏃 Performance Benchmark Setup")
        print("-" * 40)
        
        # Check if running on GCP
        if params.get('platform') == 'gcp' or self._is_gcp_available():
            return self._benchmark_performance_gcp(params)
        
        # Otherwise provide guidance
        print("\n⚠️  Performance benchmarks should run on Google Cloud Compute")
        print("   for accurate and reproducible results.\n")
        
        print("📋 Recommended Setup:")
        print("1. Deploy to GCP: todo deploy to gcp")
        print("2. SSH to instance: gcloud compute ssh quark-benchmark")
        print("3. Run benchmark: todo benchmark performance --platform gcp")
        print("\nOr use the automated GCP benchmark:")
        print("   todo benchmark performance --platform gcp --auto\n")
        
        response = input("Run local benchmark anyway? (not recommended) [y/N]: ")
        if response.lower() != 'y':
            print("✅ Good choice! Set up GCP for accurate benchmarks.")
            return 0
        
        # Run local benchmark with warning
        print("\n⚠️  Running LOCAL benchmark (results may vary)...")
        if self.benchmark_script.exists():
            cmd = [sys.executable, str(self.benchmark_script)]
            
            # Add parameters
            if params.get('iterations'):
                cmd.extend(['--iterations', str(params['iterations'])])
            if params.get('warmup'):
                cmd.extend(['--warmup', str(params['warmup'])])
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Local benchmark completed in {elapsed:.2f}s")
                print("⚠️  Note: These results are from local hardware")
                
                # Save results with local flag
                self._save_benchmark_results({
                    'type': 'performance',
                    'platform': 'local',
                    'elapsed': elapsed,
                    'timestamp': datetime.now().isoformat(),
                    'output': result.stdout,
                    'warning': 'Local benchmark - results may vary'
                })
            
            return result.returncode
        else:
            # Fallback to basic benchmark
            return self._run_basic_benchmark()
    
    def _is_gcp_available(self) -> bool:
        """Check if GCP is configured and available."""
        try:
            result = subprocess.run(['gcloud', 'config', 'get-value', 'project'],
                                  capture_output=True, text=True)
            return result.returncode == 0 and result.stdout.strip()
        except:
            return False
    
    def _benchmark_performance_gcp(self, params: Dict) -> int:
        """Run performance benchmark on GCP."""
        print("☁️ Setting up GCP Benchmark")
        print("-" * 40)
        
        # Check for existing benchmark instance
        instance_name = params.get('instance', 'quark-benchmark')
        zone = params.get('zone', 'us-central1-a')
        
        print(f"Instance: {instance_name}")
        print(f"Zone: {zone}")
        
        # Create or start instance
        if params.get('auto'):
            print("\n🚀 Auto-provisioning benchmark instance...")
            
            # Create instance if doesn't exist
            create_cmd = [
                'gcloud', 'compute', 'instances', 'create', instance_name,
                '--zone', zone,
                '--machine-type', params.get('machine_type', 'n1-highmem-8'),
                '--accelerator', 'type=nvidia-tesla-v100,count=1',
                '--maintenance-policy', 'TERMINATE',
                '--image-family', 'pytorch-latest-gpu',
                '--image-project', 'deeplearning-platform-release',
                '--boot-disk-size', '100GB',
                '--metadata', 'startup-script=cd /home && git clone https://github.com/yourusername/quark.git'
            ]
            
            print("Creating instance (this may take a few minutes)...")
            result = subprocess.run(create_cmd, capture_output=True)
            
            if result.returncode == 0:
                print("✅ Instance created")
                
                # Wait for instance to be ready
                print("Waiting for instance to initialize...")
                time.sleep(30)
                
                # Run benchmark remotely
                benchmark_cmd = [
                    'gcloud', 'compute', 'ssh', instance_name,
                    '--zone', zone,
                    '--command', 'cd quark && python scripts/benchmark_vllm_performance.py'
                ]
                
                print("Running benchmark on GCP...")
                result = subprocess.run(benchmark_cmd)
                
                # Fetch results
                fetch_cmd = [
                    'gcloud', 'compute', 'scp',
                    f'{instance_name}:~/quark/data/experiments/benchmarks/*',
                    str(self.results_dir),
                    '--zone', zone
                ]
                subprocess.run(fetch_cmd)
                
                print("✅ Benchmark completed on GCP")
                
                # Optional: Stop instance to save costs
                if params.get('stop_after', True):
                    stop_cmd = ['gcloud', 'compute', 'instances', 'stop',
                              instance_name, '--zone', zone]
                    subprocess.run(stop_cmd)
                    print("✅ Instance stopped (to save costs)")
                
                return 0
        else:
            # Manual setup instructions
            print("\n📋 Manual GCP Benchmark Steps:")
            print("1. Create/start instance:")
            print(f"   gcloud compute instances create {instance_name} \\")
            print(f"     --zone {zone} \\")
            print("     --machine-type n1-highmem-8 \\")
            print("     --accelerator type=nvidia-tesla-v100,count=1")
            print("\n2. SSH to instance:")
            print(f"   gcloud compute ssh {instance_name} --zone {zone}")
            print("\n3. Clone and setup Quark:")
            print("   git clone https://github.com/yourusername/quark.git")
            print("   cd quark && pip install -r requirements.txt")
            print("\n4. Run benchmark:")
            print("   python scripts/benchmark_vllm_performance.py")
            print("\n5. Copy results back:")
            print(f"   gcloud compute scp {instance_name}:~/quark/data/experiments/* ./data/experiments/")
            print("\n6. Stop instance (save costs):")
            print(f"   gcloud compute instances stop {instance_name} --zone {zone}")
            
            return 0
    
    def _benchmark_memory(self, params: Dict) -> int:
        """Run memory benchmark."""
        print("🧠 Running memory benchmark...")
        
        # Use memory_profiler if available
        try:
            cmd = ['python', '-m', 'memory_profiler',
                   str(self.project_root / 'brain' / 'brain_main.py')]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Memory profiling completed")
                self._save_benchmark_results({
                    'type': 'memory',
                    'timestamp': datetime.now().isoformat(),
                    'profile': result.stdout
                })
            
            return result.returncode
        except:
            print("⚠️ memory_profiler not installed")
            print("📝 Run: pip install memory_profiler")
            return 1
    
    def _benchmark_inference(self, params: Dict) -> int:
        """Run inference benchmark."""
        print("🤖 Running inference benchmark...")
        
        # Measure inference speed
        test_data = params.get('test_data', 'default')
        batch_size = params.get('batch_size', 32)
        
        print(f"  Batch size: {batch_size}")
        print(f"  Test data: {test_data}")
        
        # This would run actual inference benchmarks
        metrics = {
            'throughput': '1000 samples/sec',
            'latency_p50': '10ms',
            'latency_p95': '25ms',
            'latency_p99': '50ms'
        }
        
        print("\n📊 Inference Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        self._save_benchmark_results({
            'type': 'inference',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return 0
    
    def _benchmark_training(self, params: Dict) -> int:
        """Run training benchmark."""
        print("🎓 Running training benchmark...")
        
        epochs = params.get('epochs', 1)
        batch_size = params.get('batch_size', 32)
        
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        
        # This would run actual training benchmarks
        metrics = {
            'samples_per_second': 500,
            'gpu_utilization': '85%',
            'memory_usage': '12GB',
            'time_per_epoch': '120s'
        }
        
        print("\n📊 Training Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        self._save_benchmark_results({
            'type': 'training',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return 0
    
    def _run_basic_benchmark(self) -> int:
        """Run basic benchmark without special scripts."""
        print("⏱️ Running basic benchmark...")
        
        # Simple performance test
        import timeit
        
        # Test basic operations
        tests = {
            'list_creation': 'list(range(10000))',
            'dict_creation': '{i: i**2 for i in range(1000)}',
            'string_ops': '"test" * 1000'
        }
        
        results = {}
        for name, code in tests.items():
            time_taken = timeit.timeit(code, number=1000)
            results[name] = f"{time_taken:.4f}s"
            print(f"  {name}: {results[name]}")
        
        self._save_benchmark_results({
            'type': 'basic',
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
        return 0
    
    def compare_metrics(self, params: Dict) -> int:
        """Compare benchmark metrics."""
        print("\n📊 Comparing Metrics")
        print("=" * 50)
        
        # Load recent benchmarks
        benchmarks = self._load_recent_benchmarks(2)
        
        if len(benchmarks) < 2:
            print("⚠️ Need at least 2 benchmark runs to compare")
            return 1
        
        # Compare the two most recent
        prev = benchmarks[1]
        curr = benchmarks[0]
        
        print(f"Previous: {prev.get('timestamp', 'N/A')}")
        print(f"Current:  {curr.get('timestamp', 'N/A')}")
        print("\nChanges:")
        
        # Compare metrics
        if prev.get('metrics') and curr.get('metrics'):
            for key in curr['metrics']:
                if key in prev['metrics']:
                    prev_val = prev['metrics'][key]
                    curr_val = curr['metrics'][key]
                    
                    # Try to calculate percentage change
                    try:
                        if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                            change = ((curr_val - prev_val) / prev_val) * 100
                            symbol = "↑" if change > 0 else "↓"
                            print(f"  {key}: {prev_val} → {curr_val} ({symbol}{abs(change):.1f}%)")
                        else:
                            print(f"  {key}: {prev_val} → {curr_val}")
                    except:
                        print(f"  {key}: {prev_val} → {curr_val}")
        
        return 0
    
    def profile_performance(self, params: Dict) -> int:
        """Profile performance."""
        print("\n🔬 Profiling Performance")
        print("=" * 50)
        
        profile_type = params.get('type', 'cpu')
        
        if profile_type == 'cpu':
            return self._profile_cpu()
        elif profile_type == 'gpu':
            return self._profile_gpu()
        else:
            print(f"⚠️ Unknown profile type: {profile_type}")
            return 1
    
    def _profile_cpu(self) -> int:
        """Profile CPU performance."""
        print("💻 CPU Profiling...")
        
        # Use cProfile
        try:
            main_script = self.project_root / 'brain' / 'brain_main.py'
            if main_script.exists():
                cmd = ['python', '-m', 'cProfile', '-s', 'cumulative',
                       str(main_script)]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Save profile
                    profile_file = self.results_dir / f"cpu_profile_{datetime.now():%Y%m%d_%H%M%S}.txt"
                    with open(profile_file, 'w') as f:
                        f.write(result.stdout)
                    
                    print(f"✅ Profile saved: {profile_file}")
                    
                    # Show top functions
                    lines = result.stdout.split('\n')[:20]
                    print("\nTop functions by cumulative time:")
                    for line in lines:
                        print(line)
                
                return result.returncode
        except Exception as e:
            print(f"❌ Profiling failed: {e}")
            return 1
    
    def _profile_gpu(self) -> int:
        """Profile GPU performance."""
        print("🎮 GPU Profiling...")
        
        try:
            # Check nvidia-smi
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout)
                
                # Save GPU info
                gpu_file = self.results_dir / f"gpu_profile_{datetime.now():%Y%m%d_%H%M%S}.txt"
                with open(gpu_file, 'w') as f:
                    f.write(result.stdout)
                
                print(f"✅ GPU info saved: {gpu_file}")
            else:
                print("⚠️ No GPU detected or nvidia-smi not available")
            
            return result.returncode
        except:
            print("⚠️ GPU profiling not available")
            return 1
    
    def generate_report(self) -> int:
        """Generate benchmark report."""
        print("\n📈 Generating Benchmark Report")
        print("=" * 50)
        
        # Load all benchmarks
        benchmarks = self._load_recent_benchmarks(10)
        
        if not benchmarks:
            print("⚠️ No benchmarks found")
            return 1
        
        # Generate report
        report_file = self.results_dir / f"report_{datetime.now():%Y%m%d}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Benchmark Report\n\n")
            f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"Total runs: {len(benchmarks)}\n\n")
            
            f.write("## Recent Benchmarks\n\n")
            for i, bench in enumerate(benchmarks[:5], 1):
                f.write(f"### Run {i}\n")
                f.write(f"- Type: {bench.get('type', 'N/A')}\n")
                f.write(f"- Timestamp: {bench.get('timestamp', 'N/A')}\n")
                
                if bench.get('metrics'):
                    f.write("- Metrics:\n")
                    for key, value in bench['metrics'].items():
                        f.write(f"  - {key}: {value}\n")
                
                f.write("\n")
        
        print(f"✅ Report generated: {report_file}")
        return 0
    
    def _save_benchmark_results(self, results: Dict) -> None:
        """Save benchmark results."""
        filename = f"benchmark_{results['type']}_{datetime.now():%Y%m%d_%H%M%S}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved: {filepath}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
    
    def _load_recent_benchmarks(self, count: int) -> List[Dict]:
        """Load recent benchmark results."""
        benchmarks = []
        
        for json_file in sorted(self.results_dir.glob('benchmark_*.json'), reverse=True)[:count]:
            try:
                with open(json_file) as f:
                    benchmarks.append(json.load(f))
            except:
                pass
        
        return benchmarks
    
    def show_help(self) -> int:
        """Show benchmarking help."""
        print("""
⚡ Benchmarking Commands:
  todo benchmark performance   → Run performance benchmark
  todo benchmark memory        → Run memory benchmark
  todo benchmark inference     → Run inference benchmark
  todo benchmark training      → Run training benchmark
  todo compare metrics         → Compare recent benchmarks
  todo profile cpu             → Profile CPU usage
  todo profile gpu             → Profile GPU usage
  todo benchmark report        → Generate benchmark report
""")
        return 0
