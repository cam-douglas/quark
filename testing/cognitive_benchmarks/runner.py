import os
import sys
import argparse
import inspect
import importlib
import json
from pathlib import Path
from typing import List, Type, Dict, Any
from datetime import datetime

# Add the project root to the Python path to allow for absolute imports
# This is a bit of a hack, a better solution would be to make this a proper package
# or use a different test runner.
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from testing.cognitive_benchmarks.base_benchmark import BaseBenchmark

def discover_benchmarks(directory: Path) -> List[Type[BaseBenchmark]]:
    """
    Dynamically discover all BaseBenchmark subclasses in the given directory.
    """
    benchmarks = []
    for file in directory.glob("*.py"):
        if file.name.startswith(('_', '.')) or file.name == "runner.py":
            continue

        module_name = f"testing.cognitive_benchmarks.{file.stem}"
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseBenchmark) and obj is not BaseBenchmark:
                    benchmarks.append(obj)
        except Exception as e:
            print(f"Warning: Could not import or inspect {module_name}. Error: {e}")
    return benchmarks

def run_benchmarks(benchmark_runs: List[Dict[str, Any]], available_benchmarks: Dict[str, Type[BaseBenchmark]]):
    """
    Instantiate and run a list of benchmark configurations.
    """
    all_results = []
    print(f"Found {len(benchmark_runs)} benchmark runs in the configuration.")
    
    for run_config in benchmark_runs:
        benchmark_class_name = run_config.get("benchmark_class")
        params = run_config.get("params", {})
        
        benchmark_cls = available_benchmarks.get(benchmark_class_name)
        
        if not benchmark_cls:
            print(f"ERROR: Benchmark class '{benchmark_class_name}' not found. Skipping.")
            continue
            
        try:
            print("-" * 70)
            print(f"Starting benchmark run: {run_config.get('name', benchmark_class_name)}")
            
            # Instantiate the benchmark with the specified parameters
            instance = benchmark_cls(**params)
            
            results = instance.execute()
            all_results.append(results)
            
            # Print a summary
            print("\n--- Benchmark Summary ---")
            print(f"  Name: {results['name']}")
            for key, value in results['metrics'].items():
                print(f"  - {key}: {value}")
            print("-------------------------\n")

        except Exception as e:
            print(f"ERROR: Failed to run benchmark {benchmark_class_name}. Error: {e}")

    return all_results

def save_report(results: List[dict]):
    """
    Save the benchmark results to a JSON file with a timestamp.
    """
    if not results:
        print("No results to save.")
        return

    report_dir = project_root / "data_knowledge" / "data_repository" / "metrics"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"cognitive_benchmarks_report_{timestamp}.json"
    
    report_payload = {
        "report_generated_utc": datetime.utcnow().isoformat(),
        "benchmark_run_id": f"run_{timestamp}",
        "results": results
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_payload, f, indent=4)
        
    print(f"✅ Report saved successfully to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="QUARK Cognitive Benchmark Suite Runner")
    parser.add_argument(
        "--config", 
        type=str, 
        default="testing/cognitive_benchmarks/benchmark_config.json",
        help="Path to the benchmark configuration JSON file."
    )
    args = parser.parse_args()

    # Discover all available benchmark classes
    benchmark_dir = Path(__file__).parent
    available_benchmarks_list = discover_benchmarks(benchmark_dir)
    available_benchmarks_map = {b.__name__: b for b in available_benchmarks_list}

    # Load the benchmark configuration file
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Configuration file not found at {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    benchmark_runs = config.get("runs", [])

    if not benchmark_runs:
        print("No benchmark runs found in the configuration file.")
        return

    results = run_benchmarks(benchmark_runs, available_benchmarks_map)
    save_report(results)

if __name__ == "__main__":
    main()
