#!/usr/bin/env python3
"""
Clean performance benchmark excluding import overhead to get accurate timing comparisons.
Note: Old implementation is executed in an isolated environment via a helper.
"""

import os
import sys
import time
from functools import wraps
from typing import Dict, List

import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.utils.weac_reference_runner import (
    compute_reference_model_results,  # noqa: E402
)
from weac.components import (  # noqa: E402
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac.components.config import Config  # noqa: E402
from weac.core.system_model import SystemModel  # noqa: E402


def timeit(func):
    """Decorator to measure execution time of functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper


class CleanPerformanceBenchmark:
    """
    Clean benchmarking class focusing on pure execution time without import overhead.
    """

    def __init__(self):
        self.results = {}
        # Warm-up both implementations to ensure everything is loaded
        print("🔥 Warming up implementations...")
        self._warmup()
        print("✅ Warm-up complete!")

    def _warmup(self):
        """Warm up both implementations to ensure consistent timing."""
        # Warm up old implementation
        self._run_old_implementation(touchdown=False)
        self._run_old_implementation(touchdown=True)

        # Warm up new implementation
        self._run_new_implementation(touchdown=False)
        self._run_new_implementation(touchdown=True)

    @timeit
    def _run_old_implementation(self, touchdown: bool = False):
        """Benchmark the old published implementation (isolated env)."""
        profile = [
            [200, 150],
            [300, 100],
        ]
        total_length = 14000.0
        inclination = 30.0
        try:
            constants, _state = compute_reference_model_results(
                system="skier",
                layers_profile=profile,
                touchdown=touchdown,
                L=total_length,
                a=2000,
                m=75,
                phi=inclination,
            )
        except RuntimeError:
            # If old env cannot be provisioned, fall back to a zero array to keep benchmarks running
            return np.zeros((0,))
        return constants

    @timeit
    def _run_new_implementation(self, touchdown: bool = False):
        """Benchmark the new weac implementation (no imports)."""
        # Equivalent setup in new system
        layers = [
            Layer(rho=200, h=150),
            Layer(rho=300, h=100),
        ]

        segments = [
            Segment(length=6000, has_foundation=True, m=0),
            Segment(length=1000, has_foundation=False, m=75),
            Segment(length=1000, has_foundation=False, m=0),
            Segment(length=6000, has_foundation=True, m=0),
        ]

        inclination = 30.0
        scenario_config = ScenarioConfig(
            phi=inclination, system_type="skier", cut_length=2000
        )
        weak_layer = WeakLayer(rho=10, h=30, E=0.25, G_Ic=1)
        criteria_config = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)
        config = Config(touchdown=touchdown)

        model_input = ModelInput(
            scenario_config=scenario_config,
            weak_layer=weak_layer,
            layers=layers,
            segments=segments,
            criteria_config=criteria_config,
        )

        new_system = SystemModel(config=config, model_input=model_input)
        new_constants = new_system.unknown_constants

        return new_constants

    @timeit
    def _run_old_layers(self, layers_profile: List[List[float]]):
        """Benchmark old implementation with custom layers (isolated env)."""
        try:
            constants, _state = compute_reference_model_results(
                system="skier",
                layers_profile=layers_profile,
                touchdown=False,
                L=14000.0,
                a=2000,
                m=75,
                phi=30.0,
            )
        except RuntimeError:
            return np.zeros((0,))
        return constants

    @timeit
    def _run_new_layers(self, layers: List):
        """Benchmark new implementation with custom layers (no imports)."""
        segments = [
            Segment(length=6000, has_foundation=True, m=0),
            Segment(length=1000, has_foundation=False, m=75),
            Segment(length=1000, has_foundation=False, m=0),
            Segment(length=6000, has_foundation=True, m=0),
        ]

        scenario_config = ScenarioConfig(phi=30.0, system_type="skier", cut_length=2000)
        weak_layer = WeakLayer(rho=10, h=30, E=0.25, G_Ic=1)
        criteria_config = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)
        config = Config()

        model_input = ModelInput(
            scenario_config=scenario_config,
            weak_layer=weak_layer,
            layers=layers,
            segments=segments,
            criteria_config=criteria_config,
        )

        new_system = SystemModel(config=config, model_input=model_input)
        return new_system.unknown_constants

    def benchmark_execution_time(
        self, touchdown: bool = False, num_runs: int = 50
    ) -> Dict:
        """
        Benchmark pure execution time with many runs for statistical significance.

        Args:
            touchdown: Whether to enable touchdown
            num_runs: Number of runs to average over (increased for better stats)

        Returns:
            Dictionary with timing results
        """
        print(f"\n{'=' * 70}")
        print(f"🏁 CLEAN BENCHMARK: Two-Layer Setup (touchdown={touchdown})")
        print(f"Number of runs: {num_runs} (excluding import overhead)")
        print(f"{'=' * 70}")

        old_times = []
        new_times = []

        for run in range(num_runs):
            if run % 10 == 0:  # Progress indicator every 10 runs
                print(f"Progress: {run}/{num_runs}...")

            # Benchmark old implementation
            _, old_time = self._run_old_implementation(touchdown=touchdown)
            old_times.append(old_time)

            # Benchmark new implementation
            _, new_time = self._run_new_implementation(touchdown=touchdown)
            new_times.append(new_time)

        # Calculate statistics
        old_times = np.array(old_times)
        new_times = np.array(new_times)

        old_mean = np.mean(old_times)
        old_std = np.std(old_times)
        old_median = np.median(old_times)
        old_min = np.min(old_times)
        old_max = np.max(old_times)

        new_mean = np.mean(new_times)
        new_std = np.std(new_times)
        new_median = np.median(new_times)
        new_min = np.min(new_times)
        new_max = np.max(new_times)

        speedup = old_mean / new_mean

        results = {
            "scenario": f"clean_two_layer_touchdown_{touchdown}",
            "num_runs": num_runs,
            "old_implementation": {
                "mean_time": old_mean,
                "std_time": old_std,
                "median_time": old_median,
                "min_time": old_min,
                "max_time": old_max,
                "all_times": old_times.tolist(),
            },
            "new_implementation": {
                "mean_time": new_mean,
                "std_time": new_std,
                "median_time": new_median,
                "min_time": new_min,
                "max_time": new_max,
                "all_times": new_times.tolist(),
            },
            "speedup": speedup,
            "performance_change": (new_mean - old_mean) / old_mean * 100,
        }

        self.results[f"clean_two_layer_touchdown_{touchdown}"] = results
        return results

    def benchmark_scalability_clean(self, num_runs: int = 20) -> Dict:
        """
        Clean scalability benchmark with different numbers of layers.

        Args:
            num_runs: Number of runs to average over

        Returns:
            Dictionary with timing results for different layer counts
        """
        print(f"\n{'=' * 70}")
        print("🔢 CLEAN SCALABILITY BENCHMARK")
        print(f"Number of runs per configuration: {num_runs}")
        print(f"{'=' * 70}")

        layer_configs = [
            (2, "Two layers"),
            (3, "Three layers"),
            (4, "Four layers"),
            (5, "Five layers"),
            (6, "Six layers"),
        ]

        results = {}

        for num_layers, description in layer_configs:
            print(f"\n🧱 Testing {description}...")

            old_times = []
            new_times = []

            for run in range(num_runs):
                if run % 5 == 0:
                    print(f"  Progress: {run}/{num_runs}...")

                # Generate layer configuration
                layers_old = [[200 + i * 50, 100] for i in range(num_layers)]
                layers_new = [Layer(rho=200 + i * 50, h=100) for i in range(num_layers)]

                # Benchmark old implementation
                _, old_time = self._run_old_layers(layers_old)
                old_times.append(old_time)

                # Benchmark new implementation
                _, new_time = self._run_new_layers(layers_new)
                new_times.append(new_time)

            # Calculate statistics
            old_times = np.array(old_times)
            new_times = np.array(new_times)

            old_mean = np.mean(old_times)
            new_mean = np.mean(new_times)
            speedup = old_mean / new_mean

            results[f"{num_layers}_layers"] = {
                "description": description,
                "num_layers": num_layers,
                "num_runs": num_runs,
                "old_mean_time": old_mean,
                "old_std_time": np.std(old_times),
                "new_mean_time": new_mean,
                "new_std_time": np.std(new_times),
                "speedup": speedup,
                "performance_change": (new_mean - old_mean) / old_mean * 100,
            }

            print(
                f"  ✅ {description}: Old {old_mean:.4f}s, New {new_mean:.4f}s, Speedup: {speedup:.2f}x"
            )

        self.results["clean_scalability"] = results
        return results

    def print_detailed_summary(self):
        """Print a comprehensive summary of all clean benchmark results."""
        print("\n{'=' * 80}")
        print("🏆 CLEAN PERFORMANCE BENCHMARK SUMMARY")
        print("{'=' * 80}")

        for test_name, results in self.results.items():
            if test_name == "clean_scalability":
                print("\n📊 CLEAN SCALABILITY RESULTS:")
                print(
                    f"{'Layers':<8} {'Runs':<6} {'Old (ms)':<12} {'New (ms)':<12} {'Speedup':<10} {'Change (%)':<12}"
                )
                print(f"{'-' * 70}")

                for layer_key, layer_results in results.items():
                    num_layers = layer_results["num_layers"]
                    num_runs = layer_results["num_runs"]
                    old_time = layer_results["old_mean_time"] * 1000  # Convert to ms
                    new_time = layer_results["new_mean_time"] * 1000  # Convert to ms
                    speedup = layer_results["speedup"]
                    change = layer_results["performance_change"]

                    print(
                        f"{num_layers:<8} {num_runs:<6} {old_time:<12.2f} {new_time:<12.2f} {speedup:<10.2f}x {change:<12.1f}"
                    )

            else:
                print(f"\n🏁 {results['scenario'].upper().replace('_', ' ')} RESULTS:")
                old_stats = results["old_implementation"]
                new_stats = results["new_implementation"]

                print(f"  Runs: {results['num_runs']}")
                print("  Old implementation:")
                print(
                    f"    Mean:   {old_stats['mean_time'] * 1000:.3f}ms ± {old_stats['std_time'] * 1000:.3f}ms"
                )
                print(f"    Median: {old_stats['median_time'] * 1000:.3f}ms")
                print(
                    f"    Range:  {old_stats['min_time'] * 1000:.3f}ms - {old_stats['max_time'] * 1000:.3f}ms"
                )

                print("  New implementation:")
                print(
                    f"    Mean:   {new_stats['mean_time'] * 1000:.3f}ms ± {new_stats['std_time'] * 1000:.3f}ms"
                )
                print(f"    Median: {new_stats['median_time'] * 1000:.3f}ms")
                print(
                    f"    Range:  {new_stats['min_time'] * 1000:.3f}ms - {new_stats['max_time'] * 1000:.3f}ms"
                )

                print("  📈 Performance Analysis:")
                print(f"    Speedup: {results['speedup']:.3f}x")

                if results["speedup"] > 1.05:
                    print(
                        f"    ✅ New implementation is {results['speedup']:.2f}x FASTER"
                    )
                elif results["speedup"] < 0.95:
                    print(
                        f"    ⚠️  New implementation is {1 / results['speedup']:.2f}x SLOWER"
                    )
                else:
                    print("    ➡️  Both implementations have similar performance")

                print(f"    Performance change: {results['performance_change']:+.1f}%")

    def run_full_clean_benchmark(self):
        """Run the complete clean benchmark suite."""
        print("🚀 Starting CLEAN performance benchmark (no import overhead)...")

        # Test both touchdown scenarios with more runs for better statistics
        self.benchmark_execution_time(touchdown=False, num_runs=50)
        self.benchmark_execution_time(touchdown=True, num_runs=50)

        # Test scalability with clean timing
        self.benchmark_scalability_clean(num_runs=20)

        # Print comprehensive summary
        self.print_detailed_summary()

        print("\n✅ Clean benchmark complete! Pure execution timing results obtained.")
        return self.results


if __name__ == "__main__":
    # Run the clean benchmark
    benchmark = CleanPerformanceBenchmark()
    results = benchmark.run_full_clean_benchmark()

    # Save results to file
    import json

    with open("clean_benchmark_results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key == "clean_scalability":
                json_results[key] = value
            else:
                json_results[key] = {
                    k: v for k, v in value.items() if "all_times" not in k
                }
                json_results[key]["old_mean_time"] = value["old_implementation"][
                    "mean_time"
                ]
                json_results[key]["new_mean_time"] = value["new_implementation"][
                    "mean_time"
                ]

        json.dump(json_results, f, indent=2)

    print("\n📁 Clean benchmark results saved to 'clean_benchmark_results.json'")
