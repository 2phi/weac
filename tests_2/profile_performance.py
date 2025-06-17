#!/usr/bin/env python3
"""
Detailed profiling script to identify performance bottlenecks in weac vs weac_2.
"""

import time
import cProfile
import pstats
import io
from contextlib import contextmanager
import sys
import os
from typing import Dict, List
import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

@contextmanager
def timer_context(description: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    print(f"🔄 {description}...", end=" ")
    yield
    end = time.perf_counter()
    print(f"✅ {end - start:.4f}s")

class DetailedProfiler:
    """
    Detailed profiler for analyzing performance bottlenecks.
    """
    
    def __init__(self):
        self.results = {}
    
    def profile_new_implementation_components(self, touchdown: bool = False):
        """
        Profile individual components of the new implementation.
        """
        print(f"\n{'='*60}")
        print(f"PROFILING NEW IMPLEMENTATION COMPONENTS (touchdown={touchdown})")
        print(f"{'='*60}")
        
        from weac_2.components import ModelInput, Layer, Segment, CriteriaConfig, WeakLayer, ScenarioConfig
        from weac_2.components.config import Config
        from weac_2.core.system_model import SystemModel
        
        # Setup data
        layers = [
            Layer(rho=200, h=150),
            Layer(rho=300, h=100),
        ]
        
        segments = [
            Segment(l=6000, has_foundation=True, m=0),
            Segment(l=1000, has_foundation=False, m=75),
            Segment(l=1000, has_foundation=False, m=0),
            Segment(l=6000, has_foundation=True, m=0)
        ]
        
        inclination = 30.0
        scenario_config = ScenarioConfig(phi=inclination, system_type='skier', crack_length=2000)
        weak_layer = WeakLayer(rho=10, h=30, E=0.25, G_Ic=1)
        criteria_config = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)
        config = Config(touchdown=touchdown)
        
        # Time component creation
        with timer_context("Creating model input"):
            model_input = ModelInput(
                scenario_config=scenario_config,
                weak_layer=weak_layer,
                layers=layers,
                segments=segments,
                criteria_config=criteria_config
            )
        
        # Time system model initialization
        with timer_context("Initializing SystemModel"):
            system_model = SystemModel(config=config, model_input=model_input)
        
        # Time individual component access (these trigger cached_property calculations)
        with timer_context("Computing Eigensystem"):
            _ = system_model.eigensystem
        
        if touchdown:
            with timer_context("Computing Slab Touchdown"):
                _ = system_model.slab_touchdown
        
        with timer_context("Computing Unknown Constants"):
            constants = system_model.unknown_constants
        
        return constants
    
    def profile_old_implementation_components(self, touchdown: bool = False):
        """
        Profile individual components of the old implementation.
        """
        print(f"\n{'='*60}")
        print(f"PROFILING OLD IMPLEMENTATION COMPONENTS (touchdown={touchdown})")
        print(f"{'='*60}")
        
        import weac
        
        # Setup data
        profile = [
            [200, 150],  # Layer 1: 200 kg/m³, 150mm thick
            [300, 100],  # Layer 2: 300 kg/m³, 100mm thick
        ]
        
        # Time model creation
        with timer_context("Creating Layered model"):
            old_model = weac.Layered(system='skier', layers=profile, touchdown=touchdown)
        
        # Time segment calculation
        with timer_context("Calculating segments"):
            segments_data = old_model.calc_segments(
                L=14000.0,
                a=2000,
                m=75,
                li=None,
                mi=None,
                ki=None
            )['crack']
        
        # Time solution
        with timer_context("Assembling and solving"):
            constants = old_model.assemble_and_solve(phi=30.0, **segments_data)
        
        return constants
    
    def detailed_cprofile_analysis(self, touchdown: bool = False):
        """
        Use cProfile to get detailed function-level timing analysis.
        """
        print(f"\n{'='*60}")
        print(f"DETAILED cPROFILE ANALYSIS (touchdown={touchdown})")
        print(f"{'='*60}")
        
        # Profile new implementation
        print("\n🔍 NEW IMPLEMENTATION PROFILE:")
        new_profiler = cProfile.Profile()
        new_profiler.enable()
        self._run_new_implementation(touchdown=touchdown)
        new_profiler.disable()
        
        # Get new implementation stats
        new_stats_buffer = io.StringIO()
        new_stats = pstats.Stats(new_profiler, stream=new_stats_buffer)
        new_stats.sort_stats('cumulative')
        new_stats.print_stats(20)  # Top 20 functions
        
        print(new_stats_buffer.getvalue())
        
        # Profile old implementation
        print("\n🔍 OLD IMPLEMENTATION PROFILE:")
        old_profiler = cProfile.Profile()
        old_profiler.enable()
        self._run_old_implementation(touchdown=touchdown)
        old_profiler.disable()
        
        # Get old implementation stats
        old_stats_buffer = io.StringIO()
        old_stats = pstats.Stats(old_profiler, stream=old_stats_buffer)
        old_stats.sort_stats('cumulative')
        old_stats.print_stats(20)  # Top 20 functions
        
        print(old_stats_buffer.getvalue())
    
    def _run_new_implementation(self, touchdown: bool = False):
        """Helper to run new implementation for profiling."""
        from weac_2.components import ModelInput, Layer, Segment, CriteriaConfig, WeakLayer, ScenarioConfig
        from weac_2.components.config import Config
        from weac_2.core.system_model import SystemModel
        
        layers = [Layer(rho=200, h=150), Layer(rho=300, h=100)]
        segments = [
            Segment(l=6000, has_foundation=True, m=0),
            Segment(l=1000, has_foundation=False, m=75),
            Segment(l=1000, has_foundation=False, m=0),
            Segment(l=6000, has_foundation=True, m=0)
        ]
        
        scenario_config = ScenarioConfig(phi=30.0, system_type='skier', crack_length=2000)
        weak_layer = WeakLayer(rho=10, h=30, E=0.25, G_Ic=1)
        criteria_config = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)
        config = Config(touchdown=touchdown)
        
        model_input = ModelInput(
            scenario_config=scenario_config,
            weak_layer=weak_layer,
            layers=layers,
            segments=segments,
            criteria_config=criteria_config
        )
        
        system_model = SystemModel(config=config, model_input=model_input)
        return system_model.unknown_constants
    
    def _run_old_implementation(self, touchdown: bool = False):
        """Helper to run old implementation for profiling."""
        import weac
        
        profile = [[200, 150], [300, 100]]
        old_model = weac.Layered(system='skier', layers=profile, touchdown=touchdown)
        
        segments_data = old_model.calc_segments(
            L=14000.0, a=2000, m=75, li=None, mi=None, ki=None
        )['crack']
        
        return old_model.assemble_and_solve(phi=30.0, **segments_data)
    
    def compare_memory_usage(self, touchdown: bool = False):
        """
        Compare memory usage between implementations.
        """
        print(f"\n{'='*60}")
        print(f"MEMORY USAGE COMPARISON (touchdown={touchdown})")
        print(f"{'='*60}")
        
        try:
            import psutil
            import os
            
            # Measure old implementation memory
            process = psutil.Process(os.getpid())
            mem_before_old = process.memory_info().rss / 1024 / 1024  # MB
            
            old_result = self._run_old_implementation(touchdown=touchdown)
            
            mem_after_old = process.memory_info().rss / 1024 / 1024  # MB
            old_memory_delta = mem_after_old - mem_before_old
            
            print(f"🧠 Old implementation memory usage: {old_memory_delta:.2f} MB")
            
            # Reset and measure new implementation memory
            mem_before_new = process.memory_info().rss / 1024 / 1024  # MB
            
            new_result = self._run_new_implementation(touchdown=touchdown)
            
            mem_after_new = process.memory_info().rss / 1024 / 1024  # MB
            new_memory_delta = mem_after_new - mem_before_new
            
            print(f"🧠 New implementation memory usage: {new_memory_delta:.2f} MB")
            print(f"📊 Memory difference: {new_memory_delta - old_memory_delta:+.2f} MB")
            
        except ImportError:
            print("⚠️  psutil not available - install with 'pip install psutil' for memory profiling")
    
    def analyze_import_overhead(self):
        """
        Analyze the overhead of importing different modules.
        """
        print(f"\n{'='*60}")
        print(f"IMPORT OVERHEAD ANALYSIS")
        print(f"{'='*60}")
        
        # Time imports for new implementation
        with timer_context("Importing weac_2.components"):
            from weac_2.components import ModelInput, Layer, Segment, CriteriaConfig, WeakLayer, ScenarioConfig
        
        with timer_context("Importing weac_2.components.config"):
            from weac_2.components.config import Config
        
        with timer_context("Importing weac_2.core.system_model"):
            from weac_2.core.system_model import SystemModel
        
        # Time imports for old implementation
        with timer_context("Importing weac"):
            import weac
    
    def run_comprehensive_analysis(self):
        """
        Run all profiling analyses.
        """
        print("🚀 Starting comprehensive performance analysis...")
        
        # Analyze import overhead
        self.analyze_import_overhead()
        
        # Profile components for both touchdown scenarios
        for touchdown in [False, True]:
            self.profile_old_implementation_components(touchdown=touchdown)
            self.profile_new_implementation_components(touchdown=touchdown)
            self.compare_memory_usage(touchdown=touchdown)
            
        # Detailed profiling for touchdown=False (where we see the biggest difference)
        self.detailed_cprofile_analysis(touchdown=False)
        
        print("\n✅ Comprehensive analysis complete!")

if __name__ == "__main__":
    profiler = DetailedProfiler()
    profiler.run_comprehensive_analysis() 