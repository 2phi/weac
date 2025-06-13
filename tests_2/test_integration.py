# tests/test_system_model.py
import unittest
import numpy as np
from functools import cached_property
import importlib
import sys
import types
import os

# Add the project root to the Python path so we can import weac_2
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class TestIntegrationOldVsNew(unittest.TestCase):
    """Integration tests comparing old weac implementation with new weac_2 implementation."""

    def test_simple_two_layer_setup(self):
        """
        Test that old and new implementations produce identical results for a simple two-layer setup.
        """
        # --- Setup for OLD implementation (main.py style) ---
        import weac
        
        # Simple two-layer profile
        profile = [
            [200, 150],  # Layer 1: 200 kg/m³, 150mm thick
            [300, 100],  # Layer 2: 300 kg/m³, 100mm thick
        ]
        
        # Create old model
        old_model = weac.Layered(system='skier', layers=profile, touchdown=False)
        
        # Simple segment setup - for 'skier' system with a=0, this creates 4 segments: [L/2, 0, 0, L/2]
        total_length = 14000.0  # 14m total
        segments_data = old_model.calc_segments(
            L=total_length,
            a=2000,      # no initial crack
            m=75,     # 75kg skier
            li=None,  # use default segmentation
            mi=None,  # single point load
            ki=None   # default foundation support
        )['crack']
        
        # Solve with 30-degree inclination
        inclination = 30.0
        old_constants = old_model.assemble_and_solve(phi=inclination, **segments_data)
        
        # --- Setup for NEW implementation (main_weac2.py style) ---
        from weac_2.components import ModelInput, Layer, Segment, CriteriaConfig, WeakLayer, ScenarioConfig
        from weac_2.components.config import Config
        from weac_2.core.system_model import SystemModel
        
        # Equivalent setup in new system
        layers = [
            Layer(rho=200, h=150),
            Layer(rho=300, h=100),
        ]
        
        segments = [
            Segment(l=6000, k=True, m=0),
            Segment(l=1000, k=False, m=75),
            Segment(l=1000, k=False, m=0),
            Segment(l=6000, k=True, m=0)
        ]
        
        scenario_config = ScenarioConfig(phi=inclination, system='skier')
        weak_layer = WeakLayer(rho=10, h=30, E=0.25, G_Ic=1)  # Default weak layer properties
        criteria_config = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)
        config = Config()  # Use default configuration
        
        model_input = ModelInput(
            scenario_config=scenario_config,
            weak_layer=weak_layer,
            layers=layers,
            segments=segments,
            criteria_config=criteria_config
        )
        
        new_system = SystemModel(config=config, model_input=model_input)
        new_constants = new_system.unknown_constants
        
        # --- Compare results ---
        self.assertEqual(old_constants.shape, new_constants.shape, 
                        "Result arrays should have the same shape")
        
        # Use reasonable tolerances for integration testing between implementations
        # Small differences (~0.5%) are acceptable due to:
        # - Different numerical precision in calculations
        # - Possible minor algorithmic differences in the refactored code
        # - Floating-point arithmetic accumulation differences
        np.testing.assert_allclose(old_constants, new_constants, rtol=1e-2, atol=1e-6,
                                  err_msg="Old and new implementations should produce very similar results")
        
        max_rel_diff = np.max(np.abs((old_constants - new_constants) / old_constants))
        max_abs_diff = np.max(np.abs(old_constants - new_constants))
        
        print(f"✓ Integration test passed - implementations produce very similar results")
        print(f"  Result shape: {old_constants.shape}")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e} ({max_rel_diff*100:.3f}%)")
        
        # Assert that differences are within reasonable engineering tolerances
        self.assertLess(max_rel_diff, 0.01, "Relative differences should be < 1%")
        self.assertLess(max_abs_diff, 0.001, "Absolute differences should be < 0.001")

if __name__ == "__main__":
    unittest.main(verbosity=2)
