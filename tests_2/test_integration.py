# tests/test_system_model.py
import os
import sys
import unittest

import numpy as np

# Add the project root to the Python path so we can import weac_2
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from weac_2.logging_config import setup_logging

setup_logging()


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
        old_model = weac.Layered(system="pst-", layers=profile, touchdown=False)

        # Solve with 30-degree inclination
        inclination = 30.0

        # Simple segment setup - for 'skier' system with a=0, this creates 4 segments: [L/2, 0, 0, L/2]
        total_length = 14000.0  # 14m total
        segments_data = old_model.calc_segments(
            L=total_length,
            a=4000,  # no initial crack
            m=0,  # 75kg skier
            li=None,  # use default segmentation
            mi=None,  # single point load
            ki=None,  # default foundation support
            phi=inclination,
        )["crack"]

        old_constants = old_model.assemble_and_solve(phi=inclination, **segments_data)

        # --- Setup for NEW implementation (main_weac2.py style) ---
        from weac_2.components import (
            CriteriaConfig,
            Layer,
            ModelInput,
            ScenarioConfig,
            Segment,
            WeakLayer,
        )
        from weac_2.components.config import Config
        from weac_2.core.system_model import SystemModel

        # Equivalent setup in new system
        layers = [
            Layer(rho=200, h=150),
            Layer(rho=300, h=100),
        ]

        segments = [
            Segment(length=10000, has_foundation=True, m=0),
            Segment(length=4000, has_foundation=False, m=0),
        ]

        scenario_config = ScenarioConfig(
            phi=inclination, system_type="pst-", crack_length=4000
        )
        weak_layer = WeakLayer(
            rho=10, h=30, E=0.25, G_Ic=1
        )  # Default weak layer properties
        criteria_config = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)
        config = Config(touchdown=False)  # Use default configuration

        model_input = ModelInput(
            scenario_config=scenario_config,
            weak_layer=weak_layer,
            layers=layers,
            segments=segments,
            criteria_config=criteria_config,
        )

        new_system = SystemModel(config=config, model_input=model_input)
        new_constants = new_system.unknown_constants

        # Compare the WeakLayer attributes
        self.assertEqual(
            old_model.weak["nu"],
            new_system.weak_layer.nu,
            "Weak layer Poisson's ratio should be the same",
        )
        self.assertEqual(
            old_model.weak["E"],
            new_system.weak_layer.E,
            "Weak layer Young's modulus should be the same",
        )
        self.assertEqual(
            old_model.t,
            new_system.weak_layer.h,
            "Weak layer thickness should be the same",
        )
        self.assertEqual(
            old_model.kn,
            new_system.weak_layer.kn,
            "Weak layer normal stiffness should be the same",
        )
        self.assertEqual(
            old_model.kt,
            new_system.weak_layer.kt,
            "Weak layer shear stiffness should be the same",
        )

        # Compare the Slab properties
        self.assertEqual(
            old_model.h, new_system.slab.H, "Slab thickness should be the same"
        )
        self.assertEqual(
            old_model.zs,
            new_system.slab.z_cog,
            "Slab center of gravity should be the same",
        )

        # Compare the Layer properties
        np.testing.assert_array_equal(
            old_model.slab[:, 0] * 1e-12,
            new_system.slab.rhoi,
            "Layer density should be the same",
        )
        np.testing.assert_array_equal(
            old_model.slab[:, 1],
            new_system.slab.hi,
            "Layer thickness should be the same",
        )
        np.testing.assert_array_equal(
            old_model.slab[:, 2],
            new_system.slab.Ei,
            "Layer Young's modulus should be the same",
        )
        np.testing.assert_array_equal(
            old_model.slab[:, 3],
            new_system.slab.Gi,
            "Layer shear modulus should be the same",
        )
        np.testing.assert_array_equal(
            old_model.slab[:, 4],
            new_system.slab.nui,
            "Layer Poisson's ratio should be the same",
        )

        # Compare all the attributes of the old and new model
        self.assertEqual(
            old_model.a, new_system.scenario.crack_l, "Crack length should be the same"
        )

        # --- Compare results ---
        self.assertEqual(
            old_constants.shape,
            new_constants.shape,
            "Result arrays should have the same shape",
        )

        # Use reasonable tolerances for integration testing between implementations
        # Small differences (~0.5%) are acceptable due to:
        # - Different numerical precision in calculations
        # - Possible minor algorithmic differences in the refactored code
        # - Floating-point arithmetic accumulation differences
        np.testing.assert_allclose(
            old_constants,
            new_constants,
            rtol=1e-2,
            atol=1e-6,
            err_msg="Old and new implementations should produce very similar results",
        )

        max_rel_diff = np.max(np.abs((old_constants - new_constants) / old_constants))
        max_abs_diff = np.max(np.abs(old_constants - new_constants))

        print(
            "✓ Integration test passed - implementations produce very similar results"
        )
        print(f"  Result shape: {old_constants.shape}")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(
            f"  Max relative difference: {max_rel_diff:.2e} ({max_rel_diff * 100:.3f}%)"
        )

        # Assert that differences are within reasonable engineering tolerances
        self.assertLess(max_rel_diff, 0.001, "Relative differences should be < 0.1%")
        self.assertLess(max_abs_diff, 0.001, "Absolute differences should be < 0.001")

    def test_simple_two_layer_setup_with_touchdown(self):
        """
        Test that old and new implementations produce identical results for a simple two-layer setup with touchdown=True.
        """
        # --- Setup for OLD implementation (main.py style) ---
        import weac

        # Simple two-layer profile
        profile = [
            [200, 150],  # Layer 1: 200 kg/m³, 150mm thick
            [300, 100],  # Layer 2: 300 kg/m³, 100mm thick
        ]

        # Create old model with touchdown=True
        old_model = weac.Layered(system="pst-", layers=profile, touchdown=True)

        # Solve with 30-degree inclination
        inclination = 30.0

        # Simple segment setup - for 'skier' system with touchdown=True
        total_length = 14000.0  # 14m total
        segments_data = old_model.calc_segments(
            L=total_length,
            a=4000,  # 2m initial crack
            m=0,  # 75kg skier
            li=None,  # use default segmentation
            mi=None,  # single point load
            ki=None,  # default foundation support
            phi=inclination,
        )["crack"]

        old_constants = old_model.assemble_and_solve(phi=inclination, **segments_data)

        # --- Setup for NEW implementation (main_weac2.py style) ---
        from weac_2.components import (
            CriteriaConfig,
            Layer,
            ModelInput,
            ScenarioConfig,
            Segment,
            WeakLayer,
        )
        from weac_2.components.config import Config
        from weac_2.core.system_model import SystemModel

        # Equivalent setup in new system
        layers = [
            Layer(rho=200, h=150),
            Layer(rho=300, h=100),
        ]

        # For touchdown=True, the segmentation will be different
        # Need to match the segments that would be created by calc_segments with touchdown=True
        segments = [
            Segment(length=10000, has_foundation=True, m=0),
            Segment(length=4000, has_foundation=False, m=0),
        ]

        scenario_config = ScenarioConfig(
            phi=inclination, system_type="pst-", crack_length=4000
        )
        weak_layer = WeakLayer(
            rho=10, h=30, E=0.25, G_Ic=1
        )  # Default weak layer properties
        criteria_config = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)
        config = Config(touchdown=True)  # Use default configuration

        model_input = ModelInput(
            scenario_config=scenario_config,
            weak_layer=weak_layer,
            layers=layers,
            segments=segments,
            criteria_config=criteria_config,
        )

        new_system = SystemModel(config=config, model_input=model_input)
        new_constants = new_system.unknown_constants

        # Compare the WeakLayer attributes
        self.assertEqual(
            old_model.weak["nu"],
            new_system.weak_layer.nu,
            "Weak layer Poisson's ratio should be the same",
        )
        self.assertEqual(
            old_model.weak["E"],
            new_system.weak_layer.E,
            "Weak layer Young's modulus should be the same",
        )
        self.assertEqual(
            old_model.t,
            new_system.weak_layer.h,
            "Weak layer thickness should be the same",
        )
        self.assertEqual(
            old_model.kn,
            new_system.weak_layer.kn,
            "Weak layer normal stiffness should be the same",
        )
        self.assertEqual(
            old_model.kt,
            new_system.weak_layer.kt,
            "Weak layer shear stiffness should be the same",
        )

        # Compare the Slab Touchdown attributes
        self.assertEqual(
            old_model.tc, new_system.scenario.crack_h, "Crack height should be the same"
        )
        self.assertEqual(
            old_model.a1,
            new_system.slab_touchdown.l_AB,
            "Transition length A should be the same",
        )
        self.assertEqual(
            old_model.a2,
            new_system.slab_touchdown.l_BC,
            "Transition length B should be the same",
        )
        self.assertEqual(
            old_model.td,
            new_system.slab_touchdown.touchdown_distance,
            "Touchdown distance should be the same",
        )

        # Compare the Slab properties
        self.assertEqual(
            old_model.h, new_system.slab.H, "Slab thickness should be the same"
        )
        self.assertEqual(
            old_model.zs,
            new_system.slab.z_cog,
            "Slab center of gravity should be the same",
        )

        # Compare the Layer properties
        np.testing.assert_array_equal(
            old_model.slab[:, 0] * 1e-12,
            new_system.slab.rhoi,
            "Layer density should be the same",
        )
        np.testing.assert_array_equal(
            old_model.slab[:, 1],
            new_system.slab.hi,
            "Layer thickness should be the same",
        )
        np.testing.assert_array_equal(
            old_model.slab[:, 2],
            new_system.slab.Ei,
            "Layer Young's modulus should be the same",
        )
        np.testing.assert_array_equal(
            old_model.slab[:, 3],
            new_system.slab.Gi,
            "Layer shear modulus should be the same",
        )
        np.testing.assert_array_equal(
            old_model.slab[:, 4],
            new_system.slab.nui,
            "Layer Poisson's ratio should be the same",
        )

        # Compare all the attributes of the old and new model
        self.assertEqual(
            old_model.a, new_system.scenario.crack_l, "Crack length should be the same"
        )

        # --- Compare results ---
        self.assertEqual(
            old_constants.shape,
            new_constants.shape,
            "Result arrays should have the same shape",
        )

        # Use reasonable tolerances for integration testing between implementations
        # Small differences (~0.5%) are acceptable due to:
        # - Different numerical precision in calculations
        # - Possible minor algorithmic differences in the refactored code
        # - Floating-point arithmetic accumulation differences
        np.testing.assert_allclose(
            old_constants,
            new_constants,
            rtol=1e-2,
            atol=1e-6,
            err_msg="Old and new implementations should produce very similar results",
        )

        max_rel_diff = np.max(np.abs((old_constants - new_constants) / old_constants))
        max_abs_diff = np.max(np.abs(old_constants - new_constants))

        print(
            "✓ Integration test with touchdown passed - implementations produce very similar results"
        )
        print(f"  Result shape: {old_constants.shape}")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(
            f"  Max relative difference: {max_rel_diff:.2e} ({max_rel_diff * 100:.3f}%)"
        )

        # Assert that differences are within reasonable engineering tolerances
        self.assertLess(max_rel_diff, 0.01, "Relative differences should be < 1%")
        self.assertLess(max_abs_diff, 0.001, "Absolute differences should be < 0.001")


if __name__ == "__main__":
    unittest.main(verbosity=2)
