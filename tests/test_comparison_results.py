"""
This module contains tests that compare the results of the old and new WEAC implementations.
"""

import unittest

import numpy as np

from weac.analysis.analyzer import Analyzer
from weac.components import (
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac.components.config import Config
from weac.core.system_model import SystemModel
from tests.utils.weac_reference_runner import (  # noqa: E402
    compute_reference_model_results,
)


class TestIntegrationOldVsNew(unittest.TestCase):
    """Integration tests comparing old weac implementation with new weac implementation."""

    def test_simple_two_layer_setup(self):
        """
        Test that old and new implementations produce identical results
        for a simple two-layer setup.
        """
        # --- Setup for OLD implementation (published weac==2.6.X) ---
        profile = [
            [200, 150],
            [300, 100],
        ]
        inclination = 30.0
        total_length = 14000.0
        try:
            _, old_state, old_z, old_analysis = compute_reference_model_results(
                system="pst-",
                layers_profile=profile,
                touchdown=False,
                L=total_length,
                a=4000,
                m=0,
                phi=inclination,
            )
        except RuntimeError as exc:
            self.skipTest(f"Old weac environment unavailable: {exc}")

        # --- Setup for NEW implementation (main_weac2.py style) ---
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
            phi=inclination, system_type="pst-", cut_length=4000
        )
        weak_layer = WeakLayer(
            rho=50, h=30, E=0.25, G_Ic=1
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

        z1 = new_system.z(
            x=[0, 5000, 10000],
            C=new_constants[:, [0]],
            length=10000,
            phi=inclination,
            has_foundation=True,
        )
        z2 = new_system.z(
            x=[0, 2000, 4000],
            C=new_constants[:, [1]],
            length=4000,
            phi=inclination,
            has_foundation=False,
        )
        new_z = np.hstack([z1, z2])

        # --- Analysis for NEW implementation ---
        analyzer = Analyzer(new_system, printing_enabled=False)
        new_raster_x, new_raster_z, new_raster_xb = analyzer.rasterize_solution(num=100)
        new_z_mesh_dict = analyzer.get_zmesh(dz=2)
        new_sxx = analyzer.Sxx(new_raster_z, inclination, dz=2, unit="kPa")
        new_txz = analyzer.Txz(new_raster_z, inclination, dz=2, unit="kPa")
        new_szz = analyzer.Szz(new_raster_z, inclination, dz=2, unit="kPa")
        new_principal_stress_slab = analyzer.principal_stress_slab(
            new_raster_z, inclination, dz=2, val="max", unit="kPa", normalize=False
        )

        # Compare the WeakLayer attributes
        self.assertEqual(
            old_state["weak"]["nu"],
            new_system.weak_layer.nu,
            "Weak layer Poisson's ratio should be the same",
        )
        self.assertEqual(
            old_state["weak"]["E"],
            new_system.weak_layer.E,
            "Weak layer Young's modulus should be the same",
        )
        self.assertEqual(
            old_state["t"],
            new_system.weak_layer.h,
            "Weak layer thickness should be the same",
        )
        self.assertEqual(
            old_state["kn"],
            new_system.weak_layer.kn,
            "Weak layer normal stiffness should be the same",
        )
        self.assertEqual(
            old_state["kt"],
            new_system.weak_layer.kt,
            "Weak layer shear stiffness should be the same",
        )

        # Compare the Slab properties
        self.assertEqual(
            old_state["h"], new_system.slab.H, "Slab thickness should be the same"
        )
        self.assertEqual(
            old_state["zs"],
            new_system.slab.z_cog,
            "Slab center of gravity should be the same",
        )

        # Compare the Layer properties
        old_slab = (
            np.asarray(old_state["slab"]) if old_state["slab"] is not None else None
        )
        self.assertIsNotNone(old_slab, "Old slab data should be available")
        if old_slab is not None:
            np.testing.assert_array_equal(
                old_slab[:, 0] * 1e-12,
                new_system.slab.rhoi,
                "Layer density should be the same",
            )
            np.testing.assert_array_equal(
                old_slab[:, 1],
                new_system.slab.hi,
                "Layer thickness should be the same",
            )
            np.testing.assert_array_equal(
                old_slab[:, 2],
                new_system.slab.Ei,
                "Layer Young's modulus should be the same",
            )
            np.testing.assert_array_equal(
                old_slab[:, 3],
                new_system.slab.Gi,
                "Layer shear modulus should be the same",
            )
            np.testing.assert_array_equal(
                old_slab[:, 4],
                new_system.slab.nui,
                "Layer Poisson's ratio should be the same",
            )

        # Compare all the attributes of the old and new model
        self.assertEqual(
            old_state["a"],
            new_system.scenario.cut_length,
            "Cut length should be the same",
        )

        # Compare the z vectors
        self.assertEqual(old_z.shape, new_z.shape, "Z-vector shapes should match")
        np.testing.assert_allclose(
            old_z,
            new_z,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Old and new implementations should produce very similar z vectors",
        )

        # Compare analysis results
        np.testing.assert_allclose(
            old_analysis["raster_x"],
            new_raster_x,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Rasterized x-coordinates should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["raster_z"],
            new_raster_z,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Rasterized z-solutions should be very similar",
        )
        # For raster_xb, we need to handle NaNs
        np.testing.assert_allclose(
            old_analysis["raster_xb"],
            new_raster_xb,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Rasterized founded x-coordinates should be very similar",
            equal_nan=True,
        )
        np.testing.assert_allclose(
            old_analysis["z_mesh"][:, 0],
            new_z_mesh_dict["z"],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Z-mesh should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["z_mesh"][:, 1],
            new_z_mesh_dict["E"],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Z-mesh should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["z_mesh"][:, 2],
            new_z_mesh_dict["nu"],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Z-mesh should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["z_mesh"][:, 3],
            new_z_mesh_dict["rho"] * 1e12,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Z-mesh should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["sxx"],
            new_sxx,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Sxx stress should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["txz"],
            new_txz,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Txz stress should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["szz"],
            new_szz,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Szz stress should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["principal_stress_slab"],
            new_principal_stress_slab,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Principal slab stress should be very similar",
        )

    def test_simple_two_layer_setup_with_touchdown(self):
        """
        Test that old and new implementations produce identical results
        for a simple two-layer setup with touchdown=True.
        """
        # --- Setup for OLD implementation (published weac==2.6.X) ---
        profile = [
            [200, 150],
            [300, 100],
        ]
        inclination = 30.0
        total_length = 14000.0
        try:
            _, old_state, old_z, old_analysis = compute_reference_model_results(
                system="pst-",
                layers_profile=profile,
                touchdown=True,
                L=total_length,
                a=4000,
                m=0,
                phi=inclination,
                set_foundation={"t": 20, "E": 0.35, "nu": 0.1},
            )
        except RuntimeError as exc:
            self.skipTest(f"Old weac environment unavailable: {exc}")

        # --- Setup for NEW implementation (main_weac2.py style) ---
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
            phi=inclination, system_type="pst-", cut_length=4000
        )
        weak_layer = WeakLayer(
            rho=50, h=20, E=0.35, nu=0.1, G_Ic=1
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

        # Calculate z-vector for each segment using its actual length
        z_parts = []
        for i, segment in enumerate(new_system.scenario.segments):
            length = segment.length
            x_coords = [0, length / 2, length]
            z_segment = new_system.z(
                x=x_coords,
                C=new_constants[:, [i]],
                length=length,
                phi=inclination,
                has_foundation=segment.has_foundation,
            )
            z_parts.append(z_segment)
        new_z = np.hstack(z_parts)

        # --- Analysis for NEW implementation ---
        analyzer = Analyzer(new_system, printing_enabled=False)
        new_raster_x, new_raster_z, new_raster_xb = analyzer.rasterize_solution(num=100)
        new_z_mesh_dict = analyzer.get_zmesh(dz=2)
        new_sxx = analyzer.Sxx(new_raster_z, inclination, dz=2, unit="kPa")
        new_txz = analyzer.Txz(new_raster_z, inclination, dz=2, unit="kPa")
        new_szz = analyzer.Szz(new_raster_z, inclination, dz=2, unit="kPa")
        new_principal_stress_slab = analyzer.principal_stress_slab(
            new_raster_z, inclination, dz=2, val="max", unit="kPa", normalize=False
        )

        # Compare the WeakLayer attributes
        self.assertEqual(
            old_state["weak"]["nu"],
            new_system.weak_layer.nu,
            "Weak layer Poisson's ratio should be the same",
        )
        self.assertEqual(
            old_state["weak"]["E"],
            new_system.weak_layer.E,
            "Weak layer Young's modulus should be the same",
        )
        self.assertEqual(
            old_state["t"],
            new_system.weak_layer.h,
            "Weak layer thickness should be the same",
        )
        self.assertEqual(
            old_state["kn"],
            new_system.weak_layer.kn,
            "Weak layer normal stiffness should be the same",
        )
        self.assertEqual(
            old_state["kt"],
            new_system.weak_layer.kt,
            "Weak layer shear stiffness should be the same",
        )

        # Compare the Slab Touchdown attributes
        self.assertEqual(
            old_state["touchdown"]["tc"],
            new_system.scenario.crack_h,
            "Crack height should be the same",
        )
        self.assertEqual(
            old_state["touchdown"]["a1"],
            new_system.slab_touchdown.l_AB,
            "Transition length A should be the same",
        )
        self.assertEqual(
            old_state["touchdown"]["a2"],
            new_system.slab_touchdown.l_BC,
            "Transition length B should be the same",
        )
        self.assertEqual(
            old_state["touchdown"]["td"],
            new_system.slab_touchdown.touchdown_distance,
            "Touchdown distance should be the same",
        )

        # Compare the Slab properties
        self.assertEqual(
            old_state["h"], new_system.slab.H, "Slab thickness should be the same"
        )
        self.assertEqual(
            old_state["zs"],
            new_system.slab.z_cog,
            "Slab center of gravity should be the same",
        )

        # Compare the Layer properties
        old_slab = (
            np.asarray(old_state["slab"]) if old_state["slab"] is not None else None
        )
        self.assertIsNotNone(old_slab, "Old slab data should be available")
        if old_slab is not None:
            np.testing.assert_array_equal(
                old_slab[:, 0] * 1e-12,
                new_system.slab.rhoi,
                "Layer density should be the same",
            )
            np.testing.assert_array_equal(
                old_slab[:, 1],
                new_system.slab.hi,
                "Layer thickness should be the same",
            )
            np.testing.assert_array_equal(
                old_slab[:, 2],
                new_system.slab.Ei,
                "Layer Young's modulus should be the same",
            )
            np.testing.assert_array_equal(
                old_slab[:, 3],
                new_system.slab.Gi,
                "Layer shear modulus should be the same",
            )
            np.testing.assert_array_equal(
                old_slab[:, 4],
                new_system.slab.nui,
                "Layer Poisson's ratio should be the same",
            )

        # Compare all the attributes of the old and new model
        self.assertEqual(
            old_state["a"],
            new_system.scenario.cut_length,
            "Cut length should be the same",
        )

        # --- Compare results ---
        self.assertEqual(
            old_z.shape,
            new_z.shape,
            "Result arrays should have the same shape",
        )

        # Numerical differences lie in the absolute realm of e-12
        np.testing.assert_allclose(
            old_z,
            new_z,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Old and new implementations should produce very similar results",
        )

        # Compare analysis results
        np.testing.assert_allclose(
            old_analysis["raster_x"],
            new_raster_x,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Rasterized x-coordinates should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["raster_z"],
            new_raster_z,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Rasterized z-solutions should be very similar",
        )
        # For raster_xb, we need to handle NaNs
        np.testing.assert_allclose(
            old_analysis["raster_xb"],
            new_raster_xb,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Rasterized founded x-coordinates should be very similar",
            equal_nan=True,
        )
        np.testing.assert_allclose(
            old_analysis["z_mesh"][:, 0],
            new_z_mesh_dict["z"],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Z-mesh should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["z_mesh"][:, 1],
            new_z_mesh_dict["E"],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Z-mesh should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["z_mesh"][:, 2],
            new_z_mesh_dict["nu"],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Z-mesh should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["z_mesh"][:, 3],
            new_z_mesh_dict["rho"] * 1e12,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Z-mesh should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["sxx"],
            new_sxx,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Sxx stress should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["txz"],
            new_txz,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Txz stress should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["szz"],
            new_szz,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Szz stress should be very similar",
        )
        np.testing.assert_allclose(
            old_analysis["principal_stress_slab"],
            new_principal_stress_slab,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Principal slab stress should be very similar",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
