"""
Unit tests for the Slab class.

Tests layer assembly, property calculations, center of gravity, and physical consistency.
"""

import unittest

import numpy as np

from weac.components import Layer
from weac.constants import G_MM_S2
from weac.core.slab import Slab


class TestSlabBasicOperations(unittest.TestCase):
    """Test basic slab assembly and property calculations."""

    def test_single_layer_slab(self):
        """Test slab with a single layer."""
        layer = Layer(rho=250, h=100)
        slab = Slab([layer])

        # Check basic properties
        self.assertEqual(len(slab.layers), 1)
        self.assertEqual(
            slab.H, 100.0, "Total thickness should equal single layer thickness"
        )
        self.assertEqual(slab.hi[0], 100.0)
        self.assertEqual(slab.rhoi[0], 250e-12, "Density should be converted to t/mm³")

        # Check coordinate system (z=0 at slab midpoint)
        self.assertEqual(slab.zi_mid[0], 0.0, "Single layer midpoint should be at z=0")
        self.assertEqual(slab.zi_bottom[0], 50.0, "Bottom should be H/2 below midpoint")

    def test_multi_layer_slab(self):
        """Test slab with multiple layers."""
        layers = [
            Layer(rho=150, h=50),  # Top layer
            Layer(rho=200, h=80),  # Middle layer
            Layer(rho=300, h=70),  # Bottom layer
        ]
        slab = Slab(layers)

        # Check total thickness
        expected_H = 50 + 80 + 70
        self.assertEqual(slab.H, expected_H)

        # Check layer thicknesses
        np.testing.assert_array_equal(slab.hi, [50, 80, 70])

        # Check densities (converted to t/mm³)
        expected_rho = np.array([150, 200, 300]) * 1e-12
        np.testing.assert_array_equal(slab.rhoi, expected_rho)

        # Check coordinate system
        # Layer midpoints calculated as: H/2 - sum(hi[j:n]) + hi[j]/2
        # For H=200, hi=[50,80,70]:
        # j=0: 100 - (50+80+70) + 50/2 = 100 - 200 + 25 = -75
        # j=1: 100 - (80+70) + 80/2 = 100 - 150 + 40 = -10
        # j=2: 100 - (70) + 70/2 = 100 - 70 + 35 = 65
        expected_zi_mid = [-75, -10, 65]
        np.testing.assert_array_almost_equal(slab.zi_mid, expected_zi_mid)

        # Layer bottom coordinates
        expected_zi_bottom = [-50, 30, 100]  # Cumulative from top, centered at midpoint
        np.testing.assert_array_almost_equal(slab.zi_bottom, expected_zi_bottom)


class TestSlabCenterOfGravity(unittest.TestCase):
    """Test center of gravity calculations."""

    def test_uniform_density_slab(self):
        """Test CoG for uniform density slab."""
        layers = [
            Layer(rho=200, h=100),
            Layer(rho=200, h=100),
        ]
        slab = Slab(layers)

        # For uniform density, CoG should be at geometric center (z=0)
        self.assertAlmostEqual(
            slab.z_cog,
            0.0,
            places=5,
            msg="Uniform density slab should have CoG at geometric center",
        )

    def test_density_gradient_slab(self):
        """Test CoG for slab with density gradient."""
        layers = [
            Layer(rho=110, h=100),  # Light top layer
            Layer(rho=400, h=100),  # Heavy bottom layer
        ]
        slab = Slab(layers)

        # CoG should shift toward heavier bottom layer (positive z)
        self.assertGreater(
            slab.z_cog, 0.0, "CoG should shift toward heavier bottom layer"
        )

    def test_top_heavy_slab(self):
        """Test CoG for top-heavy slab."""
        layers = [
            Layer(rho=400, h=100),  # Heavy top layer
            Layer(rho=110, h=100),  # Light bottom layer
        ]
        slab = Slab(layers)

        # CoG should shift toward heavier top layer (negative z)
        self.assertLess(slab.z_cog, 0.0, "CoG should shift toward heavier top layer")


class TestSlabWeightCalculations(unittest.TestCase):
    """Test weight and load calculations."""

    def test_weight_load_calculation(self):
        """Test calculation of weight load per unit length."""
        layers = [Layer(rho=200, h=100, E=50, G=20)]
        slab = Slab(layers)

        # qw = sum(rho * g * h) for all layers
        expected_qw = 200e-12 * G_MM_S2 * 100  # t/mm³ * mm/s² * mm = t*mm/s²/mm² = N/mm
        self.assertAlmostEqual(slab.qw, expected_qw, places=8)

    def test_multi_layer_weight(self):
        """Test weight calculation for multiple layers."""
        layers = [
            Layer(rho=150, h=60),
            Layer(rho=250, h=80),
            Layer(rho=350, h=100),
        ]
        slab = Slab(layers)

        # Calculate expected total weight per unit length
        expected_qw = (150 * 60 + 250 * 80 + 350 * 100) * 1e-12 * G_MM_S2
        self.assertAlmostEqual(slab.qw, expected_qw, places=8)


class TestSlabVerticalCenterOfGravity(unittest.TestCase):
    """Test vertical center of gravity calculations for inclined slabs."""

    def test_vertical_cog_flat_surface(self):
        """Test vertical CoG calculation for flat surface (phi=0)."""
        layers = [Layer(rho=200, h=100)]
        slab = Slab(layers)

        x_cog, z_cog, w = slab.calc_vertical_center_of_gravity(phi=0)

        # For flat surface, should have zero displacement and weight
        self.assertEqual(x_cog, 0.0)
        self.assertEqual(z_cog, 0.0)
        self.assertEqual(w, 0.0)

    def test_vertical_cog_inclined_surface(self):
        """Test vertical CoG calculation for inclined surface."""
        layers = [
            Layer(rho=200, h=50),
            Layer(rho=300, h=100),
        ]
        slab = Slab(layers)

        x_cog, z_cog, w = slab.calc_vertical_center_of_gravity(phi=30)

        # For inclined surface, should have non-zero values
        self.assertNotEqual(
            x_cog, 0.0, "Horizontal CoG should be non-zero for inclined surface"
        )
        self.assertNotEqual(
            z_cog, 0.0, "Vertical CoG should be non-zero for inclined surface"
        )
        self.assertGreater(w, 0.0, "Weight should be positive")

    def test_vertical_cog_steep_inclination(self):
        """Test vertical CoG for steep inclination."""
        layers = [Layer(rho=250, h=80)]
        slab = Slab(layers)

        x_cog_30, _, w_30 = slab.calc_vertical_center_of_gravity(phi=30)
        x_cog_60, _, w_60 = slab.calc_vertical_center_of_gravity(phi=60)

        # Steeper inclination should result in larger displacements and weights
        self.assertGreater(
            abs(x_cog_60),
            abs(x_cog_30),
            "Steeper inclination should increase horizontal displacement",
        )
        self.assertGreater(
            w_60,
            w_30,
            "Steeper inclination should increase weight of triangular segment",
        )


class TestSlabElasticProperties(unittest.TestCase):
    """Test elastic property assembly."""

    def test_elastic_property_arrays(self):
        """Test that elastic properties are correctly assembled."""
        layers = [
            Layer(rho=200, h=100, E=30, G=12, nu=0.25),
            Layer(rho=300, h=150, E=60, G=24, nu=0.25),
        ]
        slab = Slab(layers)

        # Check Young's moduli
        np.testing.assert_array_equal(slab.Ei, [30, 60])

        # Check shear moduli
        np.testing.assert_array_equal(slab.Gi, [12, 24])

        # Check Poisson's ratios
        np.testing.assert_array_equal(slab.nui, [0.25, 0.25])

    def test_automatic_property_calculation(self):
        """Test that properties are auto-calculated when not specified."""
        layers = [Layer(rho=250, h=120)]  # Only rho and h specified
        slab = Slab(layers)

        # Properties should be auto-calculated and positive
        self.assertGreater(
            slab.Ei[0], 0, "Young's modulus should be auto-calculated and positive"
        )
        self.assertGreater(
            slab.Gi[0], 0, "Shear modulus should be auto-calculated and positive"
        )
        self.assertEqual(slab.nui[0], 0.25, "Default Poisson's ratio should be 0.25")


class TestSlabPhysicalConsistency(unittest.TestCase):
    """Test physical consistency of slab calculations."""

    def test_coordinate_system_consistency(self):
        """Test that coordinate system is consistent."""
        layers = [
            Layer(rho=150, h=80),
            Layer(rho=200, h=60),
            Layer(rho=250, h=100),
        ]
        slab = Slab(layers)

        # Total thickness should equal sum of layer thicknesses
        self.assertEqual(slab.H, sum(slab.hi))

        # Bottom of last layer should be at H/2
        self.assertAlmostEqual(slab.zi_bottom[-1], slab.H / 2, places=5)

        # Top of first layer should be at -H/2
        # (first layer bottom - first layer thickness)
        top_of_first = slab.zi_bottom[0] - slab.hi[0]
        self.assertAlmostEqual(top_of_first, -slab.H / 2, places=5)

    def test_center_of_gravity_bounds(self):
        """Test that center of gravity is within slab bounds."""
        layers = [
            Layer(rho=110, h=50),  # Very light top
            Layer(rho=500, h=50),  # Very heavy bottom
        ]
        slab = Slab(layers)

        # CoG should be within slab thickness bounds
        self.assertGreaterEqual(
            slab.z_cog, -slab.H / 2, "CoG should be within slab (above top)"
        )
        self.assertLessEqual(
            slab.z_cog, slab.H / 2, "CoG should be within slab (below bottom)"
        )

    def test_mass_conservation(self):
        """Test that mass calculations are consistent."""
        layers = [
            Layer(rho=200, h=80),
            Layer(rho=300, h=120),
        ]
        slab = Slab(layers)

        # Calculate total mass per unit length
        total_mass_per_length = sum(layer.rho * 1e-12 * layer.h for layer in layers)

        # Weight per unit length should equal mass per length times gravity
        expected_weight = total_mass_per_length * G_MM_S2
        self.assertAlmostEqual(slab.qw, expected_weight, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
