"""
Unit tests for Layer and WeakLayer components.

Tests validation, automatic property calculations, and edge cases.
"""

import unittest
from pydantic import ValidationError

from weac_2.components.layer import (
    Layer,
    WeakLayer,
    _bergfeld_youngs_modulus,
    _scapozza_youngs_modulus,
    _gerling_youngs_modulus,
)


class TestLayerPropertyCalculations(unittest.TestCase):
    """Test the layer property calculation functions."""

    def test_bergfeld_calculation(self):
        """Test Bergfeld Young's modulus calculation."""
        # Test with standard ice density
        E = _bergfeld_youngs_modulus(rho=917.0)  # Ice density
        self.assertGreater(E, 0, "Young's modulus should be positive")
        self.assertIsInstance(E, float, "Result should be a float")

        # Test with typical snow densities
        E_light = _bergfeld_youngs_modulus(rho=100.0)
        E_heavy = _bergfeld_youngs_modulus(rho=400.0)
        self.assertLess(E_light, E_heavy, "Heavier snow should have higher modulus")

    def test_scapozza_calculation(self):
        """Test Scapozza Young's modulus calculation."""
        E = _scapozza_youngs_modulus(rho=200.0)
        self.assertGreater(E, 0, "Young's modulus should be positive")

    def test_gerling_calculation(self):
        """Test Gerling Young's modulus calculation."""
        E = _gerling_youngs_modulus(rho=250.0)
        self.assertGreater(E, 0, "Young's modulus should be positive")


class TestLayer(unittest.TestCase):
    """Test the Layer class functionality."""

    def test_layer_creation_with_required_fields(self):
        """Test creating a layer with only required fields."""
        layer = Layer(rho=200.0, h=100.0)

        # Check required fields
        self.assertEqual(layer.rho, 200.0)
        self.assertEqual(layer.h, 100.0)

        # Check auto-calculated fields
        self.assertIsNotNone(layer.E, "Young's modulus should be auto-calculated")
        self.assertIsNotNone(layer.G, "Shear modulus should be auto-calculated")
        self.assertGreater(layer.E, 0, "Young's modulus should be positive")
        self.assertGreater(layer.G, 0, "Shear modulus should be positive")

        # Check default Poisson's ratio
        self.assertEqual(layer.nu, 0.25, "Default Poisson's ratio should be 0.25")

    def test_layer_creation_with_all_fields(self):
        """Test creating a layer with all fields specified."""
        layer = Layer(rho=250.0, h=150.0, nu=0.3, E=50.0, G=20.0)

        self.assertEqual(layer.rho, 250.0)
        self.assertEqual(layer.h, 150.0)
        self.assertEqual(layer.nu, 0.3)
        self.assertEqual(layer.E, 50.0, "Specified E should override auto-calculation")
        self.assertEqual(layer.G, 20.0, "Specified G should override auto-calculation")

    def test_layer_validation_errors(self):
        """Test that invalid layer parameters raise ValidationError."""
        # Negative density
        with self.assertRaises(ValidationError):
            Layer(rho=-100.0, h=100.0)

        # Zero thickness
        with self.assertRaises(ValidationError):
            Layer(rho=200.0, h=0.0)

        # Invalid Poisson's ratio (>= 0.5)
        with self.assertRaises(ValidationError):
            Layer(rho=200.0, h=100.0, nu=0.5)

        # Negative Young's modulus
        with self.assertRaises(ValidationError):
            Layer(rho=200.0, h=100.0, E=-10.0)

    def test_layer_immutability(self):
        """Test that Layer objects are immutable (frozen)."""
        layer = Layer(rho=200.0, h=100.0)

        with self.assertRaises(ValidationError):
            layer.rho = 300.0  # Should fail due to frozen=True

    def test_shear_modulus_calculation(self):
        """Test automatic shear modulus calculation from E and nu."""
        layer = Layer(rho=200.0, h=100.0, nu=0.25, E=100.0)

        # G = E / (2 * (1 + nu))
        expected_G = 100.0 / (2 * (1 + 0.25))
        self.assertAlmostEqual(layer.G, expected_G, places=5)


class TestWeakLayer(unittest.TestCase):
    """Test the WeakLayer class functionality."""

    def test_weak_layer_creation_minimal(self):
        """Test creating a weak layer with minimal required fields."""
        wl = WeakLayer(rho=50.0, h=10.0)

        # Check required fields
        self.assertEqual(wl.rho, 50.0)
        self.assertEqual(wl.h, 10.0)

        # Check auto-calculated fields
        self.assertIsNotNone(wl.E, "Young's modulus should be auto-calculated")
        self.assertIsNotNone(wl.G, "Shear modulus should be auto-calculated")
        self.assertIsNotNone(wl.kn, "Normal stiffness should be auto-calculated")
        self.assertIsNotNone(wl.kt, "Shear stiffness should be auto-calculated")

        # Check default fracture properties
        self.assertEqual(wl.G_c, 1.0)
        self.assertEqual(wl.G_Ic, 0.56)
        self.assertEqual(wl.G_IIc, 0.79)

    def test_weak_layer_stiffness_calculations(self):
        """Test weak layer stiffness calculations."""
        wl = WeakLayer(rho=100.0, h=20.0, E=10.0, nu=0.2)

        # kn = E_plane / h = E / (1 - nu²) / h
        E_plane = 10.0 / (1 - 0.2**2)
        expected_kn = E_plane / 20.0
        self.assertAlmostEqual(wl.kn, expected_kn, places=5)

        # kt = G / h
        expected_G = 10.0 / (2 * (1 + 0.2))
        expected_kt = expected_G / 20.0
        self.assertAlmostEqual(wl.kt, expected_kt, places=5)

    def test_weak_layer_custom_stiffnesses(self):
        """Test weak layer with custom stiffness values."""
        wl = WeakLayer(rho=80.0, h=15.0, kn=5.0, kt=3.0)

        self.assertEqual(wl.kn, 5.0, "Custom kn should override calculation")
        self.assertEqual(wl.kt, 3.0, "Custom kt should override calculation")

    def test_weak_layer_fracture_properties(self):
        """Test weak layer fracture property validation."""
        wl = WeakLayer(rho=90.0, h=25.0, G_c=2.5, G_Ic=1.5, G_IIc=1.8)

        self.assertEqual(wl.G_c, 2.5)
        self.assertEqual(wl.G_Ic, 1.5)
        self.assertEqual(wl.G_IIc, 1.8)

    def test_weak_layer_validation_errors(self):
        """Test weak layer validation errors."""
        # Negative fracture energy
        with self.assertRaises(ValidationError):
            WeakLayer(rho=100.0, h=20.0, G_c=-1.0)

        # Zero thickness
        with self.assertRaises(ValidationError):
            WeakLayer(rho=100.0, h=0.0)


class TestLayerPhysicalConsistency(unittest.TestCase):
    """Test physical consistency of layer calculations."""

    def test_layer_density_modulus_relationship(self):
        """Test that higher density leads to higher modulus."""
        layer_light = Layer(rho=150.0, h=100.0)
        layer_heavy = Layer(rho=350.0, h=100.0)

        self.assertLess(
            layer_light.E,
            layer_heavy.E,
            "Heavier snow should have higher Young's modulus",
        )
        self.assertLess(
            layer_light.G,
            layer_heavy.G,
            "Heavier snow should have higher shear modulus",
        )

    def test_weak_layer_thickness_stiffness_relationship(self):
        """Test that thicker weak layers have lower stiffness."""
        wl_thin = WeakLayer(rho=100.0, h=10.0)
        wl_thick = WeakLayer(rho=100.0, h=30.0)

        self.assertGreater(
            wl_thin.kn,
            wl_thick.kn,
            "Thinner weak layer should have higher normal stiffness",
        )
        self.assertGreater(
            wl_thin.kt,
            wl_thick.kt,
            "Thinner weak layer should have higher shear stiffness",
        )

    def test_poisson_ratio_bounds(self):
        """Test Poisson's ratio physical bounds."""
        # Test upper bound (must be < 0.5 for positive definite stiffness)
        with self.assertRaises(ValidationError):
            Layer(rho=200.0, h=100.0, nu=0.5)

        with self.assertRaises(ValidationError):
            Layer(rho=200.0, h=100.0, nu=0.6)

        # Test lower bound (must be >= 0)
        with self.assertRaises(ValidationError):
            Layer(rho=200.0, h=100.0, nu=-0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
