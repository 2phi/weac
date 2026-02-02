"""
Unit tests for Layer and WeakLayer components.

Tests validation, automatic property calculations, and edge cases.
"""

import unittest

import numpy as np
from pydantic import ValidationError

from weac.components.layer import (
    Layer,
    WeakLayer,
    _adam_tensile_strength,
    _bergfeld_youngs_modulus,
    _gerling_youngs_modulus,
    _scapozza_youngs_modulus,
    _sigrist_tensile_strength,
)
from weac.constants import NU


class TestLayerPropertyCalculations(unittest.TestCase):
    """Test the layer property calculation functions."""

    def test_bergfeld_calculation(self):
        """Test Bergfeld Young's modulus calculation."""
        # Test with standard ice density
        E = _bergfeld_youngs_modulus(rho=917.0)  # Ice density
        self.assertGreater(E, 0, "Young's modulus should be positive")
        self.assertTrue(np.isscalar(E), "Result should be a scalar")

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


class TestTensileStrengthCalculations(unittest.TestCase):
    """Test tensile strength calculation functions."""

    def test_sigrist_calculation_kPa(self):
        """Test Sigrist tensile strength calculation in kPa."""
        # Test with typical snow density
        ts = _sigrist_tensile_strength(rho=200.0, unit="kPa")
        self.assertGreater(ts, 0, "Tensile strength should be positive")
        self.assertTrue(np.isscalar(ts), "Result should be a scalar")

        # Test with different densities
        ts_light = _sigrist_tensile_strength(rho=100.0, unit="kPa")
        ts_heavy = _sigrist_tensile_strength(rho=400.0, unit="kPa")
        self.assertLess(ts_light, ts_heavy, "Heavier snow should have higher strength")

    def test_sigrist_calculation_MPa(self):
        """Test Sigrist tensile strength calculation in MPa."""
        ts_kPa = _sigrist_tensile_strength(rho=200.0, unit="kPa")
        ts_MPa = _sigrist_tensile_strength(rho=200.0, unit="MPa")
        self.assertAlmostEqual(
            ts_kPa, ts_MPa * 1000, places=5, msg="Unit conversion should be correct"
        )

    def test_adam_calculation_kPa(self):
        """Test Adam tensile strength calculation in kPa."""
        # Test with typical snow density
        ts = _adam_tensile_strength(rho=300.0, unit="kPa")
        self.assertGreater(ts, 0, "Tensile strength should be positive")
        self.assertTrue(np.isscalar(ts), "Result should be a scalar")

        # Test with different densities
        ts_light = _adam_tensile_strength(rho=150.0, unit="kPa")
        ts_heavy = _adam_tensile_strength(rho=450.0, unit="kPa")
        self.assertLess(ts_light, ts_heavy, "Heavier snow should have higher strength")

    def test_adam_calculation_MPa(self):
        """Test Adam tensile strength calculation in MPa."""
        ts_kPa = _adam_tensile_strength(rho=300.0, unit="kPa")
        ts_MPa = _adam_tensile_strength(rho=300.0, unit="MPa")
        self.assertAlmostEqual(
            ts_kPa, ts_MPa * 1000, places=5, msg="Unit conversion should be correct"
        )

    def test_sigrist_vs_adam_comparison(self):
        """Compare Sigrist and Adam formulations at different densities."""
        # At low densities, compare the formulations
        rho_low = 150.0
        ts_sigrist = _sigrist_tensile_strength(rho=rho_low, unit="kPa")
        ts_adam = _adam_tensile_strength(rho=rho_low, unit="kPa")
        # Both should give positive values
        self.assertGreater(ts_sigrist, 0)
        self.assertGreater(ts_adam, 0)

        # At high densities
        rho_high = 400.0
        ts_sigrist_high = _sigrist_tensile_strength(rho=rho_high, unit="kPa")
        ts_adam_high = _adam_tensile_strength(rho=rho_high, unit="kPa")
        self.assertGreater(ts_sigrist_high, 0)
        self.assertGreater(ts_adam_high, 0)


class TestLayerTensileStrength(unittest.TestCase):
    """Test Layer class tensile strength functionality."""

    def test_layer_default_tensile_strength_method(self):
        """Test that default method is 'hybrid'."""
        layer = Layer(rho=200.0, h=100.0)
        self.assertEqual(
            layer.tensile_strength_method,
            "hybrid",
            "Default method should be 'hybrid'",
        )
        self.assertGreater(
            layer.tensile_strength, 0, "Tensile strength should be calculated"
        )

    def test_layer_sigrist_method(self):
        """Test Layer with explicit Sigrist method."""
        layer = Layer(rho=200.0, h=100.0, tensile_strength_method="sigrist")
        expected_ts = _sigrist_tensile_strength(rho=200.0, unit="kPa")
        self.assertAlmostEqual(
            layer.tensile_strength,
            expected_ts,
            places=5,
            msg="Tensile strength should match Sigrist calculation",
        )

    def test_layer_adam_method(self):
        """Test Layer with explicit Adam method."""
        layer = Layer(rho=300.0, h=100.0, tensile_strength_method="adam")
        expected_ts = _adam_tensile_strength(rho=300.0, unit="kPa")
        self.assertAlmostEqual(
            layer.tensile_strength,
            expected_ts,
            places=5,
            msg="Tensile strength should match Adam calculation",
        )

    def test_layer_hybrid_method_low_density(self):
        """Test hybrid method uses Sigrist for density < 250."""
        rho = 200.0  # Below 250 threshold
        layer = Layer(rho=rho, h=100.0, tensile_strength_method="hybrid")
        expected_ts = _sigrist_tensile_strength(rho=rho, unit="kPa")
        self.assertAlmostEqual(
            layer.tensile_strength,
            expected_ts,
            places=5,
            msg="Hybrid should use Sigrist for rho < 250",
        )

    def test_layer_hybrid_method_high_density(self):
        """Test hybrid method uses Adam for density >= 250."""
        rho = 300.0  # Above 250 threshold
        layer = Layer(rho=rho, h=100.0, tensile_strength_method="hybrid")
        expected_ts = _adam_tensile_strength(rho=rho, unit="kPa")
        self.assertAlmostEqual(
            layer.tensile_strength,
            expected_ts,
            places=5,
            msg="Hybrid should use Adam for rho >= 250",
        )

    def test_layer_hybrid_method_at_threshold(self):
        """Test hybrid method behavior exactly at 250 kg/m³."""
        rho = 250.0  # Exactly at threshold
        layer = Layer(rho=rho, h=100.0, tensile_strength_method="hybrid")
        expected_ts = _adam_tensile_strength(rho=rho, unit="kPa")
        self.assertAlmostEqual(
            layer.tensile_strength,
            expected_ts,
            places=5,
            msg="Hybrid should use Adam for rho = 250",
        )

    def test_layer_custom_tensile_strength(self):
        """Test that custom tensile strength overrides calculation."""
        custom_ts = 50.0
        layer = Layer(
            rho=200.0,
            h=100.0,
            tensile_strength=custom_ts,
            tensile_strength_method="sigrist",
        )
        self.assertEqual(
            layer.tensile_strength,
            custom_ts,
            "Custom tensile strength should override calculation",
        )


class TestTensileStrengthPhysicalConsistency(unittest.TestCase):
    """Test physical consistency of tensile strength calculations."""

    def test_density_strength_relationship(self):
        """Test that higher density leads to higher tensile strength."""
        layer_light = Layer(rho=150.0, h=100.0)
        layer_heavy = Layer(rho=350.0, h=100.0)

        self.assertLess(
            layer_light.tensile_strength,
            layer_heavy.tensile_strength,
            "Heavier snow should have higher tensile strength",
        )

    def test_hybrid_continuity_around_threshold(self):
        """Test continuity of hybrid method around 250 kg/m³ threshold."""
        # Test just below threshold
        layer_below = Layer(rho=249.0, h=100.0, tensile_strength_method="hybrid")
        # Test just above threshold
        layer_above = Layer(rho=251.0, h=100.0, tensile_strength_method="hybrid")

        # Both should have positive strength
        self.assertGreater(layer_below.tensile_strength, 0)
        self.assertGreater(layer_above.tensile_strength, 0)

        # Values should be reasonably close (within an order of magnitude)
        # This is a loose check since the formulations differ
        ratio = layer_above.tensile_strength / layer_below.tensile_strength
        self.assertLess(
            ratio, 10.0, "Strength shouldn't jump by more than 10x at threshold"
        )
        self.assertGreater(
            ratio, 0.1, "Strength shouldn't drop by more than 10x at threshold"
        )

    def test_all_methods_give_positive_strength(self):
        """Test that all methods produce positive tensile strength."""
        rho_values = [100.0, 200.0, 300.0, 400.0]
        methods = ["sigrist", "adam", "hybrid"]

        for rho in rho_values:
            for method in methods:
                layer = Layer(rho=rho, h=100.0, tensile_strength_method=method)
                self.assertGreater(
                    layer.tensile_strength,
                    0,
                    f"Method {method} with rho={rho} should give positive strength",
                )

    def test_tensile_strength_density_monotonicity(self):
        """Test that tensile strength increases monotonically with density."""
        densities = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
        methods = ["sigrist", "adam", "hybrid"]

        for method in methods:
            strengths = [
                Layer(rho=rho, h=100.0, tensile_strength_method=method).tensile_strength
                for rho in densities
            ]
            # Check that each strength is greater than the previous
            for i in range(1, len(strengths)):
                self.assertGreater(
                    strengths[i],
                    strengths[i - 1],
                    f"Strength should increase with density for {method} method",
                )


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
        self.assertEqual(layer.nu, NU, "Default Poisson's ratio should be 0.25")

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
        self.assertGreater(wl.E, 0, "Young's modulus should be positive")
        self.assertGreater(wl.G, 0, "Shear modulus should be positive")
        self.assertGreater(wl.kn, 0, "Normal stiffness should be positive")
        self.assertGreater(wl.kt, 0, "Shear stiffness should be positive")

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
