"""
Unit tests for the Eigensystem class.

Tests system matrix assembly, eigenvalue/eigenvector calculations,
complementary and particular solutions.
"""

import unittest

import numpy as np

from weac.components import Layer, WeakLayer
from weac.core.eigensystem import Eigensystem
from weac.core.slab import Slab


class TestEigensystemBasicProperties(unittest.TestCase):
    """Test basic eigensystem setup and property calculations."""

    def setUp(self):
        """Set up common test data."""
        self.layers = [Layer(rho=200, h=100), Layer(rho=300, h=150)]
        self.weak_layer = WeakLayer(rho=50, h=20, E=0.5, G_Ic=1.0)
        self.slab = Slab(self.layers)
        self.eigensystem = Eigensystem(self.weak_layer, self.slab)

    def test_eigensystem_initialization(self):
        """Test that eigensystem initializes correctly."""
        self.assertIsNotNone(self.eigensystem.weak_layer)
        self.assertIsNotNone(self.eigensystem.slab)

        # Check that eigenvalue calculation was performed
        self.assertIsNotNone(
            self.eigensystem.ewC, "Complex eigenvalues should be calculated"
        )
        self.assertIsNotNone(
            self.eigensystem.ewR, "Real eigenvalues should be calculated"
        )
        self.assertIsNotNone(
            self.eigensystem.evC, "Complex eigenvectors should be calculated"
        )
        self.assertIsNotNone(
            self.eigensystem.evR, "Real eigenvectors should be calculated"
        )

    def test_laminate_stiffness_parameters(self):
        """Test calculation of laminate stiffness parameters."""
        # Check that stiffness parameters are positive
        self.assertGreater(
            self.eigensystem.A11, 0, "Extensional stiffness should be positive"
        )
        self.assertGreater(
            self.eigensystem.D11, 0, "Bending stiffness should be positive"
        )
        self.assertGreater(
            self.eigensystem.kA55, 0, "Shear stiffness should be positive"
        )

        # K0 can be negative depending on coupling
        self.assertIsInstance(self.eigensystem.K0, float)

    def test_system_matrix_properties(self):
        """Test properties of the system matrix."""
        K = self.eigensystem.K

        # Check matrix dimensions
        self.assertEqual(K.shape, (6, 6), "System matrix should be 6x6")

        # Check that it's a real matrix
        self.assertTrue(np.all(np.isreal(K)), "System matrix should be real")

        # Check specific structure (first row should be [0, 1, 0, 0, 0, 0])
        expected_first_row = [0, 1, 0, 0, 0, 0]
        np.testing.assert_array_equal(
            K[0, :],
            expected_first_row,
            "First row of system matrix has known structure",
        )

        # Check third row should be [0, 0, 0, 1, 0, 0]
        expected_third_row = [0, 0, 0, 1, 0, 0]
        np.testing.assert_array_equal(
            K[2, :],
            expected_third_row,
            "Third row of system matrix has known structure",
        )

        # Check fifth row should be [0, 0, 0, 0, 0, 1]
        expected_fifth_row = [0, 0, 0, 0, 0, 1]
        np.testing.assert_array_equal(
            K[4, :],
            expected_fifth_row,
            "Fifth row of system matrix has known structure",
        )


class TestEigensystemEigenvalueCalculations(unittest.TestCase):
    """Test eigenvalue and eigenvector calculations."""

    def setUp(self):
        """Set up test eigensystem."""
        layers = [Layer(rho=250, h=120)]
        weak_layer = WeakLayer(rho=80, h=25, E=0.3)
        slab = Slab(layers)
        self.eigensystem = Eigensystem(weak_layer, slab)

    def test_eigenvalue_classification(self):
        """Test that eigenvalues are correctly classified."""
        # Real eigenvalues should be real
        self.assertTrue(
            np.all(np.isreal(self.eigensystem.ewR)),
            "Real eigenvalues should be real numbers",
        )

        # Complex eigenvalues should have positive imaginary parts
        if len(self.eigensystem.ewC) > 0:
            self.assertTrue(
                np.all(self.eigensystem.ewC.imag > 0),
                "Complex eigenvalues should have positive imaginary parts",
            )

    def test_eigenvector_dimensions(self):
        """Test that eigenvectors have correct dimensions."""
        # Real eigenvectors
        if len(self.eigensystem.ewR) > 0:
            self.assertEqual(
                self.eigensystem.evR.shape[0],
                6,
                "Real eigenvectors should be 6-dimensional",
            )
            self.assertEqual(
                self.eigensystem.evR.shape[1],
                len(self.eigensystem.ewR),
                "Number of real eigenvectors should match number of real eigenvalues",
            )

        # Complex eigenvectors
        if len(self.eigensystem.ewC) > 0:
            self.assertEqual(
                self.eigensystem.evC.shape[0],
                6,
                "Complex eigenvectors should be 6-dimensional",
            )
            self.assertEqual(
                self.eigensystem.evC.shape[1],
                len(self.eigensystem.ewC),
                "Number of complex eigenvectors should match number of complex eigenvalues",
            )

    def test_eigenvalue_shifts(self):
        """Test eigenvalue shift arrays."""
        # Shifts should have same length as eigenvalues
        self.assertEqual(
            len(self.eigensystem.sR),
            len(self.eigensystem.ewR),
            "Real shifts should match real eigenvalues",
        )
        self.assertEqual(
            len(self.eigensystem.sC),
            len(self.eigensystem.ewC),
            "Complex shifts should match complex eigenvalues",
        )

        # Shifts should be -1 or 0
        self.assertTrue(
            np.all(np.isin(self.eigensystem.sR, [-1, 0])),
            "Real shifts should be -1 or 0",
        )
        self.assertTrue(
            np.all(np.isin(self.eigensystem.sC, [-1, 0])),
            "Complex shifts should be -1 or 0",
        )


class TestEigensystemSolutionMethods(unittest.TestCase):
    """Test complementary and particular solution methods."""

    def setUp(self):
        """Set up test eigensystem."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=60, h=15)
        slab = Slab(layers)
        self.eigensystem = Eigensystem(weak_layer, slab)

    def test_complementary_solution_bedded(self):
        """Test complementary solution for bedded segment."""
        x = 100.0  # Position
        length = 1000.0  # Segment length
        has_foundation = True  # Bedded

        zh = self.eigensystem.zh(x, length, has_foundation)

        # Should return 6x6 matrix
        self.assertEqual(
            zh.shape, (6, 6), "Complementary solution should be 6x6 matrix"
        )

        # Should be real for bedded segments
        self.assertTrue(
            np.allclose(np.imag(zh), 0.0, atol=1e-12),
            "Bedded complementary solution should be (numerically) real",
        )

    def test_complementary_solution_free(self):
        """Test complementary solution for free segment."""
        x = 50.0  # Position
        length = 500.0  # Segment length
        has_foundation = False  # Free

        zh = self.eigensystem.zh(x, length, has_foundation)

        # Should return 6x6 matrix
        self.assertEqual(
            zh.shape, (6, 6), "Complementary solution should be 6x6 matrix"
        )

        self.assertTrue(
            np.allclose(np.imag(zh), 0.0, atol=1e-12),
            "Free complementary solution should be (numerically) real",
        )

    def test_complementary_solution_at_origin(self):
        """Test complementary solution at x=0."""
        zh_bedded = self.eigensystem.zh(0.0, 1000.0, True)
        zh_free = self.eigensystem.zh(0.0, 1000.0, False)

        # At x=0, certain columns should have specific values
        # For free segments, the polynomial form gives specific patterns
        self.assertTrue(
            np.isfinite(zh_bedded).all(), "Bedded solution should be finite at origin"
        )
        self.assertTrue(
            np.isfinite(zh_free).all(), "Free solution should be finite at origin"
        )

    def test_particular_solution_bedded(self):
        """Test particular solution for bedded segment."""
        x = 200.0  # Position
        phi = 30.0  # Inclination
        has_foundation = True  # Bedded
        qs = 5.0  # Surface load

        zp = self.eigensystem.zp(x, phi, has_foundation, qs)

        # Should return 6x1 vector
        self.assertEqual(zp.shape, (6, 1), "Particular solution should be 6x1 vector")
        # Should be real
        self.assertTrue(
            np.allclose(np.imag(zp), 0.0, atol=1e-12),
            "Particular solution should be (numerically) real",
        )

    def test_particular_solution_free(self):
        """Test particular solution for free segment."""
        x = 150.0  # Position
        phi = 25.0  # Inclination
        has_foundation = False  # Free
        qs = 0.0  # No additional surface load

        zp = self.eigensystem.zp(x, phi, has_foundation, qs)

        # Should be real
        self.assertTrue(
            np.allclose(np.imag(zp), 0.0, atol=1e-12),
            "Particular solution should be (numerically) real",
        )

    def test_load_vector_calculation(self):
        """Test system load vector calculation."""
        phi = 20.0  # Inclination
        qs = 10.0  # Surface load

        q = self.eigensystem.get_load_vector(phi, qs)

        # Should return 6x1 vector
        self.assertEqual(q.shape, (6, 1), "Load vector should be 6x1")

        # Should be real
        self.assertTrue(
            np.allclose(np.imag(q), 0.0, atol=1e-12),
            "Load vector should be (numerically) real",
        )


class TestEigensystemPhysicalConsistency(unittest.TestCase):
    """Test physical consistency of eigensystem calculations."""

    def test_stiffness_scaling_with_properties(self):
        """Test that stiffness parameters scale correctly with material properties."""
        # Create two systems with different Young's moduli
        layers1 = [Layer(rho=200, h=100, E=50)]
        layers2 = [Layer(rho=200, h=100, E=100)]  # Double the modulus

        weak_layer = WeakLayer(rho=50, h=20)
        slab1 = Slab(layers1)
        slab2 = Slab(layers2)

        eig1 = Eigensystem(weak_layer, slab1)
        eig2 = Eigensystem(weak_layer, slab2)

        # Higher Young's modulus should lead to higher stiffnesses
        self.assertGreater(
            eig2.A11, eig1.A11, "Higher E should increase extensional stiffness"
        )
        self.assertGreater(
            eig2.D11, eig1.D11, "Higher E should increase bending stiffness"
        )

    def test_weak_layer_stiffness_influence(self):
        """Test that weak layer properties affect system behavior."""
        layers = [Layer(rho=250, h=120)]

        # Soft weak layer
        wl_soft = WeakLayer(rho=50, h=25, E=0.1)
        # Stiff weak layer
        wl_stiff = WeakLayer(rho=120, h=25, E=1.0)

        slab = Slab(layers)
        eig_soft = Eigensystem(wl_soft, slab)
        eig_stiff = Eigensystem(wl_stiff, slab)

        # Stiffness values should be different
        self.assertNotAlmostEqual(
            eig_soft.K[1, 0],
            eig_stiff.K[1, 0],
            msg="Different weak layer properties should affect system matrix",
        )

    def test_inclination_effect_on_loads(self):
        """Test that inclination affects load vectors correctly."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=50, h=20)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)

        # Compare load vectors for different inclinations
        q_flat = eigensystem.get_load_vector(phi=0.0, qs=0.0)
        q_inclined = eigensystem.get_load_vector(phi=30.0, qs=0.0)

        # Should be different for non-zero inclination
        self.assertFalse(
            np.allclose(q_flat, q_inclined),
            "Load vectors should differ for different inclinations",
        )

    def test_complementary_solution_continuity(self):
        """Test continuity of complementary solutions."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=50, h=20)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)

        # Test continuity for bedded segments
        x1, x2 = 100.0, 100.000001  # Very close points
        length = 1000.0

        zh1 = eigensystem.zh(x1, length, True)
        zh2 = eigensystem.zh(x2, length, True)

        # Solutions should be very close for nearby points
        self.assertTrue(
            np.allclose(zh1, zh2, atol=1e-6),
            "Complementary solutions should be continuous",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
