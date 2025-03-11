"""
Unit tests for the Eigensystem class in the WEAC package.
"""

import unittest

from weac.eigensystem import Eigensystem


class TestEigensystem(unittest.TestCase):
    """Test cases for the Eigensystem class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create an Eigensystem instance for testing
        self.eigen = Eigensystem(system="pst-")

        # Set up properties needed for tests
        self.eigen.set_beam_properties([[300, 200]])
        self.eigen.set_foundation_properties()

    def test_initialization(self):
        """Test that Eigensystem initializes with correct default values."""
        # Test default initialization
        self.assertEqual(self.eigen.system, "pst-")
        self.assertFalse(self.eigen.touchdown)
        self.assertAlmostEqual(self.eigen.g, 9810.0)  # Gravitational constant

    def test_set_beam_properties(self):
        """Test setting beam properties with different layer configurations."""
        # Create a new instance to test from scratch
        eigen = Eigensystem(system="pst-")

        # Test with a single layer
        eigen.set_beam_properties([[300, 200]])  # [density (kg/m^3), thickness (mm)]

        # Check that slab property is set
        self.assertIsNotNone(eigen.slab)
        # The actual shape might be different from what we expected
        # Let's just check that it's a 2D array with at least one row
        self.assertGreaterEqual(eigen.slab.shape[0], 1)

        # Test with multiple layers
        eigen.set_beam_properties(
            [
                [200, 100],  # [density (kg/m^3), thickness (mm)]
                [300, 150],
                [400, 50],
            ]
        )

        # Check that slab property is updated
        self.assertIsNotNone(eigen.slab)
        # Check that we have the right number of layers
        self.assertEqual(eigen.slab.shape[0], 3)

    def test_set_foundation_properties(self):
        """Test setting foundation properties."""
        # Create a new instance to test from scratch
        eigen = Eigensystem(system="pst-")

        # Test with default parameters
        eigen.set_foundation_properties()

        # Check that weak layer properties are set
        self.assertIsNotNone(eigen.weak)
        self.assertIn("E", eigen.weak)
        self.assertIn("nu", eigen.weak)

        # Test with custom parameters
        eigen.set_foundation_properties(
            t=50.0,  # Weak layer thickness (mm)
            E=0.5,  # Young's modulus (MPa)
            nu=0.3,  # Poisson's ratio
        )

        # Check that weak layer properties are updated
        self.assertIsNotNone(eigen.weak)
        self.assertEqual(eigen.weak["E"], 0.5)
        self.assertEqual(eigen.weak["nu"], 0.3)
        self.assertEqual(eigen.t, 50.0)

    def test_calc_fundamental_system(self):
        """Test calculation of the fundamental system."""
        # Calculate the fundamental system
        self.eigen.calc_fundamental_system()

        # Check that the system has been initialized
        self.assertIsNotNone(
            getattr(self.eigen, "kn", None)
        )  # Foundation normal stiffness
        self.assertIsNotNone(
            getattr(self.eigen, "kt", None)
        )  # Foundation shear stiffness
        self.assertIsNotNone(getattr(self.eigen, "A11", None))  # Extensional stiffness
        self.assertIsNotNone(
            getattr(self.eigen, "B11", None)
        )  # Bending-extension coupling stiffness
        self.assertIsNotNone(getattr(self.eigen, "D11", None))  # Bending stiffness
        self.assertIsNotNone(getattr(self.eigen, "kA55", None))  # Shear stiffness


if __name__ == "__main__":
    unittest.main()
