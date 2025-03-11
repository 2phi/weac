"""
Unit tests for the mixins module in the WEAC package.
"""

import unittest

import numpy as np

from weac.eigensystem import Eigensystem
from weac.mixins import FieldQuantitiesMixin, SlabContactMixin, SolutionMixin


class TestClass(FieldQuantitiesMixin, SolutionMixin, SlabContactMixin, Eigensystem):
    """Test class for mixin testing."""

    def __init__(self):
        """Initialize test class."""
        # Initialize parent class
        super().__init__(system="pst-", touchdown=False)

        # Create a 2D array for Z where the first index is the state variable
        # and the second index is the position
        self.Z = np.zeros((6, 5))  # 6 state variables, 5 positions
        for i in range(6):
            self.Z[i, :] = i + 1  # Each row has values [1,1,1,1,1], [2,2,2,2,2], etc.

        # Set required attributes for the mixins
        self.h = 200  # slab thickness in mm
        self.td = 0  # touchdown length
        self.t = 1  # weak layer thickness
        self.A11 = 1e6  # axial stiffness
        self.B11 = 1e4  # coupling stiffness
        self.D11 = 1e2  # bending stiffness
        self.kA55 = 1e5  # shear stiffness
        self.kn = 1e3  # normal foundation stiffness
        self.kt = 1e3  # tangential foundation stiffness
        self.system = "pst-"  # system type
        self.touchdown = False  # touchdown flag
        self.g = 9810  # gravity constant
        self.mode = "A"  # touchdown mode

        # Create slab properties array with columns:
        # density (kg/m^3), thickness (mm), Young's modulus (MPa), shear modulus (MPa), Poisson's ratio
        self.slab = np.array([[300, 200, 1e3, 4e2, 0.25]])

        self.p = 0  # surface line load
        self.phi = 0  # inclination angle


class TestFieldQuantitiesMixin(unittest.TestCase):
    """Test cases for FieldQuantitiesMixin."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_obj = TestClass()

    def test_w(self):
        """Test calculation of deflection."""
        # Test with default parameters
        result = self.test_obj.w(self.test_obj.Z)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5,))  # Should match number of positions
        self.assertTrue(np.allclose(result, 3))  # Third row of Z

        # Test with different units
        result_mm = self.test_obj.w(self.test_obj.Z, unit="mm")
        result_cm = self.test_obj.w(self.test_obj.Z, unit="cm")
        self.assertTrue(np.allclose(result_mm, result_cm * 10))

    def test_dw_dx(self):
        """Test calculation of deflection derivative."""
        # Test with default parameters
        result = self.test_obj.dw_dx(self.test_obj.Z)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5,))  # Should match number of positions
        self.assertTrue(np.allclose(result, 4))  # Fourth row of Z

    def test_psi(self):
        """Test calculation of rotation."""
        # Test with default parameters
        result = self.test_obj.psi(self.test_obj.Z)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5,))  # Should match number of positions
        self.assertTrue(np.allclose(result, 5))  # Fifth row of Z

        # Test with different units
        result_rad = self.test_obj.psi(self.test_obj.Z, unit="rad")
        result_deg = self.test_obj.psi(self.test_obj.Z, unit="degrees")
        self.assertTrue(np.allclose(result_rad, np.deg2rad(result_deg)))


class TestSolutionMixin(unittest.TestCase):
    """Test cases for SolutionMixin."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_obj = TestClass()

    def test_calc_segments(self):
        """Test calculation of segments."""
        # Test with default parameters
        crack_segments = self.test_obj.calc_segments(L=1000, a=300)

        # Check that the segments dictionary contains expected keys
        self.assertIn("crack", crack_segments)
        self.assertIn("li", crack_segments["crack"])
        self.assertIn("ki", crack_segments["crack"])
        self.assertIn("mi", crack_segments["crack"])

        # Check segment lengths
        self.assertEqual(
            len(crack_segments["crack"]["li"]), 2
        )  # Should have 2 segments for pst-
        self.assertEqual(crack_segments["crack"]["li"][0], 700)  # First segment length
        self.assertEqual(
            crack_segments["crack"]["li"][1], 300
        )  # Second segment length (crack length)


if __name__ == "__main__":
    unittest.main()
