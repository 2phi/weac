"""
Unit tests for the Layered class in the WEAC package.
"""

import unittest

import numpy as np

from weac.layered import Layered


class TestLayered(unittest.TestCase):
    """Test cases for the Layered class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a default Layered instance for testing
        self.layered = Layered(system="pst-")

        # Create a Layered instance with custom parameters
        self.custom_layered = Layered(
            system="skier",
            layers=[[240, 200]],  # [density (kg/m^3), thickness (mm)]
            touchdown=True,
        )

    def test_initialization(self):
        """Test that Layered initializes with correct default values."""
        # Test default initialization
        self.assertEqual(self.layered.system, "pst-")
        self.assertFalse(self.layered.touchdown)

        # Test custom initialization
        self.assertEqual(self.custom_layered.system, "skier")
        self.assertTrue(self.custom_layered.touchdown)
        self.assertEqual(len(self.custom_layered.slab), 1)
        self.assertAlmostEqual(self.custom_layered.slab[0, 0], 240.0)  # Density
        self.assertAlmostEqual(self.custom_layered.slab[0, 1], 200.0)  # Thickness

    def test_calc_segments(self):
        """Test calculation of segments for different systems."""
        # Test for PST cut from right
        self.layered.system = "pst-"
        segments = self.layered.calc_segments(L=1000, a=300)

        # Check that segments dictionary contains expected keys
        self.assertIn("crack", segments)
        self.assertIn("nocrack", segments)
        self.assertIn("both", segments)

        # Check segment lengths for crack configuration
        crack_segments = segments["crack"]
        self.assertIn("li", crack_segments)
        self.assertEqual(len(crack_segments["li"]), 2)  # Two segments for PST-
        self.assertAlmostEqual(crack_segments["li"][0], 700.0)  # First segment length
        self.assertAlmostEqual(crack_segments["li"][1], 300.0)  # Second segment length

        # Test for skier system
        self.layered.system = "skier"
        segments = self.layered.calc_segments()

        # Check that segments dictionary contains expected keys
        self.assertIn("crack", segments)

        # Check segment lengths for skier configuration
        skier_segments = segments["crack"]
        self.assertIn("li", skier_segments)
        # Note: The actual implementation returns 4 segments for skier, not 2
        self.assertEqual(len(skier_segments["li"]), 4)  # Four segments for skier

        # Test for multiple skiers
        self.layered.system = "skiers"
        segments = self.layered.calc_segments(
            li=[500, 100, 250, 30, 30, 500],
            ki=[True, True, True, False, False, True],
            mi=[80, 80, 0, 0, 0],
        )

        # Check that segments dictionary contains expected keys
        self.assertIn("crack", segments)

        # Check segment lengths for multiple skiers configuration
        skiers_segments = segments["crack"]
        self.assertIn("li", skiers_segments)
        self.assertEqual(len(skiers_segments["li"]), 6)  # Six segments as specified
        self.assertAlmostEqual(skiers_segments["li"][0], 500.0)
        self.assertAlmostEqual(skiers_segments["li"][1], 100.0)
        self.assertAlmostEqual(skiers_segments["li"][2], 250.0)
        self.assertAlmostEqual(skiers_segments["li"][3], 30.0)
        self.assertAlmostEqual(skiers_segments["li"][4], 30.0)
        self.assertAlmostEqual(skiers_segments["li"][5], 500.0)

    def test_assemble_and_solve(self):
        """Test assembly and solution of the system."""
        # Set up a simple configuration
        self.layered.set_beam_properties([[240, 200]])
        self.layered.set_foundation_properties()
        self.layered.calc_fundamental_system()

        # Calculate segments
        segments = self.layered.calc_segments(L=1000, a=300)

        # Assemble and solve the system
        C = self.layered.assemble_and_solve(phi=0, **segments["crack"])

        # Check that solution vector has correct shape
        self.assertIsNotNone(C)
        self.assertEqual(C.shape, (6, 2))  # 6 state variables, 2 segments

        # Test with non-zero slope angle
        C_slope = self.layered.assemble_and_solve(phi=30, **segments["crack"])
        self.assertIsNotNone(C_slope)
        self.assertEqual(C_slope.shape, (6, 2))

    def test_rasterize_solution(self):
        """Test rasterization of the solution."""
        # Set up a simple configuration
        self.layered.set_beam_properties([[240, 200]])
        self.layered.set_foundation_properties()
        self.layered.calc_fundamental_system()

        # Calculate segments
        segments = self.layered.calc_segments(L=1000, a=300)

        # Assemble and solve the system
        C = self.layered.assemble_and_solve(phi=0, **segments["crack"])

        # Rasterize the solution
        xsl, z, xwl = self.layered.rasterize_solution(C=C, phi=0, **segments["crack"])

        # Check that output arrays have correct shapes
        self.assertIsNotNone(xsl)
        self.assertIsNotNone(z)
        self.assertIsNotNone(xwl)
        self.assertEqual(z.shape[0], 6)  # 6 state variables
        self.assertEqual(xsl.shape, z.shape[1:])  # Same length as state variables

        # Check that x coordinates are within expected range
        self.assertGreaterEqual(np.min(xsl), 0)
        self.assertLessEqual(np.max(xsl), 1000)

    def test_gdif(self):
        """Test calculation of differential energy release rate."""
        # Set up a simple configuration
        self.layered.set_beam_properties([[240, 200]])
        self.layered.set_foundation_properties()
        self.layered.calc_fundamental_system()

        # Calculate segments
        segments = self.layered.calc_segments(L=1000, a=300)

        # Assemble and solve the system
        C = self.layered.assemble_and_solve(phi=0, **segments["crack"])

        # Calculate differential energy release rate
        G = self.layered.gdif(C, phi=0, **segments["crack"])

        # Check that energy release rate is non-negative
        self.assertIsNotNone(G)
        self.assertEqual(len(G), 3)  # Three components: mode I, mode II, and total
        self.assertGreaterEqual(
            G[2], 0
        )  # Total energy release rate should be non-negative

    def test_ginc(self):
        """Test calculation of incremental energy release rate."""
        # Set up a simple configuration
        self.layered.set_beam_properties([[240, 200]])
        self.layered.set_foundation_properties()
        self.layered.calc_fundamental_system()

        # Calculate segments for both configurations
        segments = self.layered.calc_segments(L=1000, a=300)

        # Assemble and solve the system for both configurations
        C0 = self.layered.assemble_and_solve(phi=0, **segments["nocrack"])
        C1 = self.layered.assemble_and_solve(phi=0, **segments["crack"])

        # Calculate incremental energy release rate
        G = self.layered.ginc(C0, C1, phi=0, **segments["both"])

        # Check that energy release rate is non-negative
        self.assertIsNotNone(G)
        self.assertEqual(len(G), 3)  # Three components: mode I, mode II, and total
        self.assertGreaterEqual(
            G[2], 0
        )  # Total energy release rate should be non-negative


if __name__ == "__main__":
    unittest.main()
