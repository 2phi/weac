"""
Unit tests for the tools module in the WEAC package.
"""

import unittest

import numpy as np

from weac.tools import bergfeld


class TestTools(unittest.TestCase):
    """Test cases for utility functions in the tools module."""

    def test_bergfeld(self):
        """Test the Bergfeld function for calculating Young's modulus from density."""
        # Test with a typical snow density
        density = 300  # kg/m^3
        young = bergfeld(density)

        # Check that the result is positive and within expected range
        self.assertGreater(young, 0)

        # Test with an array of densities
        densities = np.array([200, 300, 400])
        youngs = bergfeld(densities)

        # Check that the result is an array of the same shape
        self.assertEqual(youngs.shape, densities.shape)

        # Check that all values are positive and increasing with density
        self.assertTrue(np.all(youngs > 0))
        self.assertTrue(np.all(np.diff(youngs) > 0))

        # Test with zero density (should handle gracefully)
        zero_young = bergfeld(0)
        self.assertGreaterEqual(zero_young, 0)


if __name__ == "__main__":
    unittest.main()
