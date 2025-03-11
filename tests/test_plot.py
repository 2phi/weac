"""
Unit tests for the plot module in the WEAC package.
"""

import os
import unittest

import matplotlib.pyplot as plt

import weac.plot
from weac.layered import Layered


class TestPlot(unittest.TestCase):
    """Test cases for visualization functions in the plot module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a Layered instance for testing
        self.layered = Layered(system="pst-")
        self.layered.set_beam_properties(
            [[300, 200]]
        )  # [density (kg/m^3), thickness (mm)]
        self.layered.set_foundation_properties()
        self.layered.calc_fundamental_system()

        # Calculate segments
        self.segments = self.layered.calc_segments(L=1000, a=300)

        # Assemble and solve the system
        self.C = self.layered.assemble_and_solve(phi=0, **self.segments["crack"])

        # Rasterize the solution
        self.xsl, self.z, self.xwl = self.layered.rasterize_solution(
            C=self.C, phi=0, **self.segments["crack"]
        )

        # Create plots directory if it doesn't exist
        if not os.path.exists("plots"):
            os.makedirs("plots")

    def tearDown(self):
        """Clean up after tests."""
        # Close all matplotlib figures to avoid memory leaks
        plt.close("all")

        # Clean up plot files
        plot_files = [
            "plots/profile.png",
            "plots/cont.png",
            "plots/disp.png",
            "plots/stress.png",
        ]
        for file in plot_files:
            if os.path.exists(file):
                os.remove(file)

    def test_slab_profile(self):
        """Test plotting of slab profile."""
        # Test with default parameters
        weac.plot.slab_profile(self.layered)

        # Check that the plot file was created
        self.assertTrue(os.path.exists("plots/profile.png"))

    def test_deformed(self):
        """Test plotting of deformed slab."""
        # Test with default parameters
        weac.plot.deformed(self.layered, xsl=self.xsl, xwl=self.xwl, z=self.z, phi=0)

        # Check that the plot file was created
        self.assertTrue(os.path.exists("plots/cont.png"))

        # Test with custom parameters
        weac.plot.deformed(
            self.layered,
            xsl=self.xsl,
            xwl=self.xwl,
            z=self.z,
            phi=0,
            scale=2.0,
            field="w",
            normalize=False,
        )

        # Check that the plot file was created
        self.assertTrue(os.path.exists("plots/cont.png"))

    def test_displacements(self):
        """Test plotting of displacements."""
        # Test with default parameters
        weac.plot.displacements(
            self.layered,
            x=self.xsl,
            z=self.z,
            li=self.segments["crack"]["li"],
            ki=self.segments["crack"]["ki"],
            mi=self.segments["crack"]["mi"],  # Add mi parameter
        )

        # Check that the plot file was created
        self.assertTrue(os.path.exists("plots/disp.png"))

    def test_stresses(self):
        """Test plotting of stresses."""
        # Test with default parameters
        weac.plot.stresses(
            self.layered,
            x=self.xwl,
            z=self.z,
            li=self.segments["crack"]["li"],
            ki=self.segments["crack"]["ki"],
            mi=self.segments["crack"]["mi"],  # Add mi parameter
        )

        # Check that the plot file was created
        self.assertTrue(os.path.exists("plots/stress.png"))


if __name__ == "__main__":
    unittest.main()
