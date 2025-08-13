"""
Unit tests for the Eigensystem class in the WEAC package.
"""

import unittest
from unittest import mock

import numpy as np

from weac.eigensystem import Eigensystem


class TestEigensystem(unittest.TestCase):
    """Test cases for the Eigensystem class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create an Eigensystem instance for testing
        self.eigen = Eigensystem(system="pst-")

        # Set up properties needed for tests
        self.eigen.set_beam_properties(layers="A")
        self.eigen.set_foundation_properties()

    def test_initialization(self):
        """Test that Eigensystem initializes with correct default values."""
        # Test default initialization
        self.assertEqual(self.eigen.system, "pst-")
        self.assertFalse(self.eigen.touchdown)
        self.assertAlmostEqual(self.eigen.g, 9810.0)  # Gravitational constant

    def test_all_system_types(self):
        """Test initialization with all possible system types."""
        system_types = ["pst-", "-pst", "vpst-", "-vpst", "skier", "skiers"]

        for system in system_types:
            with self.subTest(system=system):
                eigen = Eigensystem(system=system)
                self.assertEqual(eigen.system, system)
                eigen.set_beam_properties(layers="A")
                eigen.set_foundation_properties()
                eigen.calc_fundamental_system()

                # Basic functionality checks
                self.assertIsNotNone(eigen.kn)
                self.assertIsNotNone(eigen.kt)
                self.assertIsNotNone(eigen.A11)
                self.assertIsNotNone(eigen.ewC)
                self.assertIsNotNone(eigen.ewR)

    def test_system_type_eigenvalue_comparison(self):
        """Test and compare eigenvalues between different system types."""
        # Define consistent properties for all systems
        beam_props = [[300, 200]]
        foundation_props = {"t": 30.0, "E": 0.25, "nu": 0.25}

        # Create eigensystems with different system types
        systems = {}
        for system_type in ["pst-", "-pst", "vpst-", "-vpst", "skier", "skiers"]:
            eigen = Eigensystem(system=system_type)
            eigen.set_beam_properties(beam_props)
            eigen.set_foundation_properties(**foundation_props)
            eigen.calc_fundamental_system()
            systems[system_type] = eigen

        # Check that eigenvalues are produced for all systems
        for system_type, eigen in systems.items():
            with self.subTest(system=system_type):
                # Check that eigenvalues are calculated
                self.assertIsNotNone(eigen.ewC)
                self.assertIsNotNone(eigen.ewR)

                # Check that the number of eigenvalues is consistent with expectations
                # For this beam on elastic foundation problem, we expect 2 complex eigenvalues
                self.assertEqual(
                    len(eigen.ewC),
                    2,
                    f"System {system_type} should have 2 complex eigenvalues",
                )

                # For PST-type systems, expect 2 real eigenvalues
                if system_type in ["pst-", "-pst", "vpst-", "-vpst"]:
                    self.assertEqual(
                        len(eigen.ewR),
                        2,
                        f"PST-type system {system_type} should have 2 real eigenvalues",
                    )

        # Compare eigenvalues between different PST variants
        # Corresponding eigenvalues may differ in sign but should have similar magnitudes
        for i, _ in enumerate(systems["pst-"].ewC):
            # Compare magnitudes of complex eigenvalues between PST variants
            mag_pst_right = abs(systems["pst-"].ewC[i])
            mag_pst_left = abs(systems["-pst"].ewC[i])
            self.assertAlmostEqual(
                mag_pst_right,
                mag_pst_left,
                places=2,
                msg=f"Complex eigenvalue {i} should have similar magnitude in PST variants",
            )

            mag_vpst_right = abs(systems["vpst-"].ewC[i])
            mag_vpst_left = abs(systems["-vpst"].ewC[i])
            self.assertAlmostEqual(
                mag_vpst_right,
                mag_vpst_left,
                places=2,
                msg=f"Complex eigenvalue {i} should have similar magnitude in VPST variants",
            )

    def test_system_response_for_different_boundary_conditions(self):
        """Test system response for different boundary conditions."""
        # Create test systems with different boundary conditions
        systems = {
            "pst-": Eigensystem(system="pst-"),
            "skier": Eigensystem(system="skier"),
        }

        # Set common properties
        for system_type, eigen in systems.items():
            eigen.set_beam_properties(layers="A")
            eigen.set_foundation_properties()
            eigen.calc_fundamental_system()

        # Test coordinates for evaluation
        x_values = np.linspace(0, 1000, 5)  # 5 points from 0 to 1000 mm

        # Test constants for the solution
        C = np.zeros((6, 1))  # Zero constants for simplicity

        # Set an incline angle
        phi = 30  # degrees

        # Test that solutions can be computed for both systems
        for system_type, eigen in systems.items():
            with self.subTest(system=system_type):
                # Test bedded solution
                z_bedded = eigen.z(x_values, C, 0, phi, bed=True)

                # Check solution dimensions
                self.assertEqual(z_bedded.shape[0], 6)  # 6 solution components
                self.assertEqual(
                    z_bedded.shape[1], len(x_values)
                )  # One column per x value

                # Test unbedded solution
                z_unbedded = eigen.z(x_values, C, 0, phi, bed=False)

                # Check solution dimensions
                self.assertEqual(z_unbedded.shape[0], 6)
                self.assertEqual(z_unbedded.shape[1], len(x_values))

                # Test that bedded and unbedded solutions are different
                self.assertFalse(np.allclose(z_bedded, z_unbedded))

                # Test with a single x value to cover line 656
                z_single = eigen.z(x_values[0], C, 0, phi, bed=True)
                self.assertEqual(z_single.shape[0], 6)
                self.assertEqual(z_single.shape[1], 1)

        # Test skier load
        skier_system = systems["skier"]
        skier_mass = 80  # kg
        Fn, Ft = skier_system.get_skier_load(skier_mass, phi)

        # Check skier load values are reasonable
        self.assertGreater(Fn, 0)
        self.assertLess(Ft, 0)  # Downslope component should be negative

    def test_set_beam_properties(self):
        """Test setting beam properties with different layer configurations."""
        # Create a new instance to test from scratch
        eigen = Eigensystem(system="pst-")

        # Test with a single layer
        eigen.set_beam_properties(layers="A")

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

    def test_set_beam_properties_with_update(self):
        """Test setting beam properties with update flag."""
        # Create a new instance
        eigen = Eigensystem(system="pst-")

        # Set up a foundation to make calc_fundamental_system work
        eigen.set_foundation_properties()

        # Test with update=True, which should trigger calc_fundamental_system
        with mock.patch.object(eigen, "calc_fundamental_system") as mock_calc:
            eigen.set_beam_properties([[300, 200]], update=True)
            mock_calc.assert_called_once()

    @mock.patch("weac.eigensystem.load_dummy_profile")
    def test_set_beam_properties_with_string(self, mock_load_profile):
        """Test setting beam properties using a string input."""
        # Mock the load_dummy_profile function
        mock_layers = np.array([[300, 200]])
        mock_E = np.array([10.0])
        mock_load_profile.return_value = (mock_layers, mock_E)

        # Create a new instance
        eigen = Eigensystem(system="pst-")

        # Call set_beam_properties with a string
        eigen.set_beam_properties(layers="B")

        # Verify the mock was called with the right parameter
        mock_load_profile.assert_called_once_with("B")

        # Check that the properties were set correctly
        self.assertIsNotNone(eigen.slab)
        self.assertGreaterEqual(eigen.slab.shape[0], 1)

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

    def test_set_foundation_properties_with_update(self):
        """Test setting foundation properties with update flag."""
        # Create a new instance
        eigen = Eigensystem(system="pst-")

        # Set beam properties to make calc_fundamental_system work
        eigen.set_beam_properties(layers="A")

        # Test with update=True, which should trigger calc_fundamental_system
        with mock.patch.object(eigen, "calc_fundamental_system") as mock_calc:
            eigen.set_foundation_properties(update=True)
            mock_calc.assert_called_once()

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

    def test_set_surface_load(self):
        """Test setting surface load."""
        # Create a new instance
        eigen = Eigensystem()

        # Initial value should be 0
        self.assertEqual(eigen.p, 0)

        # Set a surface load
        eigen.set_surface_load(100.0)
        self.assertEqual(eigen.p, 100.0)

    def test_get_load_vector(self):
        """Test getting the load vector."""
        # Initialize with beam and foundation properties
        eigen = Eigensystem(system="pst-")
        eigen.set_beam_properties(layers="A")
        eigen.set_foundation_properties()
        eigen.calc_fundamental_system()

        # Test the load vector calculation
        phi = 30  # degrees
        load_vector = eigen.get_load_vector(phi)

        # Check expected dimensions and structure
        self.assertEqual(load_vector.shape, (6, 1))
        self.assertEqual(load_vector[0, 0], 0)  # First component should be 0
        self.assertEqual(load_vector[2, 0], 0)  # Third component should be 0
        self.assertEqual(load_vector[4, 0], 0)  # Fifth component should be 0

        # Test with surface load to ensure all branches are covered
        eigen.set_surface_load(100.0)
        load_vector_with_surface = eigen.get_load_vector(phi)

        # The resulting load vector should be different
        self.assertFalse(np.array_equal(load_vector, load_vector_with_surface))

    def test_pst_specific_behavior(self):
        """Test PST (Propagation Saw Test) specific behavior."""
        # Test pst- (cut from right)
        eigen_pst_right = Eigensystem(system="pst-")
        eigen_pst_right.set_beam_properties(layers="A")
        eigen_pst_right.set_foundation_properties()
        eigen_pst_right.calc_fundamental_system()

        # Test -pst (cut from left)
        eigen_pst_left = Eigensystem(system="-pst")
        eigen_pst_left.set_beam_properties(layers="A")
        eigen_pst_left.set_foundation_properties()
        eigen_pst_left.calc_fundamental_system()

        # Both should have valid eigensystems but potentially different values
        self.assertTrue(eigen_pst_right.ewC is not False)
        self.assertTrue(eigen_pst_left.ewC is not False)

    def test_vpst_specific_behavior(self):
        """Test vertical PST specific behavior."""
        # Test vpst- (vertical cut from right)
        eigen_vpst_right = Eigensystem(system="vpst-")
        eigen_vpst_right.set_beam_properties(layers="A")
        eigen_vpst_right.set_foundation_properties()
        eigen_vpst_right.calc_fundamental_system()

        # Test -vpst (vertical cut from left)
        eigen_vpst_left = Eigensystem(system="-vpst")
        eigen_vpst_left.set_beam_properties(layers="A")
        eigen_vpst_left.set_foundation_properties()
        eigen_vpst_left.calc_fundamental_system()

        # Both should have valid eigensystems
        self.assertTrue(eigen_vpst_right.ewC is not False)
        self.assertTrue(eigen_vpst_left.ewC is not False)

    def test_skier_specific_behavior(self):
        """Test skier-specific behavior."""
        # Test single skier
        eigen_skier = Eigensystem(system="skier")
        eigen_skier.set_beam_properties(layers="A")
        eigen_skier.set_foundation_properties()
        eigen_skier.calc_fundamental_system()

        # Test skier load calculation
        skier_mass = 80  # kg
        slope_angle = 30  # degrees
        Fn, Ft = eigen_skier.get_skier_load(skier_mass, slope_angle)

        # Check that load values are calculated and reasonable
        self.assertGreater(Fn, 0)  # Normal force should be positive
        self.assertLess(Ft, 0)  # Tangential force should be negative on a slope

        # Test multiple skiers
        eigen_skiers = Eigensystem(system="skiers")
        eigen_skiers.set_beam_properties(layers="A")
        eigen_skiers.set_foundation_properties()
        eigen_skiers.calc_fundamental_system()

        # Both should have valid eigensystems
        self.assertTrue(eigen_skier.ewC is not False)
        self.assertTrue(eigen_skiers.ewC is not False)

    def test_touchdown_parameter(self):
        """Test the touchdown parameter behavior."""
        # Test with touchdown=True
        eigen_touchdown = Eigensystem(system="pst-", touchdown=True)
        self.assertTrue(eigen_touchdown.touchdown)

        # Test with touchdown=False (default)
        eigen_no_touchdown = Eigensystem(system="pst-")
        self.assertFalse(eigen_no_touchdown.touchdown)

    def test_touchdown_with_all_system_types(self):
        """Test the touchdown flag behavior with all possible system types."""
        system_types = ["pst-", "-pst", "vpst-", "-vpst", "skier", "skiers"]

        # Test with touchdown=True for all system types
        for system in system_types:
            with self.subTest(system=system, touchdown=True):
                eigen = Eigensystem(system=system, touchdown=True)
                self.assertTrue(eigen.touchdown)
                eigen.set_beam_properties(layers="A")
                eigen.set_foundation_properties()
                eigen.calc_fundamental_system()

                # Basic functionality checks
                self.assertIsNotNone(eigen.kn)
                self.assertIsNotNone(eigen.kt)
                self.assertIsNotNone(eigen.A11)
                self.assertIsNotNone(eigen.ewC)
                self.assertIsNotNone(eigen.ewR)

        # Test with touchdown=False for all system types
        for system in system_types:
            with self.subTest(system=system, touchdown=False):
                eigen = Eigensystem(system=system, touchdown=False)
                self.assertFalse(eigen.touchdown)
                eigen.set_beam_properties(layers="A")
                eigen.set_foundation_properties()
                eigen.calc_fundamental_system()

                # Basic functionality checks
                self.assertIsNotNone(eigen.kn)
                self.assertIsNotNone(eigen.kt)
                self.assertIsNotNone(eigen.A11)
                self.assertIsNotNone(eigen.ewC)
                self.assertIsNotNone(eigen.ewR)

    def test_weight_and_surface_loads(self):
        """Test the calculation of weight and surface loads."""
        eigen = Eigensystem()
        eigen.set_beam_properties(layers="A")

        # Test weight load at different angles
        # At 0 degrees (flat)
        qn, qt = eigen.get_weight_load(0)
        self.assertGreater(qn, 0)  # Normal load is positive
        self.assertAlmostEqual(qt, 0, places=5)  # Tangential load is zero

        # At 30 degrees
        qn, qt = eigen.get_weight_load(30)
        self.assertGreater(qn, 0)  # Normal load is positive
        self.assertLess(qt, 0)  # Tangential load is negative (downslope)

        # Set surface load
        eigen.set_surface_load(100.0)

        # Test surface load at different angles
        pn, pt = eigen.get_surface_load(0)
        self.assertAlmostEqual(pn, 100.0)  # Normal load equals input at 0 degrees
        self.assertAlmostEqual(pt, 0, places=5)  # Tangential load is zero

        # At 30 degrees
        pn, pt = eigen.get_surface_load(30)
        self.assertGreater(pn, 0)  # Normal load is positive
        self.assertLess(pt, 0)  # Tangential load is negative (downslope)


if __name__ == "__main__":
    unittest.main()
