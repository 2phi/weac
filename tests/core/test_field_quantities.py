"""
Unit tests for the FieldQuantities class.

Tests displacement calculations, stress calculations, energy release rates,
and other field quantity computations.
"""

import unittest
import numpy as np

from weac.components import Layer, WeakLayer
from weac.core.slab import Slab
from weac.core.eigensystem import Eigensystem
from weac.core.field_quantities import FieldQuantities


class TestFieldQuantitiesBasic(unittest.TestCase):
    """Test basic field quantity calculations."""

    def setUp(self):
        """Set up test eigensystem and field quantities."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=50, h=20, E=0.5)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)
        self.fq = FieldQuantities(eigensystem)

        # Create a simple test solution vector
        # [u, u', w, w', psi, psi'] at multiple points
        self.Z = np.array(
            [
                [1.0, 2.0, 3.0],  # u values at 3 points
                [0.1, 0.2, 0.3],  # u' values
                [0.5, 1.0, 1.5],  # w values
                [0.05, 0.1, 0.15],  # w' values
                [0.01, 0.02, 0.03],  # psi values
                [0.001, 0.002, 0.003],  # psi' values
            ]
        )

    def test_center_line_displacement(self):
        """Test center-line displacement calculation."""
        w_values = self.fq.w(self.Z)

        # Should return w values (row 2) in default units (mm)
        expected = self.Z[2, :]
        np.testing.assert_array_equal(
            w_values,
            expected,
            err_msg="Center-line displacement should equal w component",
        )

    def test_center_line_displacement_units(self):
        """Test center-line displacement with different units."""
        # Test different units
        w_mm = self.fq.w(self.Z, unit="mm")
        w_m = self.fq.w(self.Z, unit="m")
        w_cm = self.fq.w(self.Z, unit="cm")

        # Check unit conversions
        np.testing.assert_array_almost_equal(
            w_m * 1000,
            w_mm,
            decimal=10,
            err_msg="Meter to mm conversion should be correct",
        )
        np.testing.assert_array_almost_equal(
            w_cm * 10,
            w_mm,
            decimal=10,
            err_msg="Centimeter to mm conversion should be correct",
        )

    def test_center_line_displacement_derivative(self):
        """Test center-line displacement derivative."""
        dw_dx = self.fq.dw_dx(self.Z)

        # Should return w' values (row 3)
        expected = self.Z[3, :]
        np.testing.assert_array_equal(
            dw_dx, expected, err_msg="Displacement derivative should equal w' component"
        )

    def test_rotation_calculation(self):
        """Test rotation calculation."""
        psi_rad = self.fq.psi(self.Z, unit="rad")
        psi_deg = self.fq.psi(self.Z, unit="deg")

        # Radians should equal psi component
        expected_rad = self.Z[4, :]
        np.testing.assert_array_equal(
            psi_rad,
            expected_rad,
            err_msg="Rotation in radians should equal psi component",
        )

        # Degrees should be converted
        expected_deg = expected_rad * 180 / np.pi
        np.testing.assert_array_almost_equal(
            psi_deg,
            expected_deg,
            decimal=10,
            err_msg="Rotation conversion to degrees should be correct",
        )

    def test_rotation_derivative(self):
        """Test rotation derivative calculation."""
        dpsi_dx = self.fq.dpsi_dx(self.Z)

        # Should return psi' values (row 5)
        expected = self.Z[5, :]
        np.testing.assert_array_equal(
            dpsi_dx, expected, err_msg="Rotation derivative should equal psi' component"
        )


class TestFieldQuantitiesDisplacements(unittest.TestCase):
    """Test displacement calculations at different heights."""

    def setUp(self):
        """Set up test system."""
        layers = [Layer(rho=250, h=120)]
        weak_layer = WeakLayer(rho=60, h=25)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)
        self.fq = FieldQuantities(eigensystem)

        # Simple solution vector
        self.Z = np.array(
            [
                [2.0, 4.0],  # u values
                [0.2, 0.4],  # u' values
                [1.0, 2.0],  # w values
                [0.1, 0.2],  # w' values
                [0.05, 0.1],  # psi values
                [0.005, 0.01],  # psi' values
            ]
        )

    def test_displacement_at_different_heights(self):
        """Test horizontal displacement at different heights."""
        h0 = 30.0  # Height above centerline

        u_values = self.fq.u(self.Z, h0)

        # u = u0 + h0 * psi
        expected = self.Z[0, :] + h0 * self.Z[4, :]
        np.testing.assert_array_almost_equal(
            u_values,
            expected,
            decimal=10,
            err_msg="Displacement at height should follow u = u0 + h*psi",
        )

    def test_displacement_derivative_at_height(self):
        """Test displacement derivative at different heights."""
        h0 = 40.0

        du_dx = self.fq.du_dx(self.Z, h0)

        # du/dx = u0' + h0 * psi'
        expected = self.Z[1, :] + h0 * self.Z[5, :]
        np.testing.assert_array_almost_equal(
            du_dx,
            expected,
            decimal=10,
            err_msg="Displacement derivative should follow du/dx = u0' + h*psi'",
        )

    def test_displacement_at_centerline(self):
        """Test that displacement at centerline equals u0."""
        u_centerline = self.fq.u(self.Z, h0=0.0)

        # At centerline (h0=0), u = u0
        expected = self.Z[0, :]
        np.testing.assert_array_equal(
            u_centerline, expected, err_msg="Displacement at centerline should equal u0"
        )


class TestFieldQuantitiesStresses(unittest.TestCase):
    """Test stress and force calculations."""

    def setUp(self):
        """Set up test system with known properties."""
        layers = [Layer(rho=200, h=100, E=50, nu=0.25)]  # Known elastic properties
        weak_layer = WeakLayer(
            rho=50, h=20, E=0.5, kn=10.0, kt=5.0
        )  # Known stiffnesses
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)
        self.fq = FieldQuantities(eigensystem)

        # Test solution vector
        self.Z = np.array(
            [
                [1.0, 2.0],  # u values
                [0.1, 0.2],  # u' values
                [0.5, 1.0],  # w values
                [0.05, 0.1],  # w' values
                [0.01, 0.02],  # psi values
                [0.001, 0.002],  # psi' values
            ]
        )

    def test_axial_force_calculation(self):
        """Test axial normal force calculation."""
        N = self.fq.N(self.Z)

        # N = A11 * u' + B11 * psi'
        expected = self.fq.es.A11 * self.Z[1, :] + self.fq.es.B11 * self.Z[5, :]
        np.testing.assert_array_almost_equal(
            N,
            expected,
            decimal=10,
            err_msg="Axial force should follow N = A11*u' + B11*psi'",
        )

    def test_bending_moment_calculation(self):
        """Test bending moment calculation."""
        M = self.fq.M(self.Z)

        # M = B11 * u' + D11 * psi'
        expected = self.fq.es.B11 * self.Z[1, :] + self.fq.es.D11 * self.Z[5, :]
        np.testing.assert_array_almost_equal(
            M,
            expected,
            decimal=10,
            err_msg="Bending moment should follow M = B11*u' + D11*psi'",
        )

    def test_shear_force_calculation(self):
        """Test vertical shear force calculation."""
        V = self.fq.V(self.Z)

        # V = kA55 * (w' + psi)
        expected = self.fq.es.kA55 * (self.Z[3, :] + self.Z[4, :])
        np.testing.assert_array_almost_equal(
            V,
            expected,
            decimal=10,
            err_msg="Shear force should follow V = kA55*(w' + psi)",
        )

    def test_weak_layer_normal_stress(self):
        """Test weak layer normal stress calculation."""
        sig_MPa = self.fq.sig(self.Z, unit="MPa")
        sig_kPa = self.fq.sig(self.Z, unit="kPa")

        # sig = -kn * w
        expected_MPa = -self.fq.es.weak_layer.kn * self.Z[2, :]
        np.testing.assert_array_almost_equal(
            sig_MPa,
            expected_MPa,
            decimal=10,
            err_msg="Normal stress should follow sig = -kn*w",
        )

        # Check unit conversion
        np.testing.assert_array_almost_equal(
            sig_kPa, sig_MPa * 1000, decimal=8, err_msg="kPa should be 1000 times MPa"
        )

    def test_weak_layer_shear_stress(self):
        """Test weak layer shear stress calculation."""
        tau = self.fq.tau(self.Z, unit="MPa")

        # tau = -kt * (w' * h/2 - u(h=H/2))
        h = self.fq.es.weak_layer.h
        H = self.fq.es.slab.H
        u_surface = self.fq.u(self.Z, h0=H / 2)

        expected = -self.fq.es.weak_layer.kt * (self.Z[3, :] * h / 2 - u_surface)
        np.testing.assert_array_almost_equal(
            tau,
            expected,
            decimal=10,
            err_msg="Shear stress calculation should match expected formula",
        )


class TestFieldQuantitiesStrains(unittest.TestCase):
    """Test strain calculations."""

    def setUp(self):
        """Set up test system."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=50, h=20)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)
        self.fq = FieldQuantities(eigensystem)

        self.Z = np.array(
            [
                [1.0, 2.0],
                [0.1, 0.2],
                [0.5, 1.0],
                [0.05, 0.1],
                [0.01, 0.02],
                [0.001, 0.002],
            ]
        )

    def test_normal_strain_calculation(self):
        """Test weak layer normal strain calculation."""
        eps = self.fq.eps(self.Z)

        # eps = -w / h
        expected = -self.Z[2, :] / self.fq.es.weak_layer.h
        np.testing.assert_array_almost_equal(
            eps, expected, decimal=10, err_msg="Normal strain should follow eps = -w/h"
        )

    def test_shear_strain_calculation(self):
        """Test weak layer shear strain calculation."""
        gamma = self.fq.gamma(self.Z)

        # gamma = w'/2 - u(h=H/2)/h
        h = self.fq.es.weak_layer.h
        H = self.fq.es.slab.H
        u_surface = self.fq.u(self.Z, h0=H / 2)

        expected = self.Z[3, :] / 2 - u_surface / h
        np.testing.assert_array_almost_equal(
            gamma,
            expected,
            decimal=10,
            err_msg="Shear strain should follow gamma = w'/2 - u(H/2)/h",
        )


class TestFieldQuantitiesEnergyReleaseRates(unittest.TestCase):
    """Test energy release rate calculations."""

    def setUp(self):
        """Set up test system."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=50, h=20, kn=10.0, kt=5.0)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)
        self.fq = FieldQuantities(eigensystem)

        # Single point solution vector (crack tip)
        self.Z_tip = np.array(
            [
                [1.0],  # u
                [0.1],  # u'
                [0.5],  # w
                [0.05],  # w'
                [0.01],  # psi
                [0.001],  # psi'
            ]
        )

    def test_mode_I_energy_release_rate(self):
        """Test Mode I energy release rate calculation."""
        G_I = self.fq.Gi(self.Z_tip, unit="kJ/m^2")

        # G_I = sig^2 / (2 * kn)
        sig = self.fq.sig(self.Z_tip, unit="MPa")
        expected = sig**2 / (2 * self.fq.es.weak_layer.kn)

        np.testing.assert_array_almost_equal(
            G_I,
            expected,
            decimal=10,
            err_msg="Mode I ERR should follow G_I = sig²/(2*kn)",
        )

    def test_mode_II_energy_release_rate(self):
        """Test Mode II energy release rate calculation."""
        G_II = self.fq.Gii(self.Z_tip, unit="kJ/m^2")

        # G_II = tau^2 / (2 * kt)
        tau = self.fq.tau(self.Z_tip, unit="MPa")
        expected = tau**2 / (2 * self.fq.es.weak_layer.kt)

        np.testing.assert_array_almost_equal(
            G_II,
            expected,
            decimal=10,
            err_msg="Mode II ERR should follow G_II = tau²/(2*kt)",
        )

    def test_energy_release_rate_units(self):
        """Test energy release rate unit conversions."""
        G_I_kJ = self.fq.Gi(self.Z_tip, unit="kJ/m^2")
        G_I_J = self.fq.Gi(self.Z_tip, unit="J/m^2")
        G_I_N = self.fq.Gi(self.Z_tip, unit="N/mm")

        # Check unit conversions
        np.testing.assert_array_almost_equal(
            G_I_J, G_I_kJ * 1000, decimal=8, err_msg="J/m² should be 1000 times kJ/m²"
        )
        np.testing.assert_array_almost_equal(
            G_I_N, G_I_kJ, decimal=10, err_msg="N/mm should equal kJ/m²"
        )


class TestFieldQuantitiesPhysicalConsistency(unittest.TestCase):
    """Test physical consistency of field quantity calculations."""

    def test_displacement_continuity(self):
        """Test that displacements are continuous across heights."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=50, h=20)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)
        fq = FieldQuantities(eigensystem)

        Z = np.array([[1.0], [0.1], [0.5], [0.05], [0.01], [0.001]])

        # Test displacement at nearby heights
        h1, h2 = 30.0, 30.00001
        u1 = fq.u(Z, h1)
        u2 = fq.u(Z, h2)

        # Should be very close for nearby heights
        self.assertAlmostEqual(
            u1[0], u2[0], places=6, msg="Displacement should be continuous"
        )

    def test_stress_sign_conventions(self):
        """Test that stress sign conventions are physically reasonable."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=50, h=20)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)
        fq = FieldQuantities(eigensystem)

        # Positive deflection should give negative normal stress (compression)
        Z_positive_w = np.array([[0], [0], [1.0], [0], [0], [0]])  # Positive w
        sig_pos = fq.sig(Z_positive_w)

        self.assertLess(
            sig_pos[0], 0, "Positive deflection should give compressive stress"
        )

    def test_energy_release_rate_positivity(self):
        """Test that energy release rates are always positive."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=50, h=20)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)
        fq = FieldQuantities(eigensystem)

        # Any non-zero solution should give positive ERR
        Z_nonzero = np.array([[1.0], [0.1], [0.5], [0.05], [0.01], [0.001]])

        G_I = fq.Gi(Z_nonzero)
        G_II = fq.Gii(Z_nonzero)

        self.assertGreaterEqual(G_I[0], 0, "Mode I ERR should be non-negative")
        self.assertGreaterEqual(G_II[0], 0, "Mode II ERR should be non-negative")


if __name__ == "__main__":
    unittest.main(verbosity=2)
