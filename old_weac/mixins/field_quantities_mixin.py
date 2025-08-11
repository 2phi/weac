from __future__ import annotations

"""Mixin for field quantities."""
# Standard library imports
from functools import partial

# Third party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq

# Module imports
from old_weac.tools import calc_vertical_bc_center_of_gravity, tensile_strength_slab


class FieldQuantitiesMixin:
    """
    Mixin for field quantities.

    Provides methods for the computation of displacements, stresses,
    strains, and energy release rates from the solution vector.
    """

    # pylint: disable=no-self-use
    def w(self, Z, unit="mm"):
        """
        Get centerline deflection w.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Desired output unit. Default is mm.

        Returns
        -------
        float
            Deflection w (in specified unit) of the slab.
        """
        convert = {
            "m": 1e-3,  # meters
            "cm": 1e-1,  # centimeters
            "mm": 1,  # millimeters
            "um": 1e3,  # micrometers
        }
        return convert[unit] * Z[2, :]

    def dw_dx(self, Z):
        """
        Get first derivative w' of the centerline deflection.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            First derivative w' of the deflection of the slab.
        """
        return Z[3, :]

    def psi(self, Z, unit="rad"):
        """
        Get midplane rotation psi.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        unit : {'deg', 'degrees', 'rad', 'radians'}, optional
            Desired output unit. Default is radians.

        Returns
        -------
        psi : float
            Cross-section rotation psi (radians) of the slab.
        """
        if unit in ["deg", "degree", "degrees"]:
            psi = np.rad2deg(Z[4, :])
        elif unit in ["rad", "radian", "radians"]:
            psi = Z[4, :]
        return psi

    def dpsi_dx(self, Z):
        """
        Get first derivative psi' of the midplane rotation.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            First derivative psi' of the midplane rotation (radians/mm)
            of the slab.
        """
        return Z[5, :]

    # pylint: enable=no-self-use
    def u(self, Z, z0, unit="mm"):
        """
        Get horizontal displacement u = u0 + z0 psi.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        z0 : float
            Z-coordinate (mm) where u is to be evaluated.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Desired output unit. Default is mm.

        Returns
        -------
        float
            Horizontal displacement u (unit) of the slab.
        """
        convert = {
            "m": 1e-3,  # meters
            "cm": 1e-1,  # centimeters
            "mm": 1,  # millimeters
            "um": 1e3,  # micrometers
        }
        return convert[unit] * (Z[0, :] + z0 * self.psi(Z))

    def du_dx(self, Z, z0):
        """
        Get first derivative of the horizontal displacement.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        z0 : float
            Z-coordinate (mm) where u is to be evaluated.

        Returns
        -------
        float
            First derivative u' = u0' + z0 psi' of the horizontal
            displacement of the slab.
        """
        return Z[1, :] + z0 * self.dpsi_dx(Z)

    def N(self, Z):
        """
        Get the axial normal force N = A11 u' + B11 psi' in the slab.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            Axial normal force N (N) in the slab.
        """
        return self.A11 * Z[1, :] + self.B11 * Z[5, :]

    def M(self, Z):
        """
        Get bending moment M = B11 u' + D11 psi' in the slab.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            Bending moment M (Nmm) in the slab.
        """
        return self.B11 * Z[1, :] + self.D11 * Z[5, :]

    def V(self, Z):
        """
        Get vertical shear force V = kA55(w' + psi).

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            Vertical shear force V (N) in the slab.
        """
        return self.kA55 * (Z[3, :] + Z[4, :])

    def sig(self, Z, unit="MPa"):
        """
        Get weak-layer normal stress.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        unit : {'MPa', 'kPa'}, optional
            Desired output unit. Default is MPa.

        Returns
        -------
        float
            Weak-layer normal stress sigma (in specified unit).
        """
        convert = {"kPa": 1e3, "MPa": 1}
        return -convert[unit] * self.kn * self.w(Z)

    def tau(self, Z, unit="MPa"):
        """
        Get weak-layer shear stress.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        unit : {'MPa', 'kPa'}, optional
            Desired output unit. Default is MPa.

        Returns
        -------
        float
            Weak-layer shear stress tau (in specified unit).
        """
        convert = {"kPa": 1e3, "MPa": 1}
        return (
            -convert[unit]
            * self.kt
            * (self.dw_dx(Z) * self.t / 2 - self.u(Z, z0=self.h / 2))
        )

    def eps(self, Z):
        """
        Get weak-layer normal strain.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            Weak-layer normal strain epsilon.
        """
        return -self.w(Z) / self.t

    def gamma(self, Z):
        """
        Get weak-layer shear strain.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            Weak-layer shear strain gamma.
        """
        return self.dw_dx(Z) / 2 - self.u(Z, z0=self.h / 2) / self.t

    def Gi(self, Ztip, unit="kJ/m^2"):
        """
        Get mode I differential energy release rate at crack tip.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T
            at the crack tip.
        unit : {'N/mm', 'kJ/m^2', 'J/m^2'}, optional
            Desired output unit. Default is kJ/m^2.

        Returns
        -------
        float
            Mode I differential energy release rate (N/mm = kJ/m^2
            or J/m^2) at the crack tip.
        """
        convert = {
            "J/m^2": 1e3,  # joule per square meter
            "kJ/m^2": 1,  # kilojoule per square meter
            "N/mm": 1,  # newton per millimeter
        }
        return convert[unit] * self.sig(Ztip) ** 2 / (2 * self.kn)

    def Gii(self, Ztip, unit="kJ/m^2"):
        """
        Get mode II differential energy release rate at crack tip.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T
            at the crack tip.
        unit : {'N/mm', 'kJ/m^2', 'J/m^2'}, optional
            Desired output unit. Default is kJ/m^2 = N/mm.

        Returns
        -------
        float
            Mode II differential energy release rate (N/mm = kJ/m^2
            or J/m^2) at the crack tip.
        """
        convert = {
            "J/m^2": 1e3,  # joule per square meter
            "kJ/m^2": 1,  # kilojoule per square meter
            "N/mm": 1,  # newton per millimeter
        }
        return convert[unit] * self.tau(Ztip) ** 2 / (2 * self.kt)

    def int1(self, x, z0, z1):
        """
        Get mode I crack opening integrand at integration points xi.

        Arguments
        ---------
        x : float, ndarray
            X-coordinate where integrand is to be evaluated (mm).
        z0 : callable
            Function that returns the solution vector of the uncracked
            configuration.
        z1 : callable
            Function that returns the solution vector of the cracked
            configuration.

        Returns
        -------
        float or ndarray
            Integrant of the mode I crack opening integral.
        """
        return self.sig(z0(x)) * self.eps(z1(x)) * self.t

    def int2(self, x, z0, z1):
        """
        Get mode II crack opening integrand at integration points xi.

        Arguments
        ---------
        x : float, ndarray
            X-coordinate where integrand is to be evaluated (mm).
        z0 : callable
            Function that returns the solution vector of the uncracked
            configuration.
        z1 : callable
            Function that returns the solution vector of the cracked
            configuration.

        Returns
        -------
        float or ndarray
            Integrant of the mode II crack opening integral.
        """
        return self.tau(z0(x)) * self.gamma(z1(x)) * self.t

    def dz_dx(self, z, phi):
        """
        Get first derivative z'(x) = K*z(x) + q of the solution vector.

        z'(x) = [u'(x) u''(x) w'(x) w''(x) psi'(x), psi''(x)]^T

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            First derivative z'(x) for the solution vector (6x1).
        """
        K = self.calc_system_matrix()
        q = self.get_load_vector(phi)
        return np.dot(K, z) + q

    def dz_dxdx(self, z, phi):
        """
        Get second derivative z''(x) = K*z'(x) of the solution vector.

        z''(x) = [u''(x) u'''(x) w''(x) w'''(x) psi''(x), psi'''(x)]^T

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            Second derivative z''(x) = (K*z(x) + q)' = K*z'(x) = K*(K*z(x) + q)
            of the solution vector (6x1).
        """
        K = self.calc_system_matrix()
        q = self.get_load_vector(phi)
        dz_dx = np.dot(K, z) + q
        return np.dot(K, dz_dx)

    def du0_dxdx(self, z, phi):
        """
        Get second derivative of the horiz. centerline displacement u0''(x).

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            Second derivative of the horizontal centerline displacement
            u0''(x) (1/mm).
        """
        return self.dz_dx(z, phi)[1, :]

    def dpsi_dxdx(self, z, phi):
        """
        Get second derivative of the cross-section rotation psi''(x).

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            Second derivative of the cross-section rotation psi''(x) (1/mm^2).
        """
        return self.dz_dx(z, phi)[5, :]

    def du0_dxdxdx(self, z, phi):
        """
        Get third derivative of the horiz. centerline displacement u0'''(x).

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            Third derivative of the horizontal centerline displacement
            u0'''(x) (1/mm^2).
        """
        return self.dz_dxdx(z, phi)[1, :]

    def dpsi_dxdxdx(self, z, phi):
        """
        Get third derivative of the cross-section rotation psi'''(x).

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            Third derivative of the cross-section rotation psi'''(x) (1/mm^3).
        """
        return self.dz_dxdx(z, phi)[5, :]
