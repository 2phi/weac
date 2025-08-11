from __future__ import annotations

"""Mixin for Output."""
# Standard library imports
from functools import partial

# Third party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq

# Module imports
from old_weac.tools import calc_vertical_bc_center_of_gravity, tensile_strength_slab


class OutputMixin:
    """
    Mixin for outputs.

    Provides convenience methods for the assembly of output lists
    such as rasterized displacements or rasterized stresses.
    """

    def external_potential(self, C, phi, L, **segments):
        """
        Compute total external potential (pst only).

        Arguments
        ---------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        phi : float
            Inclination of the slab (°).
        L : float, optional
            Total length of model (mm).
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.

        Returns
        -------
        Pi_ext : float
            Total external potential (Nmm).
        """
        # Rasterize solution
        xq, zq, xb = self.rasterize_solution(C=C, phi=phi, **segments)
        _ = xq, xb
        # Compute displacements where weight loads are applied
        w0 = self.w(zq)
        us = self.u(zq, z0=self.zs)
        # Get weight loads
        qn = self.calc_qn()
        qt = self.calc_qt()
        # use +/- and us[0]/us[-1] according to system and phi
        # compute total external potential
        Pi_ext = (
            -qn * (segments["li"][0] + segments["li"][1]) * np.average(w0)
            - qn * (L - (segments["li"][0] + segments["li"][1])) * self.tc
        )
        # Ensure
        if self.system in ["pst-"]:
            ub = us[-1]
        elif self.system in ["-pst"]:
            ub = us[0]
        Pi_ext += (
            -qt * (segments["li"][0] + segments["li"][1]) * np.average(us)
            - qt * (L - (segments["li"][0] + segments["li"][1])) * ub
        )
        if self.system not in ["pst-", "-pst"]:
            print("Input error: Only pst-setup implemented at the moment.")

        return Pi_ext

    def internal_potential(self, C, phi, L, **segments):
        """
        Compute total internal potential (pst only).

        Arguments
        ---------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        phi : float
            Inclination of the slab (°).
        L : float, optional
            Total length of model (mm).
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.

        Returns
        -------
        Pi_int : float
            Total internal potential (Nmm).
        """
        # Rasterize solution
        xq, zq, xb = self.rasterize_solution(C=C, phi=phi, **segments)

        # Compute section forces
        N, M, V = self.N(zq), self.M(zq), self.V(zq)

        # Drop parts of the solution that are not a foundation
        zweak = zq[:, ~np.isnan(xb)]
        xweak = xb[~np.isnan(xb)]

        # Compute weak layer displacements
        wweak = self.w(zweak)
        uweak = self.u(zweak, z0=self.h / 2)

        # Compute stored energy of the slab (monte-carlo integration)
        n = len(xq)
        nweak = len(xweak)
        # energy share from moment, shear force, wl normal and tangential springs
        Pi_int = (
            L / 2 / n / self.A11 * np.sum([Ni**2 for Ni in N])
            + L
            / 2
            / n
            / (self.D11 - self.B11**2 / self.A11)
            * np.sum([Mi**2 for Mi in M])
            + L / 2 / n / self.kA55 * np.sum([Vi**2 for Vi in V])
            + L * self.kn / 2 / nweak * np.sum([wi**2 for wi in wweak])
            + L * self.kt / 2 / nweak * np.sum([ui**2 for ui in uweak])
        )
        # energy share from substitute rotation spring
        if self.system in ["pst-"]:
            Pi_int += 1 / 2 * M[-1] * (self.psi(zq)[-1]) ** 2
        elif self.system in ["-pst"]:
            Pi_int += 1 / 2 * M[0] * (self.psi(zq)[0]) ** 2
        else:
            print("Input error: Only pst-setup implemented at the moment.")

        return Pi_int

    def total_potential(self, C, phi, L, **segments):
        """
        Returns total differential potential

        Arguments
        ---------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        phi : float
            Inclination of the slab (°).
        L : float, optional
            Total length of model (mm).
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.

        Returns
        -------
        Pi : float
            Total differential potential (Nmm).
        """
        Pi_int = self.internal_potential(C, phi, L, **segments)
        Pi_ext = self.external_potential(C, phi, L, **segments)

        return Pi_int + Pi_ext

    def get_weaklayer_shearstress(self, x, z, unit="MPa", removeNaNs=False):
        """
        Compute weak-layer shear stress.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of unsupported
            (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
        unit : {'MPa', 'kPa'}, optional
            Stress output unit. Default is MPa.
        keepNaNs : bool
            If set, do not remove

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        sig : ndarray
            Normal stress (stress unit input).
        """
        # Convert coordinates from mm to cm and stresses from MPa to unit
        x = x / 10
        tau = self.tau(z, unit=unit)
        # Filter stresses in unspupported segments
        if removeNaNs:
            # Remove coordinate-stress pairs where no weak layer is present
            tau = tau[~np.isnan(x)]
            x = x[~np.isnan(x)]
        else:
            # Set stress NaN where no weak layer is present
            tau[np.isnan(x)] = np.nan

        return x, tau

    def get_weaklayer_normalstress(self, x, z, unit="MPa", removeNaNs=False):
        """
        Compute weak-layer normal stress.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of unsupported
            (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
        unit : {'MPa', 'kPa'}, optional
            Stress output unit. Default is MPa.
        keepNaNs : bool
            If set, do not remove

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        sig : ndarray
            Normal stress (stress unit input).
        """
        # Convert coordinates from mm to cm and stresses from MPa to unit
        x = x / 10
        sig = self.sig(z, unit=unit)
        # Filter stresses in unspupported segments
        if removeNaNs:
            # Remove coordinate-stress pairs where no weak layer is present
            sig = sig[~np.isnan(x)]
            x = x[~np.isnan(x)]
        else:
            # Set stress NaN where no weak layer is present
            sig[np.isnan(x)] = np.nan

        return x, sig

    def get_slab_displacement(self, x, z, loc="mid", unit="mm"):
        """
        Compute horizontal slab displacement.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of
            unsupported (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
        loc : {'top', 'mid', 'bot'}
            Get displacements of top, midplane or bottom of slab.
            Default is mid.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Displacement output unit. Default is mm.

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        ndarray
            Horizontal displacements (unit input).
        """
        # Coordinates (cm)
        x = x / 10
        # Locator
        z0 = {"top": -self.h / 2, "mid": 0, "bot": self.h / 2}
        # Displacement (unit)
        u = self.u(z, z0=z0[loc], unit=unit)
        # Output array
        return x, u

    def get_slab_deflection(self, x, z, unit="mm"):
        """
        Compute vertical slab displacement.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of
            unsupported (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
            Default is mid.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Displacement output unit. Default is mm.

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        ndarray
            Vertical deflections (unit input).
        """
        # Coordinates (cm)
        x = x / 10
        # Deflection (unit)
        w = self.w(z, unit=unit)
        # Output array
        return x, w

    def get_slab_rotation(self, x, z, unit="degrees"):
        """
        Compute slab cross-section rotation angle.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of
            unsupported (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
            Default is mid.
        unit : {'deg', degrees', 'rad', 'radians'}, optional
            Rotation angle output unit. Default is degrees.

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        ndarray
            Cross section rotations (unit input).
        """
        # Coordinates (cm)
        x = x / 10
        # Cross-section rotation angle (unit)
        psi = self.psi(z, unit=unit)
        # Output array
        return x, psi
