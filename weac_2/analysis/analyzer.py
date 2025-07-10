# Standard library imports
import logging
import time
from collections import defaultdict
from functools import partial, wraps
from typing import Literal

# Third party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

from weac_2.constants import G_MM_S2

# Module imports
from weac_2.core.system_model import SystemModel

logger = logging.getLogger(__name__)


def track_analyzer_call(func):
    """Decorator to track call count and execution time of Analyzer methods."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """Wrapper that adds tracking functionality."""
        if not hasattr(self, "call_stats"):
            # Safeguard in case __init__ was not called, which it should be.
            self.call_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})

        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        duration = time.perf_counter() - start_time

        func_name = func.__name__
        self.call_stats[func_name]["count"] += 1
        self.call_stats[func_name]["total_time"] += duration

        logger.debug(
            "Analyzer method '%s' called. "
            "Execution time: %.4f seconds.",
            func_name, duration
        )

        return result

    return wrapper


class Analyzer:
    """
    Provides methods for the analysis of layered slabs on compliant
    elastic foundations.
    """

    sm: SystemModel

    def __init__(self, system_model: SystemModel):
        self.sm = system_model
        self.call_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})

    def get_call_stats(self):
        """Returns the call statistics."""
        return self.call_stats

    def print_call_stats(self, message: str = "Analyzer Call Statistics"):
        """Prints the call statistics in a readable format."""
        print(f"--- {message} ---")
        if not self.call_stats:
            print("No methods have been called.")
            return

        sorted_stats = sorted(
            self.call_stats.items(),
            key=lambda item: item[1]["total_time"],
            reverse=True,
        )

        for func_name, stats in sorted_stats:
            count = stats["count"]
            total_time = stats["total_time"]
            avg_time = total_time / count if count > 0 else 0
            print(
                f"- {func_name}: "
                f"called {count} times, "
                f"total time {total_time:.4f}s, "
                f"avg time {avg_time:.4f}s"
            )
        print("---------------------------------")

    @track_analyzer_call
    def rasterize_solution(
        self,
        mode: Literal["cracked", "uncracked"] = "cracked",
        num: int = 4000,
    ):
        """
        Compute rasterized solution vector.

        Parameters:
        ---------
        mode : Literal["cracked", "uncracked"]
            Mode of the solution.
        num : int
            Number of grid points.

        Returns
        -------
        xs : ndarray
            Grid point x-coordinates at which solution vector
            is discretized.
        zs : ndarray
            Matrix with solution vectors as columns at grid
            points xs.
        x_founded : ndarray
            Grid point x-coordinates that lie on a foundation.
        """
        ki = self.sm.scenario.ki
        match mode:
            case "cracked":
                C = self.sm.unknown_constants
            case "uncracked":
                ki = np.full(len(ki), True)
                C = self.sm.uncracked_unknown_constants
        phi = self.sm.scenario.phi
        li = self.sm.scenario.li
        qs = self.sm.scenario.surface_load

        # Drop zero-length segments
        li = abs(li)
        isnonzero = li > 0
        C, ki, li = C[:, isnonzero], ki[isnonzero], li[isnonzero]

        # Compute number of plot points per segment (+1 for last segment)
        ni = np.ceil(li / li.sum() * num).astype("int")
        ni[-1] += 1

        # Provide cumulated length and plot point lists
        lic = np.insert(np.cumsum(li), 0, 0)
        nic = np.insert(np.cumsum(ni), 0, 0)

        # Initialize arrays
        issupported = np.full(ni.sum(), True)
        xs = np.full(ni.sum(), np.nan)
        zs = np.full([6, xs.size], np.nan)

        # Loop through segments
        for i, length in enumerate(li):
            # Get local x-coordinates of segment i
            xi = np.linspace(0, length, num=ni[i], endpoint=(i == li.size - 1))
            # Compute start and end coordinates of segment i
            x0 = lic[i]
            # Assemble global coordinate vector
            xs[nic[i] : nic[i + 1]] = x0 + xi
            # Mask coordinates not on foundation (including endpoints)
            if not ki[i]:
                issupported[nic[i] : nic[i + 1]] = False
            # Compute segment solution
            zi = self.sm.z(xi, C[:, [i]], length, phi, ki[i], qs=qs)
            # Assemble global solution matrix
            zs[:, nic[i] : nic[i + 1]] = zi

        # Make sure cracktips are included
        transmissionbool = [ki[j] or ki[j + 1] for j, _ in enumerate(ki[:-1])]
        for i, truefalse in enumerate(transmissionbool, start=1):
            issupported[nic[i]] = truefalse

        # Assemble vector of coordinates on foundation
        xs_supported = np.full(ni.sum(), np.nan)
        xs_supported[issupported] = xs[issupported]

        return xs, zs, xs_supported

    @track_analyzer_call
    def get_zmesh(self, dz=2):
        """
        Get z-coordinates of grid points and corresponding elastic properties.

        Arguments
        ---------
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.

        Returns
        -------
        mesh : ndarray
            Mesh along z-axis. Columns are a list of z-coordinates (mm) of
            grid points along z-axis with at least two grid points (top,
            bottom) per layer, Young's modulus of each grid point, shear
            modulus of each grid point, and Poisson's ratio of each grid
            point.
        """
        # Get z-coordinates of slab layers
        z = np.concatenate([[self.sm.slab.z0], self.sm.slab.zi_bottom])
        # Compute number of grid points per layer
        nlayer = np.ceil((z[1:] - z[:-1]) / dz).astype(np.int32) + 1
        # Calculate grid points as list of z-coordinates (mm)
        zi = np.hstack(
            [
                np.linspace(z[i], z[i + 1], n, endpoint=True)
                for i, n in enumerate(nlayer)
            ]
        )
        # Extract elastic properties for each layer, reversing to match z order
        layer_properties = {
            "E": [layer.E for layer in self.sm.slab.layers],
            "nu": [layer.nu for layer in self.sm.slab.layers],
            "rho": [
                layer.rho * 1e-12 for layer in self.sm.slab.layers
            ],  # Convert to t/mm^3
            "tensile_strength": [
                layer.tensile_strength for layer in self.sm.slab.layers
            ],
        }

        # Repeat properties for each grid point in the layer
        si = {"z": zi}
        for prop, values in layer_properties.items():
            si[prop] = np.repeat(values, nlayer)

        return si

    @track_analyzer_call
    def Sxx(self, Z, phi, dz=2, unit="kPa"):
        """
        Compute axial normal stress in slab layers.

        Arguments
        ----------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.

        Returns
        -------
        ndarray, float
            Axial slab normal stress in specified unit.
        """
        # Unit conversion dict
        convert = {"kPa": 1e3, "MPa": 1}

        # Get mesh along z-axis
        zmesh = self.get_zmesh(dz=dz)
        zi = zmesh["z"]
        rho = zmesh["rho"]

        # Get dimensions of stress field (n rows, m columns)
        n = len(zmesh["z"])
        m = Z.shape[1]

        # Initialize axial normal stress Sxx
        Sxx = np.zeros(shape=[n, m])

        # Compute axial normal stress Sxx at grid points in MPa
        for i, z in enumerate(zi):
            E = zmesh["E"][i]
            nu = zmesh["nu"][i]
            Sxx[i, :] = E / (1 - nu**2) * self.sm.fq.du_dx(Z, z)

        # Calculate weight load at grid points and superimpose on stress field
        qt = -rho * G_MM_S2 * np.sin(np.deg2rad(phi))
        for i, qi in enumerate(qt[:-1]):
            Sxx[i, :] += qi * (zi[i + 1] - zi[i])
        Sxx[-1, :] += qt[-1] * (zi[-1] - zi[-2])

        # Return axial normal stress in specified unit
        return convert[unit] * Sxx

    @track_analyzer_call
    def Txz(self, Z, phi, dz=2, unit="kPa"):
        """
        Compute shear stress in slab layers.

        Arguments
        ----------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.

        Returns
        -------
        ndarray
            Shear stress at grid points in the slab in specified unit.
        """
        # Unit conversion dict
        convert = {"kPa": 1e3, "MPa": 1}
        # Get mesh along z-axis
        zmesh = self.get_zmesh(dz=dz)
        zi = zmesh["z"]
        rho = zmesh["rho"]
        qs = self.sm.scenario.surface_load

        # Get dimensions of stress field (n rows, m columns)
        n = len(zi)
        m = Z.shape[1]

        # Get second derivatives of centerline displacement u0 and
        # cross-section rotaiton psi of all grid points along the x-axis
        du0_dxdx = self.sm.fq.du0_dxdx(Z, phi, qs=qs)
        dpsi_dxdx = self.sm.fq.dpsi_dxdx(Z, phi, qs=qs)

        # Initialize first derivative of axial normal stress sxx w.r.t. x
        dsxx_dx = np.zeros(shape=[n, m])

        # Calculate first derivative of sxx at z-grid points
        for i, z in enumerate(zi):
            E = zmesh["E"][i]
            nu = zmesh["nu"][i]
            dsxx_dx[i, :] = E / (1 - nu**2) * (du0_dxdx + z * dpsi_dxdx)

        # Calculate weight load at grid points
        qt = -rho * G_MM_S2 * np.sin(np.deg2rad(phi))

        # Integrate -dsxx_dx along z and add cumulative weight load
        # to obtain shear stress Txz in MPa
        Txz = cumulative_trapezoid(dsxx_dx, zi, axis=0, initial=0)
        Txz += cumulative_trapezoid(qt, zi, initial=0)[:, None]

        # Return shear stress Txz in specified unit
        return convert[unit] * Txz

    @track_analyzer_call
    def Szz(self, Z, phi, dz=2, unit="kPa"):
        """
        Compute transverse normal stress in slab layers.

        Arguments
        ----------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.

        Returns
        -------
        ndarray, float
            Transverse normal stress at grid points in the slab in
            specified unit.
        """
        # Unit conversion dict
        convert = {"kPa": 1e3, "MPa": 1}

        # Get mesh along z-axis
        zmesh = self.get_zmesh(dz=dz)
        zi = zmesh["z"]
        rho = zmesh["rho"]
        qs = self.sm.scenario.surface_load
        # Get dimensions of stress field (n rows, m columns)
        n = len(zi)
        m = Z.shape[1]

        # Get third derivatives of centerline displacement u0 and
        # cross-section rotaiton psi of all grid points along the x-axis
        du0_dxdxdx = self.sm.fq.du0_dxdxdx(Z, phi, qs=qs)
        dpsi_dxdxdx = self.sm.fq.dpsi_dxdxdx(Z, phi, qs=qs)

        # Initialize second derivative of axial normal stress sxx w.r.t. x
        dsxx_dxdx = np.zeros(shape=[n, m])

        # Calculate second derivative of sxx at z-grid points
        for i, z in enumerate(zi):
            E = zmesh["E"][i]
            nu = zmesh["nu"][i]
            dsxx_dxdx[i, :] = E / (1 - nu**2) * (du0_dxdxdx + z * dpsi_dxdxdx)

        # Calculate weight load at grid points
        qn = rho * G_MM_S2 * np.cos(np.deg2rad(phi))

        # Integrate dsxx_dxdx twice along z to obtain transverse
        # normal stress Szz in MPa
        integrand = cumulative_trapezoid(dsxx_dxdx, zi, axis=0, initial=0)
        Szz = cumulative_trapezoid(integrand, zi, axis=0, initial=0)
        Szz += cumulative_trapezoid(-qn, zi, initial=0)[:, None]

        # Return shear stress txz in specified unit
        return convert[unit] * Szz

    @track_analyzer_call
    def principal_stress_slab(
        self, Z, phi, dz=2, unit="kPa", val="max", normalize=False
    ):
        """
        Compute maxium or minimum principal stress in slab layers.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.
        val : str, optional
            Maximum 'max' or minimum 'min' principal stress. Default is 'max'.
        normalize : bool
            Toggle layerwise normalization to strength.

        Returns
        -------
        ndarray
            Maximum or minimum principal stress in specified unit.

        Raises
        ------
        ValueError
            If specified principal stress component is neither 'max' nor
            'min', or if normalization of compressive principal stress
            is requested.
        """
        # Raise error if specified component is not available
        if val not in ["min", "max"]:
            raise ValueError(f"Component {val} not defined.")

        # Multiplier selection dict
        m = {"max": 1, "min": -1}

        # Get axial normal stresses, shear stresses, transverse normal stresses
        Sxx = self.Sxx(Z=Z, phi=phi, dz=dz, unit=unit)
        Txz = self.Txz(Z=Z, phi=phi, dz=dz, unit=unit)
        Szz = self.Szz(Z=Z, phi=phi, dz=dz, unit=unit)

        # Calculate principal stress
        Ps = (Sxx + Szz) / 2 + m[val] * np.sqrt((Sxx - Szz) ** 2 + 4 * Txz**2) / 2

        # Raise error if normalization of compressive stresses is attempted
        if normalize and val == "min":
            raise ValueError("Can only normlize tensile stresses.")

        # Normalize tensile stresses to tensile strength
        if normalize and val == "max":
            zmesh = self.get_zmesh(dz=dz)
            tensile_strength = zmesh["tensile_strength"]
            # Normlize maximum principal stress to layers' tensile strength
            normalized_Ps = Ps / tensile_strength[:, None]
            return normalized_Ps

        # Return absolute principal stresses
        return Ps

    @track_analyzer_call
    def principal_stress_weaklayer(
        self, Z, sc=2.6, unit="kPa", val="min", normalize=False
    ):
        """
        Compute maxium or minimum principal stress in the weak layer.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        sc : float
            Weak-layer compressive strength. Default is 2.6 kPa.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.
        val : str, optional
            Maximum 'max' or minimum 'min' principal stress. Default is 'min'.
        normalize : bool
            Toggle layerwise normalization to strength.

        Returns
        -------
        ndarray
            Maximum or minimum principal stress in specified unit.

        Raises
        ------
        ValueError
            If specified principal stress component is neither 'max' nor
            'min', or if normalization of tensile principal stress
            is requested.
        """
        # Raise error if specified component is not available
        if val not in ["min", "max"]:
            raise ValueError(f"Component {val} not defined.")

        # Multiplier selection dict
        m = {"max": 1, "min": -1}

        # Get weak-layer normal and shear stresses
        sig = self.sm.fq.sig(Z, unit=unit)
        tau = self.sm.fq.tau(Z, unit=unit)

        # Calculate principal stress
        ps = sig / 2 + m[val] * np.sqrt(sig**2 + 4 * tau**2) / 2

        # Raise error if normalization of tensile stresses is attempted
        if normalize and val == "max":
            raise ValueError("Can only normlize compressive stresses.")

        # Normalize compressive stresses to compressive strength
        if normalize and val == "min":
            return ps / sc

        # Return absolute principal stresses
        return ps

    @track_analyzer_call
    def incremental_ERR(
        self, tolerance: float = 1e-6, unit: str = "kJ/m^2"
    ) -> np.ndarray:
        """
        Compute incremental energy release rate (ERR) of all cracks.

        Returns
        -------
        ndarray
            List of total, mode I, and mode II energy release rates.
        """
        li = self.sm.scenario.li
        ki = self.sm.scenario.ki
        k0 = np.ones_like(ki, dtype=bool)
        C_uncracked = self.sm.uncracked_unknown_constants
        C_cracked = self.sm.unknown_constants
        phi = self.sm.scenario.phi
        qs = self.sm.scenario.surface_load

        # Reduce inputs to segments with crack advance
        iscrack = k0 & ~ki
        C_uncracked, C_cracked, li = (
            C_uncracked[:, iscrack],
            C_cracked[:, iscrack],
            li[iscrack],
        )

        # Compute total crack lenght and initialize outputs
        da = li.sum() if li.sum() > 0 else np.nan
        Ginc1, Ginc2 = 0, 0

        # Loop through segments with crack advance
        for j, length in enumerate(li):
            # Uncracked (0) and cracked (1) solutions at integration points
            z_uncracked = partial(
                self.sm.z,
                C=C_uncracked[:, [j]],
                length=length,
                phi=phi,
                has_foundation=True,
                qs=qs,
            )
            z_cracked = partial(
                self.sm.z,
                C=C_cracked[:, [j]],
                length=length,
                phi=phi,
                has_foundation=False,
                qs=qs,
            )

            # Mode I (1) and II (2) integrands at integration points
            intGI = partial(
                self._integrand_GI, z_uncracked=z_uncracked, z_cracked=z_cracked
            )
            intGII = partial(
                self._integrand_GII, z_uncracked=z_uncracked, z_cracked=z_cracked
            )

            # Segment contributions to total crack opening integral
            Ginc1 += quad(intGI, 0, length, epsabs=tolerance, epsrel=tolerance)[0] / (
                2 * da
            )
            Ginc2 += quad(intGII, 0, length, epsabs=tolerance, epsrel=tolerance)[0] / (
                2 * da
            )

        convert = {"kJ/m^2": 1, "J/m^2": 1e3}
        return np.array([Ginc1 + Ginc2, Ginc1, Ginc2]).flatten() * convert[unit]

    @track_analyzer_call
    def differential_ERR(self, unit: str = "kJ/m^2") -> np.ndarray:
        """
        Compute differential energy release rate of all crack tips.

        Returns
        -------
        ndarray
            List of total, mode I, and mode II energy release rates.
        """
        li = self.sm.scenario.li
        ki = self.sm.scenario.ki
        C = self.sm.unknown_constants
        phi = self.sm.scenario.phi
        qs = self.sm.scenario.surface_load

        # Get number and indices of segment transitions
        ntr = len(li) - 1
        itr = np.arange(ntr)

        # Identify supported-free and free-supported transitions as crack tips
        iscracktip = [ki[j] != ki[j + 1] for j in range(ntr)]

        # Transition indices of crack tips and total number of crack tips
        ict = itr[iscracktip]
        nct = len(ict)

        # Initialize energy release rate array
        Gdif = np.zeros([3, nct])

        # Compute energy relase rate of all crack tips
        for j, idx in enumerate(ict):
            # Solution at crack tip
            z = self.sm.z(
                li[idx], C[:, [idx]], li[idx], phi, has_foundation=ki[idx], qs=qs
            )
            # Mode I and II differential energy release rates
            Gdif[1:, j] = np.concatenate(
                (self.sm.fq.Gi(z, unit=unit), self.sm.fq.Gii(z, unit=unit))
            )

        # Sum mode I and II contributions
        Gdif[0, :] = Gdif[1, :] + Gdif[2, :]

        # Adjust contributions for center cracks
        if nct > 1:
            avgmask = np.full(nct, True)  # Initialize mask
            avgmask[[0, -1]] = ki[[0, -1]]  # Do not weight edge cracks
            Gdif[:, avgmask] *= 0.5  # Weigth with half crack length

        # Return total differential energy release rate of all crack tips
        return Gdif.sum(axis=1)

    def _integrand_GI(
        self, x: float | np.ndarray, z_uncracked, z_cracked
    ) -> float | np.ndarray:
        """
        Mode I integrand for energy release rate calculation.
        """
        sig_uncracked = self.sm.fq.sig(z_uncracked(x))
        eps_cracked = self.sm.fq.eps(z_cracked(x))
        return sig_uncracked * eps_cracked * self.sm.weak_layer.h

    def _integrand_GII(
        self, x: float | np.ndarray, z_uncracked, z_cracked
    ) -> float | np.ndarray:
        """
        Mode II integrand for energy release rate calculation.
        """
        tau_uncracked = self.sm.fq.tau(z_uncracked(x))
        gamma_cracked = self.sm.fq.gamma(z_cracked(x))
        return tau_uncracked * gamma_cracked * self.sm.weak_layer.h

    @track_analyzer_call
    def total_potential(self):
        """
        Returns total differential potential

        Returns
        -------
        Pi : float
            Total differential potential (Nmm).
        """
        Pi_int = self._internal_potential()
        Pi_ext = self._external_potential()

        return Pi_int + Pi_ext

    def _external_potential(self):
        """
        Compute total external potential (pst only).

        Returns
        -------
        Pi_ext : float
            Total external potential (Nmm).
        """
        # Rasterize solution
        xq, zq, xb = self.rasterize_solution(mode="cracked", num=2000)
        _ = xq, xb
        # Compute displacements where weight loads are applied
        w0 = self.sm.fq.w(zq)
        us = self.sm.fq.u(zq, z0=self.sm.slab.z_cog)
        # Get weight loads
        qn = self.sm.scenario.qn
        qt = self.sm.scenario.qt
        # use +/- and us[0]/us[-1] according to system and phi
        # compute total external potential
        Pi_ext = (
            -qn * (self.sm.scenario.li[0] + self.sm.scenario.li[1]) * np.average(w0)
            - qn
            * (self.sm.scenario.L - (self.sm.scenario.li[0] + self.sm.scenario.li[1]))
            * self.sm.scenario.crack_h
        )
        # Ensure
        if self.sm.scenario.system_type in ["pst-"]:
            ub = us[-1]
        elif self.sm.scenario.system_type in ["-pst"]:
            ub = us[0]
        Pi_ext += (
            -qt * (self.sm.scenario.li[0] + self.sm.scenario.li[1]) * np.average(us)
            - qt
            * (self.sm.scenario.L - (self.sm.scenario.li[0] + self.sm.scenario.li[1]))
            * ub
        )
        if self.sm.scenario.system_type not in ["pst-", "-pst"]:
            print("Input error: Only pst-setup implemented at the moment.")

        return Pi_ext

    def _internal_potential(self):
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
        # Extract system parameters
        L = self.sm.scenario.L
        system_type = self.sm.scenario.system_type
        A11 = self.sm.eigensystem.A11
        B11 = self.sm.eigensystem.B11
        D11 = self.sm.eigensystem.D11
        kA55 = self.sm.eigensystem.kA55
        kn = self.sm.weak_layer.kn
        kt = self.sm.weak_layer.kt

        # Rasterize solution
        xq, zq, xb = self.rasterize_solution(mode="cracked", num=2000)

        # Compute section forces
        N, M, V = self.sm.fq.N(zq), self.sm.fq.M(zq), self.sm.fq.V(zq)

        # Drop parts of the solution that are not a foundation
        zweak = zq[:, ~np.isnan(xb)]
        xweak = xb[~np.isnan(xb)]

        # Compute weak layer displacements
        wweak = self.sm.fq.w(zweak)
        uweak = self.sm.fq.u(zweak, z0=self.sm.slab.H / 2)

        # Compute stored energy of the slab (monte-carlo integration)
        n = len(xq)
        nweak = len(xweak)
        # energy share from moment, shear force, wl normal and tangential springs
        Pi_int = (
            L / 2 / n / A11 * np.sum([Ni**2 for Ni in N])
            + L / 2 / n / (D11 - B11**2 / A11) * np.sum([Mi**2 for Mi in M])
            + L / 2 / n / kA55 * np.sum([Vi**2 for Vi in V])
            + L * kn / 2 / nweak * np.sum([wi**2 for wi in wweak])
            + L * kt / 2 / nweak * np.sum([ui**2 for ui in uweak])
        )
        # energy share from substitute rotation spring
        if system_type in ["pst-"]:
            Pi_int += 1 / 2 * M[-1] * (self.sm.fq.psi(zq)[-1]) ** 2
        elif system_type in ["-pst"]:
            Pi_int += 1 / 2 * M[0] * (self.sm.fq.psi(zq)[0]) ** 2
        else:
            print("Input error: Only pst-setup implemented at the moment.")

        return Pi_int
