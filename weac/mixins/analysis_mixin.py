from __future__ import annotations

"""Mixin for Analysis."""
# Standard library imports
from functools import partial

# Third party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq

# Module imports
from weac.tools import calc_vertical_bc_center_of_gravity, tensile_strength_slab


class AnalysisMixin:
    """
    Mixin for the analysis of model outputs.

    Provides methods for the analysis of layered slabs on compliant
    elastic foundations.
    """

    def rasterize_solution(
        self,
        C: np.ndarray,
        phi: float,
        li: list[float] | bool,
        ki: list[bool] | bool,
        num: int = 250,
        **kwargs,
    ):
        """
        Compute rasterized solution vector.

        Arguments
        ---------
        C : ndarray
            Vector of free constants.
        phi : float
            Inclination (degrees).
        li : ndarray
            List of segment lengths (mm).
        ki : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not.
        num : int
            Number of grid points.

        Returns
        -------
        xq : ndarray
            Grid point x-coordinates at which solution vector
            is discretized.
        zq : ndarray
            Matrix with solution vectors as colums at grid
            points xq.
        xb : ndarray
            Grid point x-coordinates that lie on a foundation.
        """
        # Unused arguments
        _ = kwargs

        # Drop zero-length segments
        li = abs(li)
        isnonzero = li > 0
        C, ki, li = C[:, isnonzero], ki[isnonzero], li[isnonzero]

        # Compute number of plot points per segment (+1 for last segment)
        nq = np.ceil(li / li.sum() * num).astype("int")
        nq[-1] += 1

        # Provide cumulated length and plot point lists
        lic = np.insert(np.cumsum(li), 0, 0)
        nqc = np.insert(np.cumsum(nq), 0, 0)

        # Initialize arrays
        issupported = np.full(nq.sum(), True)
        xq = np.full(nq.sum(), np.nan)
        zq = np.full([6, xq.size], np.nan)

        # Loop through segments
        for i, l in enumerate(li):
            # Get local x-coordinates of segment i
            xi = np.linspace(0, l, num=nq[i], endpoint=(i == li.size - 1))  # pylint: disable=superfluous-parens
            # Compute start and end coordinates of segment i
            x0 = lic[i]
            # Assemble global coordinate vector
            xq[nqc[i] : nqc[i + 1]] = x0 + xi
            # Mask coordinates not on foundation (including endpoints)
            if not ki[i]:
                issupported[nqc[i] : nqc[i + 1]] = False
            # Compute segment solution
            zi = self.z(xi, C[:, [i]], l, phi, ki[i])
            # Assemble global solution matrix
            zq[:, nqc[i] : nqc[i + 1]] = zi

        # Make sure cracktips are included
        transmissionbool = [ki[j] or ki[j + 1] for j, _ in enumerate(ki[:-1])]
        for i, truefalse in enumerate(transmissionbool, start=1):
            issupported[nqc[i]] = truefalse

        # Assemble vector of coordinates on foundation
        xb = np.full(nq.sum(), np.nan)
        xb[issupported] = xq[issupported]

        return xq, zq, xb

    def ginc(self, C0, C1, phi, li, ki, k0, **kwargs):
        """
        Compute incremental energy relase rate of of all cracks.

        Arguments
        ---------
        C0 : ndarray
            Free constants of uncracked solution.
        C1 : ndarray
            Free constants of cracked solution.
        phi : float
            Inclination (degress).
        li : ndarray
            List of segment lengths.
        ki : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not in the cracked configuration.
        k0 : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not in the uncracked configuration.

        Returns
        -------
        ndarray
            List of total, mode I, and mode II energy release rates.
        """
        # Unused arguments
        _ = kwargs

        # Make sure inputs are np.arrays
        li, ki, k0 = np.array(li), np.array(ki), np.array(k0)

        # Reduce inputs to segments with crack advance
        iscrack = k0 & ~ki
        C0, C1, li = C0[:, iscrack], C1[:, iscrack], li[iscrack]

        # Compute total crack lenght and initialize outputs
        da = li.sum() if li.sum() > 0 else np.nan
        Ginc1, Ginc2 = 0, 0

        # Loop through segments with crack advance
        for j, l in enumerate(li):
            # Uncracked (0) and cracked (1) solutions at integration points
            z0 = partial(self.z, C=C0[:, [j]], l=l, phi=phi, bed=True)
            z1 = partial(self.z, C=C1[:, [j]], l=l, phi=phi, bed=False)

            # Mode I (1) and II (2) integrands at integration points
            int1 = partial(self.int1, z0=z0, z1=z1)
            int2 = partial(self.int2, z0=z0, z1=z1)

            # Segement contributions to total crack opening integral
            Ginc1 += quad(int1, 0, l, epsabs=self.tol, epsrel=self.tol)[0] / (2 * da)
            Ginc2 += quad(int2, 0, l, epsabs=self.tol, epsrel=self.tol)[0] / (2 * da)

        return np.array([Ginc1 + Ginc2, Ginc1, Ginc2]).flatten()

    def gdif(self, C, phi, li, ki, unit="kJ/m^2", **kwargs):
        """
        Compute differential energy release rate of all crack tips.

        Arguments
        ---------
        C : ndarray
            Free constants of the solution.
        phi : float
            Inclination (degress).
        li : ndarray
            List of segment lengths.
        ki : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not in the cracked configuration.

        Returns
        -------
        ndarray
            List of total, mode I, and mode II energy release rates.
        """
        # Unused arguments
        _ = kwargs

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
            z = self.z(li[idx], C[:, [idx]], li[idx], phi, bed=ki[idx])
            # Mode I and II differential energy release rates
            Gdif[1:, j] = np.concatenate(
                (self.Gi(z, unit=unit), self.Gii(z, unit=unit))
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
        # Get ply (layer) coordinates
        z = self.get_ply_coordinates()
        # Compute number of grid points per layer
        nlayer = np.ceil((z[1:] - z[:-1]) / dz).astype(np.int32) + 1
        # Calculate grid points as list of z-coordinates (mm)
        zi = np.hstack(
            [
                np.linspace(z[i], z[i + 1], n, endpoint=True)
                for i, n in enumerate(nlayer)
            ]
        )
        # Get lists of corresponding elastic properties (E, nu, rho)
        si = np.repeat(self.slab[:, [2, 4, 0]], nlayer, axis=0)
        # Assemble mesh with columns (z, E, nu, rho)
        return np.column_stack([zi, si])

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
        zi = zmesh[:, 0]
        rho = 1e-12 * zmesh[:, 3]

        # Get dimensions of stress field (n rows, m columns)
        n = zmesh.shape[0]
        m = Z.shape[1]

        # Initialize axial normal stress Sxx
        Sxx = np.zeros(shape=[n, m])

        # Compute axial normal stress Sxx at grid points in MPa
        for i, (z, E, nu, _) in enumerate(zmesh):
            Sxx[i, :] = E / (1 - nu**2) * self.du_dx(Z, z)

        # Calculate weight load at grid points and superimpose on stress field
        qt = -rho * self.g * np.sin(np.deg2rad(phi))
        for i, qi in enumerate(qt[:-1]):
            Sxx[i, :] += qi * (zi[i + 1] - zi[i])
        Sxx[-1, :] += qt[-1] * (zi[-1] - zi[-2])

        # Return axial normal stress in specified unit
        return convert[unit] * Sxx

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
        zi = zmesh[:, 0]
        rho = 1e-12 * zmesh[:, 3]

        # Get dimensions of stress field (n rows, m columns)
        n = zmesh.shape[0]
        m = Z.shape[1]

        # Get second derivatives of centerline displacement u0 and
        # cross-section rotaiton psi of all grid points along the x-axis
        du0_dxdx = self.du0_dxdx(Z, phi)
        dpsi_dxdx = self.dpsi_dxdx(Z, phi)

        # Initialize first derivative of axial normal stress sxx w.r.t. x
        dsxx_dx = np.zeros(shape=[n, m])

        # Calculate first derivative of sxx at z-grid points
        for i, (z, E, nu, _) in enumerate(zmesh):
            dsxx_dx[i, :] = E / (1 - nu**2) * (du0_dxdx + z * dpsi_dxdx)

        # Calculate weight load at grid points
        qt = -rho * self.g * np.sin(np.deg2rad(phi))

        # Integrate -dsxx_dx along z and add cumulative weight load
        # to obtain shear stress Txz in MPa
        Txz = cumulative_trapezoid(dsxx_dx, zi, axis=0, initial=0)
        Txz += cumulative_trapezoid(qt, zi, initial=0)[:, None]

        # Return shear stress Txz in specified unit
        return convert[unit] * Txz

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
        zi = zmesh[:, 0]
        rho = 1e-12 * zmesh[:, 3]

        # Get dimensions of stress field (n rows, m columns)
        n = zmesh.shape[0]
        m = Z.shape[1]

        # Get third derivatives of centerline displacement u0 and
        # cross-section rotaiton psi of all grid points along the x-axis
        du0_dxdxdx = self.du0_dxdxdx(Z, phi)
        dpsi_dxdxdx = self.dpsi_dxdxdx(Z, phi)

        # Initialize second derivative of axial normal stress sxx w.r.t. x
        dsxx_dxdx = np.zeros(shape=[n, m])

        # Calculate second derivative of sxx at z-grid points
        for i, (z, E, nu, _) in enumerate(zmesh):
            dsxx_dxdx[i, :] = E / (1 - nu**2) * (du0_dxdxdx + z * dpsi_dxdxdx)

        # Calculate weight load at grid points
        qn = rho * self.g * np.cos(np.deg2rad(phi))

        # Integrate dsxx_dxdx twice along z to obtain transverse
        # normal stress Szz in MPa
        integrand = cumulative_trapezoid(dsxx_dxdx, zi, axis=0, initial=0)
        Szz = cumulative_trapezoid(integrand, zi, axis=0, initial=0)
        Szz += cumulative_trapezoid(-qn, zi, initial=0)[:, None]

        # Return shear stress txz in specified unit
        return convert[unit] * Szz

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
            # Get layer densities
            rho = self.get_zmesh(dz=dz)[:, 3]
            # Normlize maximum principal stress to layers' tensile strength
            normalized_Ps = Ps / tensile_strength_slab(rho, unit=unit)[:, None]
            return normalized_Ps

        # Return absolute principal stresses
        return Ps

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
        sig = self.sig(Z, unit=unit)
        tau = self.tau(Z, unit=unit)

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
