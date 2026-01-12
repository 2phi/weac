"""
This module defines the FieldQuantities class, which is responsible for calculating
and providing access to various physical quantities within the slab.
"""

from typing import Literal

import numpy as np

from weac.core.eigensystem import Eigensystem

LengthUnit = Literal["m", "cm", "mm", "um"]
AngleUnit = Literal["deg", "rad"]
StressUnit = Literal["Pa", "kPa", "MPa", "GPa"]
EnergyUnit = Literal["J/m^2", "kJ/m^2", "N/mm"]
Unit = Literal[LengthUnit, AngleUnit, StressUnit, EnergyUnit]

_UNIT_FACTOR: dict[str, float] = {
    "m": 1e-3,
    "cm": 1e-1,
    "mm": 1,
    "um": 1e3,
    "rad": 1,
    "deg": 180 / np.pi,
    "Pa": 1e6,
    "kPa": 1e3,
    "MPa": 1,
    "GPa": 1e-3,
    "J/m^2": 1e3,  # joule per square meter
    "kJ/m^2": 1,  # kilojoule per square meter
    "N/mm": 1,  # newton per millimeter
}


class FieldQuantities:  # pylint: disable=too-many-instance-attributes, too-many-public-methods
    """
    Convenience accessors for a 6xN solution matrix Z =
    [u, u', w, w', ψ, ψ']ᵀ.  All functions are *vectorized* along the second
    axis (x-coordinate), so they return an `ndarray` of length N.
    """

    def __init__(self, eigensystem: Eigensystem):
        self.es = eigensystem

    @staticmethod
    def _unit_factor(unit: Unit, /) -> float:
        """Return multiplicative factor associated with *unit*."""
        try:
            return _UNIT_FACTOR[unit]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported unit: {unit!r}, supported units are {_UNIT_FACTOR}"
            ) from exc

    def u(
        self,
        Z: np.ndarray,
        h0: float = 0,
        unit: LengthUnit = "mm",
    ) -> float | np.ndarray:
        """
        Compute the horizontal displacement at depth h0.
        
        Parameters:
            Z (np.ndarray): Solution matrix where rows are field components and columns are x positions; row 0 is the centerline horizontal displacement u0.
            h0 (float): Depth from the mid-plane measured in the same length unit specified by `unit` (default "mm").
            unit (LengthUnit): Output length unit; scales the result accordingly.
        
        Returns:
            np.ndarray | float: Horizontal displacement u = u0 + h0 * ψ at depth h0, expressed in `unit`; vectorized over Z's columns (1D array) or a scalar for a single position.
        """
        return self._unit_factor(unit) * (Z[0, :] + h0 * self.psi(Z))

    def du_dx(self, Z: np.ndarray, h0: float) -> float | np.ndarray:
        """
        Compute the horizontal displacement derivative u' at depth h0.
        
        Parameters:
            Z (np.ndarray): Solution matrix where rows correspond to state variables and columns to x positions.
            h0 (float): Depth offset from the mid-plane at which u' is evaluated.
        
        Returns:
            np.ndarray | float: u' = u0' + h0 * psi' evaluated along the x positions (1D array) or a scalar for a single position.
        """
        return Z[1, :] + h0 * self.dpsi_dx(Z)

    def w(self, Z: np.ndarray, unit: LengthUnit = "mm") -> float | np.ndarray:
        """
        Compute center-line vertical deflection for the slab.
        
        Parameters:
            Z (np.ndarray): Solution matrix where each column is the state vector along x; row 2 contains center-line vertical deflection values.
            unit (LengthUnit): Target length unit for the returned deflection (default "mm").
        
        Returns:
            float | np.ndarray: Center-line vertical deflection at each x position in the specified unit."""
        return self._unit_factor(unit) * Z[2, :]

    def dw_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """
        First derivative of the center-line vertical deflection w with respect to x.
        
        Parameters:
            Z (ndarray): Solution matrix where rows are field components and columns correspond to x positions; this method reads the fourth row.
        
        Returns:
            w_prime (ndarray): 1D array of w' (dw/dx) evaluated at each x position (values per unit length).
        """
        return Z[3, :]

    def psi(
        self,
        Z: np.ndarray,
        unit: AngleUnit = "rad",
    ) -> float | np.ndarray:
        """
        Compute the mid-plane rotation ψ from the solution matrix Z in the requested angular unit.
        
        Parameters:
            Z (np.ndarray): Solution matrix whose fifth row (index 4) contains ψ values in radians; may be 2D (rows × positions).
            unit (AngleUnit): Output unit for rotation (e.g., "rad", "deg").
        
        Returns:
            float | np.ndarray: ψ values converted to `unit`; returns a 1D array over x positions when Z has multiple columns.
        """
        factor = self._unit_factor(unit)
        return factor * Z[4, :]

    def dpsi_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """
        Return the first derivative of the mid-plane rotation ψ with respect to x (ψ', in radians per millimeter).
        
        Parameters:
            Z (np.ndarray): Solution matrix where each row corresponds to a field quantity and each column to an x-position.
        
        Returns:
            np.ndarray | float: ψ' values vectorized over x (1D array of radians/mm), or a scalar if Z contains a single column.
        """
        return Z[5, :]

    def N(self, Z: np.ndarray) -> float | np.ndarray:
        """
        Compute the axial normal force across x positions using N = A11 * u' + B11 * ψ'.
        
        Parameters:
            Z (numpy.ndarray): Solution array with state variables along rows and x positions along columns.
                Row 1 must contain u' (horizontal displacement derivative) and row 5 must contain ψ' (rotation derivative).
        
        Returns:
            numpy.ndarray or float: Axial normal force N in newtons (N) for each x position; a scalar if Z represents a single position.
        """
        return self.es.A11 * Z[1, :] + self.es.B11 * Z[5, :]

    def M(self, Z: np.ndarray) -> float | np.ndarray:
        """
        Compute bending moment distribution in the slab.
        
        Parameters:
            Z (np.ndarray): Solution array where rows are field components and columns are x positions; expects u' at row 1 and ψ' at row 5.
        
        Returns:
            M (np.ndarray | float): Bending moment in N·mm — a 1D array over x positions when Z has multiple columns, or a scalar for a single column.
        """
        return self.es.B11 * Z[1, :] + self.es.D11 * Z[5, :]

    def V(self, Z: np.ndarray) -> float | np.ndarray:
        """
        Compute the vertical shear force for the slab from the solution matrix.
        
        Parameters:
            Z (ndarray): Solution vector or matrix of state variables where row 3 is the center-line deflection derivative w' and row 4 is the mid-plane rotation ψ. If Z is 2D, computation is vectorized along columns and returns a 1D array over x.
        
        Returns:
            V (float or ndarray): Vertical shear force in newtons (N), equal to kA55 * (w' + ψ) evaluated for each column of Z.
        """
        return self.es.kA55 * (Z[3, :] + Z[4, :])

    def sig(self, Z: np.ndarray, unit: StressUnit = "MPa") -> float | np.ndarray:
        """
        Compute the weak-layer normal stress σ from the center-line vertical deflection.
        
        The stress is given by sig = -factor(unit) * kn * w(Z), where `kn` is the weak-layer normal stiffness and `factor(unit)` converts internal units to the requested `unit`.
        
        Parameters:
            Z (np.ndarray): Solution matrix (vectorized over x positions).
            unit (StressUnit): Desired output unit (e.g., "MPa").
        
        Returns:
            σ (float | np.ndarray): Weak-layer normal stress in the requested unit; scalar or 1D array over x matching the shape of `w(Z)`.
        """
        return -self._unit_factor(unit) * self.es.weak_layer.kn * self.w(Z)

    def tau(self, Z: np.ndarray, unit: StressUnit = "MPa") -> float | np.ndarray:
        """
        Compute the weak-layer shear stress for the given solution.
        
        Parameters:
            Z (np.ndarray): Solution matrix whose columns correspond to x positions.
            unit (StressUnit): Output stress unit (e.g., "MPa").
        
        Returns:
            Shear stress `tau` in the specified unit: a scalar or 1D array over x representing
            tau = -kt * (w' * h/2 - u at h = H/2).
        """
        return (
            -self._unit_factor(unit)
            * self.es.weak_layer.kt
            * (
                self.dw_dx(Z) * self.es.weak_layer.h / 2
                - self.u(Z, h0=self.es.slab.H / 2)
            )
        )

    def eps(self, Z: np.ndarray) -> float | np.ndarray:
        """
        Compute the weak-layer normal strain from the slab's vertical deflection.
        
        Parameters:
            Z (ndarray): Solution vector or matrix containing modal amplitudes; methods are vectorized along the second axis so Z may be 1D or 2D.
        
        Returns:
            eps (float or ndarray): Weak-layer normal strain given by eps = -w / h, where w is the center-line vertical deflection and h is the weak-layer thickness. Returns a scalar if Z is a single solution vector or a 1D array over x positions if Z is a matrix.
        """
        return -self.w(Z) / self.es.weak_layer.h

    def gamma(self, Z: np.ndarray) -> float | np.ndarray:
        """
        Compute the weak-layer shear strain γ for the provided solution matrix.
        
        Parameters:
            Z (np.ndarray): Solution matrix Z; methods are vectorized along columns so each column corresponds to an x position.
        
        Returns:
            gamma (float | np.ndarray): Weak-layer shear strain γ (dimensionless) as a scalar or a 1D array over x positions.
        """
        return (
            self.dw_dx(Z) / 2 - self.u(Z, h0=self.es.slab.H / 2) / self.es.weak_layer.h
        )

    def Gi(self, Ztip: np.ndarray, unit: EnergyUnit = "kJ/m^2") -> float | np.ndarray:
        """Mode I differential energy release rate at crack tip.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T
            at the crack tip.
        unit : {'N/mm', 'kJ/m^2', 'J/m^2'}, optional
            Desired output unit. Default is kJ/m^2.
        """
        return (
            self._unit_factor(unit) * self.sig(Ztip) ** 2 / (2 * self.es.weak_layer.kn)
        )

    def Gii(self, Ztip: np.ndarray, unit: EnergyUnit = "kJ/m^2") -> float | np.ndarray:
        """Mode II differential energy release rate at crack tip.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T
            at the crack tip.
        unit : {'N/mm', 'kJ/m^2', 'J/m^2'}, optional
            Desired output unit. Default is kJ/m^2 = N/mm.
        """
        return (
            self._unit_factor(unit) * self.tau(Ztip) ** 2 / (2 * self.es.weak_layer.kt)
        )

    def dz_dx(self, z: np.ndarray, phi: float, qs: float = 0) -> np.ndarray:
        """First derivative z'(x) = K*z(x) + q of the solution vector.

        z'(x) = [u'(x) u''(x) w'(x) w''(x) psi'(x), psi''(x)]^T

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray
            First derivative z'(x) for the solution vector (6x1).
        """
        K = self.es.K
        q = self.es.get_load_vector(phi=phi, qs=qs)
        return np.dot(K, z) + q

    def dz_dxdx(self, z: np.ndarray, phi: float, qs: float) -> np.ndarray:
        """
        Compute the second spatial derivative z''(x) of the solution vector.
        
        The returned array contains the components [u''(x), u'''(x), w''(x), w'''(x), psi''(x), psi'''(x)]^T and is computed as z'' = K * z' with z' = K*z + q.
        
        Parameters:
            z (ndarray): Solution vector with shape (6, N) containing [u, u', w, w', psi, psi'] across N positions.
            phi (float): Inclination in degrees (counterclockwise positive).
            qs (float): Shear load parameter passed to the load-vector generator.
        
        Returns:
            ndarray: Second derivative z'' with shape (6, N), matching the input z's column count.
        """
        K = self.es.K
        q = self.es.get_load_vector(phi=phi, qs=qs)
        dz_dx = np.dot(K, z) + q
        return np.dot(K, dz_dx)

    def du0_dxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
        """
        Second derivative of the horizontal centerline displacement u0 with respect to x (1/mm).
        
        Parameters:
            z (ndarray): Solution vector [u(x), u'(x), w(x), w'(x), psi(x), psi'(x)]^T sampled along x.
            phi (float): Inclination in degrees; counterclockwise positive.
            qs (float): Distributed transverse load amplitude used to form the load vector.
        
        Returns:
            float or ndarray: u0''(x) evaluated at each x position; a scalar if `z` represents a single point or a 1D array over x positions.
        """
        return self.dz_dx(z, phi, qs)[1, :]

    def dpsi_dxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
        """
        Compute the second derivative of the cross-section rotation ψ with respect to x (ψ''(x), units: 1/mm²).
        
        Parameters:
            z (ndarray): Solution vector [u, u', w, w', ψ, ψ']^T evaluated along x; may be a 1D array or 2D array where second axis indexes x positions.
            phi (float): Inclination in degrees; counterclockwise positive.
            qs (float): Shear load parameter passed to the load vector computation.
        
        Returns:
            float or ndarray: ψ''(x) as a scalar for a single x position or a 1D ndarray of values along x.
        """
        return self.dz_dx(z, phi, qs)[5, :]

    def du0_dxdxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
        """
        Return the third derivative of the horizontal centerline displacement u0 with respect to x.
        
        Parameters:
            z (ndarray): Solution vector [u(x), u'(x), w(x), w'(x), psi(x), psi'(x)]^T; if Z is a matrix, operations are vectorized along the second axis.
            phi (float): Inclination in degrees, counterclockwise positive.
            qs (float): External transverse load intensity.
        
        Returns:
            float or ndarray: u0'''(x) (mm⁻²) as a scalar for a single position or a 1D array over x positions.
        """
        return self.dz_dxdx(z, phi, qs)[1, :]

    def dpsi_dxdxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
        """
        Compute the third spatial derivative of the cross-section rotation psi.
        
        Parameters:
            z (ndarray): Solution vector [u(x), u'(x), w(x), w'(x), psi(x), psi'(x)]^T.
            phi (float): Inclination in degrees (counterclockwise positive).
            qs (float): Distributed load amplitude or shear parameter used in the load vector.
        
        Returns:
            float or ndarray: psi'''(x) (units: mm^-3), returned as a scalar for a single position or a 1D array over x positions.
        """
        return self.dz_dxdx(z, phi, qs)[5, :]