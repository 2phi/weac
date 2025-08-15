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


class FieldQuantities:
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
        """Horizontal displacement *u = u₀ + h₀ ψ* at depth h₀."""
        return self._unit_factor(unit) * (Z[0, :] + h0 * self.psi(Z))

    def du_dx(self, Z: np.ndarray, h0: float) -> float | np.ndarray:
        """Derivative u' = u₀' + h₀ ψ'."""
        return Z[1, :] + h0 * self.dpsi_dx(Z)

    def w(self, Z: np.ndarray, unit: LengthUnit = "mm") -> float | np.ndarray:
        """Center-line deflection *w*."""
        return self._unit_factor(unit) * Z[2, :]

    def dw_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative w'."""
        return Z[3, :]

    def psi(
        self,
        Z: np.ndarray,
        unit: AngleUnit = "rad",
    ) -> float | np.ndarray:
        """Rotation ψ of the mid-plane."""
        factor = self._unit_factor(unit)
        return factor * Z[4, :]

    def dpsi_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative ψ′."""
        return Z[5, :]

    def N(self, Z: np.ndarray) -> float | np.ndarray:
        """Axial normal force N = A11 u' + B11 psi' in the slab [N]"""
        return self.es.A11 * Z[1, :] + self.es.B11 * Z[5, :]

    def M(self, Z: np.ndarray) -> float | np.ndarray:
        """Bending moment M = B11 u' + D11 psi' in the slab [Nmm]"""
        return self.es.B11 * Z[1, :] + self.es.D11 * Z[5, :]

    def V(self, Z: np.ndarray) -> float | np.ndarray:
        """Vertical shear force V = kA55(w' + psi) [N]"""
        return self.es.kA55 * (Z[3, :] + Z[4, :])

    def sig(self, Z: np.ndarray, unit: StressUnit = "MPa") -> float | np.ndarray:
        """Weak-layer normal stress"""
        return -self._unit_factor(unit) * self.es.weak_layer.kn * self.w(Z)

    def tau(self, Z: np.ndarray, unit: StressUnit = "MPa") -> float | np.ndarray:
        """Weak-layer shear stress"""
        return (
            -self._unit_factor(unit)
            * self.es.weak_layer.kt
            * (
                self.dw_dx(Z) * self.es.weak_layer.h / 2
                - self.u(Z, h0=self.es.slab.H / 2)
            )
        )

    def eps(self, Z: np.ndarray) -> float | np.ndarray:
        """Weak-layer normal strain"""
        return -self.w(Z) / self.es.weak_layer.h

    def gamma(self, Z: np.ndarray) -> float | np.ndarray:
        """Weak-layer shear strain."""
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
        ndarray
            Second derivative z''(x) = (K*z(x) + q)' = K*z'(x) = K*(K*z(x) + q)
            of the solution vector (6x1).
        """
        K = self.es.K
        q = self.es.get_load_vector(phi=phi, qs=qs)
        dz_dx = np.dot(K, z) + q
        return np.dot(K, dz_dx)

    def du0_dxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
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
        return self.dz_dx(z, phi, qs)[1, :]

    def dpsi_dxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
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
        return self.dz_dx(z, phi, qs)[5, :]

    def du0_dxdxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
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
        return self.dz_dxdx(z, phi, qs)[1, :]

    def dpsi_dxdxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
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
        return self.dz_dxdx(z, phi, qs)[5, :]
