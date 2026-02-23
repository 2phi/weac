"""
This module defines the GeneralizedFieldQuantities class, which is responsible for calculating
and providing access to various physical quantities within the slab in the Generalized Model.
"""

from typing import Literal

import numpy as np

from weac.core.generalized_eigensystem import GeneralizedEigensystem

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
    "J/m^2": 1e3,
    "kJ/m^2": 1,
    "N/mm": 1,
}


class GeneralizedFieldQuantities:
    """
    Convenience accessors for a 24xN solution matrix Z =
    [u, u',v, v' w, w', ψx, ψx',ψy, ψy',ψz, ψz',thetauc, thetauc', thetaul]ᵀ.  All functions are *vectorized* along the second
    axis (x-coordinate), so they return an `ndarray` of length N.
    """

    def __init__(self, eigensystem: GeneralizedEigensystem):
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
        b0: float = 0,
        unit: LengthUnit = "mm",
    ) -> float | np.ndarray:
        """Axial displacement *u = u₀ + h₀ ψy - b₀ ψz * at depth h₀."""
        return self._unit_factor(unit) * (
            Z[0, :] + h0 * self.psiy(Z) - b0 * self.psiz(Z)
        )

    def du_dx(self, Z: np.ndarray, h0: float = 0, b0: float = 0) -> float | np.ndarray:
        """Derivative u' = u₀' + h₀ ψy'- b₀ ψz'."""
        return Z[1, :] + h0 * self.dpsiy_dx(Z) - b0 * self.dpsiz_dx(Z)

    def v(
        self,
        Z: np.ndarray,
        h0: float = 0,
        unit: LengthUnit = "mm",
    ) -> float | np.ndarray:
        """Transverse displacement *v = v₀ - h₀ ψx * at depth h₀."""
        return self._unit_factor(unit) * (Z[2, :] - h0 * self.psix(Z))

    def dv_dx(self, Z: np.ndarray, h0: float) -> float | np.ndarray:
        """Derivative v' = v₀' - h₀ ψx'."""
        return Z[3, :] - h0 * self.dpsix_dx(Z)

    def w(
        self, Z: np.ndarray, b0: float = 0, unit: LengthUnit = "mm"
    ) -> float | np.ndarray:
        """Vertical displacement *w = w₀ + b₀ ψx *."""
        return self._unit_factor(unit) * (Z[4, :] + b0 * self.psix(Z))

    def dw_dx(self, Z: np.ndarray, b0: float = 0) -> float | np.ndarray:
        """First derivative *w' = w₀' + b₀ ψx' *."""
        return Z[5, :] + b0 * self.dpsix_dx(Z)

    def psix(
        self,
        Z: np.ndarray,
        unit: AngleUnit = "rad",
    ) -> float | np.ndarray:
        """Torsional rotation ψx of the mid-plane."""
        factor = self._unit_factor(unit)
        return factor * Z[6, :]

    def dpsix_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative ψx′."""
        return Z[7, :]

    def psiy(
        self,
        Z: np.ndarray,
        unit: AngleUnit = "rad",
    ) -> float | np.ndarray:
        """Bending rotation ψy of the mid-plane around the y-axis."""
        factor = self._unit_factor(unit)
        return factor * Z[8, :]

    def dpsiy_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative ψy′."""
        return Z[9, :]

    def psiz(
        self,
        Z: np.ndarray,
        unit: AngleUnit = "rad",
    ) -> float | np.ndarray:
        """Bending rotation ψz of the mid-plane around the z-axis."""
        factor = self._unit_factor(unit)
        return factor * Z[10, :]

    def dpsiz_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative ψz′."""
        return Z[11, :]

    def theta_uc(self, Z: np.ndarray) -> float | np.ndarray:
        """Linear amplitude of axial cosine shaped displacements in the weak layer."""
        return Z[12, :]

    def dtheta_uc_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative of the constant amplitude of axial cosine shaped displacements in the weak layer."""
        return Z[13, :]

    def theta_ul(self, Z: np.ndarray) -> float | np.ndarray:
        """Linear amplitude of axial cosine shaped displacements in the weak layer."""
        return Z[14, :]

    def dtheta_ul_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative of the linear amplitude of axial cosine shaped displacements in the weak layer."""
        return Z[15, :]

    def theta_vc(self, Z: np.ndarray) -> float | np.ndarray:
        """Linear amplitude of axial cosine shaped displacements in the weak layer."""
        return Z[16, :]

    def dtheta_vc_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative of the constant amplitude of out-of-plane cosine shaped displacements in the weak layer."""
        return Z[17, :]

    def theta_vl(self, Z: np.ndarray) -> float | np.ndarray:
        """Linear amplitude of axial cosine shaped displacements in the weak layer."""
        return Z[18, :]

    def dtheta_vl_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative of the linear amplitude of out-of-plane cosine shaped displacements in the weak layer."""
        return Z[19, :]

    def theta_wc(self, Z: np.ndarray) -> float | np.ndarray:
        """Linear amplitude of axial cosine shaped displacements in the weak layer."""
        return Z[20, :]

    def dtheta_wc_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative of the constant amplitude of vertical cosine shaped displacements in the weak layer."""
        return Z[21, :]

    def theta_wl(self, Z: np.ndarray) -> float | np.ndarray:
        """Linear amplitude of axial cosine shaped displacements in the weak layer."""
        return Z[22, :]

    def dtheta_wl_dx(self, Z: np.ndarray) -> float | np.ndarray:
        """First derivative of the linear amplitude of vertical cosine shaped displacements in the weak layer."""
        return Z[23, :]

    def Nx(self, Z: np.ndarray, has_foundation: bool) -> float | np.ndarray:
        """Axial normal force Nx = A11 b u' + B11 b psiy'  in the slab [N]"""
        Nx_slab = self.es.A11 * self.es.slab.b * Z[
            1, :
        ] + self.es.slab.b * self.es.B11 * self.dpsiy_dx(Z)
        if has_foundation:
            return Nx_slab + (
                self.es.weak_layer.E
                * (
                    3 * self.es.slab.b * np.pi * self.es.weak_layer.nu * Z[4, :]
                    - 12
                    * self.es.weak_layer.h
                    * self.es.weak_layer.nu
                    * self.theta_vl(Z)
                    - 12 * self.es.slab.b * self.es.weak_layer.nu * self.theta_wc(Z)
                    + self.es.slab.b
                    * self.es.weak_layer.h
                    * (-1 + self.es.weak_layer.nu)
                    * (
                        2 * np.pi * Z[1, :]
                        + 6 * self.dtheta_uc_dx(Z)
                        + self.es.slab.H * np.pi * self.dpsiy_dx(Z)
                    )
                )
            ) / (
                6 * np.pi * (-1 + self.es.weak_layer.nu + 2 * self.es.weak_layer.nu**2)
            )
        else:
            return Nx_slab

    def Vy(self, Z: np.ndarray, has_foundation: bool) -> float | np.ndarray:
        """Vertical out-of-plane shear force Vy = kA66 b (-psiz + v')- kB66 b psix'  [N]"""
        if not has_foundation:
            return self.es.slab.b * self.es.kA55 * (
                -self.psiz(Z) + Z[3, :]
            ) - self.es.slab.b * self.es.kB55 * self.dpsix_dx(Z)
        else:
            return (
                self.es.slab.b * self.es.kA55 * (-self.psiz(Z) + Z[3, :])
                - self.es.slab.b * self.es.kB55 * self.dpsix_dx(Z)
                + (
                    self.es.weak_layer.h
                    * self.es.weak_layer.E
                    * (
                        12 * self.theta_ul(Z)
                        + self.es.slab.b
                        * (
                            -2 * np.pi * self.psiz(Z)
                            + 2 * np.pi * Z[3, :]
                            + 6 * self.dtheta_vc_dx(Z)
                            - self.es.slab.H * np.pi * self.dpsix_dx(Z)
                        )
                    )
                )
                / (12 * np.pi * (1 + self.es.weak_layer.nu))
            )

    def Vz(self, Z: np.ndarray, has_foundation: bool) -> float | np.ndarray:
        """Vertical shear force V = kA55(w' + psiy) [N]"""
        if not has_foundation:
            return self.es.slab.b * self.es.kA55 * (self.psiy(Z) + Z[5, :])
        else:
            return self.es.slab.b * self.es.kA55 * (self.psiy(Z) + Z[5, :]) + (
                self.es.slab.b
                * self.es.weak_layer.E
                * (
                    -6 * np.pi * Z[0, :]
                    + 24 * self.theta_uc(Z)
                    - 3 * self.es.slab.H * np.pi * self.psiy(Z)
                    + 4 * np.pi * self.es.weak_layer.h * Z[5, :]
                    + 12 * self.es.weak_layer.h * self.dtheta_wc_dx(Z)
                )
            ) / (24 * np.pi * (1 + self.es.weak_layer.nu))

    def Mx(self, Z: np.ndarray, has_foundation: bool) -> float | np.ndarray:
        """Torsional moment Mx = kA55 * b^3/12 * psix' + kB66 b (psiz - v0') + kD66 b psix' in the slab [Nmm]"""
        if not has_foundation:
            return (
                self.es.slab.b * self.es.kB55 * (self.psiz(Z) - Z[3, :])
                + (self.es.slab.b**3 * self.es.kA55 * self.dpsix_dx(Z)) / 12
                + self.es.slab.b * self.es.kD55 * self.dpsix_dx(Z)
            )
        else:
            return (
                self.es.slab.b * self.es.kB55 * (self.psiz(Z) - Z[3, :])
                + (self.es.slab.b**3 * self.es.kA55 * self.dpsix_dx(Z)) / 12
                + self.es.slab.b * self.es.kD55 * self.dpsix_dx(Z)
                - (
                    self.es.slab.b**2
                    * self.es.weak_layer.E
                    * (
                        -24 * self.es.weak_layer.h * self.theta_ul(Z)
                        + 3
                        * self.es.slab.b
                        * np.pi
                        * (-2 + self.es.weak_layer.h)
                        * self.psiz(Z)
                        + 12
                        * (-2 + self.es.weak_layer.h)
                        * self.es.weak_layer.h
                        * self.dtheta_wl_dx(Z)
                        + self.es.slab.b
                        * np.pi
                        * (-3 + self.es.weak_layer.h)
                        * self.es.weak_layer.h
                        * self.dpsix_dx(Z)
                    )
                )
                / (144 * np.pi * self.es.weak_layer.h * (1 + self.es.weak_layer.nu))
                - (
                    self.es.slab.H
                    * self.es.weak_layer.h
                    * self.es.weak_layer.E
                    * (
                        12 * self.theta_ul(Z)
                        + self.es.slab.b
                        * (
                            -2 * np.pi * self.psiz(Z)
                            + 2 * np.pi * Z[3, :]
                            + 6 * self.dtheta_vc_dx(Z)
                            - self.es.slab.H * np.pi * self.dpsix_dx(Z)
                        )
                    )
                )
                / (24 * np.pi * (1 + self.es.weak_layer.nu))
            )

    def My(self, Z: np.ndarray, has_foundation: bool) -> float | np.ndarray:
        """Bending moment My = B11 b u' + D11 b psiy' in the slab[Nmm]"""
        if not has_foundation:
            return self.es.slab.b * self.es.B11 * Z[
                1, :
            ] + self.es.slab.b * self.es.D11 * self.dpsiy_dx(Z)
        else:
            return (
                self.es.slab.b * self.es.B11 * Z[1, :]
                + self.es.slab.b * self.es.D11 * self.dpsiy_dx(Z)
                + (
                    self.es.slab.H
                    * self.es.weak_layer.E
                    * (
                        3 * self.es.slab.b * np.pi * self.es.weak_layer.nu * Z[4, :]
                        - 12
                        * self.es.weak_layer.h
                        * self.es.weak_layer.nu
                        * self.theta_vl(Z)
                        - 12 * self.es.slab.b * self.es.weak_layer.nu * self.theta_wc(Z)
                        + self.es.slab.b
                        * self.es.weak_layer.h
                        * (-1 + self.es.weak_layer.nu)
                        * (
                            2 * np.pi * Z[1, :]
                            + 6 * self.dtheta_uc_dx(Z)
                            + self.es.slab.H * np.pi * self.dpsiy_dx(Z)
                        )
                    )
                )
                / (
                    12
                    * np.pi
                    * (-1 + self.es.weak_layer.nu + 2 * self.es.weak_layer.nu**2)
                )
            )

    def Mz(self, Z: np.ndarray, has_foundation: bool) -> float | np.ndarray:
        """Bending moment Mz = A11 b^3/12 * psiz' in the slab [Nmm]"""
        if not has_foundation:
            return (self.es.A11 * self.es.slab.b**3 * self.dpsiz_dx(Z)) / 12
        else:
            return (self.es.A11 * self.es.slab.b**3 * self.dpsiz_dx(Z)) / 12 - (
                self.es.slab.b**2
                * self.es.weak_layer.E
                * (
                    -24 * self.es.weak_layer.nu * self.theta_wl(Z)
                    + 3 * self.es.slab.b * np.pi * self.es.weak_layer.nu * self.psix(Z)
                    + 2
                    * self.es.weak_layer.h
                    * (-1 + self.es.weak_layer.nu)
                    * (
                        6 * self.dtheta_ul_dx(Z)
                        - self.es.slab.b * np.pi * self.dpsiz_dx(Z)
                    )
                )
            ) / (
                72 * np.pi * (-1 + self.es.weak_layer.nu + 2 * self.es.weak_layer.nu**2)
            )

    def Nx_c_weakLayer(self, Z: np.ndarray) -> float | np.ndarray:
        """Generalized axial force associated to theta_uc [N]"""
        return (
            self.es.weak_layer.E
            * (
                4 * self.es.slab.b * self.es.weak_layer.nu * Z[4, :]
                - 2
                * np.pi
                * self.es.weak_layer.h
                * self.es.weak_layer.nu
                * self.theta_vl(Z)
                + self.es.slab.b
                * self.es.weak_layer.h
                * (-1 + self.es.weak_layer.nu)
                * (
                    2 * Z[1, :]
                    + np.pi * self.dtheta_uc_dx(Z)
                    + self.es.slab.H * self.dpsiy_dx(Z)
                )
            )
        ) / (2 * np.pi * (1 + self.es.weak_layer.nu) * (-1 + 2 * self.es.weak_layer.nu))

    def Nx_l_weakLayer(self, Z: np.ndarray) -> float | np.ndarray:
        """Generalized axial force associated to theta_ul [N]"""
        return (
            self.es.slab.b
            * self.es.weak_layer.E
            * (
                2 * self.es.slab.b * self.psiz(Z)
                + np.pi * self.es.weak_layer.h * self.dtheta_wl_dx(Z)
                + self.es.slab.b * self.es.weak_layer.h * self.dpsix_dx(Z)
            )
        ) / (12 * np.pi * (1 + self.es.weak_layer.nu))

    def Vy_c_weakLayer(self, Z: np.ndarray) -> float | np.ndarray:
        """Generalized axial force associated to theta_vc [N]"""
        return (
            self.es.weak_layer.h
            * self.es.weak_layer.E
            * (
                2 * np.pi * self.theta_ul(Z)
                + self.es.slab.b
                * (
                    -2 * self.psiz(Z)
                    + 2 * Z[3, :]
                    + np.pi * self.dtheta_vc_dx(Z)
                    - self.es.slab.H * self.dpsix_dx(Z)
                )
            )
        ) / (4 * np.pi * (1 + self.es.weak_layer.nu))

    def Vy_l_weakLayer(self, Z: np.ndarray) -> float | np.ndarray:
        """Generalized axial force associated to theta_vl [N]"""
        return (
            self.es.slab.b
            * self.es.weak_layer.h
            * self.es.weak_layer.E
            * self.dtheta_vl_dx(Z)
        ) / (12 * (1 + self.es.weak_layer.nu))

    def Vz_c_weakLayer(self, Z: np.ndarray) -> float | np.ndarray:
        """Generalized axial force associated to theta_wc [N]"""
        return (
            self.es.slab.b
            * self.es.weak_layer.E
            * (
                -4 * Z[0, :]
                - 2 * self.es.slab.H * self.psiy(Z)
                + 2 * self.es.weak_layer.h * Z[5, :]
                + np.pi * self.es.weak_layer.h * self.dtheta_wc_dx(Z)
            )
        ) / (4 * np.pi * (1 + self.es.weak_layer.nu))

    def Vz_l_weakLayer(self, Z: np.ndarray) -> float | np.ndarray:
        """Generalized axial force associated to theta_wl [N]"""
        return (
            self.es.slab.b
            * self.es.weak_layer.E
            * (
                2 * self.es.slab.b * self.es.weak_layer.nu * self.psix(Z)
                + self.es.weak_layer.h
                * (-1 + self.es.weak_layer.nu)
                * (np.pi * self.dtheta_ul_dx(Z) - self.es.slab.b * self.dpsiz_dx(Z))
            )
        ) / (6 * np.pi * (1 + self.es.weak_layer.nu) * (-1 + 2 * self.es.weak_layer.nu))

    def sig_zz(
        self,
        Z: np.ndarray,
        h0: float | None = None,
        b0: float = 0,
        unit: StressUnit = "MPa",
    ) -> float | np.ndarray:
        """Weak-layer normal stress"""
        if h0 is None:
            h0 = self.es.slab.H / 2 + self.es.weak_layer.h / 2
        return (
            self._unit_factor(unit)
            * (
                self.es.weak_layer.E
                * (
                    2
                    * self.es.weak_layer.h
                    * self.es.weak_layer.nu
                    * np.cos(
                        (np.pi * (self.es.slab.H + self.es.weak_layer.h - 2 * h0))
                        / (2 * self.es.weak_layer.h)
                    )
                    * self.theta_vl(Z)
                    - (1 - self.es.weak_layer.nu)
                    * (
                        -(
                            np.pi
                            * np.cos(
                                (np.pi * (self.es.slab.H - 2 * h0))
                                / (2 * self.es.weak_layer.h)
                            )
                            * (
                                self.es.slab.b * self.theta_wc(Z)
                                + 2 * b0 * self.theta_wl(Z)
                            )
                        )
                        + self.es.slab.b * (Z[4, :] + b0 * self.psix(Z))
                    )
                    + self.es.weak_layer.nu
                    * (
                        self.es.weak_layer.h
                        * np.cos(
                            (np.pi * (self.es.slab.H + self.es.weak_layer.h - 2 * h0))
                            / (2 * self.es.weak_layer.h)
                        )
                        * (
                            self.es.slab.b * self.dtheta_uc_dx(Z)
                            + 2 * b0 * self.dtheta_ul_dx(Z)
                        )
                        + (
                            self.es.slab.b
                            * (self.es.slab.H + 2 * self.es.weak_layer.h - 2 * h0)
                            * (
                                2 * Z[1, :]
                                + self.es.slab.H * self.dpsiy_dx(Z)
                                - 2 * b0 * self.dpsiz_dx(Z)
                            )
                        )
                        / 4
                    )
                )
            )
            / (
                self.es.slab.b
                * self.es.weak_layer.h
                * (1 - 2 * self.es.weak_layer.nu)
                * (1 + self.es.weak_layer.nu)
            )
        )

    def tau_yz(
        self, Z: np.ndarray, h0: float = 0, b0: float = 0, unit: StressUnit = "MPa"
    ) -> float | np.ndarray:
        """Weak-layer shear stress"""
        return (
            self._unit_factor(unit)
            * (
                self.es.weak_layer.E
                * (
                    -(
                        (
                            np.pi
                            * np.sin((np.pi * h0) / self.es.weak_layer.h)
                            * (
                                self.theta_vc(Z)
                                + (2 * b0 * self.theta_vl(Z)) / self.es.slab.b
                            )
                        )
                        / self.es.weak_layer.h
                    )
                    + (
                        2
                        * np.cos((np.pi * h0) / self.es.weak_layer.h)
                        * self.theta_wl(Z)
                    )
                    / self.es.slab.b
                    + (1 - (self.es.weak_layer.h / 2 + h0) / self.es.weak_layer.h)
                    * self.psix(Z)
                    - (Z[2, :] - (self.es.slab.H * self.psix(Z)) / 2)
                    / self.es.weak_layer.h
                )
            )
            / (2 * (1 + self.es.weak_layer.nu))
        )

    def tau_xz(
        self, Z: np.ndarray, h0: float = 0, b0: float = 0, unit: StressUnit = "MPa"
    ) -> float | np.ndarray:
        """Weak-layer shear stress"""
        return (
            self._unit_factor(unit)
            * (
                self.es.weak_layer.E
                * (
                    -(
                        (
                            np.pi
                            * np.sin((np.pi * h0) / self.es.weak_layer.h)
                            * (
                                self.theta_uc(Z)
                                + (2 * b0 * self.theta_ul(Z)) / self.es.slab.b
                            )
                        )
                        / self.es.weak_layer.h
                    )
                    - (
                        Z[0, :]
                        + (self.es.slab.H * self.psiy(Z)) / 2
                        - b0 * self.psiz(Z)
                    )
                    / self.es.weak_layer.h
                    + np.cos((np.pi * h0) / self.es.weak_layer.h)
                    * (
                        self.dtheta_wc_dx(Z)
                        + (2 * b0 * self.dtheta_wl_dx(Z)) / self.es.slab.b
                    )
                    + (1 - (self.es.weak_layer.h / 2 + h0) / self.es.weak_layer.h)
                    * (Z[5, :] + b0 * self.dpsix_dx(Z))
                )
            )
            / (2 * (1 + self.es.weak_layer.nu))
        )

    def eps_zz(self, Z: np.ndarray, h0: float = 0, b0: float = 0) -> float | np.ndarray:
        """Weak-layer normal strain"""
        return (
            -(
                (
                    np.pi
                    * np.sin((np.pi * h0) / self.es.weak_layer.h)
                    * (self.theta_wc(Z) + (2 * b0 * self.theta_wl(Z)) / self.es.slab.b)
                )
                / self.es.weak_layer.h
            )
            - (Z[4, :] + b0 * self.psix(Z)) / self.es.weak_layer.h
        )

    def gamma_yz(
        self, Z: np.ndarray, h0: float = 0, b0: float = 0
    ) -> float | np.ndarray:
        """Weak-layer shear strain."""
        return (
            -(
                (
                    np.pi
                    * np.sin((np.pi * h0) / self.es.weak_layer.h)
                    * (self.theta_vc(Z) + (2 * b0 * self.theta_vl(Z)) / self.es.slab.b)
                )
                / self.es.weak_layer.h
            )
            + (2 * np.cos((np.pi * h0) / self.es.weak_layer.h) * self.theta_wl(Z))
            / self.es.slab.b
            + (1 - (self.es.weak_layer.h / 2 + h0) / self.es.weak_layer.h)
            * self.psix(Z)
            - (Z[2, :] - (self.es.slab.H * self.psix(Z)) / 2) / self.es.weak_layer.h
        )

    def gamma_xz(
        self, Z: np.ndarray, h0: float = 0, b0: float = 0
    ) -> float | np.ndarray:
        """Weak-layer shear strain."""
        return (
            -(
                (
                    np.pi
                    * np.sin((np.pi * h0) / self.es.weak_layer.h)
                    * (self.theta_uc(Z) + (2 * b0 * self.theta_ul(Z)) / self.es.slab.b)
                )
                / self.es.weak_layer.h
            )
            - (Z[0, :] + (self.es.slab.H * self.psiy(Z)) / 2 - b0 * self.psiz(Z))
            / self.es.weak_layer.h
            + np.cos((np.pi * h0) / self.es.weak_layer.h)
            * (self.dtheta_wc_dx(Z) + (2 * b0 * self.dtheta_wl_dx(Z)) / self.es.slab.b)
            + (1 - (self.es.weak_layer.h / 2 + h0) / self.es.weak_layer.h)
            * (Z[5, :] + b0 * self.dpsix_dx(Z))
        )

    def Gi(
        self,  # pylint: disable=unused-argument
        Z_tip: np.ndarray,
        unit: EnergyUnit = "kJ/m^2",
    ) -> float | np.ndarray:
        """Mode I differential energy release rate at crack tip."""
        return (
            self._unit_factor(unit)
            * 1
            / 2
            * self.es.weak_layer.kn
            * (self.w(Z_tip) ** 2 + self.es.slab.b**2 / 12 * self.psix(Z_tip) ** 2)
        )

    def Gii(
        self,  # pylint: disable=unused-argument
        Z_tip: np.ndarray,
        unit: EnergyUnit = "kJ/m^2",
    ) -> float | np.ndarray:
        """Mode II differential energy release rate at crack tip."""
        return (
            self._unit_factor(unit)
            * self.es.weak_layer.G
            / 2
            * self.es.weak_layer.h
            * (
                (
                    self.dw_dx(Z_tip) / 2
                    - self.u(Z_tip, h0=self.es.slab.H / 2) / self.es.weak_layer.h
                )
                ** 2
                + self.es.slab.b**2
                / 12
                * (self.psiz(Z_tip) / self.es.weak_layer.h + self.dpsix_dx(Z_tip) / 2)
                ** 2
            )
        )

    def Giii(
        self,  # pylint: disable=unused-argument
        Z_tip: np.ndarray,
        unit: EnergyUnit = "kJ/m^2",
    ) -> float | np.ndarray:
        """Mode III differential energy release rate at crack tip."""
        return (
            self._unit_factor(unit)
            * self.es.weak_layer.G
            / self.es.weak_layer.h
            / 2
            * (
                (self.v(Z_tip, h0=self.es.slab.H / 2))
                + self.psix(Z_tip) * self.es.weak_layer.h / 2
            )
            ** 2
        )

    def dz_dx(self, z: np.ndarray, phi: float, qs: float = 0) -> np.ndarray:
        """First derivative z'(x) = K*z(x) + q of the solution vector."""
        K = self.es.K
        q = self.es.get_load_vector(phi=phi, qs=qs)
        return np.dot(K, z) + q

    def dz_dxdx(self, z: np.ndarray, phi: float, qs: float) -> np.ndarray:
        """Second derivative z''(x) = K*z'(x) of the solution vector."""
        K = self.es.K
        q = self.es.get_load_vector(phi=phi, qs=qs)
        dz_dx = np.dot(K, z) + q
        return np.dot(K, dz_dx)

    def du0_dxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
        """Second derivative of the horiz. centerline displacement u0''(x)."""
        return self.dz_dx(z, phi, qs)[1, :]

    def dpsi_dxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
        """Second derivative of the cross-section rotation psi''(x)."""
        return self.dz_dx(z, phi, qs)[5, :]

    def du0_dxdxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
        """Third derivative of the horiz. centerline displacement u0'''(x)."""
        return self.dz_dxdx(z, phi, qs)[1, :]

    def dpsi_dxdxdx(self, z: np.ndarray, phi: float, qs: float) -> float | np.ndarray:
        """Third derivative of the cross-section rotation psi'''(x)."""
        return self.dz_dxdx(z, phi, qs)[5, :]
