"""
Mechanical properties of snow-pack layers.

* `Layer` - a regular slab layer (no foundation springs)
* `WeakLayer` - a slab layer that also acts as a Winkler-type foundation
"""

import logging

from pydantic import BaseModel, ConfigDict, Field

from weac_2.constants import CB0, CB1, CG0, CG1, NU, RHO0

logger = logging.getLogger(__name__)


def bergfeld(rho: float, C_0: float = CB0, C_1: float = CB1) -> float:
    """Young's modulus from Bergfeld et al. (2023) - returns MPa.

    Arguments
    ---------
    rho : float or ndarray
        Density (kg/m^3).
    C0 : float, optional
        Multiplicative constant of Young modulus parametrization
        according to Bergfeld et al. (2023). Default is 6.5.
    C1 : float, optional
        Exponent of Young modulus parameterization according to
        Bergfeld et al. (2023). Default is 4.4.
    """
    return C_0 * 1e3 * (rho / RHO0) ** C_1


def scapozza(rho: float) -> float:
    """Young's modulus from Scapazzo - return MPa
    `rho` in [kg/m^3]"""
    rho = rho * 1e-12  # Convert to [t/mm^3]
    rho_0 = RHO0 * 1e-12  # Desity of ice in [t/mm^3]
    return 5.07e3 * (rho / rho_0) ** 5.13


def gerling(rho: float, C_0: float = CG0, C_1: float = CG1) -> float:
    """Young's modulus according to Gerling et al. 2017.

    Arguments
    ---------
    rho : float or ndarray
        Density (kg/m^3).
    C0 : float, optional
        Multiplicative constant of Young modulus parametrization
        according to Gerling et al. (2017). Default is 6.0.
    C1 : float, optional
        Exponent of Young modulus parameterization according to
        Gerling et al. (2017). Default is 4.6.
    """
    return C_0 * 1e-10 * rho**C_1


class Layer(BaseModel):
    """
    Regular slab layer (no foundation springs).

    Attributes
    ----------
    rho : float
        Density of the layer [kg m⁻³].
    h : float
        Height/Thickness of the layer [mm].
    nu : float
        Poisson's ratio [-] Defaults to `weac_2.constants.NU`).
    E : float, optional
        Young's modulus E [MPa].  If omitted it is derived from ``rho``.
    G : float, optional
        Shear modulus G [MPa].  If omitted it is derived from ``E`` and ``nu``.
    """

    # has to be provided
    rho: float = Field(..., gt=0, description="Density of the Slab  [kg m⁻³]")
    h: float = Field(..., gt=0, description="Height/Thickness of the slab  [mm]")
    nu: float = Field(default=NU, ge=0, lt=0.5, description="Poisson's ratio [-]")

    # derived if not provided
    E: float | None = Field(default=None, gt=0, description="Young's modulus [MPa]")
    G: float | None = Field(default=None, gt=0, description="Shear modulus [MPa]")

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def model_post_init(self, _ctx):
        object.__setattr__(self, "E", self.E or bergfeld(self.rho))
        object.__setattr__(self, "G", self.G or self.E / (2 * (1 + self.nu)))


class WeakLayer(BaseModel):
    """
    Weak layer that also behaves as a Winkler foundation.

    Attributes
    ----------
    rho : float
        Density of the layer [kg m⁻³].
    h : float
        Height/Thickness of the layer [mm].
    nu : float
        Poisson's ratio [-] Defaults to `weac_2.constants.NU`).
    E : float, optional
        Young's modulus E [MPa].  If omitted it is derived from ``rho``.
    G : float, optional
        Shear modulus G [MPa].  If omitted it is derived from ``E`` and ``nu``.
    kn : float, optional
        Normal (compression) spring stiffness kₙ [N mm⁻³].  If omitted it is
        computed as ``E_plane / t`` where
        ``E_plane = E / (1 - nu²)``.
    kt : float, optional
        Shear spring stiffness kₜ [N mm⁻³].  If omitted it is ``G / t``.
    G_c : float
        Total fracture energy Gc [J/m^2].  Default 1.0 J/m^2.
    G_Ic : float
        Mode-I fracture toughness GIc [J/m^2].  Default 0.56 J/m^2.
    G_IIc : float
        Mode-II fracture toughness GIIc [J/m^2].  Default 0.79 J/m^2.
    """

    rho: float = Field(..., gt=0, description="Density of the Slab  [kg m⁻³]")
    h: float = Field(..., gt=0, description="Height/Thickness of the slab  [mm]")
    nu: float = Field(default=NU, ge=0, lt=0.5, description="Poisson's ratio [-]")
    E: float = Field(default=None, gt=0, description="Young's modulus [MPa]")
    G: float = Field(default=None, gt=0, description="Shear modulus [MPa]")
    # Winkler springs (can be overridden by caller)
    kn: float = Field(default=None, description="Normal stiffness  [N mm⁻³]")
    kt: float = Field(default=None, description="Shear  stiffness  [N mm⁻³]")
    # fracture-mechanics parameters
    G_c: float = Field(
        default=1.0, gt=0, description="Total fracture energy Gc [J/m^2]"
    )
    G_Ic: float = Field(
        default=0.56, gt=0, description="Mode-I fracture toughness GIc [J/m^2]"
    )
    G_IIc: float = Field(
        default=0.79, gt=0, description="Mode-II fracture toughness GIIc [J/m^2]"
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def model_post_init(self, _ctx):
        object.__setattr__(self, "E", self.E or bergfeld(self.rho))
        object.__setattr__(self, "G", self.G or self.E / (2 * (1 + self.nu)))
        E_plane = self.E / (1 - self.nu**2)  # plane-strain Young
        object.__setattr__(self, "kn", self.kn or E_plane / self.h)
        object.__setattr__(self, "kt", self.kt or self.G / self.h)


if __name__ == "__main__":
    ly1 = Layer(rho=180, h=120)  # E,G,k auto-computed
    ly2 = Layer(rho=250, h=80, E=50.0)  # override E, derive G
    wl = WeakLayer(rho=170, h=30)  # full set incl. kn, kt

    print(wl.model_dump())
