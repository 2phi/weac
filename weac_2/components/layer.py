"""
Mechanical properties of snow-pack layers.

* `Layer`    – a regular slab layer (no foundation springs)
* `WeakLayer` – a slab layer that also acts as a Winkler-type foundation
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict
from weac_2.constants import CB0, CB1, CG0, CG1, K_SHEAR, NU, RHO0

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
    rho = rho * 1e-12               # Convert to [t/mm^3]
    rho_0 = RHO0 * 1e-12            # Desity of ice in [t/mm^3]
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

class _BaseLayer(BaseModel):
    """
    Common base for all snow layers.

    Attributes
    ----------
    rho : float
        Density of the layer [kg m⁻³].
    h : float
        Height/Thickness of the layer [mm].
    nu : float
        Poisson’s ratio ν [–] Defaults to ``weac_2.constants.NU``).

    E : float, optional
        Young’s modulus E [MPa].  If omitted it is derived from ``rho``.
    G : float, optional
        Shear modulus G [MPa].  If omitted it is derived from ``E`` and ``nu``.
    k : float, optional
        Mindlin shear-correction factor k [-].  Defaults to
        ``weac_2.constants.K_SHEAR``.
    """
    # has to be provided
    rho: float = Field(..., gt=0, description="Density of the Slab  [kg m⁻³]")
    h: float = Field(..., gt=0, description="Height/Thickness of the slab  [mm]")
    nu: float = Field(default=NU, ge=0, lt=0.5, description="Poisson's ratio [-]")

    # derived if not provided
    E: float | None = Field(default=None, gt=0, description="Young's modulus [MPa]")
    G: float | None = Field(default=None, gt=0, description="Shear modulus [MPa]")
    k: float | None = Field(default=None, description="Mindlin k  [-]")

    model_config = ConfigDict(frozen=True, extra='forbid',)

    def model_post_init(self, _ctx):
        object.__setattr__(self, "E", self.E or bergfeld(self.rho))
        object.__setattr__(self, "G", self.G or self.E / (2 * (1 + self.nu)))
        object.__setattr__(self, "k", self.k or K_SHEAR)


class Layer(_BaseLayer):
    """
    Regular slab layer (no foundation springs).

    Attributes
    ----------
    rho, h, nu, E, G, k
        See ``_BaseLayer`` for full descriptions.
    """
    pass

class WeakLayer(_BaseLayer):
    """
    Weak layer that also behaves as a Winkler foundation.

    Attributes
    ----------
    rho, h, nu, E, G, k
        Inherited from ``_BaseLayer``.
    kn : float, optional
        Normal (compression) spring stiffness kₙ [N mm⁻³].  If omitted it is
        computed as ``E_plane / t`` where
        ``E_plane = E / (1 - nu²)``.
    kt : float, optional
        Shear spring stiffness kₜ [N mm⁻³].  If omitted it is ``G / t``.
    G_c : float
        Total fracture energy Gc [MPa m½].  Default 1 MPa m½.
    G_Ic : float
        Mode-I fracture toughness GIc [MPa m½].  Default 1 MPa m½.
    G_IIc : float
        Mode-II fracture toughness GIIc [MPa m½].  Default 1 MPa m½.
    """
    # Winkler springs (can be overridden by caller)
    kn: float | None = Field(default=None, description="Normal stiffness  [N mm⁻³]")
    kt: float | None = Field(default=None, description="Shear  stiffness  [N mm⁻³]")

    # fracture-mechanics parameters
    G_c: float = Field(default=1.0, gt=0, description="Gc  [MPa m½]")
    G_Ic: float = Field(default=1.0, gt=0, description="GIc [MPa m½]")
    G_IIc:float = Field(default=1.0, gt=0, description="GIIc[MPa m½]")

    def model_post_init(self, _ctx):
        super().model_post_init(_ctx)      # fills E, G, k

        E_plane = self.E / (1 - self.nu**2)    # plane-strain Young
        object.__setattr__(self, "kn", self.kn or E_plane / self.h)
        object.__setattr__(self, "kt", self.kt or self.G / self.h)

if __name__ == "__main__":
    ly1 = Layer(rho=180, h=120)           # E,G,k auto-computed
    ly2 = Layer(rho=250, h= 80, E=50.0)   # override E, derive G
    wl  = WeakLayer(rho=170, h=30)        # full set incl. kn, kt

    print(wl.model_dump())