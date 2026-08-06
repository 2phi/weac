"""
Mechanical properties of the weak layer (Winkler foundation).
"""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from weac.components.layer import (
    _bergfeld_youngs_modulus,
    _gerling_youngs_modulus,
    _scapozza_youngs_modulus,
)
from weac.constants import CS0, CS1, G_MM_S2, NU, RHO_ICE
from weac.utils.snow_types import GrainType, HandHardness


def _schottner_fc_dh_youngs_modulus(
    rho: float, C_0: float = CS0, C_1: float = CS1
) -> float:
    """Young's modulus from Schöttner et al. FC&DH law — returns MPa.

    E = C_S0 * (rho / rho_ice) ** C_S1 for faceted crystals & depth hoar.

    Arguments
    ---------
    rho : float
        Density (kg/m^3).
    C_0 : float, optional
        Prefactor of Young modulus parametrization (default CS0 = 2.72e4 MPa).
    C_1 : float, optional
        Exponent of Young modulus parameterization (default CS1 = 5.4).
    """
    return C_0 * (rho / RHO_ICE) ** C_1


def _collapse_height(h: float) -> float:
    """
    Based on data from Herwijnen (van Herwijnen, 2016)
    `Estimating the effective elastic modulus and specific fracture energy of
    snowpack layers from field experiments`
    Data collection 2005 - 2016.

    Arguments:
    ----------
    h : float
        Height/Thickness of the layer [mm].
    """
    return 4.70 * (1 - np.exp(-h / 7.78))


class WeakLayer(BaseModel):
    """
    Weak layer that also behaves as a Winkler foundation.

    Attributes
    ----------
    rho : float
        Density of the layer [kg m⁻³].
    h : float
        Height/Thickness of the layer [mm].
    f : float
        Resultant force of the layer [N/mm]
    nu : float
        Poisson's ratio [-] Defaults to `weac.constants.NU`).
    E : float, optional
        Young's modulus E [MPa].  If omitted it is derived from ``rho``.
    G : float, optional
        Shear modulus G [MPa].  If omitted it is derived from ``E`` and ``nu``.
    kn : float, optional
        Normal (compression) spring stiffness kₙ [N mm⁻³].  If omitted is
        computed as ``E_plane / h`` where
        ``E_plane = E / (1 - nu²)``.
    kt : float, optional
        Shear spring stiffness kₜ [N mm⁻³].  If omitted it is ``G / h``.
    G_Ic : float
        Mode-I fracture toughness GIc [J/m^2].  Default 0.56 J/m^2.
    G_IIc : float
        Mode-II fracture toughness GIIc [J/m^2].  Default 0.79 J/m^2.
    """

    rho: float = Field(
        default=150, gt=0, description="Density of the Weak Layer  [kg m⁻³]"
    )
    h: float = Field(
        default=20, gt=0, description="Height/Thickness of the weak layer  [mm]"
    )
    f: float | None = Field(
        default=None, description="Weight density of the weak layer [N/mm^3]"
    )
    collapse_height: float = Field(
        default=0.0, ge=0, description="Collapse height [mm]"
    )
    nu: float = Field(default=NU, ge=0, lt=0.5, description="Poisson's ratio [-]")

    E: float = Field(default=0.0, ge=0, description="Young's modulus [MPa]")
    G: float = Field(default=0.0, ge=0, description="Shear modulus [MPa]")
    # Winkler springs (can be overridden by caller)
    kn: float = Field(default=0.0, description="Normal stiffness  [N mm⁻³]")
    kt: float = Field(default=0.0, description="Shear  stiffness  [N mm⁻³]")
    # fracture-mechanics parameters
    G_Ic: float = Field(
        default=0.56, gt=0, description="Mode-I fracture toughness GIc [J/m^2]"
    )
    G_IIc: float = Field(
        default=0.79, gt=0, description="Mode-II fracture toughness GIIc [J/m^2]"
    )
    sigma_c: float = Field(default=6.16, gt=0, description="Tensile strength [kPa]")
    tau_c: float = Field(default=5.09, gt=0, description="Shear strength [kPa]")
    sigma_comp: float = Field(
        default=2.6, gt=0, description="Compressive strength [kPa]"
    )
    E_method: Literal["schottner_fc_dh", "bergfeld", "scapazzo", "gerling"] = Field(
        default="schottner_fc_dh",
        description="Method to calculate the Young's modulus",
    )
    constitutive_model: Literal["PlaneStrain", "PlaneStress", "Uniaxial"] = Field(
        default="PlaneStrain",
        description="Marks how interlinked the weak layer is in out-of-plane direction.",
    )
    grain_type: GrainType | None = Field(default=None, description="Grain type")
    grain_size: float | None = Field(default=None, description="Grain size [mm]")
    hand_hardness: HandHardness | None = Field(
        default=None, description="Hand hardness"
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def model_post_init(self, _ctx):  # pylint: disable=arguments-differ
        if self.E_method == "schottner_fc_dh":
            object.__setattr__(
                self, "E", self.E or _schottner_fc_dh_youngs_modulus(self.rho)
            )
        elif self.E_method == "bergfeld":
            object.__setattr__(self, "E", self.E or _bergfeld_youngs_modulus(self.rho))
        elif self.E_method == "scapazzo":
            object.__setattr__(self, "E", self.E or _scapozza_youngs_modulus(self.rho))
        elif self.E_method == "gerling":
            object.__setattr__(self, "E", self.E or _gerling_youngs_modulus(self.rho))
        else:
            raise ValueError(f"Invalid E_method: {self.E_method}")
        object.__setattr__(
            self, "collapse_height", self.collapse_height or _collapse_height(self.h)
        )

        # Validate that collapse height is smaller than layer height
        if self.collapse_height >= self.h:
            raise ValueError(
                f"Collapse height ({self.collapse_height:.2f} mm) must be smaller than "
                f"layer height ({self.h:.2f} mm). Consider reducing collapse_height or "
                f"increasing layer thickness."
            )

        if self.constitutive_model == "PlaneStrain":
            nu_eff = self.nu
            E_eff = self.E
        elif self.constitutive_model == "PlaneStress":
            nu_eff = self.nu / (1 + self.nu)
            E_eff = self.E * (1 + 2 * self.nu) / ((1 + self.nu) ** 2)
        elif self.constitutive_model == "Uniaxial":
            nu_eff = 0
            E_eff = self.E
        object.__setattr__(self, "nu", nu_eff)
        object.__setattr__(self, "E", E_eff)
        object.__setattr__(self, "G", self.G or self.E / (2 * (1 + self.nu)))
        E_plane = self.E / (1 - self.nu**2)  # plane-strain Young
        object.__setattr__(self, "kn", self.kn or E_plane / self.h)
        object.__setattr__(self, "kt", self.kt or self.G / self.h)
        object.__setattr__(
            self, "f", self.f if self.f is not None else self.rho * 1e-12 * G_MM_S2
        )

    @model_validator(mode="after")
    def validate_positive_E_G(self):
        """Validate that E and G are positive."""
        if self.E <= 0:
            raise ValueError("E must be positive")
        if self.G <= 0:
            raise ValueError("G must be positive")
        return self


if __name__ == "__main__":
    wl = WeakLayer(rho=170, h=30)  # full set incl. kn, kt
    print(wl.model_dump())
