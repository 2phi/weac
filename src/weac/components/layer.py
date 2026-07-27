"""
Mechanical properties of snow-pack layers.

* `Layer` - a regular slab layer (no foundation springs)
* `WeakLayer` - re-exported from `weac.components.weak_layer` for compatibility
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from weac.constants import CB0, CB1, CG0, CG1, NU, RHO_ICE
from weac.utils.snow_types import GrainType, HandHardness


def _bergfeld_youngs_modulus(rho: float, C_0: float = CB0, C_1: float = CB1) -> float:
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
    return C_0 * 1e3 * (rho / RHO_ICE) ** C_1


def _scapozza_youngs_modulus(rho: float) -> float:
    """Young's modulus from Scapozzo et al. (2019) - return MPa
    `rho` in [kg/m^3]"""
    rho = rho * 1e-12  # Convert to [t/mm^3]
    rho_0 = RHO_ICE * 1e-12  # Density of ice in [t/mm^3]
    return 5.07e3 * (rho / rho_0) ** 5.13


def _gerling_youngs_modulus(rho: float, C_0: float = CG0, C_1: float = CG1) -> float:
    """Young's modulus according to Gerling et al. (2017).

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


def _sigrist_tensile_strength(rho, unit: Literal["kPa", "MPa"] = "kPa"):
    """
    Estimate the tensile strength of a slab layer from its density.

    Uses the density parametrization of Sigrist (2006).

    Arguments
    ---------
    rho : ndarray, float
        Layer density (kg/m^3).
    unit : str, optional
        Desired output unit of the layer strength. Default is 'kPa'.

    Returns
    -------
    ndarray
        Tensile strength in specified unit.
    """
    convert = {"kPa": 1, "MPa": 1e-3}
    # Sigrist's equation is given in kPa
    return convert[unit] * 240 * (rho / RHO_ICE) ** 2.44


def _adam_tensile_strength(rho, unit: Literal["kPa", "MPa"] = "kPa"):
    """
    Estimate the tensile strength of a slab layer from its density.

    Uses the density parametrization of Adam (2025).

    Arguments
    ---------
    rho : ndarray, float
        Layer density (kg/m^3).
    unit : str, optional
        Desired output unit of the layer strength. Default is 'kPa'.

    Returns
    -------
    ndarray
        Tensile strength in specified unit.
    """
    convert = {"kPa": 1e3, "MPa": 1}
    TS_0 = 1.0  # [MPa]
    kappa = 3.45  # [-]
    # Adam's equation is given in MPa
    return TS_0 * (rho / RHO_ICE) ** kappa * convert[unit]


# # TODO: Compressive Strength from Schöttner
# def _schotter_compressive_strength(rho, unit: Literal["kPa", "MPa"] = "kPa"):
#     """
#     Estimate the compressive strength of a slab layer from its density.
#     On the compressive strength of weak snow layers of depth hoar - Schöttner (2025).

#     Uses the density parametrization of Schöttner (2025).
#     """
#     convert = {"kPa": 1e3, "MPa": 1}
#     CS_0 = 11.0  # [MPa]
#     CS_1 = 5.4  # [-]
#     return CS_0 * (rho / RHO_ICE) ** CS_1 * convert[unit]


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
        Poisson's ratio [-] Defaults to `weac.constants.NU`).
    E : float, optional
        Young's modulus E [MPa].  If omitted it is derived from ``rho``.
    G : float, optional
        Shear modulus G [MPa].  If omitted it is derived from ``E`` and ``nu``.
    tensile_strength: float
        Tensile strength [kPa].
    tensile_strength_method: Literal["sigrist", "adam", "hybrid"]
        Method to calculate the tensile strength.
    """

    # has to be provided
    rho: float = Field(default=150, gt=0, description="Density of the Slab  [kg m⁻³]")
    h: float = Field(
        default=200, gt=0, description="Height/Thickness of the slab  [mm]"
    )

    # derived if not provided
    nu: float = Field(default=NU, ge=0, lt=0.5, description="Poisson's ratio [-]")
    E: float = Field(default=0.0, ge=0, description="Young's modulus [MPa]")
    G: float = Field(default=0.0, ge=0, description="Shear modulus [MPa]")
    tensile_strength: float = Field(
        default=0.0, ge=0, description="Tensile strength [kPa]"
    )
    tensile_strength_method: Literal["sigrist", "adam", "hybrid"] = Field(
        default="hybrid",
        description="Method to calculate the tensile strength",
    )
    E_method: Literal["bergfeld", "scapazzo", "gerling"] = Field(
        default="bergfeld",
        description="Method to calculate the Young's modulus",
    )
    grain_type: GrainType | None = Field(default=None, description="Grain type")
    grain_size: float | None = Field(default=None, description="Grain size [mm]")
    hand_hardness: HandHardness | None = Field(
        default=None, description="Hand hardness"
    )

    def model_post_init(self, _ctx):  # pylint: disable=arguments-differ
        if self.E_method == "bergfeld":
            object.__setattr__(self, "E", self.E or _bergfeld_youngs_modulus(self.rho))
        elif self.E_method == "scapazzo":
            object.__setattr__(self, "E", self.E or _scapozza_youngs_modulus(self.rho))
        elif self.E_method == "gerling":
            object.__setattr__(self, "E", self.E or _gerling_youngs_modulus(self.rho))
        else:
            raise ValueError(f"Invalid E_method: {self.E_method}")
        object.__setattr__(self, "G", self.G or self.E / (2 * (1 + self.nu)))

        if not self.tensile_strength:
            if self.tensile_strength_method == "sigrist":
                ts_value = _sigrist_tensile_strength(self.rho, unit="kPa")
            elif self.tensile_strength_method == "adam":
                ts_value = _adam_tensile_strength(self.rho, unit="kPa")
            elif self.tensile_strength_method == "hybrid":
                # Use Sigrist for rho < 250, Adam for rho >= 250
                if self.rho < 250:
                    ts_value = _sigrist_tensile_strength(self.rho, unit="kPa")
                else:
                    ts_value = _adam_tensile_strength(self.rho, unit="kPa")
            else:
                raise ValueError(
                    f"Invalid tensile_strength_method: {self.tensile_strength_method}"
                )
            object.__setattr__(self, "tensile_strength", ts_value)

    @model_validator(mode="after")
    def validate_positive_E_G(self):
        """Validate that E and G are positive."""
        if self.E <= 0:
            raise ValueError("E must be positive")
        if self.G <= 0:
            raise ValueError("G must be positive")
        return self


def __getattr__(name: str):
    """Lazy re-export to avoid circular imports with weak_layer."""
    if name == "WeakLayer":
        from weac.components.weak_layer import WeakLayer

        return WeakLayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Layer",
    "WeakLayer",
    "_adam_tensile_strength",
    "_bergfeld_youngs_modulus",
    "_gerling_youngs_modulus",
    "_scapozza_youngs_modulus",
    "_sigrist_tensile_strength",
]


if __name__ == "__main__":
    ly1 = Layer(rho=180, h=120)  # E,G,k auto-computed
    ly2 = Layer(rho=250, h=80, E=50.0)  # override E, derive G

    print(ly1.model_dump())
    print(ly2.model_dump())
