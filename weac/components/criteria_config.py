"""
Module for configuring failure-mode interaction criteria and stress failure envelope selection.

Main fields:
- fn, fm: interaction exponents for normal (sigma) and shear (tau) stresses (> 0).
- gn, gm: interaction exponents for mode-I (G_I) and mode-II (G_II) energy release rates (> 0).
- stress_envelope_method: one of {"adam_unpublished", "schottner", "mede_s-RG1", "mede_s-RG2", "mede_s-FCDH"}.
- scaling_factor, order_of_magnitude: positive scalars applied to the stress envelope.

Typical usage:
    from weac.components.criteria_config import CriteriaConfig

    config = CriteriaConfig(
        stress_envelope_method="schottner",
        scaling_factor=1.0,
        order_of_magnitude=1.0,
    )

See also:
- weac.analysis.criteria_evaluator for how these parameters influence failure checks.
"""

from typing import Literal
from pydantic import BaseModel, Field


class CriteriaConfig(BaseModel):
    """
    Parameters defining the interaction between different failure modes.

    Args:
    -----
        fn : float
            Failure mode interaction exponent for normal stress (sigma). Default is 2.0.
        fm : float
            Failure mode interaction exponent for shear stress (tau). Default is 2.0.
        gn : float
            Failure mode interaction exponent for closing energy release rate (G_I). Default is 5.0.
        gm : float
            Failure mode interaction exponent for shearing energy release rate (G_II). Default is 2.22.
    """

    fn: float = Field(
        default=2.0,
        gt=0,
        description="Failure mode interaction exponent for normal stress (sigma)",
    )
    fm: float = Field(
        default=2.0,
        gt=0,
        description="Failure mode interaction exponent for shear stress (tau)",
    )
    gn: float = Field(
        default=1 / 0.2,
        gt=0,
        description="Failure mode interaction exponent for closing energy release rate (G_I)",
    )
    gm: float = Field(
        default=1 / 0.45,
        gt=0,
        description="Failure mode interaction exponent for shearing energy release rate (G_II)",
    )
    stress_envelope_method: Literal[
        "adam_unpublished", "schottner", "mede_s-RG1", "mede_s-RG2", "mede_s-FCDH"
    ] = Field(
        default="adam_unpublished",
        description="Method to calculate the stress failure envelope",
    )
    scaling_factor: float = Field(
        default=1,
        gt=0,
        description="Scaling factor for stress envelope",
    )
    order_of_magnitude: float = Field(
        default=1,
        gt=0,
        description="Order of magnitude for stress envelope",
    )
