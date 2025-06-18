"""
TODO: blabla
"""

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
        default=5.0,
        gt=0,
        description="Failure mode interaction exponent for closing energy release rate (G_I)",
    )
    gm: float = Field(
        default=2.22,
        gt=0,
        description="Failure mode interaction exponent for shearing energy release rate (G_II)",
    )
