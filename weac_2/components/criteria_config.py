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
        fn : float = 1.0
            Failure mode interaction exponent for normal stress.
        fm : float = 1.0
            Failure mode interaction exponent for normal strain.
        gn : float = 1.0
            Failure mode interaction exponent for closing energy release rate.
        gm : float = 1.0
            Failure mode interaction exponent for shearing energy release rate.
    """
    fn: float = Field(default=1, gt=0, description="Failure mode interaction exponent for normal stress")
    fm: float = Field(default=1, gt=0, description="Failure mode interaction exponent for normal strain")
    gn: float = Field(default=1, gt=0, description="Failure mode interaction exponent for closing energy release rate")
    gm: float = Field(default=1, gt=0, description="Failure mode interaction exponent for shearing energy release rate")
