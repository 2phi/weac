"""
This module defines the configuration for the WEAC simulation.
The configuration is used to set runtime parameters for the WEAC simulation.
In general, the configuration should only be changed by the developers and is
static for the users with the most stable configuration.

We utilize the pydantic library to define the configuration.

Pydantic syntax is for a field:
field_name: type = Field(..., gt=0, description="Description")
- typing, default value, conditions, description
"""

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Config(BaseModel):
    """
    Configuration for the WEAC simulation.

    Attributes
    ----------
    touchdown : bool
        Consider Touchdown of the Slab on the Collapse Weak Layer
    """

    touchdown: bool = Field(
        default=False, description="Whether to calculate the touchdown of the slab"
    )


if __name__ == "__main__":
    config = Config()
    print(config.model_dump_json(indent=2))
