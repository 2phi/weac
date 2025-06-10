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
from typing import Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Config(BaseModel):
    """
    Configuration for the WEAC simulation.
    """
    youngs_modulus_method: Literal['bergfeld', 'scapazzo', 'gerling'] = Field(default='adam_unpublished', description="Method to calculate the density of the snowpack")
    stress_failure_envelope_method: Literal['adam_unpublished', 'adam_published'] = Field(default='bergfeld', description="Method to calculate the stress failure envelope")

if __name__ == "__main__":
    config = Config()
    print(config.model_dump_json(indent=2))