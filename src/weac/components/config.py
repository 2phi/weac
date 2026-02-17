"""
Configuration for the WEAC simulation.
These settings control runtime parameters for WEAC.
In general, developers maintain these defaults; end users should see a stable configuration.

We utilize the pydantic library to define the configuration.

Pydantic syntax is for a field:
field_name: type = Field(..., gt=0, description="Description")
- typing, default value, constraints, description
"""

from typing import Literal

from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Configuration for the WEAC simulation.

    Attributes
    ----------
    touchdown : bool
        Whether slab touchdown on the collapsed weak layer is considered.
    backend : Literal["classic", "generalized"]
        Selects which eigensystem implementation to use under the hood.
    """

    touchdown: bool = Field(
        default=False, description="Whether to include slab touchdown in the analysis"
    )
    backend: Literal["classic", "generalized"] = Field(
        default="classic",
        description=(
            "Eigensystem backend: 'classic' uses the current WEAC solver; "
            "'generalized' routes to an advanced adapter (initially identical behavior)."
        ),
    )
    b: float = Field(
        default = 300, ge=1, description= f"Out-of-plane width of the model in [mm]"
    )


if __name__ == "__main__":
    config = Config()
    print(config.model_dump_json(indent=2))
