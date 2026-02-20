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

from pydantic import BaseModel, Field, model_validator

from weac.components.scenario_config import TouchdownMode


class Config(BaseModel):
    """
    Configuration for the WEAC simulation.

    Attributes
    ----------
    touchdown : bool
        Whether slab touchdown on the collapsed weak layer is considered.
    backend : Literal["classic", "generalized"]
        Selects which eigensystem implementation to use under the hood.
    forced_touchdown_mode : TouchdownMode | None
        If set, forces the touchdown mode instead of calculating it from l_AB/l_BC.
        This avoids floating-point precision issues when the mode boundary values
        are recalculated with different scenario parameters.
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

    forced_touchdown_mode: TouchdownMode | None = Field(
        default=None,
        description="Force a specific touchdown mode instead of auto-calculating",
    )

    @model_validator(mode="after")
    def validate_touchdown_backend_compatibility(self):
        if self.touchdown and self.backend == "generalized":
            raise ValueError(
                "Slab touchdown is only available for the classic backend. "
                "Set backend='classic' or disable touchdown."
            )
        return self


if __name__ == "__main__":
    config = Config()
    print(config.model_dump_json(indent=2))
