"""
This module defines the ScenarioConfig class, which contains the configuration for a given scenario.
"""

from typing import Literal, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np

SystemType = Literal[
    "skier", "skiers", "pst-", "-pst", "rot", "trans", "vpst-", "-vpst"
]

TouchdownMode = Literal["A_free_hanging", "B_point_contact", "C_in_contact"]


class ScenarioConfig(BaseModel):
    """
    Configuration for the overall scenario, such as slope angle.

    Attributes
    ----------
    phi : float, optional
        Slope angle in degrees (counterclockwise positive).
    theta float, optional
        Rotation of the slab around its axis (counterclockwise positive)
    system_type : SystemType
        Type of system. Allowed values are:
        - skier: single skier in-between two segments
        - skiers: multiple skiers spread over the slope
        - pst-: positive PST: down-slope + slab-normal cuts
        - -pst: negative PST: up-slope + slab-normal cuts
        - rot: rotation: rotation of the slab
        - trans: translation: translation of the slab
        - vpst-: positive VPST: down-slope + vertical cuts
        - -vpst: negative VPST: up-slope + vertical cuts
    cut_length : float, optional
        Cut length for PST/VPST [mm].
    stiffness_ratio : float, optional
        Stiffness ratio between collapsed and uncollapsed weak layer.
    surface_load : float, optional
        Surface line-load on slab [N/mm] (force per mm of out-of-plane width)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_type: SystemType = Field(
        default="skiers",
        description="Type of system, '-pst', 'pst-', ....; \n"
        "skier: single skier in-between two segments, \n"
        "skiers: multiple skiers spread over the slope, \n"
        "pst-: positive PST: down-slope + slab-normal cuts, \n"
        "-pst: negative PST: up-slope + slab-normal cuts, \n"
        "rot: rotation: rotation of the slab, \n"
        "trans: translation: translation of the slab, \n"
        "vpst-: positive VPST: down-slope + vertical cuts, \n"
        "-vpst: negative VPST: up-slope + vertical cuts, \n",
    )
    phi: float = Field(
        default=0.0,
        ge=-90.0,
        le=90.0,
        description="Slope angle in degrees (counterclockwise positive)",
    )
    theta: float = Field(
        default=0.0,
        ge=-90.0,
        le=90.0,
        description="Rotation angle in degrees (counterclockwise positive)",
    )
    cut_length: float = Field(
        default=0.0, ge=0, description="Cut length of performed PST or VPST [mm]"
    )
    stiffness_ratio: float = Field(
        default=1000.0,
        gt=0.0,
        description="Stiffness ratio between collapsed and uncollapsed weak layer",
    )
    surface_load: float = Field(
        default=0.0,
        ge=0.0,
        description="Surface line-load on slab [N/mm], e.g. evenly spaced weights, "
        "Adam et al. (2024)",
    )
    load_vector_left: np.ndarray = Field( 
        default_factory = lambda: np.zeros((6,1)),
        description="Load vector on the left side of the configuration to model external loading in mode III experiments.")


    load_vector_right: np.ndarray = Field( 
        default_factory = lambda: np.zeros((6,1)),
        description="Load vector on the right side of the configuration to model external loading in mode III experiments.")


    # @field_validator("load_vector_left", "load_vector_right", mode="after")
    # def check_load_vector_shape(cls, value: Any) -> Any:
    #     # Convert to numpy array if needed
    #     arr = np.asarray(value, dtype=np.float64)
    #     # Ensure correct shape (6, 1)
    #     if arr.shape != (6, 1):
    #         # Try to reshape if possible
    #         arr = arr.reshape(-1, 1)
    #         if arr.shape[0] != 6:
    #             raise ValueError(f"load vectors must have shape (6, 1), got {arr.shape}")
    #     return arr 