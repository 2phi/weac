from typing import Literal

from pydantic import BaseModel, Field


class ScenarioConfig(BaseModel):
    """
    Configuration for the overall scenario, such as slope angle.

    Attributes
    ----------
    phi: float, optional
        Slope angle in degrees.
    system : Literal['skier', 'skiers', 'pst-', '-pst', 'rot', 'trans', 'vpst-', '-vpst'], optional
        Type of system, '-pst', '+pst', ....
    cut_length : float
        Cut Length from PST [mm]
    stiffness_factor : float, optional
        Stiffness ratio between collapsed and uncollapsed weak layer
    surface_load : float, optional
        Surface load on slab [N/mm]
    """

    phi: float = Field(
        default=0.0,
        ge=-50.0,
        le=50.0,
        description="Slope angle in degrees, counterclockwise positive",
    )
    system_type: Literal[
        "skier", "skiers", "pst-", "-pst", "rot", "trans", "vpst-", "-vpst"
    ] = Field(default="skiers", description="Type of system, '-pst', '+pst', ....")
    cut_length: float = Field(default=0.0, ge=0, description="Initial cut length [mm]")
    stiffness_ratio: float = Field(
        default=1000.0,
        gt=0.0,
        description="Stiffness ratio between collapsed and uncollapsed weak layer",
    )
    surface_load: float = Field(
        default=0.0,
        ge=0.0,
        description="Surface load on slab [N/mm], e.g. evenly spaced weights, Adam et al. (2024)",
    )
