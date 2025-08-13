from typing import Literal

from pydantic import BaseModel, Field


class ScenarioConfig(BaseModel):
    """
    Configuration for the overall scenario, such as slope angle.

    Attributes
    ----------
    phi: float, optional
        Slope angle in degrees.
    system_type : Literal['skier', 'skiers', 'pst-', '-pst', 'rot', 'trans', 'vpst-', '-vpst'], optional
        Type of system, '-pst', 'pst-', ....
    cut_length : float
        Cut Length from PST [mm]
    stiffness_ratio : float, optional
        Stiffness ratio between collapsed and uncollapsed weak layer
    surface_load : float, optional
        Surface load on slab [N/mm]
    """

    system_type: Literal[
        "skier", "skiers", "pst-", "-pst", "rot", "trans", "vpst-", "-vpst"
    ] = Field(
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
        description="Surface line-load on slab [N/mm], e.g. evenly spaced weights, Adam et al. (2024)",
    )
