from typing import Literal
from pydantic import BaseModel, Field

class ScenarioConfig(BaseModel):
    """
    Configuration for the overall scenario, such as slope angle.

    Attributes
    ----------
    phi: float, optional
        Slope angle in degrees.
    system : Literal['skier', 'skiers', 'pst-', 'pst+', 'rot', 'trans', 'vpst-', '-vpst'], optional
        Type of system, '-pst', '+pst', ....
    crack_length : float
        Crack Length from PST [mm]
    collapse_factor : float, optional
        Fractional collapse factor (0 <= f < 1)
    stiffness_factor : float, optional
        Stiffness ratio between collapsed and uncollapsed weak layer    
    qs : float, optional
        Surface load on slab [N/mm]
    """
    phi: float = Field(default=0, gt=-90, lt=90,description="Slope angle in degrees, counterclockwise positive")
    system_type: Literal['skier', 'skiers', 'pst-', 'pst+', 'rot', 'trans', 'vpst-', '-vpst'] = Field(default='skiers', description="Type of system, '-pst', '+pst', ....")
    crack_length: float = Field(default=0.0, ge=0, description="Initial crack length [mm]")
    collapse_factor: float = Field(default=0.5, ge=0.0, lt=1.0, description="Fractional collapse factor (0 <= f < 1)")
    stiffness_ratio: float = Field(default=1000, gt=0.0, description="Stiffness ratio between collapsed and uncollapsed weak layer")
    qs: float = Field(default=0.0, ge=0.0, description="Surface load on slab [N/mm], e.g. evenly spaced weights, Adam et al. (2024)")
    
