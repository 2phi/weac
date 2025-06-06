from typing import Literal
from pydantic import BaseModel, Field

class ScenarioConfig(BaseModel):
    """
    Configuration for the overall scenario, such as slope angle.

    Attributes
    ----------
    phi (float): 
        Slope angle in degrees.
    touchdown:
    
    system:
    
    crack_length:
    
    collapse_factor:
    
    stiffness_factor:
    
    surface_load:
    
    """
    phi: float = Field(0, description="Slope angle in degrees, counterclockwise positive")
    touchdown: bool = Field(False, description="Whether to calculate the touchdown")
    # TODO: add more descriptive/human-readable system names
    system: Literal['skier', 'skiers', 'pst-', 'pst+', 'rot', 'trans'] = Field('skiers', description="Type of system, '-pst', '+pst', ....")
    crack_length: float | None = Field(None, ge=0, description="Initial crack length in metres")
    collapse_factor: float = Field(0.5, ge=0.0, lt=1.0, description="Fractional collapse factor (0 <= f < 1)")
    stiffness_ratio: float = Field(1000, gt=0.0, description="Stiffness ratio between collapsed and uncollapsed weak layer")
    surface_load: float = Field(0.0, ge=0.0, description="Surface load on slab [N/mm], e.g. evenly spaced weights, Adam et al. (2024)")
    
