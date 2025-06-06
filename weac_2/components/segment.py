from pydantic import BaseModel, Field

class Segment(BaseModel):
    """
    Defines a segment of the snow slab, its length, foundation support, and applied loads.

    Args:
        length (float): Segment length in mm.
        fractured (bool): Boolean indicating whether the segment is fractured or not.
        skier_weight (float): Skier weight at segments right edge in kg. Defaults to 0.
        surface_load (float): Surface load in kPa. Defaults to 0.
    """
    l: float = Field(..., gt=0, description="Segment length in mm")
    k: bool = Field(..., description="Boolean indicating whether the segment is fractured or not")
    m: float = Field(0, ge=0, description="Skier weight at segment right edge in kg")
