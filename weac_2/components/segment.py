from pydantic import BaseModel, Field

class Segment(BaseModel):
    """
    Defines a segment of the snow slab, its length, foundation support, and applied loads.

    Args:
        length : float
            Segment length [mm]
        fractured: bool
            Indicating whether the segment is supported or free hanging.
        skier_weight : float
            Skier weight at segments right edge in kg
    """
    l: float = Field(..., gt=0, description="Segment length in mm")
    k: bool = Field(..., description="Boolean indicating whether the segment is fractured or not")
    m: float = Field(default=0, ge=0, description="Skier weight at segment right edge in kg")
