from pydantic import BaseModel, Field

class Segment(BaseModel):
    """
    Defines a segment of the snow slab, its length, foundation support, and applied loads.

    Args:
        length : float
            Segment length [mm]
        has_foundation: bool
            Indicating whether the segment is supported or free hanging.
        m : float
            Skier weight at segments right edge in kg
    """
    length: float = Field(..., ge=0, description="Segment length in mm")
    has_foundation: bool = Field(default=True, description="Boolean indicating whether the segment is fractured or not")
    m: float = Field(default=0, ge=0, description="Skier weight at segment right edge in kg")
