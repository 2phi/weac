from pydantic import BaseModel, Field


class Segment(BaseModel):
    """
    Defines a snow-slab segment: its length, foundation support, and applied loads.

    Args:
        length: float
            Segment length in millimeters [mm].
        has_foundation: bool
            Whether the segment is supported (foundation present) or cracked/free-hanging (no foundation).
        m: float
            Skier weight at the segment's right edge in kg.
    """

    length: float = Field(default=5e3, ge=0, description="Segment length in mm")
    has_foundation: bool = Field(
        default=True,
        description="Whether the segment is supported (foundation present) or cracked/free-hanging (no foundation)",
    )
    m: float = Field(
        default=0, ge=0, description="Skier weight at the segment's right edge in kg"
    )
