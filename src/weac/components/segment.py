"""
This module defines the Segment class, which represents a segment of the snowpack.
"""

from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, field_validator, PlainSerializer, WithJsonSchema
from pydantic import ConfigDict
from numpy.typing import NDArray


def _serialize_ndarray(arr: np.ndarray) -> list:
    """Serialize numpy array to nested list for JSON compatibility."""
    return arr.tolist()


NumpyArray = Annotated[
    np.ndarray,
    PlainSerializer(_serialize_ndarray, return_type=list),
    WithJsonSchema({"type": "array", "items": {"type": "array", "items": {"type": "number"}}}),
]

class Segment(BaseModel):
    """
    Defines a snow-slab segment: its length, foundation support, and applied loads.

    Attributes
    ----------
    length: float
        Segment length in millimeters [mm].
    has_foundation: bool
        Whether the segment is supported (foundation present) or cracked/free-hanging
        (no foundation).
    is_loaded: bool
        Whether additional loading is applied at the segment's top side.
    m: float
        Skier mass at the segment's right edge [kg].
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    length: float = Field(default=5e3, ge=0, description="Segment length in [mm]")
    has_foundation: bool = Field(
        default=True,
        description="Whether the segment is supported (foundation present) or "
        "cracked/free-hanging (no foundation)",
    )
    is_loaded: bool = Field(
        default=True, description="Whether additional loading is applied at the segment's top side"
    )
    m: float = Field(
        default=0, ge=0, description="Skier mass at the segment's right edge in [kg]"
    )

    f: NumpyArray = Field(
        default_factory=lambda: np.zeros((6,1), dtype=float),
        description=(
            "Load vector acting on the right side of the segment. "
            "Includes the six section forces [Nx, Vy, Vz, Mx, My, Mz]."
        ),
    )

    @field_validator("f")
    def ensure_ndarray(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        # allows passing lists/tuples etc. and enforces float dtype
        return np.asarray(v, dtype=float)
