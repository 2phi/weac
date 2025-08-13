"""JSON serialization helpers for tests."""

from __future__ import annotations

import numpy as np


def json_default(o: object) -> object:
    """Custom JSON serializer for numpy data types."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):  # covers np.int64, np.float64, np.bool_, etc.
        return o.item()
    return str(o)
