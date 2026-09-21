"""Convert field-profile depths to WEAC slope-normal thicknesses."""

import numpy as np


def plumb_to_slope_normal(phi_deg: float) -> float:
    """Scale vertical (plumb) depth/thickness to distance along slope normal.

    Field profiles typically report depths from the surface along the vertical.
    WEAC slab layer thicknesses are measured normal to the slope. The plumb-line
    depth ``d_v`` is converted to slope-normal depth ``d_n`` by
    ``d_n = d_v * cos(phi)``, where ``phi`` is the slope angle from horizontal.
    """
    phi = np.deg2rad(float(phi_deg))
    c = float(np.cos(phi))
    if c <= 1e-6:
        raise ValueError(
            f"Slope angle too close to ±90° ({phi_deg}°); cannot convert vertical "
            "depths to slope-normal."
        )
    return c
