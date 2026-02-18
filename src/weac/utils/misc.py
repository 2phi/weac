"""
This module contains miscellaneous utility functions.
"""

from typing import Literal

import numpy as np

from numpy.typing import NDArray
from weac.components import Layer
from weac.constants import G_MM_S2, LSKI_MM


def decompose_to_xyz(
    f: NDArray[np.float64] | float,
    phi: NDArray[np.float64] | float,
    theta: NDArray[np.float64] | float = 0,
):
    """
    Resolve a gravity-type force/line-load into its x'-component (downslope), y'-component (cross-slope) and z'-component (into-slope)  with respect to an inclined surface.
    Fully vectorized; if input contains arrays, output matches shape.

    Parameters
    ----------
    f : float | NDArray[np.float64]
        is interpreted as a vertical load magnitude
        acting straight downward (global z negative).
    phi : float | NDArray[np.float64]
        Surface dip angle `in degrees`, measured between horizontal plane and the slabs axis (local x'-axis).
        Positive `phi` means the surface slopes upward in +x.
    theta: float | NDArray[np.float64]
        rotation angle `in degrees`, measured between the x'z-plane and the x'z'-plane. Positive `theta` means the slab is rotated counterclockwise around the slab axis (local x'-axis).

    Returns
    -------
    f_x', f_y', f_z': float | NDArray[np.float64]
        Magnitudes of the x'-component (downslope), y'-component (cross-slope) and z'-component (into-slope) components, respectively.
    """
    f = np.asarray(f, dtype=float)
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)

    phi = np.deg2rad(phi)  # Convert inclination to rad
    theta = np.deg2rad(theta)  # Convert rotation to rad
    f_x = -f * np.sin(phi)  # x'-component
    f_y = f * np.sin(theta)  # y'-component
    f_z = f * np.cos(phi) * np.cos(theta)  # z'-component

    return (*(np.squeeze(x) for x in (f_x, f_y, f_z)),)


def get_skier_point_load(m: float) -> float:
    """
    Calculate skier point load.

    Arguments
    ---------
    m : float
        Skier weight [kg].

    Returns
    -------
    f : float
        Skier load [N/mm].
    """
    F = 1e-3 * m * G_MM_S2 / LSKI_MM  # Total skier
    return F


def load_dummy_profile(
    profile_id: Literal[
        "a", "b", "c", "d", "e", "f", "h", "soft", "medium", "hard", "comp"
    ],
) -> list[Layer]:
    """Define standard layering types for comparison."""
    soft_layer = Layer(rho=180, h=120, E=5)
    medium_layer = Layer(rho=270, h=120, E=30)
    hard_layer = Layer(rho=350, h=120, E=93.8)

    tested_layers = [
        Layer(rho=350, h=120),
        Layer(rho=270, h=120),
        Layer(rho=180, h=120),
    ]

    # Database (top to bottom)
    database = {
        # Layered
        "a": [hard_layer, medium_layer, soft_layer],
        "b": [soft_layer, medium_layer, hard_layer],
        "c": [hard_layer, soft_layer, hard_layer],
        "d": [soft_layer, hard_layer, soft_layer],
        "e": [hard_layer, soft_layer, soft_layer],
        "f": [soft_layer, soft_layer, hard_layer],
        "tested": tested_layers,
        # Homogeneous
        "h": [medium_layer, medium_layer, medium_layer],
        "soft": [soft_layer, soft_layer, soft_layer],
        "medium": [medium_layer, medium_layer, medium_layer],
        "hard": [hard_layer, hard_layer, hard_layer],
        # Comparison
        "comp": [
            Layer(rho=240, h=200, E=5.23),
        ],
    }

    # Load profile
    try:
        profile = database[profile_id.lower()]
    except KeyError:
        raise ValueError(f"Profile {profile_id} is not defined.") from None
    return profile


def isnotebook() -> bool:
    """
    Check if code is running in a Jupyter notebook environment.

    Returns
    -------
    bool
        True if running in Jupyter notebook, False otherwise.
    """
    try:
        # Check if we're in IPython
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel

        if get_ipython() is None:
            return False

        # Check if we're specifically in a notebook (not just IPython terminal)
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            return True  # Jupyter notebook
        if get_ipython().__class__.__name__ == "TerminalInteractiveShell":
            return False  # IPython terminal
        return False  # Other IPython environments
    except ImportError:
        return False  # IPython not available
