import numpy as np
from typing import Tuple

from weac_2.constants import G_MM_S2, LSKI_MM
from weac_2.components import Layer


def decompose_to_normal_tangential(f: float, phi: float) -> Tuple[float, float]:
    """
    Resolve a gravity-type force/line-load into its tangential (downslope) and
    normal (into-slope) components with respect to an inclined surface.

    Parameters
    ----------
    f_vec : float
        is interpreted as a vertical load magnitude
        acting straight downward (global y negative).
    phi : float
        Surface dip angle `in degrees`, measured from horizontal.
        Positive `phi` means the surface slopes upward in +x.

    Returns
    -------
    f_norm, f_tan : float
        Magnitudes of the tangential ( + downslope ) and normal
        ( + into-slope ) components, respectively.
    """
    # Convert units
    phi = np.deg2rad(phi)  # Convert inclination to rad
    # Split into components
    f_norm = f * np.cos(phi)  # Normal direction
    f_tan = -f * np.sin(phi)  # Tangential direction
    return f_norm, f_tan


def get_skier_point_load(m: float):
    """
    Calculate skier point load.

    Arguments
    ---------
    m : float
        Skier weight (kg).

    Returns
    -------
    f : float
        Skier load (N).
    """
    F = 1e-3 * np.array(m) * G_MM_S2 / LSKI_MM  # Total skier
    return F


def load_dummy_profile(profile_id):
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
        from IPython import get_ipython

        if get_ipython() is None:
            return False

        # Check if we're specifically in a notebook (not just IPython terminal)
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            return True  # Jupyter notebook
        elif get_ipython().__class__.__name__ == "TerminalInteractiveShell":
            return False  # IPython terminal
        else:
            return False  # Other IPython environments
    except ImportError:
        return False  # IPython not available
