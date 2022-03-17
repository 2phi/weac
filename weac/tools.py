# pylint: disable=C0103
"""Helper functions for the WEak Layer AntiCrack nucleation model."""

# Standard library imports
from timeit import default_timer as timer

# Third party imports
import numpy as np


def time():
    """Return current time in milliseconds."""
    return 1e3*timer()


def isnotebook():
    """Identify shell environment."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        if shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        # None of the above: other type ?
        return False
    except NameError:
        return False      # Probably standard Python interpreter


def load_dummy_profile(profile_id):
    """Define standard layering types for comparison."""
    # Layers [density (kg/m^3), thickness (mm), Young's modulus (N/mm^2)]
    soft = [180., 120., 5]
    medium = [270., 120., 30]
    hard = [350., 120., 93.8]

    # Database (top to bottom)
    database = {
        # Layered
        'a':      [hard,   medium, soft],
        'b':      [soft,   medium, hard],
        'c':      [hard,   soft,   hard],
        'd':      [soft,   hard,   soft],
        'e':      [hard,   soft,   soft],
        'f':      [soft,   soft,   hard],
        # Homogeneous
        'soft':   [soft,   soft,   soft],
        'medium': [medium, medium, medium],
        'hard':   [hard,   hard,   hard],
        # Comparison
        'comp':   [[240., 200., 5.23], ]
    }

    # Load profile
    try:
        profile = np.array(database[profile_id.lower()])
    except KeyError:
        raise ValueError(f'Profile {profile_id} is not defined.') from None

    # Prepare output
    layers = profile[:, 0:2]
    E = profile[:, 2]

    return layers, E


def calc_center_of_gravity(layers):
    """
    Calculate z-coordinate of the center of gravity.

    Arguments
    ---------
    layers : list
        2D list of layer densities and thicknesses. Columns are
        density (kg/m^3) and thickness (mm). One row corresponds
        to one layer.

    Returns
    -------
    H : float
        Total slab thickness (mm).
    zs : float
        Z-coordinate of center of gravity (mm).
    """
    # Layering info for center of gravity calculation (bottom to top)
    n = layers.shape[0]                 # Number of layers
    rho = np.flipud(layers[:, 0])       # Layer densities
    h = np.flipud(layers[:, 1])          # Layer thicknesses
    H = sum(h)                          # Total slab thickness
    # Layer center coordinates (bottom to top)
    zi = [H/2 - sum(h[0:j]) - h[j]/2 for j in range(n)]
    # Z-coordinate of the center of gravity
    zs = sum(zi*h*rho)/sum(h*rho)
    # Return slab thickness and center of gravity
    return H, zs


def scapozza(rho):
    """
    Compute Young's modulus (MPa) from density (kg/m^3).

    Arguments
    ---------
    rho : float or ndarray
        Density (kg/m^3).

    Returns
    -------
    E : float or ndarray
        Young's modulus (MPa).
    """
    rho = rho*1e-12                 # Convert to t/mm^3
    rho0 = 917e-12                  # Desity of ice in t/mm^3
    E = 5.07e3*(rho/rho0)**5.13   # Young's modulus in MPa
    return E


def gerling(rho, C0=6.0, C1=4.6):
    """
    Compute Young's modulus from density according to Gerling et al. 2017.

    Arguments
    ---------
    rho : float or ndarray
        Density (kg/m^3).
    C0 : float, optional
        Multiplicative constant of Young modulus parametrization
        according to Gerling et al. (2017). Default is 6.0.
    C1 : float, optional
        Exponent of Young modulus parameterization according to
        Gerling et al. (2017). Default is 4.6.

    Returns
    -------
    E : float or ndarray
        Young's modulus (MPa).
    """
    return C0*1e-10*rho**C1
