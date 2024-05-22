# pylint: disable=C0103
"""Helper functions for the WEak Layer AntiCrack nucleation model."""

# Standard library imports
from timeit import default_timer as timer
from IPython import get_ipython

# Third party imports
import numpy as np
import weac


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
    # soft = [120., 120., 0.3]
    # medium = [180., 120., 1.5]
    # hard = [270., 120., 7.5]


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
        'h':      [medium,   medium,   medium],
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
    layers : ndarray
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
    n = layers.shape[0]                    # Number of layers
    rho = 1e-12*np.flipud(layers[:, 0])    # Layer densities (kg/m^3 -> t/mm^3)
    h = np.flipud(layers[:, 1])            # Layer thicknesses
    H = sum(h)                             # Total slab thickness
    # Layer center coordinates (bottom to top)
    zi = [H/2 - sum(h[0:j]) - h[j]/2 for j in range(n)]
    # Z-coordinate of the center of gravity
    zs = sum(zi*h*rho)/sum(h*rho)
    # Return slab thickness and center of gravity
    return H, zs


def calc_vertical_bc_center_of_gravity(slab, phi):
    """
    Calculate center of gravity of triangular slab segements for vertical PSTs.

    Parameters
    ----------
    slab : ndarray
        List of layer densities, thicknesses, and elastic properties.
        Columns are density (kg/m^3), thickness (mm), Young's modulus
        (MPa), shear modulus (MPa), and Poisson's ratio. One row corresponds
        to one layer.
    phi : fload
        Slope angle (deg).

    Returns
    -------
    xs : float
        Horizontal coordinate of center of gravity (mm).
    zs : float
        Vertical coordinate of center of gravity (mm).
    w : ndarray
        Weight of the slab segment that is cut off or added (t).
    """
    # Convert slope angle to radians
    phi = np.deg2rad(phi)
    
    # Catch flat-field case
    if phi == 0:
        xs = 0
        zs = 0
        w = 0
    else:
        # Layering info for center of gravity calculation (top to bottom)
        n = slab.shape[0]               # Number of slab
        rho = 1e-12*slab[:, 0]          # Layer densities (kg/m^3 -> t/mm^3)
        hi = slab[:, 1]                 # Layer thicknesses
        H = sum(hi)                     # Total slab thickness
        # Layer coordinates z_i (top to bottom)
        z = np.array([-H/2 + sum(hi[0:j]) for j in range(n + 1)])
        zi = z[:-1]                         # z_i
        zii = z[1:]                         # z_{i+1}
        # Center of gravity of all layers (top to bottom)
        zsi = zi + hi/3*(3/2*H - zi - 2*zii)/(H - zi - zii)
        # Surface area of all layers (top to bottom)
        Ai = hi/2*(H - zi - zii)*np.tan(phi)
        # Center of gravity in vertical direction
        zs = sum(zsi*rho*Ai)/sum(rho*Ai)
        # Center of gravity in horizontal direction
        xs = (H/2 - zs)*np.tan(phi/2)
        # Weight of added or cut off slab segments (t)
        w = sum(Ai*rho)
    
    # Return center of gravity and weight of slab segment
    return xs, zs, w

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


def bergfeld(rho, rho0=917, C0=6.5, C1=4.4):
    """
    Compute Young's modulus from density according to Bergfeld et al. (2023).

    Arguments
    ---------
    rho : float or ndarray
        Density (kg/m^3).
    rho0 : float, optional
        Density of ice (kg/m^3). Default is 917.
    C0 : float, optional
        Multiplicative constant of Young modulus parametrization
        according to Bergfeld et al. (2023). Default is 6.5.
    C1 : float, optional
        Exponent of Young modulus parameterization according to
        Bergfeld et al. (2023). Default is 4.4.

    Returns
    -------
    E : float or ndarray
        Young's modulus (MPa).
    """
    return C0*1e3*(rho/rho0)**C1


def tensile_strength_slab(rho, unit='kPa'):
    """
    Estimate the tensile strenght of a slab layer from its density.

    Uses the density parametrization of Sigrist (2006).

    Arguments
    ---------
    rho : ndarray, float
        Layer density (kg/m^3).
    unit : str, optional
        Desired output unit of the layer strength. Default is 'kPa'.

    Returns
    -------
    ndarray
        Tensile strenght in specified unit.
    """
    convert = {
            'kPa': 1,
            'MPa': 1e-3
        }
    rho_ice = 917
    # Sigrist's equation is given in kPa
    return convert[unit]*240*(rho/rho_ice)**2.44

def touchdown_distance(
        layers: np.ndarray | str | None = None,
        C0: float = 6.5,
        C1: float = 4.4,
        Ewl: float = 0.25,
        t: float = 10,
        phi: float = 0):
    """
    Calculate cut length at first contanct and steady-state touchdown distance.

    Arguments
    ---------
    layers : list, optional
        2D list of layer densities and thicknesses. Columns are
        density(kg/m ^ 3) and thickness(mm). One row corresponds
        to one layer. Default is [[240, 200], ].
    C0 : float, optional
        Multiplicative constant of Young modulus parametrization
        according to Bergfeld et al. (2023). Default is 6.5.
    C1 : float, optional
        Exponent of Young modulus parameterization according to
        Bergfeld et al. (2023). Default is 4.4.
    Ewl : float, optional
        Young's modulus of the weak layer (MPa). Default is 0.25.
    t : float, optional
        Thickness of the weak layer (mm). Default is 10.
    phi : float, optional
        Inclination of the slab (°). Default is 0.
        
    Returns
    -------
    first_contact : float
        Cut length at first contact (mm).
    full_contact : float
        Cut length at which the slab comes into full contact (more than
        a singular point) with the base layer (mm).
    steady_state : float
        Steady-state touchdown distance (mm).
    """
    # Check if layering is defined
    layers = layers if layers else [[240, 200], ]

    # Initialize model with user input
    touchdown = weac.Layered(system='pst-', touchdown=True)

    # Set material properties
    touchdown.set_foundation_properties(E=Ewl, t=t, update=True)
    touchdown.set_beam_properties(layers=layers, C0=C0, C1=C1, update=True)

    # Assemble very long dummy PST to compute crack length where the slab
    # first comes in contact with base layer after weak-layer collapse
    touchdown.calc_segments(L=1e5, a=0, phi=phi)
    first_contact = touchdown.calc_a1()

    # Compute ut length at which the slab comes into full contact (more
    # than a singular point) with the base layer
    full_contact = touchdown.calc_a2()

    # Compute steady-state touchdown distance in a dummy PST with a cut
    # of 5 times the first contact distance
    touchdown.calc_segments(L=1e5, a=5*first_contact, phi=phi)
    steady_state = touchdown.calc_lC()

    # Return first-contact cut length, full-contact cut length,
    # and steady-state touchdown distance (mm)
    return first_contact, full_contact, steady_state
