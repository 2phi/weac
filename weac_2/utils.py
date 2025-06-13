import numpy as np
from typing import Tuple

from weac_2.constants import G_MM_S2, LSKI_MM

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
        phi = np.deg2rad(phi)                   # Convert inclination to rad
        # Split into components
        f_norm = f*np.cos(phi)                 # Normal direction
        f_tan = -f*np.sin(phi)                # Tangential direction
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
        F = 1e-3*np.array(m)*G_MM_S2/LSKI_MM   # Total skier
        return F

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
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            return True  # Jupyter notebook
        elif get_ipython().__class__.__name__ == 'TerminalInteractiveShell':
            return False  # IPython terminal
        else:
            return False  # Other IPython environments
    except ImportError:
        return False  # IPython not available
