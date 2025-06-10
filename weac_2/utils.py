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
        f_tan, f_norm : float
            Magnitudes of the tangential ( + downslope ) and normal
            ( + into-slope ) components, respectively.
        """
        # Convert units
        phi = np.deg2rad(phi)                   # Convert inclination to rad
        # Split into components
        f_tan = -f*np.sin(phi)                # Tangential direction
        f_norm = f*np.cos(phi)                 # Normal direction
        return f_tan, f_norm
    
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
