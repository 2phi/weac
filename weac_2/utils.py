


import numpy as np


def split_q(q: float, phi: float) -> tuple[float, float]:
        """
        Splits a line-load intensity from gravitational forces into:
        Tangential component is taken positive downslope.
        Normal component is normal to surface layer.
        
        Returns
        -------
        q_n, q_t: [float, float]
            normal and tangential component
        """
        # Convert units
        phi = np.deg2rad(phi)                   # Convert inclination to rad
        # Split into components
        q_n = q*np.cos(phi)                 # Normal direction
        q_t = -q*np.sin(phi)                # Tangential direction
        return q_n, q_t
