from typing import List
import numpy as np

from weac_2.constants import G_MM_S2
from weac_2.components import Layer

class Slab():
    """
    Parameters of all layers assembled into a slab,
    provided as np.ndarray for easier access.
    
    Coordinate frame: 
    - z-axis points downward (first index: top layer, last index: bottom layer)
    - z = 0 is set at the mid-point of the slabs thickness

    Attributes
    ----------
    zi_mid: np.ndarray
        z-coordinate of the layer i mid-point
    zi_bottom: np.ndarray
        z-coordinate of the layer i (boundary towards bottom)
    rhoi: np.ndarray
        densities of the layer i [t/mm^3]
    hi: np.ndarray
        thickness of the layer i [mm]
    Ei: np.ndarray
        Young's modulus of the layer i [MPa]
    Gi: np.ndarray
        Shear Modulus of the layer i [MPa]
    nui: np.ndarray
        Poisson Ratio of the layer i [-]
    H: float
        Total slab thickness (i.e. assembled layers) [mm]
    z_cog: float
        z-coordinate of Center of Gravity [mm]
    qw: float
        Weight Load of the slab [N/mm]
    """
    # Input data
    layers: List[Layer]
    
    rhoi: np.ndarray            # densities of the layer i [t/mm^3]
    hi: np.ndarray              # thickness of the layer i [mm]
    Ei: np.ndarray              # Young's modulus of the layer i [MPa]
    Gi: np.ndarray              # Shear Modulus of the layer i [MPa]
    nui: np.ndarray             # Poisson Ratio of the layer i [-]
    
    # Derived Values
    zi_mid: np.ndarray          # z-coordinate of the layer i mid-point
    zi_bottom: np.ndarray       # z-coordinate of the layer i (boundary towards bottom)
    H: float                    # Total slab thickness (i.e. assembled layers) [mm]
    z_cog: float                # z-coordinate of Center of Gravity [mm]
    qw: float                   # Weight Load of the slab [N/mm]

    
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        self._calc_slab_params()

    def _calc_slab_params(self) -> None:
        n = len(self.layers)  # Number of layers
        rhoi = np.array([ly.rho for ly in self.layers]) * 1e-12 # Layer densities (kg/m^3 -> t/mm^3)
        hi = np.array([ly.h for ly in self.layers]) # Layer thickness
        Ei = np.array([ly.E for ly in self.layers])
        Gi = np.array([ly.G for ly in self.layers])
        nui = np.array([ly.nu for ly in self.layers])
        
        H = hi.sum()
        
        zi_mid = [H / 2 - sum(hi[0:j]) - hi[j] / 2 for j in range(n)]
        zi_bottom = np.cumsum(hi) - H/2
        z_cog = sum(zi_mid * hi * rhoi) / sum(hi * rhoi)
        
        qw = sum(rhoi*G_MM_S2*hi)     # Line load [N/mm]
        
        self.rhoi = rhoi
        self.hi = hi
        self.Ei = Ei
        self.Gi = Gi
        self.nui = nui
        
        self.zi_mid = zi_mid
        self.zi_bottom = zi_bottom
        
        self.H = H
        self.z_cog = z_cog
        self.qw = qw
    
    def calc_vertical_center_of_gravity(self, phi: float):
        """
        TODO: No idea what this does.
        Calculate center of gravity of triangular slab segments for vertical PSTs.

        Parameters
        ----------
        phi : float
            Slope angle [deg]

        Returns
        -------
        x_cog : float
            Horizontal coordinate of center of gravity [mm]
        z_cog : float
            Vertical coordinate of center of gravity [mm]
        w : ndarray
            Weight of the slab segment that is cut off or added [t]
        """
        # Convert slope angle to radians
        phi = np.deg2rad(phi)

        # Catch flat-field case
        if phi == 0:
            x_cog = 0
            z_cog = 0
            w = 0
        else:
            n = len(self.hi)
            rho = self.rhoi  # [t/mm^3]
            hi = self.hi     # [mm]
            H = self.H       # [mm]
            # Layer coordinates z_i (top to bottom)
            z = np.array([-H / 2 + sum(hi[0:j]) for j in range(n + 1)])
            zi = z[:-1]
            zii = z[1:]
            # Center of gravity of all layers (top to bottom)
            zsi = zi + hi / 3 * (3 / 2 * H - zi - 2 * zii) / (H - zi - zii)
            # Surface area of all layers (top to bottom)
            Ai = hi / 2 * (H - zi - zii) * np.tan(phi)
            # Center of gravity in vertical direction
            z_cog = sum(zsi * rho * Ai) / sum(rho * Ai)
            # Center of gravity in horizontal direction
            x_cog = (H / 2 - z_cog) * np.tan(phi / 2)
            # Weight of added or cut off slab segments (t)
            w = sum(Ai * rho)

        # Return center of gravity and weight of slab segment
        return x_cog, z_cog, w
