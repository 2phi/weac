
from typing import List
import numpy as np

from weac_2.components import Layer

class Slab():
    """
    Parameters of the assembled layered system.
    
    Layer z-coordinates (top to bottom) in coordinate system with
    downward pointing z-axis (z-list will be negative to positive).
    z = 0 is set at the mid-point of the slabs thickness.
    
    Attributes
    ----------
    zi_mid
    zi_top
    rhoi
    hi
    Ei
    Gi
    H
    z_cog
    """
    # Input data
    layers: List[Layer]
    
    # Derived Values
    # Z-Coordinates with z=0 at the midpoint of the whole slab
    zi_mid: np.ndarray          # z-coordinate of the layer i mid-point
    zi_top: np.ndarray          # z-coordinate of the layer i (boundary towards surface)
    rhoi: np.ndarray            # densities of the layer i [t/mm^3]
    hi: np.ndarray              # thickness of the layer i [mm]
    Ei: np.ndarray              # Young's modulus of the layer i [MPa]
    Gi: np.ndarray              # Shear Modulus of the layer i [MPa]
    nui: np.ndarray             # Poisson Ratio of the layer i [-]

    H: float                    # Total slab thickness (i.e. assembled layers) [mm]
    z_cog: float                # z-coordinate of Center of Gravity [mm]
    
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        self._calc_slab_params()

    def _calc_slab_params(self):
        """
        Calculates:
        zi:              z-coordinate of the layer i mid-point, with z=0 at the midpoint of the whole slab
        rhoi:            densities in [t/mm^3] of the layer i
        slab_thickness:  Slab thickness (all layers excluding weaklayer)
        z_cog:           z-coordinate center of gravity of the slab
        """
        n = len(self.layers)  # Number of layers
        rhoi = np.array([ly.rho for ly in self.layers]) * 1e-12 # Layer densities (kg/m^3 -> t/mm^3)
        hi = np.array([ly.h for ly in self.layers]) # Layer thickness
        Ei = np.array([ly.E for ly in self.layers])
        Gi = np.array([ly.G for ly in self.layers])
        nui = np.array([ly.nu for ly in self.layers])
        
        H = hi.sum()
        
        zi_mid = [H / 2 - sum(hi[0:j]) - hi[j] / 2 for j in range(n)]
        zi_top = np.cumsum(hi) - H/2
        z_cog = sum(zi_mid * hi * rhoi) / sum(hi * rhoi)
        
        self.rhoi = rhoi
        self.hi = hi
        self.Ei = Ei
        self.Gi = Gi
        self.nui = nui
        
        self.zi_mid = zi_mid
        self.zi_top = zi_top
        
        self.H = H
        self.z_cog = z_cog