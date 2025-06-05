"""
This module defines the system properties for the WEAC simulation.
The system properties are used to define the system of the WEAC simulation.
The Eigenvalue problem is solved for the system properties and the mechanical properties are calculated.
"""
import logging
import numpy as np
from typing import List

from weac_2.constants import G_MM_S2, LSKI_MM, ROMBERG_TOL
from weac_2.components import Layer, WeakLayer, Segment
from weac_2.core.slab import Slab

logger = logging.getLogger(__name__)


class Eigensystem():
    """
    Base class for a layered beam on an elastic foundation.

    Provides geometry, material and loading attributes, and methods
    for the assembly of the eigensystem.
    
    """
    # Input data
    system: str
    touchdown: bool
    weak_layer: WeakLayer
    slab: Slab
    
    # System properties
    A11: float
    B11: float
    D11: float
    kA55: float
    K0: float
    ewC: float
    ewR: float
    evC: float
    evR: float
    sR: float
    sC: float
    
    def __init__(self, system: str, touchdown: bool, weak_layer: WeakLayer, slab: Slab):
        self.system = system
        self.touchdown = touchdown
        self.slab = slab
        self.weak_layer = weak_layer
        
        self._calc_laminate_stiffness_parameters(self.slab, self.weak_layer)
        self._calc_ev_ew_of_system_matrix()
    
    def calc_eigensystem(self):
        """Calculate the fundamental system of the problem."""
        self._calc_laminate_stiffness_parameters()
        self._calc_eigensystem()
        
    def _calc_laminate_stiffness_parameters(self, slab: Slab, weak_layer: WeakLayer):
        """
        Provide ABD matrix.

        Return plane-strain laminate stiffness matrix (ABD matrix).
        """
        # Append z_{N+1} at top of weak layer
        zis = np.concatenate(([-self.slab.H/2] , self.slab.zi_top))
        
        # Initialize stiffness components
        A11, B11, D11, kA55 = 0, 0, 0, 0
        # Add layerwise contributions
        for i in range(len(zis) - 1):
            E = self.slab.Ei[i]
            G = self.slab.Gi[i]
            nu = self.slab.nui[i]
            A11 = A11 + E/(1 - nu**2)*(zis[i+1] - zis[i])
            B11 = B11 + 1/2*E/(1 - nu**2)*(zis[i+1]**2 - zis[i]**2)
            D11 = D11 + 1/3*E/(1 - nu**2)*(zis[i+1]**3 - zis[i]**3)
            kA55 = kA55 + self.k*G*(zis[i+1] - zis[i])

        self.A11 = A11
        self.B11 = B11
        self.D11 = D11
        self.kA55 = kA55
        self.K0 = B11**2 - A11*D11

    def _calc_eigensystem(self):
        """Calculate eigenvalues and eigenvectors of the system matrix."""
        # Calculate eigenvalues (ew) and eigenvectors (ev)
        ew, ev = np.linalg.eig(self.calc_system_matrix())
        # Classify real and complex eigenvalues
        real = (ew.imag == 0) & (ew.real != 0)  # real eigenvalues
        cmplx = ew.imag > 0                   # positive complex conjugates
        # Eigenvalues
        self.ewC = ew[cmplx]
        self.ewR = ew[real].real
        # Eigenvectors
        self.evC = ev[:, cmplx]
        self.evR = ev[:, real].real
        # Prepare positive eigenvalue shifts for numerical robustness
        self.sR, self.sC = np.zeros(self.ewR.shape), np.zeros(self.ewC.shape)
        self.sR[self.ewR > 0], self.sC[self.ewC > 0] = -1, -1
    
    def _calc_ev_ew_of_system_matrix(self):
        pass
