"""
This module defines the system properties for the WEAC simulation.
The system properties are used to define the system of the WEAC simulation.
The Eigenvalue problem is solved for the system properties and the mechanical properties are calculated.
"""
import logging
from typing import Literal
import numpy as np
from numpy.typing import NDArray

from weac_2.constants import K_SHEAR
from weac_2.components import WeakLayer
from weac_2.core.slab import Slab

logger = logging.getLogger(__name__)


class Eigensystem():
    """
    Calculates system properties and solves the eigenvalue problem for a layered beam on an elastic foundation (Winkler model).
    
    Attributes
    ----------
    weak_layer: WeakLayer
    slab: Slab
    
    System properties
    -----------------
    A11: float          # extensional stiffness
    B11: float          # coupling stiffness
    D11: float          # bending stiffness
    kA55: float         # shear stiffness
    K0: float           # foundation stiffness
    
    Eigenvalues and Eigenvectors
    ----------------------------
    ewC: NDArray[np.complex128]     # shape (k): Complex Eigenvalues
    ewR: NDArray[np.float64]        # shape (k): Real Eigenvalues
    evC: NDArray[np.complex128]     # shape (6, k): Complex Eigenvectors
    evR: NDArray[np.float64]        # shape (6, k): Real Eigenvectors
    sR: NDArray[np.float64]         # shape (k): Real positive eigenvalue shifts (for numerical robustness)
    sC: NDArray[np.float64]         # shape (k): Complex positive eigenvalue shifts (for numerical robustness)
    """
    # Input data
    weak_layer: WeakLayer
    slab: Slab
    
    # System properties
    A11: float          # extensional stiffness
    B11: float          # coupling stiffness
    D11: float          # bending stiffness
    kA55: float         # shear stiffness
    K0: float           # foundation stiffness
    
    # Eigenvalues and Eigenvectors
    ewC: NDArray[np.complex128]     # shape (k): Complex Eigenvalues
    ewR: NDArray[np.float64]        # shape (k): Real Eigenvalues
    evC: NDArray[np.complex128]     # shape (6, k): Complex Eigenvectors
    evR: NDArray[np.float64]        # shape (6, k): Real Eigenvectors
    sR: NDArray[np.float64]         # shape (k): Real positive eigenvalue shifts (for numerical robustness)
    sC: NDArray[np.float64]         # shape (k): Complex positive eigenvalue shifts (for numerical robustness)
    
    def __init__(self, weak_layer: WeakLayer, slab: Slab):
        self.slab = slab
        self.weak_layer = weak_layer
        
        self.calc_eigensystem()
    
    def calc_eigensystem(self):
        """Calculate the fundamental system of the problem."""
        self._calc_laminate_stiffness_parameters()
        K = self._assemble_system_matrix()
        self._calc_eigenvalues_and_eigenvectors(K)
        
    def _calc_laminate_stiffness_parameters(self):
        """
        Provide ABD matrix.

        Return plane-strain laminate stiffness matrix (ABD matrix).
        """
        # Append z_{1} at top of surface layer
        zis = np.concatenate(([-self.slab.H/2] , self.slab.zi_bottom))
        
        # Initialize stiffness components
        A11, B11, D11, kA55 = 0, 0, 0, 0
        # Add layerwise contributions
        for i in range(len(zis) - 1):
            E = self.slab.Ei[i]
            G = self.slab.Gi[i]
            nu = self.slab.nui[i]
            A11 += E/(1 - nu**2)*(zis[i+1] - zis[i])
            B11 += 1/2*E/(1 - nu**2)*(zis[i+1]**2 - zis[i]**2)
            D11 += 1/3*E/(1 - nu**2)*(zis[i+1]**3 - zis[i]**3)
            kA55 += K_SHEAR*G*(zis[i+1] - zis[i])

        self.A11 = A11
        self.B11 = B11
        self.D11 = D11
        self.kA55 = kA55
        self.K0 = B11**2 - A11*D11
    
    def _assemble_system_matrix(self) -> NDArray[np.float64]:
        """
        Assemble first-order ODE system matrix K.

        Using the solution vector z = [u, u', w, w', psi, psi']
        the ODE system is written in the form Az' + Bz = d
        and rearranged to z' = -(A^-1)Bz + (A^-1)d = Kz + q

        Returns
        -------
        NDArray[np.float64]
            System matrix K (6x6).
        """
        kn = self.weak_layer.kn
        kt = self.weak_layer.kt
        H = self.slab.H          # total slab thickness
        h = self.weak_layer.h    # weak layer thickness

        # Abbreviations (MIT t/2 im GGW, MIT w' in Kinematik)
        K21 = kt*(-2*self.D11 + self.B11*(H + h))/(2*self.K0)
        K24 = (2*self.D11*kt*h
            - self.B11*kt*h*(H + h)
            + 4*self.B11*self.kA55)/(4*self.K0)
        K25 = (-2*self.D11*H*kt
            + self.B11*H*kt*(H + h)
            + 4*self.B11*self.kA55)/(4*self.K0)
        K43 = kn/self.kA55
        K61 = kt*(2*self.B11 - self.A11*(H + h))/(2*self.K0)
        K64 = (-2*self.B11*kt*h
            + self.A11*kt*h*(H + h)
            - 4*self.A11*self.kA55)/(4*self.K0)
        K65 = (2*self.B11*H*kt
            - self.A11*H*kt*(H + h)
            - 4*self.A11*self.kA55)/(4*self.K0)

        # System matrix
        K = [[0,    1,    0,    0,    0,    0],
            [K21,  0,    0,  K24,  K25,    0],
            [0,    0,    0,    1,    0,    0],
            [0,    0,  K43,    0,    0,   -1],
            [0,    0,    0,    0,    0,    1],
            [K61,  0,    0,  K64,  K65,    0]]

        return np.array(K, dtype=np.float64)

    def _calc_eigenvalues_and_eigenvectors(self, system_matrix: NDArray[np.float64]):
        """Calculate eigenvalues and eigenvectors of the system matrix."""
        # Calculate eigenvalues (ew) and eigenvectors (ev)
        ew, ev = np.linalg.eig(system_matrix)
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
        # 1. Keep small-positive eigenvalues away from zero, to not have a near-singular matrix
        self.sR, self.sC = np.zeros(self.ewR.shape), np.zeros(self.ewC.shape)
        self.sR[self.ewR > 0], self.sC[self.ewC > 0] = -1, -1
