"""
This module defines the system properties for the WEAC simulation.
The system properties are used to define the system of the WEAC simulation.
The Eigenvalue problem is solved for the system properties and the mechanical properties are calculated.
"""
import logging
from typing import Literal
import numpy as np
from numpy.typing import NDArray

from weac_2.utils import decompose_to_normal_tangential
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

    K: NDArray          # System Matrix
    
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
        self.K = self._assemble_system_matrix()
        self._calc_eigenvalues_and_eigenvectors(self.K)
        
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

        # Abbreviations (MIT h/2 im GGW, MIT w' in Kinematik)
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
        """
        Calculate eigenvalues and eigenvectors of the system matrix.
        """
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

    def zh(self, x: float, l: float = 0, k: bool = True) -> NDArray:
        """
        Compute bedded or free complementary solution at position x.

        Arguments
        ---------
        x : float
            Horizontal coordinate (mm).
        l : float, optional
            Segment length (mm). Default is 0.
        k : bool
            Indicates whether segment has foundation or not. Default
            is True.

        Returns
        -------
        zh : ndarray
            Complementary solution matrix (6x6) at position x.
        """
        if k:
            zh = np.concatenate([
                # Real
                self.evR*np.exp(self.ewR*(x + l*self.sR)),
                # Complex
                np.exp(self.ewC.real*(x + l*self.sC))*(
                       self.evC.real*np.cos(self.ewC.imag*x)
                       - self.evC.imag*np.sin(self.ewC.imag*x)),
                # Complex
                np.exp(self.ewC.real*(x + l*self.sC))*(
                       self.evC.imag*np.cos(self.ewC.imag*x)
                       + self.evC.real*np.sin(self.ewC.imag*x))], axis=1)
        else:
            # Abbreviations
            H14 = 3*self.B11/self.A11*x**2
            H24 = 6*self.B11/self.A11*x
            H54 = -3*x**2 + 6*self.K0/(self.A11*self.kA55)
            # Complementary solution matrix of free segments
            zh = np.array(
                [[0,      0,      0,    H14,      1,      x],
                 [0,      0,      0,    H24,      0,      1],
                 [1,      x,   x**2,   x**3,      0,      0],
                 [0,      1,    2*x, 3*x**2,      0,      0],
                 [0,     -1,   -2*x,    H54,      0,      0],
                 [0,      0,     -2,   -6*x,      0,      0]])

        return zh

    def zp(self, x: float, phi: float = 0, k=True, qs: float = 0) -> NDArray:
        """
        Compute bedded or free particular integrals at position x.

        Arguments
        ---------
        x : float
            Horizontal coordinate (mm).
        phi : float
            Inclination (degrees).
        k : bool
            Indicates whether segment has foundation (True) or not
            (False). Default is True.
        qs : float
            additional surface load weight

        Returns
        -------
        zp : ndarray
            Particular integral vector (6x1) at position x.
        """
        # Get weight and surface loads
        qw_n, qw_t = decompose_to_normal_tangential(f=self.slab.qw, phi=phi)
        qs_n, qs_t = decompose_to_normal_tangential(f=qs, phi=phi)

        # Weak Layer properties
        kn = self.weak_layer.kn
        kt = self.weak_layer.kt
        h = self.weak_layer.h
        
        # Slab properties
        H = self.slab.H
        z_cog = self.slab.z_cog
        
        # Laminate stiffnesses
        A11 = self.A11
        B11 = self.B11
        kA55 = self.kA55
        K0 = self.K0

        # Assemble particular integral vectors
        if k:
            zp = np.array([
                [(qw_t + qs_t)/kt + H*qw_t*(H + h - 2*z_cog)/(4*kA55)
                    + H*qs_t*(2*H + h)/(4*kA55)],
                [0],
                [(qw_n + qs_n)/kn],
                [0],
                [-(qw_t*(H + h - 2*z_cog) + qs_t*(2*H + h))/(2*kA55)],
                [0]])
        else:
            zp = np.array([
                [(-3*(qw_t + qs_t)/A11 - B11*(qw_n + qs_n)*x/K0)/6*x**2],
                [(-2*(qw_t + qs_t)/A11 - B11*(qw_n + qs_n)*x/K0)/2*x],
                [-A11*(qw_n + qs_n)*x**4/(24*K0)],
                [-A11*(qw_n + qs_n)*x**3/(6*K0)],
                [A11*(qw_n + qs_n)*x**3/(6*K0)
                 + ((z_cog - B11/A11)*qw_t - H*qs_t/2 - (qw_n + qs_n)*x)/kA55],
                [(qw_n + qs_n)*(A11*x**2/(2*K0) - 1/kA55)]])

        return zp
    
    def get_load_vector(self, phi: float, qs: float = 0) -> NDArray:
        """
        Compute sytem load vector q.

        Using the solution vector z = [u, u', w, w', psi, psi']
        the ODE system is written in the form Az' + Bz = d
        and rearranged to z' = -(A ^ -1)Bz + (A ^ -1)d = Kz + q

        Arguments
        ---------
        phi : float
            Inclination [deg]. Counterclockwise positive.
        qs : float
            Surface Load [N/mm]

        Returns
        -------
        ndarray
            System load vector q (6x1).
        """
        # Get weight and surface loads
        qw_n, qw_t = decompose_to_normal_tangential(f=self.slab.qw, phi=phi)
        qs_n, qs_t = decompose_to_normal_tangential(f=qs, phi=phi)

        return np.array([
            [0],
            [(self.B11*(self.h*qs_t - 2*qw_t*self.slab.z_cog)
            + 2*self.D11*(qw_t + qs_t))/(2*self.K0)],
            [0],
            [-(qw_n + qs_n)/self.kA55],
            [0],
            [-(self.A11*(self.h*qs_t - 2*qw_t*self.slab.z_cog)
            + 2*self.B11*(qw_t + qs_t))/(2*self.K0)]
        ])
