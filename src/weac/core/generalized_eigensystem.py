"""
Adapter backend for a generalized eigensystem (composition only).

This class intentionally does NOT inherit from the classic Eigensystem.
It exposes the same public API surface that callers expect (zh, zp,
get_load_vector, calc_eigensystem, assemble_system_matrix, eigen data,
stiffness parameters), but you are free to replace internals with a
12/24-DOF formulation without being constrained by the 6-DOF base class.

Initially, all calls delegate to the classic implementation to preserve
behavior. Replace the internals incrementally with the generalized
(OOP-based) formulas.
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from weac.core.slab import Slab
from weac.constants import SHEAR_CORRECTION_FACTOR, G_MM_S2
from weac.components import WeakLayer
from weac.utils.misc import decompose_to_xyz

logger = logging.getLogger(__name__)


class GeneralizedEigensystem:
    """
    Composition-based adapter for gradual introduction of generalized physics.
    Calculates system properties and solves the eigenvalue problem
    for a layered beam on an isotropic elastic layer under generalized loading conditions.

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
    kB55: float         # higher-order shear stiffness
    kD55: float         # higher-order bending shear stiffness
    
    K: float           # system matrix


    Eigenvalues and Eigenvectors
    ----------------------------
    ewC: NDArray[np.complex128]     # shape (k): Complex Eigenvalues
    ewR: NDArray[np.float64]        # shape (k): Real Eigenvalues
    evC: NDArray[np.complex128]     # shape (24, k): Complex Eigenvectors
    evR: NDArray[np.float64]        # shape (24, k): Real Eigenvectors
    sR: NDArray[np.float64]         # shape (k): Real positive eigenvalue shifts
                                    # (for numerical robustness)
    sC: NDArray[np.float64]         # shape (k): Complex positive eigenvalue shifts
                                    # (for numerical robustness)
    """

    # Input data
    weak_layer: WeakLayer
    slab: Slab

    # System properties
    A11: float  # extensional stiffness
    B11: float  # coupling stiffness
    D11: float  # bending stiffness
    kA55: float  # shear stiffness
    kB55: float  # higher-order shear stiffness
    kD55: float  # higher-order bending shear stiffness
    K0: float # stiffness determinant

    K: NDArray  # System Matrix

    # Eigenvalues and Eigenvectors
    ewC: NDArray[np.complex128]  # shape (k): Complex Eigenvalues
    ewR: NDArray[np.float64]  # shape (k): Real Eigenvalues
    evC: NDArray[np.complex128]  # shape (24, k): Complex Eigenvectors
    evR: NDArray[np.float64]  # shape (24, k): Real Eigenvectors
    sR: NDArray[
        np.float64
    ]  # shape (k): Real positive eigenvalue shifts (for numerical robustness)
    sC: NDArray[
        np.float64
    ]  # shape (k): Complex positive eigenvalue shifts (for numerical robustness)

    def __init__(self, weak_layer: WeakLayer, slab: Slab):
        # Store references only; no delegation, no eigen-decomposition at init
        self.weak_layer = weak_layer
        self.slab = slab

        self.calc_eigensystem()

    # Public API expected by downstream components
    def calc_eigensystem(self):
        """Calculate generalized fundamental system (if needed).
        """
        self._calc_laminate_stiffness_parameters()
        self.K = self.assemble_system_matrix()
        self.ewC, self.ewR, self.evC, self.evR, self.sR, self.sC = (
            self.calc_eigenvalues_and_eigenvectors(self.K)
        )

    def _calc_laminate_stiffness_parameters(self):
        """
        Provide ABD matrix.

        Return plane-strain laminate stiffness matrix (ABD matrix).
        """
        # Append z_{1} at top of surface layer
        zis = np.concatenate(([-self.slab.H / 2], self.slab.zi_bottom))

        # Initialize stiffness components
        A11, B11, D11, kA55, kB55, kD55 = 0, 0, 0, 0, 0, 0
        # Add layerwise contributions
        for i in range(len(zis) - 1):
            E = self.slab.Ei[i]
            G = self.slab.Gi[i]
            nu = self.slab.nui[i]
            A11 += E / (1 - nu**2) * (zis[i + 1] - zis[i])
            B11 += 1 / 2 * E / (1 - nu**2) * (zis[i + 1] ** 2 - zis[i] ** 2)
            D11 += 1 / 3 * E / (1 - nu**2) * (zis[i + 1] ** 3 - zis[i] ** 3)
            kA55 += SHEAR_CORRECTION_FACTOR * G * (zis[i + 1] - zis[i])
            kB55 += 1 /  2 * SHEAR_CORRECTION_FACTOR * G * (zis[i + 1] ** 2 - zis[i] ** 2)
            kD55 += 1 /  3 * SHEAR_CORRECTION_FACTOR * G * (zis[i + 1] ** 3 - zis[i] ** 3)

        self.A11 = A11
        self.B11 = B11
        self.D11 = D11
        self.kA55 = kA55
        self.kB55 = kB55
        self.kD55 = kD55
        self.K0 = B11**2 - A11 * D11

        

    def assemble_system_matrix(self) -> NDArray[np.float64]:
        """Assemble generalized first-order ODE system matrix K (if needed)."""
        Ew = self.weak_layer.E
        nuw = self.weak_layer.nu
        H = self.slab.H
        h = self.weak_layer.h
        b = self.slab.b

        A11 = self.A11
        B11 = self.B11
        D11 = self.D11
        kA55 = self.kA55
        kB55 = self.kB55
        kD55 = self.kD55
        Pi = np.pi
        # TODO: Compute and implement system matrix for unsupported structures for stress computation


        c0201=(-3*(2*D11 - B11*H)*Pi**2*Ew*(-1 + 2*nuw))/(h*(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c0206=(4*H*kA55*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 6*D11*Ew*(-8 + Pi**2*(-1 + 4*nuw)) + 3*B11*(H*Ew*(8 + Pi**2 - 4*Pi**2*nuw) + 8*kA55*Pi**2*(-1 + nuw + 2*nuw**2)))/(8*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 24*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 2*(4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c0209=(2*H*Ew*(3*D11*Pi**2*(1 - 2*nuw) + 2*kA55*(-6 + Pi**2)*h**2*(-1 + nuw)) + 3*B11*Pi**2*(-1 + 2*nuw)*(H**2*Ew + 8*kA55*h*(1 + nuw)))/(2*h*(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c0213=(3*(2*D11 - B11*H)*Pi**3*Ew*(-1 + 2*nuw))/(h*(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c0222=(6*(2*D11 - B11*H)*Pi*Ew)/(4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) - 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) + A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) - 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) + 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2))


        c0403=(3*Ew*(36*Pi**2*(2*kD55*Pi**2 - kB55*(-8 + Pi**2)*h)*(1 + nuw) + b**2*(Pi**2*(-6 + Pi**2)*h*Ew + 6*kA55*Pi**4*(1 + nuw)) - 3*H*((48 - 14*Pi**2 + Pi**4)*h**2*Ew + 12*kB55*Pi**4*(1 + nuw))))/(h*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0407=(-3*Ew*(-((-1 + 2*nuw)*(24*Pi**2*h*(-3*kD55*(8 + Pi**2) + 2*kB55*(-6 + Pi**2)*h)*(1 + nuw) + 3*H**2*((48 - 14*Pi**2 + Pi**4)*h**2*Ew + 12*kB55*Pi**4*(1 + nuw)) + 4*H*((-6 + Pi**2)**2*h**3*Ew - 18*kD55*Pi**4*(1 + nuw) + 18*kB55*Pi**4*h*(1 + nuw)))) + b**2*(-24*kB55*Pi**4*(-1 + nuw**2) + (8 + Pi**2)*h*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) + H*(Pi**2*(-6 + Pi**2)*h*Ew + 6*kA55*Pi**4*(-1 + nuw + 2*nuw**2)))))/(2*h*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0412=-1/4*(72*Pi**2*(-1 + nuw + 2*nuw**2)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw))) + b**2*(-24*kA55*Pi**2*(-1 + nuw + 2*nuw**2)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) + Ew*(36*kB55*Pi**2*(1 + nuw)*(8 + Pi**2*(-1 + 4*nuw)) - (-6 + Pi**2)*h*(4*(-6 + Pi**2)*h*Ew*(-1 + 2*nuw) + 24*kA55*Pi**2*(-1 + nuw + 2*nuw**2) - 3*H*Ew*(8 + Pi**2*(-1 + 4*nuw))))))/((-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0416=(6*b*Pi*Ew*(H*(-6 + Pi**2)*h*Ew + 12*kB55*Pi**2*(1 + nuw)))/((-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0417=(-3*Pi*Ew*(b**2*(Pi**2*(-6 + Pi**2)*h*Ew + 6*kA55*Pi**4*(1 + nuw)) - 12*(H*(-6 + Pi**2)*h**2*Ew + 3*H*kB55*Pi**4*(1 + nuw) - 6*Pi**2*(kD55*Pi**2 - 2*kB55*h)*(1 + nuw))))/(h*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0423=(-6*Pi*Ew*(-72*(H*kB55 - 2*kD55)*Pi**2*h*(-1 + nuw + 2*nuw**2) + b**2*(12*kB55*Pi**4*(-1 + nuw**2) + 12*kA55*Pi**2*h*(-1 + nuw + 2*nuw**2) + (-6 + Pi**2)*h*Ew*(H*Pi**2*(-1 + nuw) + 2*h*(-1 + 2*nuw)))))/(b*h*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0602=(3*Ew*(8 + Pi**2*(-1 + 4*nuw)))/(2*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))
        
        c0605=(6*Pi**2*Ew*(-1 + nuw))/(h*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))
        
        c0610=(3*(-8*kA55*Pi**2*(-1 + nuw + 2*nuw**2) + H*Ew*(8 + Pi**2*(-1 + 4*nuw))))/(4*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))
        
        c0614=(6*Pi*Ew)/((-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))
        
        c0619=(24*Pi*Ew*nuw)/(b*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))
        
        c0621=(-6*Pi**3*Ew*(-1 + nuw))/(h*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c0803=(-18*Ew*(6*H*kA55*Pi**4*(1 + nuw) - 12*kB55*Pi**4*(1 + nuw) + (-8 + Pi**2)*h*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))))/(h*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0807=(3*Ew*(2*b**2*Pi**2*(-1 + nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) + (-1 + 2*nuw)*(18*H**2*kA55*Pi**4*(1 + nuw) + 4*h*(-9*kB55*Pi**2*(8 + Pi**2)*(1 + nuw) + (-6 + Pi**2)*h*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))) + H*(-36*kB55*Pi**4*(1 + nuw) + 3*h*((48 - 14*Pi**2 + Pi**4)*h*Ew + 12*kA55*Pi**4*(1 + nuw))))))/(h*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0812=(-3*b**2*Ew*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*(8 + Pi**2*(-1 + 4*nuw)))/(2*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0816=(12*b*Pi*Ew*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))/((-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0817=(36*Pi*Ew*(3*H*kA55*Pi**4*(1 + nuw) - 6*kB55*Pi**4*(1 + nuw) + 2*h*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))))/(h*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0823=(-12*Pi**3*Ew*(-36*(H*kA55 - 2*kB55)*h*(-1 + nuw + 2*nuw**2) + b**2*(-1 + nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))))/(b*h*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        
        c1001=(3*(2*B11 - A11*H)*Pi**2*Ew*(-1 + 2*nuw))/(h*(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1006=(-8*kA55*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 6*B11*Ew*(8 + Pi**2 - 4*Pi**2*nuw) - 3*A11*(H*Ew*(8 + Pi**2 - 4*Pi**2*nuw) + 8*kA55*Pi**2*(-1 + nuw + 2*nuw**2)))/(8*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 24*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 2*(4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1009=-1/2*(6*B11*H*Pi**2*Ew*(1 - 2*nuw) + 8*kA55*(-6 + Pi**2)*h**2*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + 2*nuw)*(H**2*Ew + 8*kA55*h*(1 + nuw)))/(h*(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1013=(-3*(2*B11 - A11*H)*Pi**3*Ew*(-1 + 2*nuw))/(h*(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1022=(6*(2*B11 - A11*H)*Pi*Ew)/(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2))

        c1204=(-6*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))/(b**2*((-6 + Pi**2)*h*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1208=(3*(4*H*(-6 + Pi**2)*h*Ew*(-1 + 2*nuw) + 48*kB55*Pi**2*(-1 + nuw + 2*nuw**2) + b**2*Ew*(-8 + Pi**2*(-1 + 4*nuw))))/(4*b**2*((-6 + Pi**2)*h*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1211=(3*(-1 + 2*nuw)*(b**2*Pi**2*Ew + 4*(-6 + Pi**2)*h**2*Ew + 24*kA55*Pi**2*h*(1 + nuw)))/(2*b**2*h*((-6 + Pi**2)*h*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1215=(3*Pi**3*Ew*(-1 + 2*nuw))/(b*h*((-6 + Pi**2)*h*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1224=(-6*Pi*Ew)/(b*((-6 + Pi**2)*h*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1401=(3*(4*D11 + H*(-4*B11 + A11*H))*Pi*Ew*(-1 + 2*nuw))/(h*(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))
        
        c1406=-1/2*(Pi*(48*B11**2*(-1 + nuw + 2*nuw**2) - 48*A11*D11*(-1 + nuw + 2*nuw**2) + 4*D11*h*Ew*(7 - 19*nuw + 12*nuw**2) + A11*H*h*(-1 + nuw)*(H*Ew*(-7 + 12*nuw) - 24*kA55*(-1 + nuw + 2*nuw**2)) + 4*B11*h*(-1 + nuw)*(H*Ew*(7 - 12*nuw) + 12*kA55*(-1 + nuw + 2*nuw**2))))/(h*(-1 + nuw)*(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1409=(-3*Pi*(-1 + 2*nuw)*(4*B11*(H**2*Ew + 4*kA55*h*(1 + nuw)) - H*(4*D11*Ew + A11*H**2*Ew + 8*A11*kA55*h*(1 + nuw))))/(2*h*(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))
        
        c1413=-1/2*(Pi**4*(-1 + 2*nuw)*(4*D11*h*Ew*(-1 + nuw) - 4*B11*H*h*Ew*(-1 + nuw) + A11*H**2*h*Ew*(-1 + nuw) - 12*B11**2*(-1 + nuw + 2*nuw**2) + 12*A11*D11*(-1 + nuw + 2*nuw**2)))/(h**2*(-1 + nuw)*(-4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) - A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1420=(-2*nuw)/(b - b*nuw)

        c1422=(-6*(4*D11 + H*(-4*B11 + A11*H))*Ew)/(4*D11*(-6 + Pi**2)*h*Ew*(-1 + nuw) - 4*B11*H*(-6 + Pi**2)*h*Ew*(-1 + nuw) + A11*H**2*(-6 + Pi**2)*h*Ew*(-1 + nuw) - 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) + 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2))
        
        c1604=(-18*Pi*(1 + nuw)*(-1 + 2*nuw)*(A11 + 2*kA55*(-1 + nuw) - 2*A11*nuw))/(b*(-1 + nuw)*((-6 + Pi**2)*h*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1608=(Pi*(-12*A11*(-1 + nuw + 2*nuw**2)*(b**2 + 3*H*h*(-1 + 2*nuw)) + h*(-1 + nuw)*(b**2*Ew*(-7 + 12*nuw) + 144*kB55*(-1 + nuw + 2*nuw**2))))/(4*b*h*(-1 + nuw)*((-6 + Pi**2)*h*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))
        
        c1611=(-3*Pi*(-1 + 2*nuw)*(12*A11*h*(-1 + nuw + 2*nuw**2) - (-1 + nuw)*(b**2*Ew + 24*kA55*h*(1 + nuw))))/(2*b*h*(-1 + nuw)*((-6 + Pi**2)*h*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1615=((-1 + 2*nuw)*(h*(b**2*Pi**4 + 12*(-6 + Pi**2)*h**2)*Ew*(-1 + nuw) + 3*A11*Pi**2*(b**2*Pi**2 + 12*h**2)*(-1 + nuw + 2*nuw**2)))/(2*b**2*h**2*(-1 + nuw)*((-6 + Pi**2)*h*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1618=(3 - 6*nuw)/(b - b*nuw)

        c1624=(-6*Ew)/((-6 + Pi**2)*h*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2))

        c1803=(-6*Pi*Ew*(18*(H**2*kA55*Pi**2 - 4*H*kB55*Pi**2 + 4*kD55*Pi**2 + H*kA55*(-8 + Pi**2)*h - 2*kB55*(-8 + Pi**2)*h)*(1 + nuw) + b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))))/(h*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))
        
        c1807=-((18*Pi*(-1 + nuw + 2*nuw**2)*(-4*kD55*Pi**2*(3*H + 7*h)*Ew + 2*kB55*(6*H**2*Pi**2 + H*(-24 + 17*Pi**2)*h + 4*(-6 + Pi**2)*h**2)*Ew + 96*kB55**2*Pi**2*(1 + nuw) - kA55*(H*(3*H**2*Pi**2 + 2*H*(-12 + 5*Pi**2)*h + 4*(-6 + Pi**2)*h**2)*Ew + 96*kD55*Pi**2*(1 + nuw))) - b**2*Pi*(6*kA55*Pi**2*(-1 + nuw + 2*nuw**2)*((3*H + 7*h)*Ew + 24*kA55*(1 + nuw)) + Ew*(-72*kB55*Pi**2*(-1 + nuw**2) + (-6 + Pi**2)*h*(-1 + 2*nuw)*(7*h*Ew + 24*kA55*(1 + nuw)) + 3*H*((-6 + Pi**2)*h*Ew*(-1 + 2*nuw) + 12*kA55*Pi**2*(-1 + nuw**2)))))/(h*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw))))))
        
        c1812=(-9*b**2*(H*kA55 - 2*kB55)*Pi*Ew*(1 + nuw)*(8 + Pi**2*(-1 + 4*nuw)))/((-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c1816=-((-36*Pi**2*(-1 + nuw + 2*nuw**2)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw))) + 2*b**2*(6*kA55*Pi**2*(-1 + nuw + 2*nuw**2)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) + Ew*(-36*H*kA55*Pi**2*(1 + nuw) + 72*kB55*Pi**2*(1 + nuw) + (-6 + Pi**2)*h*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))))/(b*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw))))))

        c1817=-((-(b**2*Pi**4*(h*Ew + 6*kA55*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))) + 18*Pi**2*(1 + nuw)*(-4*kD55*Pi**4*h*Ew + 4*kB55*h*(H*Pi**4 + 12*h)*Ew + 24*kB55**2*Pi**4*(1 + nuw) - kA55*(H*h*(H*Pi**4 + 24*h)*Ew + 24*kD55*Pi**4*(1 + nuw))))/(h**2*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))) 
        
        c1823=(24*Ew*(18*(H**2*kA55 - 4*H*kB55 + 4*kD55)*Pi**2*h*(-1 + nuw + 2*nuw**2) + b**2*(-3*H*kA55*Pi**4*(-1 + nuw**2) + 6*kB55*Pi**4*(-1 + nuw**2) + h*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))))/(b*h*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2002=(24*nuw)/(b*Pi - 2*b*Pi*nuw)
        
        c2005=(-48*nuw)/(b*Pi*h - 2*b*Pi*h*nuw)
        
        c2010=(12*H*nuw)/(b*Pi - 2*b*Pi*nuw)

        c2014=(12*nuw)/(b - 2*b*nuw)
        
        c2019=Pi**2/h**2 + (24*(-1 + nuw))/(b**2*(-1 + 2*nuw))
        
        c2202=-((Pi*(24*kA55*(1 + nuw) + h*Ew*(1 + 12*nuw)))/(h*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))))
        
        c2205=(-12*Pi*Ew*(-1 + nuw))/(h*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))
        
        c2210=-1/2*(Pi*(24*H*kA55*(1 + nuw) + H*h*Ew*(1 + 12*nuw) - 24*kA55*h*(-1 + nuw + 2*nuw**2)))/(h*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))
        
        c2214=(-12*Ew)/((-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))
        
        c2219=(-48*Ew*nuw)/(b*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))
        
        c2221=(2*Pi**4*(-1 + nuw)*(h*Ew + 6*kA55*(1 + nuw)))/(h**2*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))
        
        c2403=(-6*Pi*(-72*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw))) + b**2*(24*kA55*(1 + nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) + Ew*(-18*H*kA55*Pi**2*(1 + nuw) + 36*kB55*Pi**2*(1 + nuw) + h*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))))))/(b*h*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2407=(-3*Pi*(2*b**4*Ew*(-1 + nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 72*(H + h)*(-1 + nuw + 2*nuw**2)*(H**2*kA55*(-6 + Pi**2)*h*Ew - 4*H*kB55*(-6 + Pi**2)*h*Ew + 4*kD55*(-6 + Pi**2)*h*Ew - 24*kB55**2*Pi**2*(1 + nuw) + 24*kA55*kD55*Pi**2*(1 + nuw)) - b**2*(-1 + 2*nuw)*(-18*H**2*kA55*Pi**2*Ew*(1 + nuw) + 12*h*(1 + nuw)*(3*kB55*(8 + Pi**2)*Ew + 2*kA55*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))) + H*(24*kA55*(1 + nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) + Ew*(36*kB55*Pi**2*(1 + nuw) + h*((-6 + Pi**2)*h*Ew - 12*kA55*(12 + Pi**2)*(1 + nuw)))))))/(b*h*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2412=-1/2*(-(b**3*Pi*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*(24*kA55*(1 + nuw) + h*Ew*(1 + 12*nuw))) + 72*b*Pi*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw))))/(h*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2416=(-12*b**2*Ew*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)))/((-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2417=(-36*b*Ew*(3*H*kA55*Pi**4*(1 + nuw) - 6*kB55*Pi**4*(1 + nuw) + 2*h*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))))/(h*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2423=-((-2*b**4*Pi**4*(-1 + nuw)*(h*Ew + 6*kA55*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) + 216*Pi**2*h**2*(-1 + nuw + 2*nuw**2)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw))) + 12*b**2*(12*kB55*Pi**2*h*Ew*(6*h*(1 - 2*nuw) + H*Pi**2*(-6 + Pi**2)*(-1 + nuw))*(1 + nuw) + 72*kB55**2*Pi**6*(-1 + nuw)*(1 + nuw)**2 - (-6 + Pi**2)*h*Ew*(12*kD55*Pi**4*(-1 + nuw**2) + h**2*(-1 + 2*nuw)*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))) - 3*kA55*Pi**2*(1 + nuw)*(24*kD55*Pi**4*(-1 + nuw**2) + h*(12*kA55*Pi**2*h*(-1 + nuw + 2*nuw**2) + Ew*(12*H*h*(1 - 2*nuw) + H**2*Pi**2*(-6 + Pi**2)*(-1 + nuw) + 2*(-6 + Pi**2)*h**2*(-1 + 2*nuw))))))/(b**2*h**2*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw))*((-6 + Pi**2)*h*Ew + 6*kA55*Pi**2*(1 + nuw)) - 18*Pi**2*(1 + nuw)*(4*H*kB55*(-6 + Pi**2)*h*Ew - 4*kD55*(-6 + Pi**2)*h*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(H**2*(-6 + Pi**2)*h*Ew + 24*kD55*Pi**2*(1 + nuw))))))

 
 

        SystemMatrixC = [[    0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [c0201,     0,     0,     0,     0, c0206,     0,     0, c0209,     0,     0,     0, c0213,     0,     0,     0,     0,     0,     0,     0,     0, c0222,     0,     0],
                         [    0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    0,     0, c0403,     0,     0,     0, c0407,     0,     0,     0,     0, c0412,     0,     0,     0, c0416, c0417,     0,     0,     0,     0,     0, c0423,     0],
                         [    0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    0, c0602,     0,     0, c0605,     0,     0,     0,     0, c0610,     0,     0,     0, c0614,     0,     0,     0,     0, c0619,     0, c0621,     0,     0,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    0,     0, c0803,     0,     0,     0, c0807,     0,     0,     0,     0, c0812,     0,     0,     0, c0816, c0817,     0,     0,     0,     0,     0, c0823,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [c1001,     0,     0,     0,     0, c1006,     0,     0, c1009,     0,     0,     0, c1013,     0,     0,     0,     0,     0,     0,     0,     0, c1022,     0,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    0,     0,     0, c1204,     0,     0,     0, c1208,     0,     0, c1211,     0,     0,     0, c1215,     0,     0,     0,     0,     0,     0,     0,     0, c1224],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [c1401,     0,     0,     0,     0, c1406,     0,     0, c1409,     0,     0,     0, c1413,     0,     0,     0,     0,     0,     0, c1420,     0, c1422,     0,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    0,     0,     0, c1604,     0,     0,     0, c1608,     0,     0, c1611,     0,     0,     0, c1615,     0,     0, c1618,     0,     0,     0,     0,     0, c1624],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0],
                         [    0,     0, c1803,     0,     0,     0, c1807,     0,     0,     0,     0, c1812,     0,     0,     0, c1816, c1817,     0,     0,     0,     0,     0, c1823,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0],
                         [    0, c2002,     0,     0, c2005,     0,     0,     0,     0, c2010,     0,     0,     0, c2014,     0,     0,     0,     0, c2019,     0,     0,     0,     0,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0],
                         [    0, c2202,     0,     0, c2205,     0,     0,     0,     0, c2210,     0,     0,     0, c2214,     0,     0,     0,     0, c2219,     0, c2221,     0,     0,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1],
                         [    0,     0, c2403,     0,     0,     0, c2407,     0,     0,     0,     0, c2412,     0,     0,     0, c2416, c2417,     0,     0,     0,     0,     0, c2423,     0],]
        return np.array(SystemMatrixC, dtype=np.float64)
    


    def calc_eigenvalues_and_eigenvectors(
        self, system_matrix: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.float64],
        NDArray[np.complex128],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Calculate eigenvalues and eigenvectors of the system matrix.

        Parameters:
        -----------
        system_matrix: NDArray          # system_matrix size (6x6) of the eigenvalue problem

        Return:
        -------
        ewC: NDArray[np.complex128]     # shape (k): Complex Eigenvalues
        ewR: NDArray[np.float64]        # shape (g): Real Eigenvalues
        evC: NDArray[np.complex128]     # shape (24, k): Complex Eigenvectors
        evR: NDArray[np.float64]        # shape (24, g): Real Eigenvectors
        sR: NDArray[np.float64]         # shape (k): Real positive eigenvalue shifts
                                        # (for numerical robustness)
        sC: NDArray[np.float64]         # shape (g): Complex positive eigenvalue shifts
                                        # (for numerical robustness)
        """
        # Calculate eigenvalues (ew) and eigenvectors (ev)
        ew, ev = np.linalg.eig(system_matrix)
        # Classify real and complex eigenvalues
        real = (ew.imag == 0) & (ew.real != 0)  # real eigenvalues
        cmplx = ew.imag > 0  # positive complex conjugates
        # Eigenvalues
        ewC = ew[cmplx]
        ewR = ew[real].real
        # Eigenvectors
        evC = ev[:, cmplx]
        evR = ev[:, real].real
        # Prepare positive eigenvalue shifts for numerical robustness
        # 1. Keep small-positive eigenvalues away from zero, to not have a near-singular matrix
        sR, sC = np.zeros(ewR.shape), np.zeros(ewC.shape)
        sR[ewR > 0], sC[ewC > 0] = -1, -1
        return ewC, ewR, evC, evR, sR, sC


    def zh(self, x: float, length: float = 0, has_foundation: bool = True) -> NDArray:
        """
        Compute bedded or free complementary solution at position x.

        Arguments
        ---------
        x : float
            Horizontal coordinate (mm).
        length : float, optional
            Segment length (mm). Default is 0.
        has_foundation : bool
            Indicates whether segment has foundation or not. Default
            is True.

        Returns
        -------
        zh : ndarray
            Complementary solution matrix (24x24) at position x.
        """
        if has_foundation:
            zh = np.concatenate(
                [
                    # Real
                    self.evR * np.exp(self.ewR * (x + length * self.sR)),
                    # Complex
                    np.exp(self.ewC.real * (x + length * self.sC))
                    * (
                        self.evC.real * np.cos(self.ewC.imag * x)
                        - self.evC.imag * np.sin(self.ewC.imag * x)
                    ),
                    # Complex
                    np.exp(self.ewC.real * (x + length * self.sC))
                    * (
                        self.evC.imag * np.cos(self.ewC.imag * x)
                        + self.evC.real * np.sin(self.ewC.imag * x)
                    ),
                ],
                axis=1,
            )
        else:
            zh = np.zeros((24,24))
            # Abbreviations
            H0101=1.
            H0102=x
            H0104=(self.B11*self.kA55)/(2*self.B11**2 - 2*self.A11*self.D11)*x**2
            H0105=(self.B11*self.kA55)/(2*self.B11**2 - 2*self.A11*self.D11)*x**2

            H0202=1.
            H0204=(self.B11*self.kA55*x)/(self.B11**2 - self.A11*self.D11)
            H0205=(self.B11*self.kA55*x)/(self.B11**2 - self.A11*self.D11)

            H0307=1.
            H0308=x - (2*self.kA55*x**3)/(self.A11*self.slab.b**2)
            H0310=(2*self.kB55*x**3)/(self.A11*self.slab.b**2)
            H0311=(2*self.kA55*x**3)/(self.A11*self.slab.b**2)
            H0312=x**2/2

            H0408=1. - (6*self.kA55*x**2)/(self.A11*self.slab.b**2)
            H0410=(6*self.kB55*x**2)/(self.A11*self.slab.b**2)
            H0411=(6*self.kA55*x**2)/(self.A11*self.slab.b**2)
            H0412=x

            H0503=1.
            H0504=(6*self.B11**2*x - 6*self.A11*self.D11*x + self.A11*self.kA55*x**3)/(6*self.B11**2 - 6*self.A11*self.D11)
            H0505=(self.A11*self.kA55*x**3)/(6*self.B11**2 - 6*self.A11*self.D11)
            H0506=-1/2*x**2

            H0604=(2*self.B11**2 - 2*self.A11*self.D11 + self.A11*self.kA55*x**2)/(2*self.B11**2 - 2*self.A11*self.D11)
            H0605=(self.A11*self.kA55*x**2)/(2*self.B11**2 - 2*self.A11*self.D11)
            H0606=-x

            H0709=1.
            H0710=x

            H0810=1.

            H0904=(self.A11*self.kA55*x**2)/(-2*self.B11**2 + 2*self.A11*self.D11)
            H0905=(-2*self.B11**2 + 2*self.A11*self.D11 + self.A11*self.kA55*x**2)/(-2*self.B11**2 + 2*self.A11*self.D11)
            H0906=x

            H1004=(self.A11*self.kA55*x)/(-self.B11**2 + self.A11*self.D11)
            H1005=(self.A11*self.kA55*x)/(-self.B11**2 + self.A11*self.D11)
            H1006=1.

            H1108=(-6*self.kA55*x**2)/(self.A11*self.slab.b**2)
            H1110=(6*self.kB55*x**2)/(self.A11*self.slab.b**2)
            H1111=1. + (6*self.kA55*x**2)/(self.A11*self.slab.b**2)
            H1112=x

            H1208=(-12*self.kA55*x)/(self.A11*self.slab.b**2)
            H1210=(12*self.kB55*x)/(self.A11*self.slab.b**2)
            H1211=(12*self.kA55*x)/(self.A11*self.slab.b**2)
            H1212=1.
            # Complementary solution matrix of free segments
            zh[0:12,0:12] = np.array(
                [
                    [H0101, H0102,     0, H0104, H0105,     0,     0,     0,     0,      0,     0,     0],
                    [    0, H0202,     0, H0204, H0205,     0,     0,     0,     0,     0,     0,     0],
                    [    0,     0,     0,     0,     0,     0, H0307, H0308,     0, H0310, H0311, H0312],
                    [    0,     0,     0,     0,     0,     0,     0, H0408,     0, H0410, H0411, H0412],
                    [    0,     0, H0503, H0504, H0505, H0506,     0,     0,    0,     0,     0,     0],
                    [    0,     0,     0, H0604, H0605, H0606,     0,     0,    0,     0,     0,     0],
                    [    0,     0,     0,     0,     0,     0,     0,     0, H0709, H0710,    0,    0],
                    [    0,     0,     0,     0,     0,     0,     0,     0,     0, H0810,    0,     0],
                    [    0,     0,     0, H0904, H0905, H0906,     0,     0,     0,     0,     0,     0],
                    [    0,     0,     0, H1004, H1005, H1006,     0,     0,     0,     0,     0,     0],
                    [    0,     0,     0,     0,     0,     0,     0, H1108,     0, H1110, H1111, H1112],
                    [    0,     0,     0,     0,     0,     0,     0, H1208,     0, H1210, H1211, H1212],
                ],dtype=np.float64)

        return zh

    def zp(self, x: float, phi: float = 0, theta: float = 0, has_foundation: bool = True, qs: float = 0) -> NDArray:
        """Return particular integral vector (24x1)."""
        # Get weight and surface  loads
        qw_x, qw_y, qw_z = decompose_to_xyz(f=self.slab.qw, phi=phi,theta=theta)
        qs_x, qs_y, qs_z = decompose_to_xyz(f=qs, phi=phi,theta=theta)
        f_x,f_y,f_z = decompose_to_xyz(f=self.weak_layer.f, phi=phi, theta=theta)
        
        z_w = self.slab.z_cog

        z_s = -self.slab.H/2
        y_s = 0
        # Clean up code by adapting mx,my,mz and f_x,f_y,f_z
        my = qw_x * z_w + qs_x * z_s
        mz = -qs_x * y_s
        mx = -qw_y * z_w + qs_z * y_s - qs_y * z_s
        if has_foundation:

            zp01=(self.slab.H*(-2*my + self.slab.H*(qs_x + qw_x)))/(4*self.kA55) + (self.weak_layer.h*(2*qs_x + 2*qw_x + f_x*self.weak_layer.h)*(1 + self.weak_layer.nu))/self.weak_layer.E

            zp02=0

            zp03=(self.weak_layer.h*(1 + self.weak_layer.nu)*(2*self.slab.b**4*np.pi**8*(2*qs_y + 2*qw_y + f_y*self.weak_layer.h)*(-1 + self.weak_layer.nu)**2 + 6*(6*self.slab.H**2*np.pi**4*(-8 + np.pi**2)*(qs_y + qw_y) + 4*self.weak_layer.h*(3*mx*np.pi**4*(-8 + np.pi**2) + self.weak_layer.h*(192*f_y*self.weak_layer.h - 48*np.pi**2*(qs_y + qw_y + f_y*self.weak_layer.h) - 6*np.pi**4*(2*qs_y + 2*qw_y + f_y*self.weak_layer.h) + np.pi**6*(2*qs_y + 2*qw_y + f_y*self.weak_layer.h))) + 3*self.slab.H*(-8 + np.pi**2)*(4*mx*np.pi**4 + self.weak_layer.h*(-32*f_y*self.weak_layer.h + np.pi**4*(4*qs_y + 4*qw_y + f_y*self.weak_layer.h))))*(self.weak_layer.h - 2*self.weak_layer.h*self.weak_layer.nu)**2 + self.slab.b**2*np.pi**4*(6*self.slab.H**2*np.pi**4*(qs_y + qw_y) + 4*self.weak_layer.h*(3*mx*np.pi**4 + self.weak_layer.h*(2*(-24 + 3*np.pi**2 + np.pi**4)*qs_y + 2*(-24 + 3*np.pi**2 + np.pi**4)*qw_y + f_y*(-48 + 3*np.pi**2 + np.pi**4)*self.weak_layer.h)) + 3*self.slab.H*(4*mx*np.pi**4 + self.weak_layer.h*(-32*f_y*self.weak_layer.h + np.pi**4*(4*qs_y + 4*qw_y + f_y*self.weak_layer.h))))*(1 - 3*self.weak_layer.nu + 2*self.weak_layer.nu**2)))/(self.weak_layer.E*(self.slab.b**2*np.pi**4*(-1 + self.weak_layer.nu) + 6*(-8 + np.pi**2)*self.weak_layer.h**2*(-1 + 2*self.weak_layer.nu))*(2*self.slab.b**2*np.pi**4*(-1 + self.weak_layer.nu) + (-96 + np.pi**4)*self.weak_layer.h**2*(-1 + 2*self.weak_layer.nu)))

            zp04=0

            zp05=(np.pi**2*self.weak_layer.h*(2*qs_z + 2*qw_z + f_z*self.weak_layer.h)*(1 + self.weak_layer.nu)*(-1 + 2*self.weak_layer.nu)*(24*self.weak_layer.h**2*(-1 + self.weak_layer.nu) + self.slab.b**2*np.pi**2*(-1 + 2*self.weak_layer.nu)))/(2*self.weak_layer.E*(24*self.weak_layer.h**2*(np.pi**2*(-1 + self.weak_layer.nu)**2 - 8*self.weak_layer.nu**2) + self.slab.b**2*np.pi**4*(1 - 3*self.weak_layer.nu + 2*self.weak_layer.nu**2)))

            zp06=0

            zp07=(6*self.weak_layer.h*(4*mx*np.pi**4 + 2*self.slab.H*np.pi**4*(qs_y + qw_y) + self.weak_layer.h*(-32*f_y*self.weak_layer.h + np.pi**4*(2*qs_y + 2*qw_y + f_y*self.weak_layer.h)))*(1 + self.weak_layer.nu)*(-1 + 2*self.weak_layer.nu))/(self.weak_layer.E*(2*self.slab.b**2*np.pi**4*(-1 + self.weak_layer.nu) + (-96 + np.pi**4)*self.weak_layer.h**2*(-1 + 2*self.weak_layer.nu)))

            zp08=0

            zp09=(2*my - self.slab.H*(qs_x + qw_x))/(2*self.kA55)

            zp10=0

            zp11=(24*mz*np.pi**2*self.weak_layer.h*(self.slab.b**2*np.pi**2 + 12*self.weak_layer.h**2)*(1 + self.weak_layer.nu))/(self.slab.b**4*np.pi**4*self.weak_layer.E + 48*self.weak_layer.h**3*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu)) + 4*self.slab.b**2*np.pi**2*self.weak_layer.h*((3 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu)))

            zp12=0

            zp13=(8*f_x*self.weak_layer.h**2*(1 + self.weak_layer.nu))/(np.pi**3*self.weak_layer.E)

            zp14=0

            zp15=(288*self.slab.b*mz*np.pi*self.weak_layer.h**3*(1 + self.weak_layer.nu))/(self.slab.b**4*np.pi**4*self.weak_layer.E + 48*self.weak_layer.h**3*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu)) + 4*self.slab.b**2*np.pi**2*self.weak_layer.h*((3 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu)))

            zp16=0

            zp17=(-16*np.pi*self.weak_layer.h**2*(1 + self.weak_layer.nu)*((6*mx + 3*self.slab.H*(qs_y + qw_y) + self.weak_layer.h*(3*qs_y + 3*qw_y + f_y*self.weak_layer.h))*(-1 + 2*self.weak_layer.nu) + self.slab.b**2*(f_y - f_y*self.weak_layer.nu)))/(self.weak_layer.E*(2*self.slab.b**2*np.pi**4*(-1 + self.weak_layer.nu) + (-96 + np.pi**4)*self.weak_layer.h**2*(-1 + 2*self.weak_layer.nu)))

            zp18=0

            zp19=(-24*self.slab.b*np.pi*self.weak_layer.h**2*(2*qs_z + 2*qw_z + f_z*self.weak_layer.h)*self.weak_layer.nu*(1 + self.weak_layer.nu)*(-1 + 2*self.weak_layer.nu))/(self.weak_layer.E*(24*self.weak_layer.h**2*(np.pi**2*(-1 + self.weak_layer.nu)**2 - 8*self.weak_layer.nu**2) + self.slab.b**2*np.pi**4*(1 - 3*self.weak_layer.nu + 2*self.weak_layer.nu**2)))

            zp20=0

            zp21=(4*f_z*self.weak_layer.h**2*(1 + self.weak_layer.nu)*(-1 + 2*self.weak_layer.nu))/(np.pi**3*self.weak_layer.E*(-1 + self.weak_layer.nu))

            zp22=0

            zp23=(12*self.slab.b*np.pi*self.weak_layer.h**2*(2*qs_y + 2*qw_y + f_y*self.weak_layer.h)*(1 + self.weak_layer.nu)*(-1 + 2*self.weak_layer.nu))/(self.weak_layer.E*(self.slab.b**2*np.pi**4*(-1 + self.weak_layer.nu) + 6*(-8 + np.pi**2)*self.weak_layer.h**2*(-1 + 2*self.weak_layer.nu)))

            zp24=0



            zp = np.array([[zp01],[zp02],[zp03],[zp04],[zp05],[zp06],[zp07],[zp08], [zp09],[zp10],[zp11],[zp12],[zp13],[zp14],[zp15],[zp16],[zp17],[zp18], [zp19],[zp20],[zp21],[zp22],[zp23],[zp24]],dtype = np.double)
        else:
            zp01=(x**2*(3*self.B11*my - 3*self.D11*(qs_x + qw_x) + self.B11*(qs_z + qw_z)*x))/(-6*self.B11**2 + 6*self.A11*self.D11)

            zp02=(x*(2*self.B11*my - 2*self.D11*(qs_x + qw_x) + self.B11*(qs_z + qw_z)*x))/(-2*self.B11**2 + 2*self.A11*self.D11)

            zp03=(x**2*(-((12*self.kB55*mx + (self.slab.b**2*self.kA55 + 12*self.kD55)*(qs_y + qw_y))/(self.slab.b**2*self.kA55*self.kA55 - 12*self.kB55**2 + 12*self.kA55*self.kD55)) + (x*(-4*mz + (qs_y + qw_y)*x))/(self.A11*self.slab.b**2)))/2

            zp04=x*(-((12*self.kB55*mx + (self.slab.b**2*self.kA55 + 12*self.kD55)*(qs_y + qw_y))/(self.slab.b**2*self.kA55*self.kA55 - 12*self.kB55**2 + 12*self.kA55*self.kD55)) + (2*x*(-3*mz + (qs_y + qw_y)*x))/(self.A11*self.slab.b**2))

            zp05=-1/24*(x**2*(12*self.B11**2*(qs_z + qw_z) - 12*self.A11*self.D11*(qs_z + qw_z) - 4*self.B11*self.kA55*(qs_x + qw_x)*x + self.A11*self.kA55*x*(4*my + (qs_z + qw_z)*x)))/((self.B11**2 - self.A11*self.D11)*self.kA55)

            zp06=-1/6*(x*(6*self.B11**2*(qs_z + qw_z) - 6*self.A11*self.D11*(qs_z + qw_z) - 3*self.B11*self.kA55*(qs_x + qw_x)*x + self.A11*self.kA55*x*(3*my + (qs_z + qw_z)*x)))/((self.B11**2 - self.A11*self.D11)*self.kA55)

            zp07=(-6*(self.kA55*mx + self.kB55*(qs_y + qw_y))*x**2)/(self.slab.b**2*self.kA55*self.kA55 - 12*self.kB55**2 + 12*self.kA55*self.kD55)

            zp08=(-12*(self.kA55*mx + self.kB55*(qs_y + qw_y))*x)/(self.slab.b**2*self.kA55*self.kA55 - 12*self.kB55**2 + 12*self.kA55*self.kD55)

            zp09=(x**2*(3*self.A11*my - 3*self.B11*(qs_x + qw_x) + self.A11*(qs_z + qw_z)*x))/(6*(self.B11**2 - self.A11*self.D11))

            zp10=(x*(2*self.A11*my - 2*self.B11*(qs_x + qw_x) + self.A11*(qs_z + qw_z)*x))/(2*(self.B11**2 - self.A11*self.D11))

            zp11=(2*x**2*(-3*mz + (qs_y + qw_y)*x))/(self.A11*self.slab.b**2)

            zp12=(6*x*(-2*mz + (qs_y + qw_y)*x))/(self.A11*self.slab.b**2)




            zp = np.array([
                [zp01],
                [zp02],
                [zp03],
                [zp04],
                [zp05],
                [zp06],
                [zp07],
                [zp08],
                [zp09],
                [zp10],
                [zp11],
                [zp12],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0]],dtype=np.float64)
        return zp
        




    def get_load_vector(self, phi: float, theta: float, qs: float = 0,has_foundation: bool = True) -> NDArray:
        """Return generalized load vector q if your pipeline needs it."""
        qw_x, qw_y, qw_z = decompose_to_xyz(self.slab.qw,phi,theta)
        qs_x,qs_y,qs_z = decompose_to_xyz(qs,phi,theta)
        f_x,f_y,f_z = decompose_to_xyz(self.weak_layer.f,phi,theta)
        z_w = self.slab.z_cog
        z_s = -self.slab.H/2
        y_s = 0
        my = -qw_x * z_w - qs_x * z_s
        mz = -qs_x * y_s
        mx = -qw_y * z_w + qs_z * y_s - qs_y * z_s

    
        q01=0

        q02=(self.slab.H*(-6 + np.pi**2)*(2*my + self.slab.H*(qs_x + qw_x))*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + 3*self.B11*(4*my*np.pi**2 - self.slab.b*f_x*self.slab.H*(-8 + np.pi**2)*self.weak_layer.h)*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2) + 6*self.D11*(-8*self.slab.b*f_x*self.weak_layer.h + np.pi**2*(2*qs_x + 2*qw_x + self.slab.b*f_x*self.weak_layer.h))*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2))/(self.slab.b*(-4*self.D11*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + 4*self.B11*self.slab.H*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) - self.A11*self.slab.H**2*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + 12*self.B11**2*np.pi**2*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2) - 12*self.A11*self.D11*np.pi**2*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2)))

        q03=0

        q04=(-3*(1 + self.weak_layer.nu)*(72*self.slab.b*f_y*self.kD55*np.pi**2*(-8 + np.pi**2)*self.weak_layer.h*(1 + self.weak_layer.nu) - 36*self.kB55*np.pi**2*(4*mx*np.pi**2 + self.slab.b*f_y*self.slab.H*(-8 + np.pi**2)*self.weak_layer.h)*(1 + self.weak_layer.nu) + 2*self.slab.b**2*np.pi**2*(qs_y + qw_y)*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu)) + self.slab.b**3*f_y*(-8 + np.pi**2)*self.weak_layer.h*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu)) + 6*np.pi**2*(self.slab.H*(-6 + np.pi**2)*(-2*mx + self.slab.H*(qs_y + qw_y))*self.weak_layer.h*self.weak_layer.E + 24*self.kD55*np.pi**2*(qs_y + qw_y)*(1 + self.weak_layer.nu))))/(self.slab.b**3*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu))**2 - 18*self.slab.b*np.pi**2*(1 + self.weak_layer.nu)*(4*self.slab.H*self.kB55*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E - 4*self.kD55*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 24*self.kB55**2*np.pi**2*(1 + self.weak_layer.nu) - self.kA55*(self.slab.H**2*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 24*self.kD55*np.pi**2*(1 + self.weak_layer.nu))))

        q05=0

        q06=(-3*(-8*self.slab.b*f_z*self.weak_layer.h + np.pi**2*(2*qs_z + 2*qw_z + self.slab.b*f_z*self.weak_layer.h))*(1 + self.weak_layer.nu))/(self.slab.b*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu)))

        q07=0

        q08=(36*np.pi**2*(1 + self.weak_layer.nu)*((-6 + np.pi**2)*(2*mx - self.slab.H*(qs_y + qw_y))*self.weak_layer.h*self.weak_layer.E + 3*self.kA55*(4*mx*np.pi**2 + self.slab.b*f_y*self.slab.H*(-8 + np.pi**2)*self.weak_layer.h)*(1 + self.weak_layer.nu) - 6*self.kB55*(-8*self.slab.b*f_y*self.weak_layer.h + np.pi**2*(2*qs_y + 2*qw_y + self.slab.b*f_y*self.weak_layer.h))*(1 + self.weak_layer.nu)))/(self.slab.b**3*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu))**2 - 18*self.slab.b*np.pi**2*(1 + self.weak_layer.nu)*(4*self.slab.H*self.kB55*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E - 4*self.kD55*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 24*self.kB55**2*np.pi**2*(1 + self.weak_layer.nu) - self.kA55*(self.slab.H**2*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 24*self.kD55*np.pi**2*(1 + self.weak_layer.nu))))

        q09=0

        q10=(-3*self.A11*(4*my*np.pi**2 - self.slab.b*f_x*self.slab.H*(-8 + np.pi**2)*self.weak_layer.h)*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2) - 2*((-6 + np.pi**2)*(2*my + self.slab.H*(qs_x + qw_x))*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + 3*self.B11*(-8*self.slab.b*f_x*self.weak_layer.h + np.pi**2*(2*qs_x + 2*qw_x + self.slab.b*f_x*self.weak_layer.h))*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2)))/(self.slab.b*(-4*self.D11*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + 4*self.B11*self.slab.H*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) - self.A11*self.slab.H**2*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + 12*self.B11**2*np.pi**2*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2) - 12*self.A11*self.D11*np.pi**2*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2)))

        q11=0

        q12=(36*mz*np.pi**2*(1 + self.weak_layer.nu)*(-1 + 2*self.weak_layer.nu))/(self.slab.b**3*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + 3*self.A11*np.pi**2*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2)))

        q13=0

        q14=(np.pi*(1 + self.weak_layer.nu)*(-1 + 2*self.weak_layer.nu)*(12*(-2*self.B11*my + self.A11*self.slab.H*my - 2*self.D11*(qs_x + qw_x) + self.B11*self.slab.H*(qs_x + qw_x))*self.weak_layer.E*(-1 + self.weak_layer.nu) + self.slab.b*f_x*(4*self.D11*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) - 4*self.B11*self.slab.H*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + self.A11*self.slab.H**2*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) - 48*self.B11**2*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2) + 48*self.A11*self.D11*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2))))/(self.slab.b*self.weak_layer.E*(-1 + self.weak_layer.nu)*(-4*self.D11*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + 4*self.B11*self.slab.H*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) - self.A11*self.slab.H**2*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + 12*self.B11**2*np.pi**2*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2) - 12*self.A11*self.D11*np.pi**2*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2)))

        q15=0

        q16=(36*mz*np.pi*(1 + self.weak_layer.nu)*(-1 + 2*self.weak_layer.nu))/(self.slab.b**2*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E*(-1 + self.weak_layer.nu) + 3*self.A11*np.pi**2*(-1 + self.weak_layer.nu + 2*self.weak_layer.nu**2)))

        q17=0

        q18=(-2*np.pi*(1 + self.weak_layer.nu)*(216*np.pi**2*(-(self.slab.H*self.kA55*mx) + 2*self.kB55*mx + self.slab.H*self.kB55*(qs_y + qw_y) - 2*self.kD55*(qs_y + qw_y))*self.weak_layer.E*(1 + self.weak_layer.nu) - 6*self.slab.b**2*(qs_y + qw_y)*self.weak_layer.E*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu)) + self.slab.b**3*f_y*(self.weak_layer.h*self.weak_layer.E + 24*self.kA55*(1 + self.weak_layer.nu))*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu)) - 18*self.slab.b*f_y*np.pi**2*(1 + self.weak_layer.nu)*(4*self.slab.H*self.kB55*self.weak_layer.h*self.weak_layer.E - 4*self.kD55*self.weak_layer.h*self.weak_layer.E + 96*self.kB55**2*(1 + self.weak_layer.nu) - self.kA55*(self.slab.H**2*self.weak_layer.h*self.weak_layer.E + 96*self.kD55*(1 + self.weak_layer.nu)))))/(self.slab.b*self.weak_layer.E*(self.slab.b**2*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu))**2 - 18*np.pi**2*(1 + self.weak_layer.nu)*(4*self.slab.H*self.kB55*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E - 4*self.kD55*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 24*self.kB55**2*np.pi**2*(1 + self.weak_layer.nu) - self.kA55*(self.slab.H**2*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 24*self.kD55*np.pi**2*(1 + self.weak_layer.nu)))))

        q19=0

        q20=0

        q21=0

        q22=(-2*np.pi*(1 + self.weak_layer.nu)*(-6*(qs_z + qw_z)*self.weak_layer.E + self.slab.b*f_z*(self.weak_layer.h*self.weak_layer.E + 24*self.kA55*(1 + self.weak_layer.nu))))/(self.slab.b*self.weak_layer.E*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu)))

        q23=0

        q24=(-36*np.pi*(1 + self.weak_layer.nu)*((-6 + np.pi**2)*(2*mx - self.slab.H*(qs_y + qw_y))*self.weak_layer.h*self.weak_layer.E + 3*self.kA55*(4*mx*np.pi**2 + self.slab.b*f_y*self.slab.H*(-8 + np.pi**2)*self.weak_layer.h)*(1 + self.weak_layer.nu) - 6*self.kB55*(-8*self.slab.b*f_y*self.weak_layer.h + np.pi**2*(2*qs_y + 2*qw_y + self.slab.b*f_y*self.weak_layer.h))*(1 + self.weak_layer.nu)))/(self.slab.b**2*((-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 6*self.kA55*np.pi**2*(1 + self.weak_layer.nu))**2 - 18*np.pi**2*(1 + self.weak_layer.nu)*(4*self.slab.H*self.kB55*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E - 4*self.kD55*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 24*self.kB55**2*np.pi**2*(1 + self.weak_layer.nu) - self.kA55*(self.slab.H**2*(-6 + np.pi**2)*self.weak_layer.h*self.weak_layer.E + 24*self.kD55*np.pi**2*(1 + self.weak_layer.nu))))
        q = np.array([
            [q01],[q02],[q03][q04],[q05],[q06],[q07],[q08],[q09],[q10],[q11],[q12],[q13],[q14],[q15],[q16],[q17][q18],[q19],[q20],[q21],[q22],[q23],[q24]],dtype=np.double),
        # TODO: Determine the load vector for unsupported structures
        return q







