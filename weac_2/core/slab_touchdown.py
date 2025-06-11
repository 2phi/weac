import numpy as np
from typing import Literal
from scipy.optimize import brentq

from weac_2.core.eigensystem import Eigensystem
from weac_2.core.scenario import Scenario

class SlabTouchdown:
    """
    Handling the touchdown situation in a PST.
    Calculations follow paper Rosendahl et al. (2024)
        `The effect of slab touchdown on anticrack arrest in propagation saw tests`
    
    Types of Touchdown:
        `A_free_hanging` : Slab is free hanging (not in contact with the collapsed weak layer)
            touchdown_l `=` crack_l -> the unsupported segment (touchdown_l) equals the crack length
        `B_point_contact` : End of slab is in contact with the collapsed weak layer
            touchdown_l `=` crack_l -> the unsupported segment (touchdown_l) equals the crack length
        `C_in_contact` : more of the slab is in contact with the collapsed weak layer
            touchdown_l `<` crack_l -> the unsupported segment (touchdown_l) i striclty smaller than the crack length
    
    The Module does:
    1. Calculation of Zones of modes `[A_free_hanging, B_point_contact, C_in_contact]`::
    
        |+++++++++++++++++++|-------A-------|-------B-------|--------C-------- [...]
        | supported segment | free-hanging  | point contact |  in contact
                            0             `aAB`           `aBC`
        through calculation of boundary touchdown_l `aAB` and `aBC`
    
    Parameters:
    -----------
    scenario: `Scenario`
    eigensystem: `Eigensystem`
    
    Attributes:
    -----------
    aAB: float
    aAC: float
    mode: Literal["A_free_hanging", "B_point_contact", "C_in_contact"]
    touchdown_l: float
    """
    # Inputs
    scenario: Scenario
    eigensystem: Eigensystem
    
    # Attributes
    aAB: float
    aAC: float
    mode: Literal["A_free_hanging", "B_point_contact", "C_in_contact"]  # Three types of contact with collapsed weak layer
    touchdown_l: float

    def __init__(self, scenario: Scenario, eigensystem: Eigensystem):
        self.scenario = scenario
        self.eigensystem = eigensystem
        
        self._setup_touchdown_system()

    def _setup_touchdown_system(self):
        """Calculate touchdown"""
        self._calc_touchdown_mode()
        self._calc_touchdown_length()

    def _calc_touchdown_mode(self):
        """Calculate touchdown-mode from thresholds"""
        # Calculate stage transitions
        self.aAB = self._calc_aAB()
        self.aAC = self._calc_aBC()
        # Assign stage
        if self.scenario.crack_l <= self.aAB:
            mode = "A_free_hanging"
        elif self.aAB < self.scenario.crack_l <= self.aAC:
            mode = "B_point_contact"
        elif self.aAC < self.scenario.crack_l:
            mode = "C_in_contact"
        self.mode = mode

    def _calc_touchdown_length(self):
        """Calculate touchdown length"""
        if self.mode in ["A_free_hanging"]:
            self.touchdown_l = self.scenario.crack_l
        elif self.mode in ["B_point_contact"]:
            self.touchdown_l = self.scenario.crack_l
        elif self.mode in ["C_in_contact"]:
            self.touchdown_l = self._calc_touchdown_length_C()

    def _calc_aAB(self):
        """
        Calc transition lengths aAB

        Returns
        -------
        aAB : float
            Length of the crack for transition of stage A to stage B [mm]
        """
        # Unpack variables
        bs = -(self.eigensystem.B11**2 / self.eigensystem.A11 - self.eigensystem.D11)
        ss = self.eigensystem.kA55
        H = self.scenario.slab.H
        crack_h = self.scenario.crack_h
        qn = self.scenario.calc_normal_load()

        # Create polynomial expression
        def polynomial(x):
            # Spring stiffness supported segment
            kRl = self.substitute_stiffness(H - x, "supported", "rot")
            kNl = self.substitute_stiffness(H - x, "supported", "trans")
            c1 = 1 / (8 * bs)
            c2 = 1 / (2 * kRl)
            c3 = 1 / (2 * ss)
            c4 = 1 / kNl
            c5 = -crack_h / qn
            return c1 * x**4 + c2 * x**3 + c3 * x**2 + c4 * x + c5

        # Find root
        aAB = brentq(polynomial, H / 1000, 999 / 1000 * H)

        return aAB

    def _calc_aBC(self):
        """
        Calc transition lengths aBC

        Returns
        -------
        aAC : float
            Length of the crack for transition of stage B to stage C [mm]
        """
        # Unpack variables
        bs = -(self.B11**2 / self.A11 - self.D11)
        ss = self.kA55
        H = self.scenario.slab.H
        crack_h = self.scenario.crack_h
        qn = self.scenario.calc_normal_load()

        # Create polynomial function
        def polynomial(x):
            # Spring stiffness supported segment
            kRl = self.substitute_stiffness(H - x, "supported", "rot")
            kNl = self.substitute_stiffness(H - x, "supported", "trans")
            c1 = ss**2 * kRl * kNl * qn
            c2 = 6 * ss**2 * bs * kNl * qn
            c3 = 30 * bs * ss * kRl * kNl * qn
            c4 = 24 * bs * qn * (2 * ss**2 * kRl + 3 * bs * ss * kNl)
            c5 = 72 * bs * (bs * qn * (ss**2 + kRl * kNl) - ss**2 * kRl * kNl * crack_h)
            c6 = 144 * bs * ss * (bs * kRl * qn - bs * ss * kNl * crack_h)
            c7 = -144 * bs**2 * ss * kRl * kNl * crack_h
            return (
                c1 * x**6 + c2 * x**5 + c3 * x**4 + c4 * x**3 + c5 * x**2 + c6 * x + c7
            )

        # Find root
        aAC = brentq(polynomial, H / 1000, 999 / 1000 * H)

        return aAC

    def _calc_touchdown_length_C(self):
        """
        Calculate the length of the touchdown element in mode C
        when the slab is in contact.
        """
        # Unpack variables
        bs = -(self.eigensystem.B11**2 / self.eigensystem.A11 - self.eigensystem.D11)
        ss = self.eigensystem.kA55
        H = self.scenario.slab.H
        crack_l = self.scenario.crack_l
        crack_h = self.scenario.crack_h
        qn = self.scenario.calc_normal_load()

        def polynomial(x):
            # Spring stiffness supported segment
            kRl = self.substitute_stiffness(H - crack_l, "supported", "rot")
            kNl = self.substitute_stiffness(H - crack_l, "supported", "trans")
            # Spring stiffness rested segment
            kRr = self.substitute_stiffness(crack_l - x, "rested", "rot")
            # define constants
            c1 = ss**2 * kRl * kNl * qn
            c2 = 6 * ss * kNl * qn * (bs * ss + kRl * kRr)
            c3 = 30 * bs * ss * kNl * qn * (kRl + kRr)
            c4 = (
                24
                * bs
                * qn
                * (2 * ss**2 * kRl + 3 * bs * ss * kNl + 3 * kRl * kRr * kNl)
            )
            c5 = (
                72
                * bs
                * (
                    bs * qn * (ss**2 + kNl * (kRl + kRr))
                    + ss * kRl * (2 * kRr * qn - ss * kNl * crack_h)
                )
            )
            c6 = (
                144
                * bs
                * ss
                * (bs * qn * (kRl + kRr) - kNl * crack_h * (bs * ss + kRl * kRr))
            )
            c7 = -144 * bs**2 * ss * kNl * crack_h * (kRl + kRr)
            return (
                c1 * x**6 + c2 * x**5 + c3 * x**4 + c4 * x**3 + c5 * x**2 + c6 * x + c7
            )

        # Find root
        lC = brentq(polynomial, crack_l / 1000, 999 / 1000 * crack_l)

        return lC

    def substitute_stiffness(self, H, support="rested", dof="rot"):
        """
        Calc substitute stiffness for beam on elastic foundation.

        Arguments
        ---------
        H : float
            Total length of the PST-column (mm).
        support : string
            Type of segment foundation. Defaults to 'rested'.
        dof : string
            Type of substitute spring, either 'rot' or 'trans'. Defaults to 'rot'.

        Returns
        -------
        k : stiffness of substitute spring.
        """
        # adjust system to substitute system
        if dof in ["rot"]:
            tempsys = self.system
            self.system = "rot"
        if dof in ["trans"]:
            tempsys = self.system
            self.system = "trans"

        # Change eigensystem for rested segment
        if support in ["rested"]:
            tempkn = self.kn
            tempkt = self.kt
            self.kn = self.ratio * self.kn
            self.kt = self.ratio * self.kt
            K = self.eigensystem._assemble_system_matrix()
            self.eigensystem._calc_eigenvalues_and_eigenvectors(K)

        # prepare list of segment characteristics
        segments = {
            "li": np.array([H, 0.0]),
            "mi": np.array([0]),
            "ki": np.array([True, True]),
        }
        # solve system of equations
        constants = self.assemble_and_solve(phi=0, **segments)
        # calculate stiffness
        _, z_pst, _ = self.rasterize_solution(C=constants, phi=0, num=1, **segments)
        if dof in ["rot"]:
            k = abs(1 / self.psi(z_pst)[0])
        if dof in ["trans"]:
            k = abs(1 / self.w(z_pst)[0])

        # Reset to previous system and eigensystem
        self.system = tempsys
        if support in ["rested"]:
            self.kn = tempkn
            self.kt = tempkt
            self.calc_system_matrix()
            self.calc_eigensystem()

        return k
