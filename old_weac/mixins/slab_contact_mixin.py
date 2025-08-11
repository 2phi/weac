from __future__ import annotations

"""Mixin for slab contact."""
# Standard library imports
from functools import partial

# Third party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq

# Module imports
from old_weac.tools import calc_vertical_bc_center_of_gravity, tensile_strength_slab


class SlabContactMixin:
    """
    Mixin for handling the touchdown situation in a PST.

    Provides Methods for the calculation of substitute spring stiffnesses,
    cracklength-tresholds and element lengths.
    """

    # pylint: disable=too-many-instance-attributes

    def calc_touchdown_system(self, L, a, cf, phi, ratio=1000):
        """Calculate touchdown"""
        self.set_touchdown_attributes(L, a, cf, phi, ratio)
        self.calc_touchdown_mode()
        self.calc_touchdown_distance()

    def set_touchdown_attributes(self, L, a, cf, phi, ratio):
        """Set class attributes for touchdown consideration"""
        self.set_columnlength(L)
        self.set_cracklength(a)
        self.set_phi(phi)
        self.set_tc(cf)
        self.set_stiffness_ratio(ratio)

    def calc_touchdown_mode(self):
        """Calculate touchdown-mode from thresholds"""
        if self.touchdown:
            # Calculate stage transitions
            a1 = self.calc_a1()
            a2 = self.calc_a2()
            self.a1 = a1
            self.a2 = a2
            # Assign stage
            if self.a <= a1:
                mode = "A"
            elif a1 < self.a <= a2:
                mode = "B"
            elif a2 < self.a:
                mode = "C"
            self.mode = mode
        else:
            self.mode = "A"

    def calc_touchdown_distance(self):
        """Calculate touchdown distance"""
        if self.mode in ["A"]:
            self.td = self.calc_lA()
        elif self.mode in ["B"]:
            self.td = self.calc_lB()
        elif self.mode in ["C"]:
            self.td = self.calc_lC()

    def set_columnlength(self, L):
        """
        Set cracklength.

        Arguments
        ---------
        L : float
            Column length of a PST (mm).
        """
        self.L = L

    def set_cracklength(self, a):
        """
        Set cracklength.

        Arguments
        ---------
        a : float
            Cracklength in a PST (mm).
        """
        self.a = a

    def set_phi(self, phi):
        """
        Set inclination of the slab.

        Arguments
        ---------
        phi : float
            Inclination of the slab (°).
        """
        self.phi = phi

    def set_tc(self, cf):
        """
        Set height of the crack.

        Arguments
        ---------
        cf : float
            Collapse-factor. Ratio of the crack height to the
            uncollapsed weak-layer height.
        """
        # subtract displacement under constact load from collapsed wl height
        qn = self.calc_qn()
        # TODO: replaced with Adam formula
        # self.tc = cf * self.t - qn / self.kn
        collapse_height = 4.70 * (1 - np.exp(-self.t / 7.78))
        self.tc = collapse_height - qn / self.kn

    def set_stiffness_ratio(self, ratio=1000):
        """
        Set ratio between collapsed and uncollapsed weak-layer stiffness.

        Parameters
        ----------
        ratio : int, optional
            Stiffness ratio between collapsed and uncollapsed weak layer.
            Default is 1000.
        """
        self.ratio = ratio

    def calc_a1(self):
        """
        Calc transition lengths a1 (aAB).

        Returns
        -------
        a1 : float
            Length of the crack for transition of stage A to stage B (mm).
        """
        # Unpack variables
        bs = -(self.B11**2 / self.A11 - self.D11)
        ss = self.kA55
        L = self.L
        tc = self.tc
        qn = self.calc_qn()

        # Create polynomial expression
        def polynomial(x):
            # Spring stiffness supported segment
            kRl = self.substitute_stiffness(L - x, "supported", "rot")
            kNl = self.substitute_stiffness(L - x, "supported", "trans")
            c1 = 1 / (8 * bs)
            c2 = 1 / (2 * kRl)
            c3 = 1 / (2 * ss)
            c4 = 1 / kNl
            c5 = -tc / qn
            return c1 * x**4 + c2 * x**3 + c3 * x**2 + c4 * x + c5

        # Find root
        a1 = brentq(polynomial, L / 1000, 999 / 1000 * L)

        return a1

    def calc_a2(self):
        """
        Calc transition lengths a2 (aBC).

        Returns
        -------
        a2 : float
            Length of the crack for transition of stage B to stage C (mm).
        """
        # Unpack variables
        bs = -(self.B11**2 / self.A11 - self.D11)
        ss = self.kA55
        L = self.L
        tc = self.tc
        qn = self.calc_qn()

        # Create polynomial function
        def polynomial(x):
            # Spring stiffness supported segment
            kRl = self.substitute_stiffness(
                L - x, "supported", "rot"
            )  # rotational spring stiffness
            kNl = self.substitute_stiffness(
                L - x, "supported", "trans"
            )  # linear spring stiffness
            c1 = ss**2 * kRl * kNl * qn
            c2 = 6 * ss**2 * bs * kNl * qn
            c3 = 30 * bs * ss * kRl * kNl * qn
            c4 = 24 * bs * qn * (2 * ss**2 * kRl + 3 * bs * ss * kNl)
            c5 = 72 * bs * (bs * qn * (ss**2 + kRl * kNl) - ss**2 * kRl * kNl * tc)
            c6 = 144 * bs * ss * (bs * kRl * qn - bs * ss * kNl * tc)
            c7 = -144 * bs**2 * ss * kRl * kNl * tc
            return (
                c1 * x**6 + c2 * x**5 + c3 * x**4 + c4 * x**3 + c5 * x**2 + c6 * x + c7
            )

        # Find root
        a2 = brentq(polynomial, L / 1000, 999 / 1000 * L)

        return a2

    def calc_lA(self):
        """
        Calculate the length of the touchdown element in mode A.
        """
        lA = self.a

        return lA

    def calc_lB(self):
        """
        Calculate the length of the touchdown element in mode B.
        """
        lB = self.a

        return lB

    def calc_lC(self):
        """
        Calculate the length of the touchdown element in mode C.
        """
        # Unpack variables
        bs = -(self.B11**2 / self.A11 - self.D11)
        ss = self.kA55
        L = self.L
        a = self.a
        tc = self.tc
        qn = self.calc_qn()

        # Spring stiffness supported segment
        kRl = self.substitute_stiffness(L - a, "supported", "rot")
        kNl = self.substitute_stiffness(L - a, "supported", "trans")

        def polynomial(x):
            # Spring stiffness rested segment
            kRr = self.substitute_stiffness(a - x, "rested", "rot")
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
                    + ss * kRl * (2 * kRr * qn - ss * kNl * tc)
                )
            )
            c6 = (
                144
                * bs
                * ss
                * (bs * qn * (kRl + kRr) - kNl * tc * (bs * ss + kRl * kRr))
            )
            c7 = -144 * bs**2 * ss * kNl * tc * (kRl + kRr)
            return (
                c1 * x**6 + c2 * x**5 + c3 * x**4 + c4 * x**3 + c5 * x**2 + c6 * x + c7
            )

        # Find root
        lC = brentq(polynomial, a / 1000, 999 / 1000 * a)

        return lC

    def calc_qn(self):
        """
        Calc total surface normal load.

        Returns
        -------
        float
            Total surface normal load (N/mm).
        """
        return self.get_weight_load(self.phi)[0] + self.get_surface_load(self.phi)[0]

    def calc_qt(self):
        """
        Calc total surface tangential load.

        Returns
        -------
        float
            Total surface tangential load (N/mm).
        """
        return self.get_weight_load(self.phi)[1] + self.get_surface_load(self.phi)[1]

    def substitute_stiffness(self, L, support="rested", dof="rot"):
        """
        Calc substitute stiffness for beam on elastic foundation.

        Arguments
        ---------
        L : float
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
            self.calc_system_matrix()
            self.calc_eigensystem()

        # prepare list of segment characteristics
        segments = {
            "li": np.array([L, 0.0]),
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
