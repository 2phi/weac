from __future__ import annotations

"""Mixin for solution."""
# Standard library imports
from functools import partial

# Third party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq

# Module imports
from weac.tools import calc_vertical_bc_center_of_gravity, tensile_strength_slab


class SolutionMixin:
    """
    Mixin for the solution of boundary value problems.

    Provides methods for the assembly of the system of equations
    and for the computation of the free constants.
    """

    def calc_segments(
        self,
        li: list[float] | list[int] | bool = False,
        mi: list[float] | list[int] | bool = False,
        ki: list[bool] | bool = False,
        k0: list[bool] | bool = False,
        L: float = 1e4,
        a: float = 0,
        m: float = 0,
        phi: float = 0,
        cf: float = 0.5,
        ratio: float = 1000,
        **kwargs,
    ):
        """
        Assemble lists defining the segments.

        This includes length (li), foundation (ki, k0), and skier
        weight (mi).

        Arguments
        ---------
        li : squence, optional
            List of lengths of segements(mm). Used for system 'skiers'.
        mi : squence, optional
            List of skier weigths (kg) at segement boundaries. Used for
            system 'skiers'.
        ki : squence, optional
            List of one bool per segement indicating whether segement
            has foundation (True) or not (False) in the cracked state.
            Used for system 'skiers'.
        k0 : squence, optional
            List of one bool per segement indicating whether segement
            has foundation(True) or not (False) in the uncracked state.
            Used for system 'skiers'.
        L : float, optional
            Total length of model (mm). Used for systems 'pst-', '-pst',
            'vpst-', '-vpst', and 'skier'.
        a : float, optional
            Crack length (mm). Used for systems 'pst-', '-pst',  'pst-',
            '-pst', and 'skier'.
        phi : float, optional
            Inclination (degree).
        m : float, optional
            Weight of skier (kg) in the axial center of the model.
            Used for system 'skier'.
        cf : float, optional
            Collapse factor. Ratio of the crack height to the uncollapsed
            weak-layer height. Used for systems 'pst-', '-pst'. Default is 0.5.
        ratio : float, optional
            Stiffness ratio between collapsed and uncollapsed weak layer.
            Default is 1000.

        Returns
        -------
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.
        """

        _ = kwargs  # Unused arguments

        # Precompute touchdown properties
        self.calc_touchdown_system(L=L, a=a, cf=cf, phi=phi, ratio=ratio)

        # Assemble list defining the segments
        if self.system == "skiers":
            li = np.array(li)  # Segment lengths
            mi = np.array(mi)  # Skier weights
            ki = np.array(ki)  # Crack
            k0 = np.array(k0)  # No crack
        elif self.system == "pst-":
            li = np.array([L - self.a, self.td])  # Segment lengths
            mi = np.array([0])  # Skier weights
            ki = np.array([True, False])  # Crack
            k0 = np.array([True, True])  # No crack
        elif self.system == "-pst":
            li = np.array([self.td, L - self.a])  # Segment lengths
            mi = np.array([0])  # Skier weights
            ki = np.array([False, True])  # Crack
            k0 = np.array([True, True])  # No crack
        elif self.system == "vpst-":
            li = np.array([L - a, self.td])  # Segment lengths
            mi = np.array([0])  # Skier weights
            ki = np.array([True, False])  # Crack
            k0 = np.array([True, True])  # No crack
        elif self.system == "-vpst":
            li = np.array([self.td, L - a])  # Segment lengths
            mi = np.array([0])  # Skier weights
            ki = np.array([False, True])  # Crack
            k0 = np.array([True, True])  # No crack
        elif self.system == "skier":
            lb = (L - self.a) / 2  # Half bedded length
            lf = self.a / 2  # Half free length
            li = np.array([lb, lf, lf, lb])  # Segment lengths
            mi = np.array([0, m, 0])  # Skier weights
            ki = np.array([True, False, False, True])  # Crack
            k0 = np.array([True, True, True, True])  # No crack
        else:
            raise ValueError(f"System {self.system} is not implemented.")

        # Fill dictionary
        segments = {
            "nocrack": {"li": li, "mi": mi, "ki": k0},
            "crack": {"li": li, "mi": mi, "ki": ki},
            "both": {"li": li, "mi": mi, "ki": ki, "k0": k0},
        }
        return segments

    def assemble_and_solve(self, phi, li, mi, ki):
        """
        Compute free constants for arbitrary beam assembly.

        Assemble LHS from supported and unsupported segments in the form
        [  ]   [ zh1  0   0  ...  0   0   0  ][   ]   [    ]   [     ]  left
        [  ]   [ zh1 zh2  0  ...  0   0   0  ][   ]   [    ]   [     ]  mid
        [  ]   [  0  zh2 zh3 ...  0   0   0  ][   ]   [    ]   [     ]  mid
        [z0] = [ ... ... ... ... ... ... ... ][ C ] + [ zp ] = [ rhs ]  mid
        [  ]   [  0   0   0  ... zhL zhM  0  ][   ]   [    ]   [     ]  mid
        [  ]   [  0   0   0  ...  0  zhM zhN ][   ]   [    ]   [     ]  mid
        [  ]   [  0   0   0  ...  0   0  zhN ][   ]   [    ]   [     ]  right
        and solve for constants C.

        Arguments
        ---------
        phi : float
            Inclination (degrees).
        li : ndarray
            List of lengths of segements (mm).
        mi : ndarray
            List of skier weigths (kg) at segement boundaries.
        ki : ndarray
            List of one bool per segement indicating whether segement
            has foundation (True) or not (False).

        Returns
        -------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        """
        # --- CATCH ERRORS ----------------------------------------------------

        # No foundation
        if not any(ki):
            raise ValueError("Provide at least one supported segment.")
        # Mismatch of number of segements and transisions
        if len(li) != len(ki) or len(li) - 1 != len(mi):
            raise ValueError(
                "Make sure len(li)=N, len(ki)=N, and "
                "len(mi)=N-1 for a system of N segments."
            )

        if self.system not in ["pst-", "-pst", "vpst-", "-vpst", "rot", "trans"]:
            # Boundary segments must be on foundation for infinite BCs
            if not all([ki[0], ki[-1]]):
                raise ValueError(
                    "Provide supported boundary segments in "
                    "order to account for infinite extensions."
                )
            # Make sure infinity boundary conditions are far enough from skiers
            if li[0] < 5e3 or li[-1] < 5e3:
                print(
                    (
                        "WARNING: Boundary segments are short. Make sure "
                        "the complementary solution has decayed to the "
                        "boundaries."
                    )
                )

        # --- PREPROCESSING ---------------------------------------------------

        # Determine size of linear system of equations
        nS = len(li)  # Number of beam segments

        nDOF = 6  # Number of free constants per segment

        # Add dummy segment if only one segment provided
        if nS == 1:
            li.append(0)
            ki.append(True)
            mi.append(0)
            nS = 2

        # Assemble position vector
        pi = np.full(nS, "m")
        pi[0], pi[-1] = "l", "r"

        # Initialize matrices
        zh0 = np.zeros([nS * 6, nS * nDOF])
        zp0 = np.zeros([nS * 6, 1])
        rhs = np.zeros([nS * 6, 1])

        # --- ASSEMBLE LINEAR SYSTEM OF EQUATIONS -----------------------------

        # Loop through segments to assemble left-hand side
        for i in range(nS):
            # Length, foundation and position of segment i
            l, k, pos = li[i], ki[i], pi[i]
            # Transmission conditions at left and right segment ends
            zhi = self.eqs(
                zl=self.zh(x=0, l=l, bed=k), zr=self.zh(x=l, l=l, bed=k), k=k, pos=pos
            )
            zpi = self.eqs(
                zl=self.zp(x=0, phi=phi, bed=k),
                zr=self.zp(x=l, phi=phi, bed=k),
                k=k,
                pos=pos,
            )
            # Rows for left-hand side assembly
            start = 0 if i == 0 else 3
            stop = 6 if i == nS - 1 else 9
            # Assemble left-hand side
            zh0[(6 * i - start) : (6 * i + stop), i * nDOF : (i + 1) * nDOF] = zhi
            zp0[(6 * i - start) : (6 * i + stop)] += zpi

        # Loop through loads to assemble right-hand side
        for i, m in enumerate(mi, start=1):
            # Get skier loads
            Fn, Ft = self.get_skier_load(m, phi)
            # Right-hand side for transmission from segment i-1 to segment i
            rhs[6 * i : 6 * i + 3] = np.vstack([Ft, -Ft * self.h / 2, Fn])
        # Set rhs so that complementary integral vanishes at boundaries
        if self.system not in ["pst-", "-pst", "rested"]:
            rhs[:3] = self.bc(self.zp(x=0, phi=phi, bed=ki[0]))
            rhs[-3:] = self.bc(self.zp(x=li[-1], phi=phi, bed=ki[-1]))

        # Set rhs for vertical faces
        if self.system in ["vpst-", "-vpst"]:
            # Calculate center of gravity and mass of
            # added or cut off slab segement
            xs, zs, m = calc_vertical_bc_center_of_gravity(self.slab, phi)
            # Convert slope angle to radians
            phi = np.deg2rad(phi)
            # Translate inbto section forces and moments
            N = -self.g * m * np.sin(phi)
            M = -self.g * m * (xs * np.cos(phi) + zs * np.sin(phi))
            V = self.g * m * np.cos(phi)
            # Add to right-hand side
            rhs[:3] = np.vstack([N, M, V])  # left end
            rhs[-3:] = np.vstack([N, M, V])  # right end

        # Loop through segments to set touchdown conditions at rhs
        for i in range(nS):
            # Length, foundation and position of segment i
            l, k, pos = li[i], ki[i], pi[i]
            # Set displacement BC in stage B
            if not k and bool(self.mode in ["B"]):
                if i == 0:
                    rhs[:3] = np.vstack([0, 0, self.tc])
                if i == (nS - 1):
                    rhs[-3:] = np.vstack([0, 0, self.tc])
            # Set normal force and displacement BC for stage C
            if not k and bool(self.mode in ["C"]):
                N = self.calc_qt() * (self.a - self.td)
                if i == 0:
                    rhs[:3] = np.vstack([-N, 0, self.tc])
                if i == (nS - 1):
                    rhs[-3:] = np.vstack([N, 0, self.tc])

        # Rhs for substitute spring stiffness
        if self.system in ["rot"]:
            # apply arbitrary moment of 1 at left boundary
            rhs = rhs * 0
            rhs[1] = 1
        if self.system in ["trans"]:
            # apply arbitrary force of 1 at left boundary
            rhs = rhs * 0
            rhs[2] = 1

        # --- SOLVE -----------------------------------------------------------
        # Solve z0 = zh0*C + zp0 = rhs for constants, i.e. zh0*C = rhs - zp0
        C = np.linalg.solve(zh0, rhs - zp0)
        # Sort (nDOF = 6) constants for each segment into columns of a matrix
        return C.reshape([-1, nDOF]).T

    def bc(self, z, k=False, pos="mid"):
        """
        Provide equations for free (pst) or infinite (skiers) ends.

        Arguments
        ---------
        z : ndarray
            Solution vector (6x1) at a certain position x.
        l : float, optional
            Length of the segment in consideration. Default is zero.
        k : boolean
            Indicates whether segment has foundation(True) or not (False).
            Default is False.
        pos : {'left', 'mid', 'right', 'l', 'm', 'r'}, optional
            Determines whether the segement under consideration
            is a left boundary segement (left, l), one of the
            center segement (mid, m), or a right boundary
            segement (right, r). Default is 'mid'.

        Returns
        -------
        bc : ndarray
            Boundary condition vector (lenght 3) at position x.
        """

        # Set boundary conditions for PST-systems
        if self.system in ["pst-", "-pst"]:
            if not k:
                if self.mode in ["A"]:
                    # Free end
                    bc = np.array([self.N(z), self.M(z), self.V(z)])
                elif self.mode in ["B"] and pos in ["r", "right"]:
                    # Touchdown right
                    bc = np.array([self.N(z), self.M(z), self.w(z)])
                elif self.mode in ["B"] and pos in ["l", "left"]:  # Kann dieser Block
                    # Touchdown left                                # verschwinden? Analog zu 'A'
                    bc = np.array([self.N(z), self.M(z), self.w(z)])
                elif self.mode in ["C"] and pos in ["r", "right"]:
                    # Spring stiffness
                    kR = self.substitute_stiffness(self.a - self.td, "rested", "rot")
                    # Touchdown right
                    bc = np.array([self.N(z), self.M(z) + kR * self.psi(z), self.w(z)])
                elif self.mode in ["C"] and pos in ["l", "left"]:
                    # Spring stiffness
                    kR = self.substitute_stiffness(self.a - self.td, "rested", "rot")
                    # Touchdown left
                    bc = np.array([self.N(z), self.M(z) - kR * self.psi(z), self.w(z)])
            else:
                # Free end
                bc = np.array([self.N(z), self.M(z), self.V(z)])
        # Set boundary conditions for PST-systems with vertical faces
        elif self.system in ["-vpst", "vpst-"]:
            bc = np.array([self.N(z), self.M(z), self.V(z)])
        # Set boundary conditions for SKIER-systems
        elif self.system in ["skier", "skiers"]:
            # Infinite end (vanishing complementary solution)
            bc = np.array([self.u(z, z0=0), self.w(z), self.psi(z)])
        # Set boundary conditions for substitute spring calculus
        elif self.system in ["rot", "trans"]:
            bc = np.array([self.N(z), self.M(z), self.V(z)])
        else:
            raise ValueError(
                f"Boundary conditions not defined forsystem of type {self.system}."
            )

        return bc

    def eqs(self, zl, zr, k=False, pos="mid"):
        """
        Provide boundary or transmission conditions for beam segments.

        Arguments
        ---------
        zl : ndarray
            Solution vector (6x1) at left end of beam segement.
        zr : ndarray
            Solution vector (6x1) at right end of beam segement.
        k : boolean
            Indicates whether segment has foundation(True) or not (False).
            Default is False.
        pos: {'left', 'mid', 'right', 'l', 'm', 'r'}, optional
            Determines whether the segement under consideration
            is a left boundary segement (left, l), one of the
            center segement (mid, m), or a right boundary
            segement (right, r). Default is 'mid'.

        Returns
        -------
        eqs : ndarray
            Vector (of length 9) of boundary conditions (3) and
            transmission conditions (6) for boundary segements
            or vector of transmission conditions (of length 6+6)
            for center segments.
        """
        if pos in ("l", "left"):
            eqs = np.array(
                [
                    self.bc(zl, k, pos)[0],  # Left boundary condition
                    self.bc(zl, k, pos)[1],  # Left boundary condition
                    self.bc(zl, k, pos)[2],  # Left boundary condition
                    self.u(zr, z0=0),  # ui(xi = li)
                    self.w(zr),  # wi(xi = li)
                    self.psi(zr),  # psii(xi = li)
                    self.N(zr),  # Ni(xi = li)
                    self.M(zr),  # Mi(xi = li)
                    self.V(zr),
                ]
            )  # Vi(xi = li)
        elif pos in ("m", "mid"):
            eqs = np.array(
                [
                    -self.u(zl, z0=0),  # -ui(xi = 0)
                    -self.w(zl),  # -wi(xi = 0)
                    -self.psi(zl),  # -psii(xi = 0)
                    -self.N(zl),  # -Ni(xi = 0)
                    -self.M(zl),  # -Mi(xi = 0)
                    -self.V(zl),  # -Vi(xi = 0)
                    self.u(zr, z0=0),  # ui(xi = li)
                    self.w(zr),  # wi(xi = li)
                    self.psi(zr),  # psii(xi = li)
                    self.N(zr),  # Ni(xi = li)
                    self.M(zr),  # Mi(xi = li)
                    self.V(zr),
                ]
            )  # Vi(xi = li)
        elif pos in ("r", "right"):
            eqs = np.array(
                [
                    -self.u(zl, z0=0),  # -ui(xi = 0)
                    -self.w(zl),  # -wi(xi = 0)
                    -self.psi(zl),  # -psii(xi = 0)
                    -self.N(zl),  # -Ni(xi = 0)
                    -self.M(zl),  # -Mi(xi = 0)
                    -self.V(zl),  # -Vi(xi = 0)
                    self.bc(zr, k, pos)[0],  # Right boundary condition
                    self.bc(zr, k, pos)[1],  # Right boundary condition
                    self.bc(zr, k, pos)[2],
                ]
            )  # Right boundary condition
        else:
            raise ValueError(
                (
                    f"Invalid position argument {pos} given. "
                    "Valid segment positions are l, m, and r, "
                    "or left, mid and right."
                )
            )
        return eqs
