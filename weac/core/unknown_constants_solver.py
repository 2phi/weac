"""
This module defines the system model for the WEAC simulation.
The system model is the heart of the WEAC simulation. All data sources are bundled into the system model.
The system model initializes and calculates all the parameterizations and passes relevant data to the different components.

We utilize the pydantic library to define the system model.
"""

import logging
from typing import Literal, Optional

import numpy as np
from numpy.linalg import LinAlgError

from weac.constants import G_MM_S2
from weac.core.eigensystem import Eigensystem
from weac.core.field_quantities import FieldQuantities
from weac.core.scenario import Scenario

# from weac.constants import G_MM_S2, LSKI_MM
from weac.utils.misc import decompose_to_normal_tangential, get_skier_point_load

logger = logging.getLogger(__name__)


class UnknownConstantsSolver:
    """
    This class solves the unknown constants for the WEAC simulation.
    """

    @classmethod
    def solve_for_unknown_constants(
        cls,
        scenario: Scenario,
        eigensystem: Eigensystem,
        system_type: Literal[
            "skier", "skiers", "pst-", "-pst", "vpst-", "-vpst", "rot", "trans"
        ],
        touchdown_distance: Optional[float] = None,
        touchdown_mode: Optional[
            Literal["A_free_hanging", "B_point_contact", "C_in_contact"]
        ] = None,
        collapsed_weak_layer_kR: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute free constants *C* for system. \\
        Assemble LHS from supported and unsupported segments in the form::
        
            [  ]   [ zh1  0   0  ...  0   0   0  ][   ]   [    ]   [     ]  (left)
            [  ]   [ zh1 zh2  0  ...  0   0   0  ][   ]   [    ]   [     ]  (mid)
            [  ]   [  0  zh2 zh3 ...  0   0   0  ][   ]   [    ]   [     ]  (mid)
            [z0] = [ ... ... ... ... ... ... ... ][ C ] + [ zp ] = [ rhs ]  (mid)
            [  ]   [  0   0   0  ... zhL zhM  0  ][   ]   [    ]   [     ]  (mid)
            [  ]   [  0   0   0  ...  0  zhM zhN ][   ]   [    ]   [     ]  (mid)
            [  ]   [  0   0   0  ...  0   0  zhN ][   ]   [    ]   [     ]  (right)
        
        and solve for constants C.
        
        Returns
        -------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        """
        logger.debug("Starting solve unknown constants")
        phi = scenario.phi
        qs = scenario.surface_load
        li = scenario.li
        ki = scenario.ki
        mi = scenario.mi

        # Determine size of linear system of equations
        nS = len(li)  # Number of beam segments
        nDOF = 6  # Number of free constants per segment
        logger.debug(f"Number of segments: {nS}, DOF per segment: {nDOF}")

        # Assemble position vector
        pi = np.full(nS, "m")
        pi[0], pi[-1] = "length", "r"

        # Initialize matrices
        Zh0 = np.zeros([nS * 6, nS * nDOF])
        Zp0 = np.zeros([nS * 6, 1])
        rhs = np.zeros([nS * 6, 1])
        logger.debug(
            f"Initialized Zh0 shape: {Zh0.shape}, Zp0 shape: {Zp0.shape}, rhs shape: {rhs.shape}"
        )

        # LHS: Transmission & Boundary Conditions between segments
        for i in range(nS):
            # Length, foundation and position of segment i
            length, has_foundation, pos = li[i], ki[i], pi[i]

            logger.debug(
                f"Assembling segment {i}: length={length}, has_foundation={has_foundation}, pos={pos}"
            )
            # Matrix of Size one of: (l: [9,6], m: [12,6], r: [9,6])
            Zhi = cls._setup_conditions(
                zl=eigensystem.zh(x=0, length=length, has_foundation=has_foundation),
                zr=eigensystem.zh(
                    x=length, length=length, has_foundation=has_foundation
                ),
                eigensystem=eigensystem,
                has_foundation=has_foundation,
                pos=pos,
                touchdown_mode=touchdown_mode,
                system_type=system_type,
                collapsed_weak_layer_kR=collapsed_weak_layer_kR,
            )
            # Vector of Size one of: (l: [9,1], m: [12,1], r: [9,1])
            zpi = cls._setup_conditions(
                zl=eigensystem.zp(x=0, phi=phi, has_foundation=has_foundation, qs=qs),
                zr=eigensystem.zp(
                    x=length, phi=phi, has_foundation=has_foundation, qs=qs
                ),
                eigensystem=eigensystem,
                has_foundation=has_foundation,
                pos=pos,
                touchdown_mode=touchdown_mode,
                system_type=system_type,
                collapsed_weak_layer_kR=collapsed_weak_layer_kR,
            )

            # Rows for left-hand side assembly
            start = 0 if i == 0 else 3
            stop = 6 if i == nS - 1 else 9
            # Assemble left-hand side
            Zh0[(6 * i - start) : (6 * i + stop), i * nDOF : (i + 1) * nDOF] = Zhi
            Zp0[(6 * i - start) : (6 * i + stop)] += zpi
            logger.debug(f"Segment {i}: Zhi shape: {Zhi.shape}, zpi shape: {zpi.shape}")

        # Loop through loads to assemble right-hand side
        for i, m in enumerate(mi, start=1):
            # Get skier point-load
            F = get_skier_point_load(m)
            Fn, Ft = decompose_to_normal_tangential(f=F, phi=phi)
            # Right-hand side for transmission from segment i-1 to segment i
            rhs[6 * i : 6 * i + 3] = np.vstack([Ft, -Ft * scenario.slab.H / 2, Fn])
            logger.debug(f"Load {i}: m={m}, F={F}, Fn={Fn}, Ft={Ft}")
            logger.debug(f"RHS {rhs[6 * i : 6 * i + 3]}")
        # Set RHS so that Complementary Integral vanishes at boundaries
        if system_type not in ["pst-", "-pst", "rested"]:
            logger.debug(f"Pre RHS {rhs[:3]}")
            rhs[:3] = cls._boundary_conditions(
                eigensystem.zp(x=0, phi=phi, has_foundation=ki[0], qs=qs),
                eigensystem,
                False,
                "mid",
                system_type,
                touchdown_mode,
                collapsed_weak_layer_kR,
            )
            logger.debug(f"Post RHS {rhs[:3]}")
            rhs[-3:] = cls._boundary_conditions(
                eigensystem.zp(x=li[-1], phi=phi, has_foundation=ki[-1], qs=qs),
                eigensystem,
                False,
                "mid",
                system_type,
                touchdown_mode,
                collapsed_weak_layer_kR,
            )
            logger.debug(f"Post RHS {rhs[-3:]}")
            logger.debug("Set complementary integral vanishing at boundaries.")

        # Set rhs for vertical faces
        if system_type in ["vpst-", "-vpst"]:
            # Calculate center of gravity and mass of added or cut off slab segement
            x_cog, z_cog, m = scenario.slab.calc_vertical_center_of_gravity(phi)
            logger.debug(
                f"Vertical center of gravity: x_cog={x_cog}, z_cog={z_cog}, m={m}"
            )
            # Convert slope angle to radians
            phi = np.deg2rad(phi)
            # Translate into section forces and moments
            N = -G_MM_S2 * m * np.sin(phi)
            M = -G_MM_S2 * m * (x_cog * np.cos(phi) + z_cog * np.sin(phi))
            V = G_MM_S2 * m * np.cos(phi)
            # Add to right-hand side
            rhs[:3] = np.vstack([N, M, V])  # left end
            rhs[-3:] = np.vstack([N, M, V])  # right end
            logger.debug(f"Vertical faces: N={N}, M={M}, V={V}")

        # Loop through segments to set touchdown conditions at rhs
        for i in range(nS):
            # Length, foundation and position of segment i
            length, has_foundation, pos = li[i], ki[i], pi[i]
            # Set displacement BC in stage B
            if not has_foundation and bool(touchdown_mode in ["B_point_contact"]):
                if i == 0:
                    rhs[:3] = np.vstack([0, 0, scenario.crack_h])
                if i == (nS - 1):
                    rhs[-3:] = np.vstack([0, 0, scenario.crack_h])
            # Set normal force and displacement BC for stage C
            if not has_foundation and bool(touchdown_mode in ["C_in_contact"]):
                N = scenario.qt * (scenario.crack_length - touchdown_distance)
                if i == 0:
                    rhs[:3] = np.vstack([-N, 0, scenario.crack_h])
                if i == (nS - 1):
                    rhs[-3:] = np.vstack([N, 0, scenario.crack_h])

        # Rhs for substitute spring stiffness
        if system_type in ["rot"]:
            # apply arbitrary moment of 1 at left boundary
            rhs = rhs * 0
            rhs[1] = 1
        if system_type in ["trans"]:
            # apply arbitrary force of 1 at left boundary
            rhs = rhs * 0
            rhs[2] = 1

        # Solve z0 = Zh0*C + Zp0 = rhs for constants, i.e. Zh0*C = rhs - Zp0
        try:
            C = np.linalg.solve(Zh0, rhs - Zp0)
        except LinAlgError as e:
            raise e
        # Sort (nDOF = 6) constants for each segment into columns of a matrix
        return C.reshape([-1, nDOF]).T

    @classmethod
    def _setup_conditions(
        cls,
        zl: np.ndarray,
        zr: np.ndarray,
        eigensystem: Eigensystem,
        has_foundation: bool,
        pos: Literal["l", "r", "m", "left", "right", "mid"],
        system_type: Literal[
            "skier", "skiers", "pst-", "-pst", "vpst-", "-vpst", "rot", "trans"
        ],
        touchdown_mode: Optional[
            Literal["A_free_hanging", "B_point_contact", "C_in_contact"]
        ] = None,
        collapsed_weak_layer_kR: Optional[float] = None,
    ) -> np.ndarray:
        """
        Provide boundary or transmission conditions for beam segments.

        Arguments
        ---------
        zl : ndarray
            Solution vector (6x1) or (6x6) at left end of beam segement.
        zr : ndarray
            Solution vector (6x1) or (6x6) at right end of beam segement.
        has_foundation : boolean
            Indicates whether segment has foundation(True) or not (False).
            Default is False.
        pos: {'left', 'mid', 'right', 'l', 'm', 'r'}, optional
            Determines whether the segement under consideration
            is a left boundary segement (left, l), one of the
            center segement (mid, m), or a right boundary
            segement (right, r). Default is 'mid'.

        Returns
        -------
        conditions : ndarray
            `zh`: Matrix of Size one of: (`l: [9,6], m: [12,6], r: [9,6]`)

            `zp`: Vector of Size one of: (`l: [9,1], m: [12,1], r: [9,1]`)
        """
        fq = FieldQuantities(eigensystem=eigensystem)
        if pos in ("l", "left"):
            bcs = cls._boundary_conditions(
                zl,
                eigensystem,
                has_foundation,
                pos,
                system_type,
                touchdown_mode,
                collapsed_weak_layer_kR,
            )  # Left boundary condition
            conditions = np.array(
                [
                    bcs[0],
                    bcs[1],
                    bcs[2],
                    fq.u(zr, h0=0),  # ui(xi = li)
                    fq.w(zr),  # wi(xi = li)
                    fq.psi(zr),  # psii(xi = li)
                    fq.N(zr),  # Ni(xi = li)
                    fq.M(zr),  # Mi(xi = li)
                    fq.V(zr),  # Vi(xi = li)
                ]
            )
        elif pos in ("m", "mid"):
            conditions = np.array(
                [
                    -fq.u(zl, h0=0),  # -ui(xi = 0)
                    -fq.w(zl),  # -wi(xi = 0)
                    -fq.psi(zl),  # -psii(xi = 0)
                    -fq.N(zl),  # -Ni(xi = 0)
                    -fq.M(zl),  # -Mi(xi = 0)
                    -fq.V(zl),  # -Vi(xi = 0)
                    fq.u(zr, h0=0),  # ui(xi = li)
                    fq.w(zr),  # wi(xi = li)
                    fq.psi(zr),  # psii(xi = li)
                    fq.N(zr),  # Ni(xi = li)
                    fq.M(zr),  # Mi(xi = li)
                    fq.V(zr),  # Vi(xi = li)
                ]
            )
        elif pos in ("r", "right"):
            bcs = cls._boundary_conditions(
                zr,
                eigensystem,
                has_foundation,
                pos,
                system_type,
                touchdown_mode,
                collapsed_weak_layer_kR,
            )  # Right boundary condition
            conditions = np.array(
                [
                    -fq.u(zl, h0=0),  # -ui(xi = 0)
                    -fq.w(zl),  # -wi(xi = 0)
                    -fq.psi(zl),  # -psii(xi = 0)
                    -fq.N(zl),  # -Ni(xi = 0)
                    -fq.M(zl),  # -Mi(xi = 0)
                    -fq.V(zl),  # -Vi(xi = 0)
                    bcs[0],
                    bcs[1],
                    bcs[2],
                ]
            )
        logger.debug(f"Boundary Conditions at pos {pos}: {conditions.shape}")
        return conditions

    @classmethod
    def _boundary_conditions(
        cls,
        z,
        eigensystem: Eigensystem,
        has_foundation: bool,
        pos: Literal["l", "r", "m", "left", "right", "mid"],
        system_type: Literal[
            "skier", "skiers", "pst-", "-pst", "vpst-", "-vpst", "rot", "trans"
        ],
        touchdown_mode: Optional[
            Literal["A_free_hanging", "B_point_contact", "C_in_contact"]
        ] = None,
        collapsed_weak_layer_kR: Optional[float] = None,
    ):
        """
        Provide equations for free (pst) or infinite (skiers) ends.

        Arguments
        ---------
        z : ndarray
            Solution vector (6x1) at a certain position x.
        l : float, optional
            Length of the segment in consideration. Default is zero.
        has_foundation : boolean
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
        fq = FieldQuantities(eigensystem=eigensystem)
        # Set boundary conditions for PST-systems
        if system_type in ["pst-", "-pst"]:
            if not has_foundation:
                if touchdown_mode in ["A_free_hanging"]:
                    # Free end
                    bc = np.array([fq.N(z), fq.M(z), fq.V(z)])
                elif touchdown_mode in ["B_point_contact"] and pos in ["r", "right"]:
                    # Touchdown right
                    bc = np.array([fq.N(z), fq.M(z), fq.w(z)])
                elif touchdown_mode in ["B_point_contact"] and pos in ["l", "left"]:
                    # Touchdown left
                    bc = np.array([fq.N(z), fq.M(z), fq.w(z)])
                elif touchdown_mode in ["C_in_contact"] and pos in ["r", "right"]:
                    # Spring stiffness
                    kR = collapsed_weak_layer_kR
                    # Touchdown right
                    bc = np.array([fq.N(z), fq.M(z) + kR * fq.psi(z), fq.w(z)])
                elif touchdown_mode in ["C_in_contact"] and pos in ["l", "left"]:
                    # Spring stiffness
                    kR = collapsed_weak_layer_kR
                    # Touchdown left
                    bc = np.array([fq.N(z), fq.M(z) - kR * fq.psi(z), fq.w(z)])
                else:
                    # Touchdown not enabled
                    bc = np.array([fq.N(z), fq.M(z), fq.V(z)])
            else:
                # Free end
                bc = np.array([fq.N(z), fq.M(z), fq.V(z)])
        # Set boundary conditions for PST-systems with vertical faces
        elif system_type in ["-vpst", "vpst-"]:
            bc = np.array([fq.N(z), fq.M(z), fq.V(z)])
        # Set boundary conditions for SKIER-systems
        elif system_type in ["skier", "skiers"]:
            # Infinite end (vanishing complementary solution)
            bc = np.array([fq.u(z, h0=0), fq.w(z), fq.psi(z)])
        # Set boundary conditions for substitute spring calculus
        elif system_type in ["rot", "trans"]:
            bc = np.array([fq.N(z), fq.M(z), fq.V(z)])
        else:
            raise ValueError(
                f"Boundary conditions not defined for system of type {system_type}."
            )

        return bc
