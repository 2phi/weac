"""
This module defines the system model for the WEAC simulation.
The system model is the heart of the WEAC simulation. All data sources are bundled into the system model.
The system model initializes and calculates all the parameterizations and passes relevant data to the different components.

We utilize the pydantic library to define the system model.
"""
import logging
import numpy as np
from typing import List

from weac_2.components import Config, WeakLayer, Segment, ScenarioConfig, CriteriaOverrides, ModelInput
from weac_2.core.slab import Slab
from weac_2.core.eigensystem import Eigensystem
from weac_2.core.scenario import Scenario

logger = logging.getLogger(__name__)

class SystemModel:
    """
    This class is the heart of the WEAC simulation. All data sources are bundled into the system model.
    """
    config: Config
    criteria_overrides: CriteriaOverrides

    weak_layer: WeakLayer
    slab: Slab
    eigensystem: Eigensystem
    
    scenario: Scenario
    C_constants: np.ndarray
    
    def __init__(self, model_input: ModelInput, config: Config):
        self.config = config
        self.criteria_overrides = model_input.criteria_overrides

        self.weak_layer = model_input.weak_layer
        self.slab = Slab(layers=model_input.layers)
        self.eigensystem = Eigensystem(weak_layer=self.weak_layer, slab=self.slab)

        self.scenario = Scenario(scenario_config=model_input.scenario_config, segments=model_input.segments, weak_layer=self.weak_layer, slab=self.slab)
        self.C_constants = self.solve_for_unknown_constants()

    def solve_for_unknown_constants(self) -> np.ndarray:
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
        phi = self.scenario.scenario_config.phi
        li = self.scenario.li
        ki = self.scenario.ki
        mi = self.scenario.mi
        
        # Determine size of linear system of equations
        nS = len(li)  # Number of beam segments
        nDOF = 6  # Number of free constants per segment
        
        # Assemble position vector
        pi = np.full(nS, "m")
        pi[0], pi[-1] = "l", "r"

        # Initialize matrices
        zh0 = np.zeros([nS * 6, nS * nDOF])
        zp0 = np.zeros([nS * 6, 1])
        rhs = np.zeros([nS * 6, 1])
        
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

        # Solve z0 = zh0*C + zp0 = rhs for constants, i.e. zh0*C = rhs - zp0
        C = np.linalg.solve(zh0, rhs - zp0)
        # Sort (nDOF = 6) constants for each segment into columns of a matrix
        return C.reshape([-1, nDOF]).T


    def z(self, x, C, l, phi, bed=True):
        """
        Assemble solution vector at positions x.

        Arguments
        ---------
        x : float or squence
            Horizontal coordinate (mm). Can be sequence of length N.
        C : ndarray
            Vector of constants (6xN) at positions x.
        l : float
            Segment length (mm).
        phi : float
            Inclination (degrees).
        bed : bool
            Indicates whether segment has foundation (True) or not
            (False). Default is True.

        Returns
        -------
        z : ndarray
            Solution vector (6xN) at position x.
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            z = np.concatenate([
                np.dot(self.zh(xi, l, bed), C)
                + self.zp(xi, phi, bed) for xi in x], axis=1)
        else:
            z = np.dot(self.zh(x, l, bed), C) + self.zp(x, phi, bed)

        return z

    def zh(self, x, l=0, bed=True):
        """
        Compute bedded or free complementary solution at position x.

        Arguments
        ---------
        x : float
            Horizontal coordinate (mm).
        l : float, optional
            Segment length (mm). Default is 0.
        bed : bool
            Indicates whether segment has foundation or not. Default
            is True.

        Returns
        -------
        zh : ndarray
            Complementary solution matrix (6x6) at position x.
        """
        if bed:
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

    def zp(self, x, phi, bed=True):
        """
        Compute bedded or free particular integrals at position x.

        Arguments
        ---------
        x : float
            Horizontal coordinate (mm).
        phi : float
            Inclination (degrees).
        bed : bool
            Indicates whether segment has foundation (True) or not
            (False). Default is True.

        Returns
        -------
        zp : ndarray
            Particular integral vector (6x1) at position x.
        """
        # Get weight and surface loads
        qn, qt = self.get_weight_load(phi)
        pn, pt = self.get_surface_load(phi)

        # Set foundation stiffnesses
        kn = self.kn
        kt = self.kt

        # Unpack laminate stiffnesses
        A11 = self.A11
        B11 = self.B11
        kA55 = self.kA55
        K0 = self.K0

        # Unpack geometric properties
        h = self.slab.H
        z_cog = self.slab.z_cog
        t = self.weak_layer.h

        # Assemble particular integral vectors
        if bed:
            zp = np.array([
                [(qt + pt)/kt + h*qt*(h + t - 2*z_cog)/(4*kA55)
                    + h*pt*(2*h + t)/(4*kA55)],
                [0],
                [(qn + pn)/kn],
                [0],
                [-(qt*(h + t - 2*z_cog) + pt*(2*h + t))/(2*kA55)],
                [0]])
        else:
            zp = np.array([
                [(-3*(qt + pt)/A11 - B11*(qn + pn)*x/K0)/6*x**2],
                [(-2*(qt + pt)/A11 - B11*(qn + pn)*x/K0)/2*x],
                [-A11*(qn + pn)*x**4/(24*K0)],
                [-A11*(qn + pn)*x**3/(6*K0)],
                [A11*(qn + pn)*x**3/(6*K0)
                 + ((z_cog - B11/A11)*qt - h*pt/2 - (qn + pn)*x)/kA55],
                [(qn + pn)*(A11*x**2/(2*K0) - 1/kA55)]])

        return zp

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
                "Boundary conditions not defined for" f"system of type {self.system}."
            )

        return bc
