"""
This module defines the system model for the WEAC simulation.
The system model is the heart of the WEAC simulation. All data sources are bundled into the system model.
The system model initializes and calculates all the parameterizations and passes relevant data to the different components.

We utilize the pydantic library to define the system model.
"""
import logging
from functools import cached_property
from collections.abc import Sequence
import numpy as np
from typing import List, Optional, Union, Iterable, Tuple, Literal

# from weac_2.constants import G_MM_S2, LSKI_MM
from weac_2.utils import decompose_to_normal_tangential, get_skier_point_load
from weac_2.constants import G_MM_S2
from weac_2.components import Config, WeakLayer, Segment, ScenarioConfig, CriteriaConfig, ModelInput, Layer
from weac_2.core.slab import Slab
from weac_2.core.eigensystem import Eigensystem
from weac_2.core.scenario import Scenario
from weac_2.core.field_quantities import FieldQuantities

logger = logging.getLogger(__name__)

class SystemModel():
    """
    This class is the heart of the WEAC simulation. All data sources are bundled into the system model.
    """
    config: Config

    weak_layer: WeakLayer
    slab: Slab
    eigensystem: Eigensystem
    
    scenario: Scenario
    C_constants: np.ndarray
    
    def __init__(self, model_input: ModelInput, config: Config):
        self.config = config

        # Setup the Entirty of the Eigenproblem
        self.weak_layer = model_input.weak_layer
        self.slab = Slab(layers=model_input.layers)
        # self.eigensystem = Eigensystem(weak_layer=self.weak_layer, slab=self.slab)
        self.fq = FieldQuantities(eigensystem=self.eigensystem)

        # Solve for a specific Scenario
        self.scenario = Scenario(scenario_config=model_input.scenario_config, segments=model_input.segments, weak_layer=self.weak_layer, slab=self.slab)
        # self.C_constants = self._solve_for_unknown_constants()
        
        self.__dict__['_eigensystem_cache'] = None
        self.__dict__['_C_constants_cache']   = None
    
    @cached_property
    def eigensystem(self) -> Eigensystem:                 # heavy
        return Eigensystem(weak_layer=self.weak_layer, slab=self.slab)

    @cached_property
    def C_constants(self) -> np.ndarray:                  # medium
        return self._solve_for_unknown_constants()

    # Changes that affect the *slab*  -> rebuild everything
    def update_slab_layers(self, new_layers: List[Layer]):
        self.slab.layers = new_layers
        self._invalidate_eigensystem()

    # Changes that affect the *weak layer*  -> rebuild everything
    def update_weak_layer(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.weak_layer, k, v)
        self._invalidate_eigensystem()

    # Changes that affect the *scenario*  -> only rebuild C constants
    def update_scenario(self, **kwargs):
        """
        Update fields on `scenario_config` (if present) or on the
        Scenario object itself, then refresh and invalidate constants.
        """
        logger.debug("Updating Scenario...")
        for k, v in kwargs.items():
            if hasattr(self.scenario.scenario_config, k):
                setattr(self.scenario.scenario_config, k, v)
            elif hasattr(self.scenario, k):
                setattr(self.scenario, k, v)
            else:
                raise AttributeError(f"Unknown scenario field '{k}'")

        # Pull new values through & recompute segment lengths, etc.
        logger.debug(f"Old Phi: {self.scenario.phi}")
        self.scenario.refresh_from_config()
        logger.debug(f"New Phi: {self.scenario.phi}")
        self._invalidate_constants()

    def _invalidate_eigensystem(self):
        self.__dict__.pop('eigensystem', None)
        self.__dict__.pop('C_constants', None)

    def _invalidate_constants(self):
        self.__dict__.pop('C_constants', None)

    def z(self, x: Union[float, Sequence[float], np.ndarray], C: np.ndarray, l: float, phi: float, k: bool = True, qs: float = 0) -> np.ndarray:
        """
        Assemble solution vector at positions x.

        Arguments
        ---------
        x : float or sequence
            Horizontal coordinate (mm). Can be sequence of length N.
        C : ndarray
            Vector of constants (6xN) at positions x.
        l : float
            Segment length (mm).
        phi : float
            Inclination (degrees).
        k : bool
            Indicates whether segment has foundation (True) or not
            (False). Default is True.
        qs : float
            Surface Load [N/mm]

        Returns
        -------
        z : ndarray
            Solution vector (6xN) at position x.
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            z = np.concatenate([
                np.dot(self.eigensystem.zh(xi, l, k), C)
                + self.eigensystem.zp(xi, phi, k, qs) for xi in x], axis=1)
        else:
            z = np.dot(self.eigensystem.zh(x, l, k), C) + self.eigensystem.zp(x, phi, k, qs)

        return z

    def _solve_for_unknown_constants(self) -> np.ndarray:
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
        system = self.scenario.system
        phi = self.scenario.phi
        qs = self.scenario.qs
        li = self.scenario.li
        ki = self.scenario.ki
        mi = self.scenario.mi
        
        # Determine size of linear system of equations
        nS = len(li)  # Number of beam segments
        nDOF = 6  # Number of free constants per segment
        logger.debug(f"Number of segments: {nS}, DOF per segment: {nDOF}")
        
        # Assemble position vector
        pi = np.full(nS, "m")
        pi[0], pi[-1] = "l", "r"

        # Initialize matrices
        Zh0 = np.zeros([nS * 6, nS * nDOF])
        Zp0 = np.zeros([nS * 6, 1])
        rhs = np.zeros([nS * 6, 1])
        logger.debug(f"Initialized Zh0 shape: {Zh0.shape}, Zp0 shape: {Zp0.shape}, rhs shape: {rhs.shape}")
        
        # LHS: Transmission & Boundary Conditions between segments
        for i in range(nS):
            # Length, foundation and position of segment i
            l, k, pos = li[i], ki[i], pi[i]
            
            logger.debug(f"Assembling segment {i}: l={l}, k={k}, pos={pos}")
            # Matrix of Size one of: (l: [9,6], m: [12,6], r: [9,6])
            Zhi = self._setup_conditions(
                zl=self.eigensystem.zh(x=0, l=l, k=k),
                zr=self.eigensystem.zh(x=l, l=l, k=k),
                k=k,
                pos=pos,
                system=system,
            )
            # Vector of Size one of: (l: [9,1], m: [12,1], r: [9,1])
            zpi = self._setup_conditions(
                zl=self.eigensystem.zp(x=0, phi=phi, k=k, qs=qs),
                zr=self.eigensystem.zp(x=l, phi=phi, k=k, qs=qs),
                k=k,
                pos=pos,
                system=system,
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
            rhs[6 * i : 6 * i + 3] = np.vstack([Ft, -Ft * self.slab.H / 2, Fn])
            logger.debug(f"Load {i}: m={m}, F={F}, Fn={Fn}, Ft={Ft}")
            logger.debug(f"RHS {rhs[6 * i : 6 * i + 3]}")
        # Set RHS so that Complementary Integral vanishes at boundaries
        if system not in ["pst-", "-pst", "rested"]:
            logger.debug(f"Pre RHS {rhs[:3]}")
            rhs[:3] = self._boundary_conditions(self.eigensystem.zp(x=0, phi=phi, k=ki[0], qs=qs), k=False, pos="mid", system=system)
            logger.debug(f"Post RHS {rhs[:3]}")
            rhs[-3:] = self._boundary_conditions(self.eigensystem.zp(x=li[-1], phi=phi, k=ki[-1], qs=qs), k=False, pos="mid", system=system)
            logger.debug("Set complementary integral vanishing at boundaries.")
        
        # Set rhs for vertical faces
        if system in ["vpst-", "-vpst"]:
            # Calculate center of gravity and mass of
            # added or cut off slab segement
            x_cog, z_cog, m = self.slab.calc_vertical_center_of_gravity(phi)
            # Convert slope angle to radians
            phi = np.deg2rad(phi)
            # Translate inbto section forces and moments
            N = - G_MM_S2 * m * np.sin(phi)
            M = - G_MM_S2 * m * (x_cog * np.cos(phi) + z_cog * np.sin(phi))
            V = G_MM_S2 * m * np.cos(phi)
            # Add to right-hand side
            rhs[:3] = np.vstack([N, M, V])   # left end
            rhs[-3:] = np.vstack([N, M, V])  # right end
            logger.info(f"Vertical faces: N={N}, M={M}, V={V}")

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
                N = self.scenario.calc_tangential_load() * (self.a - self.td)
                if i == 0:
                    rhs[:3] = np.vstack([-N, 0, self.tc])
                if i == (nS - 1):
                    rhs[-3:] = np.vstack([N, 0, self.tc])

        # Rhs for substitute spring stiffness
        if system in ["rot"]:
            # apply arbitrary moment of 1 at left boundary
            rhs = rhs * 0
            rhs[1] = 1
        if system in ["trans"]:
            # apply arbitrary force of 1 at left boundary
            rhs = rhs * 0
            rhs[2] = 1

        # Solve z0 = Zh0*C + Zp0 = rhs for constants, i.e. Zh0*C = rhs - Zp0
        C = np.linalg.solve(Zh0, rhs - Zp0)
        # Sort (nDOF = 6) constants for each segment into columns of a matrix
        return C.reshape([-1, nDOF]).T

    def _setup_conditions(self, zl: np.ndarray, zr: np.ndarray, k: bool, pos: Literal['l','r','m','left','right','mid'] , system: Literal['skier', 'skiers', 'pst-', 'pst+', 'rot', 'trans']) -> np.ndarray:
        """
        Provide boundary or transmission conditions for beam segments.

        Arguments
        ---------
        zl : ndarray
            Solution vector (6x1) or (6x6) at left end of beam segement.
        zr : ndarray
            Solution vector (6x1) or (6x6) at right end of beam segement.
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
        conditions : ndarray
            `zh`: Matrix of Size one of: (`l: [9,6], m: [12,6], r: [9,6]`)
            
            `zp`: Vector of Size one of: (`l: [9,1], m: [12,1], r: [9,1]`)
        """
        if pos in ("l", "left"):
            bcs = self._boundary_conditions(zl, k, pos, system)  # Left boundary condition
            conditions = np.array(
                [
                    bcs[0],  
                    bcs[1],
                    bcs[2],
                    self.fq.u(zr, h0=0),             # ui(xi = li)
                    self.fq.w(zr),                   # wi(xi = li)
                    self.fq.psi(zr),                 # psii(xi = li)
                    self.fq.N(zr),                   # Ni(xi = li)
                    self.fq.M(zr),                   # Mi(xi = li)
                    self.fq.V(zr),                   # Vi(xi = li)
                ]
            )
        elif pos in ("m", "mid"):
            conditions = np.array(
                [
                    -self.fq.u(zl, h0=0),  # -ui(xi = 0)
                    -self.fq.w(zl),  # -wi(xi = 0)
                    -self.fq.psi(zl),  # -psii(xi = 0)
                    -self.fq.N(zl),  # -Ni(xi = 0)
                    -self.fq.M(zl),  # -Mi(xi = 0)
                    -self.fq.V(zl),  # -Vi(xi = 0)
                    self.fq.u(zr, h0=0),  # ui(xi = li)
                    self.fq.w(zr),  # wi(xi = li)
                    self.fq.psi(zr),  # psii(xi = li)
                    self.fq.N(zr),  # Ni(xi = li)
                    self.fq.M(zr),  # Mi(xi = li)
                    self.fq.V(zr),  # Vi(xi = li)
                ]
            )
        elif pos in ("r", "right"):
            bcs = self._boundary_conditions(zr, k, pos, system) # Right boundary condition
            conditions = np.array(
                [
                    -self.fq.u(zl, h0=0),  # -ui(xi = 0)
                    -self.fq.w(zl),  # -wi(xi = 0)
                    -self.fq.psi(zl),  # -psii(xi = 0)
                    -self.fq.N(zl),  # -Ni(xi = 0)
                    -self.fq.M(zl),  # -Mi(xi = 0)
                    -self.fq.V(zl),  # -Vi(xi = 0)
                    bcs[0],
                    bcs[1],
                    bcs[2],
                ]
            )  
        logger.debug(f"Boundary Conditions at pos {pos}: {conditions.shape}")
        return conditions

    def _boundary_conditions(self, z, k: bool, pos: Literal['l','r','m','left','right','mid'], system: Literal['skier', 'skiers', 'pst-', 'pst+', 'rot', 'trans']):
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
        if system in ["pst-", "-pst"]:
            if not k:
                if self.mode in ["A"]:
                    # Free end
                    bc = np.array([self.fq.N(z), self.fq.M(z), self.fq.V(z)])
                elif self.mode in ["B"] and pos in ["r", "right"]:
                    # Touchdown right
                    bc = np.array([self.fq.N(z), self.fq.M(z), self.fq.w(z)])
                elif self.mode in ["B"] and pos in ["l", "left"]:  # Kann dieser Block
                    # Touchdown left                                # verschwinden? Analog zu 'B'
                    bc = np.array([self.fq.N(z), self.fq.M(z), self.fq.w(z)])
                elif self.mode in ["C"] and pos in ["r", "right"]:
                    # Spring stiffness
                    kR = self.substitute_stiffness(self.a - self.td, "rested", "rot")
                    # Touchdown right
                    bc = np.array([self.fq.N(z), self.fq.M(z) + kR * self.fq.psi(z), self.w(z)])
                elif self.mode in ["C"] and pos in ["l", "left"]:
                    # Spring stiffness
                    kR = self.substitute_stiffness(self.a - self.td, "rested", "rot")
                    # Touchdown left
                    bc = np.array([self.fq.N(z), self.fq.M(z) - kR * self.fq.psi(z), self.w(z)])
            else:
                # Free end
                bc = np.array([self.fq.N(z), self.fq.M(z), self.fq.V(z)])
        # Set boundary conditions for PST-systems with vertical faces
        elif system in ["-vpst", "vpst-"]:
            bc = np.array([self.fq.N(z), self.fq.M(z), self.fq.V(z)])
        # Set boundary conditions for SKIER-systems
        elif system in ["skier", "skiers"]:
            # Infinite end (vanishing complementary solution)
            bc = np.array([self.fq.u(z, h0=0), self.fq.w(z), self.fq.psi(z)])
        # Set boundary conditions for substitute spring calculus
        elif system in ["rot", "trans"]:
            bc = np.array([self.fq.N(z), self.fq.M(z), self.fq.V(z)])
        else:
            raise ValueError(
                "Boundary conditions not defined for" f"system of type {system}."
            )

        return bc
