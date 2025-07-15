"""
This module defines the system model for the WEAC simulation.
The system model is the heart of the WEAC simulation. All data sources are bundled into the system model.
The system model initializes and calculates all the parameterizations and passes relevant data to the different components.

We utilize the pydantic library to define the system model.
"""

import copy
import logging
from collections.abc import Sequence
from functools import cached_property
from typing import List, Optional, Union

import numpy as np

# from weac_2.constants import G_MM_S2, LSKI_MM
from weac_2.components import (
    Config,
    Layer,
    Segment,
    ModelInput,
    ScenarioConfig,
    WeakLayer,
)
from weac_2.core.eigensystem import Eigensystem
from weac_2.core.field_quantities import FieldQuantities
from weac_2.core.scenario import Scenario
from weac_2.core.slab import Slab
from weac_2.core.slab_touchdown import SlabTouchdown
from weac_2.core.unknown_constants_solver import UnknownConstantsSolver

logger = logging.getLogger(__name__)


class SystemModel:
    """
    The heart of the WEAC simulation system for avalanche release modeling.

    This class orchestrates all components of the WEAC simulation, including slab mechanics,
    weak layer properties, touchdown calculations, and the solution of unknown constants
    for beam-on-elastic-foundation problems.

    The SystemModel follows a lazy evaluation pattern using cached properties, meaning
    expensive calculations (eigensystem, touchdown, unknown constants) are only computed
    when first accessed and then cached for subsequent use.

    **Extracting Unknown Constants:**

    The primary output of the SystemModel is the `unknown_constants` matrix, which contains
    the solution constants for the beam segments:

    ```python
    # Basic usage
    system = SystemModel(model_input=model_input, config=config)
    constants = system.unknown_constants  # Shape: (6, N) where N = number of segments

    # Each column represents the 6 constants for one segment:
    # constants[:, i] = [C1, C2, C3, C4, C5, C6] for segment i
    # These constants define the beam deflection solution within that segment
    ```

    **Calculation Flow:**

    1. **Eigensystem**: Computes eigenvalues/eigenvectors for the beam-foundation system
    2. **Slab Touchdown** (if enabled): Calculates touchdown behavior and updates segment lengths
    3. **Unknown Constants**: Solves the linear system for beam deflection constants

    **Touchdown Behavior:**

    When `config.touchdown=True`, the system automatically:
    - Calculates touchdown mode (A: free-hanging, B: point contact, C: in contact)
    - Determines touchdown length based on slab-foundation interaction
    - **Redefines scenario segments** to use touchdown length instead of crack length
    - This matches the behavior of the original WEAC implementation

    **Performance Notes:**

    - First access to `unknown_constants` triggers all necessary calculations
    - Subsequent accesses return cached results instantly
    - Use `update_*` methods to modify parameters and invalidate caches as needed

    **Example Usage:**

    ```python
    from weac_2.components import ModelInput, Layer, Segment, Config
    from weac_2.core.system_model import SystemModel

    # Define system components
    layers = [Layer(rho=200, h=150), Layer(rho=300, h=100)]
    segments = [Segment(length=10000, has_foundation=True, m=0), Segment(length=4000, has_foundation=False, m=0)]

    # Create system
    system = SystemModel(model_input=model_input, config=Config(touchdown=True))

    # Solve system and extract results
    constants = system.unknown_constants      # Solution constants (6 x N_segments)
    touchdown_info = system.slab_touchdown    # Touchdown analysis (if enabled)
    eigensystem = system.eigensystem          # Eigenvalue problem solution
    ```

    Attributes:
        config: Configuration settings including touchdown enable/disable
        slab: Slab properties (thickness, material properties per layer)
        weak_layer: Weak layer properties (stiffness, thickness, etc.)
        scenario: Scenario definition (segments, loads, boundary conditions)
        eigensystem: Eigenvalue problem solution (computed lazily)
        slab_touchdown: Touchdown analysis results (computed lazily if enabled)
        unknown_constants: Solution constants matrix (computed lazily)
    """

    config: Config
    slab: Slab
    weak_layer: WeakLayer
    eigensystem: Eigensystem
    fq: FieldQuantities

    scenario: Scenario
    slab_touchdown: Optional[SlabTouchdown]
    unknown_constants: np.ndarray
    uncracked_scenario: Scenario
    uncracked_unknown_constants: np.ndarray

    def __init__(self, model_input: ModelInput, config: Config = Config()):
        self.config = config
        self.weak_layer = model_input.weak_layer
        self.slab = Slab(layers=model_input.layers)
        self.scenario = Scenario(
            scenario_config=model_input.scenario_config,
            segments=model_input.segments,
            weak_layer=self.weak_layer,
            slab=self.slab,
        )
        self.fq = FieldQuantities(eigensystem=self.eigensystem)
        logger.info("Scenario setup")

        # At this point only the system is initialized
        # The solution to the system (unknown_constants) are only computed
        # when required by the user (at runtime)

        self.__dict__["_eigensystem_cache"] = None
        self.__dict__["_unknown_constants_cache"] = None
        self.__dict__["_slab_touchdown_cache"] = None
        self.__dict__["_uncracked_unknown_constants_cache"] = None

    @cached_property
    def eigensystem(self) -> Eigensystem:  # heavy
        logger.info("Solving for Eigensystem")
        return Eigensystem(weak_layer=self.weak_layer, slab=self.slab)

    @cached_property
    def slab_touchdown(self) -> Optional[SlabTouchdown]:
        if self.config.touchdown:
            logger.info("Solving for Slab Touchdown")
            slab_touchdown = SlabTouchdown(
                scenario=self.scenario, eigensystem=self.eigensystem
            )

            logger.info(
                f"Original crack_length: {self.scenario.crack_length}, touchdown_distance: {slab_touchdown.touchdown_distance}"
            )

            new_segments = copy.deepcopy(self.scenario.segments)
            if (
                self.scenario.system_type == "pst-"
                or self.scenario.system_type == "vpst-"
            ):
                new_segments[-1].length = slab_touchdown.touchdown_distance
            elif (
                self.scenario.system_type == "-pst"
                or self.scenario.system_type == "-vpst"
            ):
                new_segments[0].length = slab_touchdown.touchdown_distance

            # Create new scenario with updated segments
            self.scenario = Scenario(
                scenario_config=self.scenario.scenario_config,
                segments=new_segments,
                weak_layer=self.weak_layer,
                slab=self.slab,
            )
            logger.info(
                f"Updated scenario with new segment lengths: {[seg.length for seg in new_segments]}"
            )

            return slab_touchdown
        return None

    @cached_property
    def unknown_constants(self) -> np.ndarray:
        """
        Solve for the unknown constants matrix defining beam deflection in each segment.

        This is the core solution of the WEAC beam-on-elastic-foundation problem.
        The unknown constants define the deflection, slope, moment, and shear force
        distributions within each beam segment.

        Returns:
        --------
        np.ndarray: Solution constants matrix of shape (6, N_segments)
            Each column contains the 6 constants for one segment:
            [C1, C2, C3, C4, C5, C6]

            These constants are used in the general solution:
            u(x) = Σ Ci * φi(x) + up(x)

            Where φi(x) are the homogeneous solutions and up(x)
            is the particular solution.

        Notes:
            - For touchdown systems, segment lengths are automatically adjusted
              based on touchdown calculations before solving
            - The solution accounts for boundary conditions, load transmission
              between segments, and foundation support
            - Results are cached after first computation for performance

        Example:
            ```python
            system = SystemModel(model_input, config)
            C = system.unknown_constants  # Shape: (6, 2) for 2-segment system

            # Constants for first segment
            segment_0_constants = C[:, 0]

            # Use with eigensystem to compute field quantities
            x = 1000  # Position in mm
            segment_length = system.scenario.li[0]
            ```
        """
        if self.slab_touchdown is not None:
            logger.info("Solving for Unknown Constants")
            return UnknownConstantsSolver.solve_for_unknown_constants(
                scenario=self.scenario,
                eigensystem=self.eigensystem,
                system_type=self.scenario.system_type,
                touchdown_distance=self.slab_touchdown.touchdown_distance,
                touchdown_mode=self.slab_touchdown.touchdown_mode,
                collapsed_weak_layer_kR=self.slab_touchdown.collapsed_weak_layer_kR,
            )
        else:
            logger.info("Solving for Unknown Constants")
            return UnknownConstantsSolver.solve_for_unknown_constants(
                scenario=self.scenario,
                eigensystem=self.eigensystem,
                system_type=self.scenario.system_type,
                touchdown_distance=None,
                touchdown_mode=None,
                collapsed_weak_layer_kR=None,
            )

    @cached_property
    def uncracked_unknown_constants(self) -> np.ndarray:
        new_segments = copy.deepcopy(self.scenario.segments)
        for i, seg in enumerate(new_segments):
            seg.has_foundation = True
        self.uncracked_scenario = Scenario(
            scenario_config=self.scenario.scenario_config,
            segments=new_segments,
            weak_layer=self.weak_layer,
            slab=self.slab,
        )

        logger.info("Solving for Uncracked Unknown Constants")
        if self.slab_touchdown is not None:
            return UnknownConstantsSolver.solve_for_unknown_constants(
                scenario=self.uncracked_scenario,
                eigensystem=self.eigensystem,
                system_type=self.scenario.system_type,
                touchdown_distance=self.slab_touchdown.touchdown_distance,
                touchdown_mode=self.slab_touchdown.touchdown_mode,
                collapsed_weak_layer_kR=self.slab_touchdown.collapsed_weak_layer_kR,
            )
        else:
            logger.info("Solving for Uncracked Unknown Constants")
            return UnknownConstantsSolver.solve_for_unknown_constants(
                scenario=self.uncracked_scenario,
                eigensystem=self.eigensystem,
                system_type=self.scenario.system_type,
                touchdown_distance=None,
                touchdown_mode=None,
                collapsed_weak_layer_kR=None,
            )

    # Changes that affect the *weak layer*  -> rebuild everything
    def update_weak_layer(self, weak_layer: WeakLayer):
        self.weak_layer = weak_layer
        self.scenario = Scenario(
            scenario_config=self.scenario.scenario_config,
            segments=self.scenario.segments,
            weak_layer=weak_layer,
            slab=self.slab,
        )
        self._invalidate_eigensystem()

    # Changes that affect the *slab*  -> rebuild everything
    def update_layers(self, new_layers: List[Layer]):
        slab = Slab(layers=new_layers)
        self.slab = slab
        self.scenario = Scenario(
            scenario_config=self.scenario.scenario_config,
            segments=self.scenario.segments,
            weak_layer=self.weak_layer,
            slab=slab,
        )
        self._invalidate_eigensystem()

    # Changes that affect the *scenario*  -> only rebuild C constants
    def update_scenario(
        self,
        segments: Optional[List[Segment]] = None,
        scenario_config: Optional[ScenarioConfig] = None,
    ):
        """
        Update fields on `scenario_config` (if present) or on the
        Scenario object itself, then refresh and invalidate constants.
        """
        logger.debug("Updating Scenario...")
        if segments is None:
            segments = self.scenario.segments
        if scenario_config is None:
            scenario_config = self.scenario.scenario_config
        self.scenario = Scenario(
            scenario_config=scenario_config,
            segments=segments,
            weak_layer=self.weak_layer,
            slab=self.slab,
        )
        if self.config.touchdown:
            self._invalidate_slab_touchdown()
        self._invalidate_constants()

    def toggle_touchdown(self, touchdown: bool):
        if self.config.touchdown != touchdown:
            self.config.touchdown = touchdown
            self._invalidate_slab_touchdown()
            self._invalidate_constants()

    def _invalidate_eigensystem(self):
        self.__dict__.pop("eigensystem", None)
        self.__dict__.pop("unknown_constants", None)
        self.__dict__.pop("slab_touchdown", None)

    def _invalidate_slab_touchdown(self):
        self.__dict__.pop("slab_touchdown", None)

    def _invalidate_constants(self):
        self.__dict__.pop("unknown_constants", None)
        self.__dict__.pop("uncracked_unknown_constants", None)

    def z(
        self,
        x: Union[float, Sequence[float], np.ndarray],
        C: np.ndarray,
        length: float,
        phi: float,
        has_foundation: bool = True,
        qs: float = 0,
    ) -> np.ndarray:
        """
        Assemble solution vector at positions x.

        Arguments
        ---------
        x : float or sequence
            Horizontal coordinate (mm). Can be sequence of length N.
        C : ndarray
            Vector of constants (6xN) at positions x.
        length : float
            Segment length (mm).
        phi : float
            Inclination (degrees).
        has_foundation : bool
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
            z = np.concatenate(
                [
                    np.dot(self.eigensystem.zh(xi, length, has_foundation), C)
                    + self.eigensystem.zp(xi, phi, has_foundation, qs)
                    for xi in x
                ],
                axis=1,
            )
        else:
            z = np.dot(
                self.eigensystem.zh(x, length, has_foundation), C
            ) + self.eigensystem.zp(x, phi, has_foundation, qs)

        return z
