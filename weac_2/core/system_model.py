"""
This module defines the system model for the WEAC simulation.
The system model is the heart of the WEAC simulation. All data sources are bundled into the system model.
The system model initializes and calculates all the parameterizations and passes relevant data to the different components.

We utilize the pydantic library to define the system model.
"""
import logging
import copy
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
from weac_2.core.slab_touchdown import SlabTouchdown
from weac_2.core.field_quantities import FieldQuantities
from weac_2.core.unknown_constants_solver import UnknownConstantsSolver

logger = logging.getLogger(__name__)

class SystemModel():
    """
    This class is the heart of the WEAC simulation. All data sources are bundled into the system model.
    """
    config: Config

    slab: Slab
    weak_layer: WeakLayer
    eigensystem: Eigensystem
    
    field_quantities: FieldQuantities
    
    scenario: Scenario
    slab_touchdown: Optional[SlabTouchdown]
    unknown_constants_solver: UnknownConstantsSolver
    unknown_constants: np.ndarray
    
    def __init__(self, model_input: ModelInput, config: Config):
        self.config = config

        # Setup the Entirty of the Eigenproblem
        self.weak_layer = model_input.weak_layer
        self.slab = Slab(layers=model_input.layers)
        
        self.eigensystem = Eigensystem(weak_layer=self.weak_layer, slab=self.slab)
        
        self.fq = FieldQuantities(eigensystem=self.eigensystem)
        self.scenario = Scenario(scenario_config=model_input.scenario_config, segments=model_input.segments, weak_layer=self.weak_layer, slab=self.slab)
        
        # Setup the Touchdown if needed - SlabTouchdown now handles all collapsed logic internally
        if config.touchdown:
            self.slab_touchdown = SlabTouchdown(scenario=self.scenario, eigensystem=self.eigensystem)
        else:
            self.slab_touchdown = None

        self.unknown_constants_solver = UnknownConstantsSolver()
        
        self.__dict__['_eigensystem_cache'] = None
        self.__dict__['_unknown_constants_cache'] = None
        self.__dict__['_slab_touchdown_cache'] = None

    @cached_property
    def eigensystem(self) -> Eigensystem:                 # heavy
        return Eigensystem(weak_layer=self.weak_layer, slab=self.slab)

    @cached_property
    def slab_touchdown(self) -> Optional[SlabTouchdown]:
        if self.config.touchdown:
            return SlabTouchdown(scenario=self.scenario, eigensystem=self.eigensystem)
        return None

    @cached_property
    def unknown_constants(self) -> np.ndarray:                  # medium
        return self.unknown_constants_solver._solve_for_unknown_constants(scenario=self.scenario, eigensystem=self.eigensystem, system_type=self.scenario.system_type, touchdown_l=self.slab_touchdown.touchdown_l, touchdown_mode=self.slab_touchdown.mode, collapsed_weak_layer_kR=self.slab_touchdown.collapsed_weak_layer_kR)

    # Changes that affect the *weak layer*  -> rebuild everything
    def update_weak_layer(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.weak_layer, k, v)
        self._invalidate_eigensystem()

    # Changes that affect the *slab*  -> rebuild everything
    def update_slab_layers(self, new_layers: List[Layer]):
        self.slab.layers = new_layers
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
        self.__dict__.pop('unknown_constants', None)
        self.__dict__.pop('slab_touchdown', None)

    def _invalidate_slab_touchdown(self):
        self.__dict__.pop('slab_touchdown', None)

    def _invalidate_constants(self):
        self.__dict__.pop('unknown_constants', None)

    def _solve_for_unknown_constants(self) -> np.ndarray:
        """Solve for unknown constants using the UnknownConstantsSolver."""
        return self.unknown_constants_solver._solve_for_unknown_constants(
            scenario=self.scenario, 
            eigensystem=self.eigensystem, 
            system_type=self.scenario.system_type
        )

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
