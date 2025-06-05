"""
This module defines the system model for the WEAC simulation.
The system model is the heart of the WEAC simulation. All data sources are bundled into the system model.
The system model initializes and calculates all the parameterizations and passes relevant data to the different components.

We utilize the pydantic library to define the system model.
"""
import logging
import numpy as np
from typing import List

from weac_2.components import Config, WeakLayer, Segment, Scenario, CriteriaOverrides, ModelInput
from weac_2.core.slab import Slab
from weac_2.core.eigensystem import Eigensystem

logger = logging.getLogger(__name__)

class SystemModel:
    """
    This class is the heart of the WEAC simulation. All data sources are bundled into the system model.
    """
    config: Config
    weak_layer: WeakLayer
    slab: Slab
    segments: List[Segment]
    scenario: Scenario
    criteria_overrides: CriteriaOverrides
    eigensystem: Eigensystem
    
    unknown_constants: np.ndarray
    
    def __init__(self, model_input: ModelInput, config: Config):
        self.config = config
        self.weak_layer = model_input.weak_layer
        self.slab = Slab(layers=model_input.layers)
        self.segments = model_input.segments
        self.scenario = model_input.scenario
        self.criteria_overrides = model_input.criteria_overrides

        self.unknown_constants = np.array([])

        self.eigensystem = Eigensystem(system=self.scenario.system, touchdown=self.scenario.touchdown, slab=self.slab, weak_layer=self.weak_layer)

    def solve_for_unknown_constants(self):
        pass
