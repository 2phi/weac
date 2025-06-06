

from typing import List, Literal
import numpy as np

from weac_2.utils import split_q

from weac_2.components import ScenarioConfig, Segment, WeakLayer
from weac_2.core.slab import Slab

class Scenario:
    """
    Sets up the scenario on which the eigensystem is solved.
    
    Arguments
    ---------
    scenario_config: ScenarioConfig
    segments: List[Segment]
    weak_layer: WeakLayer
    slab: Slab
    
    Attributes
    ----------
    li : List[float]
        length of segment i [mm]
    ki : List[bool]
        booleans indicating foundation support for segment i
    mi : List[float]
        skier masses (kg) on boundary of segment i and i+1 [kg]
    
    L : float
        Length of the model [mm]
    crack_h: float
        Height of the crack [mm]
    """
    # Inputs
    scenario_config: ScenarioConfig
    segments: List[Segment]
    weak_layer: WeakLayer
    slab: Slab
    
    # Attributes
    li: np.ndarray               # length of segment i [mm]
    ki: np.ndarray               # booleans indicating foundation support for segment i
    mi: np.ndarray               # skier masses (kg) on boundary of segment i and i+1 [kg]
    
    L: float                     # Length of the model [mm]
    crack_h: float               # Height of the crack [mm]
    
    def __init__(self, scenario_config: ScenarioConfig, segments: List[Segment], weak_layer: WeakLayer, slab: Slab):
        self.scenario_config = scenario_config
        self.segments = segments
        self.weak_layer = weak_layer
        self.slab = slab
        
        self._setup_scenario()
        self._calc_crack_height()

    def _setup_scenario(self):
        self.li = np.array([seg.l for seg in self.segments])
        self.ki = np.array([seg.k for seg in self.segments])
        # masses that act *between* segments: take all but the last one
        self.mi = np.array([seg.m for seg in self.segments[:-1]])
        
        # Add dummy segment if only one segment provided
        if len(self.li) == 1:
            self.li.append(0)
            self.ki.append(True)
            self.mi.append(0)
        
        # Calculate the total slab length
        self.L = np.sum(self.li)

    def _calc_crack_height(self):
        # Surface Load & Weight Load
        qw = self.slab.weight_load
        qs = self.scenario_config.surface_load
        
        # Normal components of forces
        phi = self.scenario_config.phi
        qwn, _ = split_q(qw, phi)
        qsn, _ = split_q(qs, phi)
        qn = qwn + qsn
        
        # Crack Height: Difference between collapsed weak layer and
        #               Weak Layer (Winkler type) under slab load
        cf = self.scenario_config.collapse_factor
        self.crack_h = cf * self.weak_layer.h - qn / self.weak_layer.kn
