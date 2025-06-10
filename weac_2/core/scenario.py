

from typing import List, Literal
import numpy as np

from weac_2.utils import decompose_to_normal_tangential

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
    
    system : Literal[
    phi : float
        Angle of slab in positive in counter-clockwise direction [deg]
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
    
    system: Literal['skier', 'skiers', 'pst-', 'pst+', 'rot', 'trans']
    touchdown: bool              # Considering Touchdown or not
    phi: float                   # Angle in [deg]
    qs: float                    # Line-Load [N/mm]
    L: float                     # Length of the model [mm]
    crack_h: float               # Height of the crack [mm]
    
    def __init__(self, scenario_config: ScenarioConfig, segments: List[Segment], weak_layer: WeakLayer, slab: Slab):
        self.scenario_config = scenario_config
        self.segments = segments
        self.weak_layer = weak_layer
        self.slab = slab
        
        self.system = scenario_config.system
        self.phi = scenario_config.phi
        self.qs = scenario_config.qs
        
        self._setup_scenario()
        self._calc_crack_height()
    
    def refresh_from_config(self):
        """Pull changed values out of scenario_config
           and recompute derived attributes."""
        self.system = self.scenario_config.system
        self.phi    = self.scenario_config.phi
        self.qs     = self.scenario_config.qs

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
        """
        Crack Height: Difference between collapsed weak layer and
            Weak Layer (Winkler type) under slab load
        """
        qn = self.calc_normal_load()
        
        cf = self.scenario_config.collapse_factor
        self.crack_h = cf * self.weak_layer.h - qn / self.weak_layer.kn
    
    def calc_tangential_load(self):
        # Surface Load & Weight Load
        qw = self.slab.qw
        qs = self.qs
        
        # Normal components of forces
        phi = self.phi
        qwn, _ = decompose_to_normal_tangential(qw, phi)
        qsn, _ = decompose_to_normal_tangential(qs, phi)
        qn = qwn + qsn
        return qn
    
    def calc_normal_load(self):
        # Surface Load & Weight Load
        qw = self.slab.qw
        qs = self.qs
        
        # Normal components of forces
        phi = self.phi
        _, qwt = decompose_to_normal_tangential(qw, phi)
        _, qst = decompose_to_normal_tangential(qs, phi)
        qt = qwt + qst
        return qt
