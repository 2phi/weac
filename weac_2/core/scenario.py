from typing import List, Literal
import numpy as np
import logging

from weac_2.utils import decompose_to_normal_tangential

from weac_2.components import ScenarioConfig, Segment, WeakLayer
from weac_2.core.slab import Slab

logger = logging.getLogger(__name__)

class Scenario:
    """
    Sets up the scenario on which the eigensystem is solved.
    
    Parameters
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
    
    system_type : Literal['skier', 'skiers', 'pst-', 'pst+', 'rot', 'trans']
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
    
    system_type: Literal['skier', 'skiers', 'pst-', '-pst', 'vpst-', '-vpst', 'rot', 'trans']
    phi: float                   # Angle in [deg]
    qs: float                    # Line-Load [N/mm]
    qw: float                    # Weight Load [N/mm]
    qn: float                    # Normal Load [N/mm]
    qt: float                    # Tangential Load [N/mm]
    L: float                     # Length of the model [mm]
    crack_h: float               # Height of the crack [mm]
    crack_l: float               # Length of the crack [mm]

    def __init__(self, scenario_config: ScenarioConfig, segments: List[Segment], weak_layer: WeakLayer, slab: Slab):
        self.scenario_config = scenario_config
        self.segments = segments
        self.weak_layer = weak_layer
        self.slab = slab
        
        self.system_type = scenario_config.system_type
        self.phi = scenario_config.phi
        self.qs = scenario_config.qs
        
        self._setup_scenario()
        self._calc_normal_load()
        self._calc_tangential_load()
        self._calc_crack_height()
        self.crack_l = scenario_config.crack_length

    def refresh_from_config(self):
        """Pull changed values out of scenario_config
           and recompute derived attributes."""
        self.system_type = self.scenario_config.system_type
        self.phi    = self.scenario_config.phi
        self.qs     = self.scenario_config.qs

        self._setup_scenario()
        self._calc_crack_height()

    def _calc_tangential_load(self):
        """
        Total Tangential Load (Surface Load + Weight Load)
        
        Returns:
        --------
        qt : float
            Tangential Component of Load [N/mm]
        """
        # Surface Load & Weight Load
        qw = self.slab.qw
        qs = self.qs
        
        # Normal components of forces
        phi = self.phi
        _, qwt = decompose_to_normal_tangential(qw, phi)
        _, qst = decompose_to_normal_tangential(qs, phi)
        qt = qwt + qst
        self.qt = qt
    
    def _calc_normal_load(self):
        """
        Total Normal Load (Surface Load + Weight Load)
        
        Returns:
        --------
        qn : float
            Normal Component of Load [N/mm]
        """
        # Surface Load & Weight Load
        qw = self.slab.qw
        qs = self.qs
        
        # Normal components of forces
        phi = self.phi
        qwn, _ = decompose_to_normal_tangential(qw, phi)
        qsn, _ = decompose_to_normal_tangential(qs, phi)
        qn = qwn + qsn
        self.qn = qn

    def _setup_scenario(self):
        self.li = np.array([seg.length for seg in self.segments])
        self.ki = np.array([seg.has_foundation for seg in self.segments])
        # masses that act *between* segments: take all but the last one
        self.mi = np.array([seg.m for seg in self.segments[:-1]])
        
        # Add dummy segment if only one segment provided
        if len(self.li) == 1:
            self.li = np.append(self.li, 0)
            self.ki = np.append(self.ki, True)
            self.mi = np.append(self.mi, 0)
        
        # Calculate the total slab length
        self.L = np.sum(self.li)

    def _calc_crack_height(self):
        """
        Crack Height: Difference between collapsed weak layer and
            Weak Layer (Winkler type) under slab load
        """
        cf = self.scenario_config.collapse_factor
        self.crack_h = cf * self.weak_layer.h - self.qn / self.weak_layer.kn
