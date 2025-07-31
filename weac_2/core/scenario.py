import logging
from typing import List, Literal, Sequence, Union

import numpy as np

from weac_2.components import ScenarioConfig, Segment, WeakLayer
from weac_2.core.slab import Slab
from weac_2.utils.misc import decompose_to_normal_tangential

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

    system_type : Literal['skier', 'skiers', 'pst-', '-pst', 'rot', 'trans']
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
    li: np.ndarray  # length of segment i [mm]
    ki: np.ndarray  # booleans indicating foundation support for segment i
    mi: np.ndarray  # skier masses (kg) on boundary of segment i and i+1 [kg]

    cum_sum_li: np.ndarray  # cumulative sum of segment lengths [mm]

    system_type: Literal[
        "skier", "skiers", "pst-", "-pst", "vpst-", "-vpst", "rot", "trans"
    ]
    phi: float  # Angle in [deg]
    surface_load: float  # Surface Line-Load [N/mm]
    qw: float  # Weight Line-Load [N/mm]
    qn: float  # Total Normal Line-Load [N/mm]
    qt: float  # Total Tangential Line-Load [N/mm]
    L: float  # Length of the model [mm]
    crack_h: float  # Height of the crack [mm]
    crack_l: float  # Length of the crack [mm]

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        segments: List[Segment],
        weak_layer: WeakLayer,
        slab: Slab,
    ):
        self.scenario_config = scenario_config
        self.segments = segments
        self.weak_layer = weak_layer
        self.slab = slab

        self.system_type = scenario_config.system_type
        self.phi = scenario_config.phi
        self.surface_load = scenario_config.surface_load

        self._setup_scenario()
        self._calc_normal_load()
        self._calc_tangential_load()
        self._calc_crack_height()
        self.crack_length = scenario_config.crack_length

    def refresh_from_config(self):
        """Pull changed values out of scenario_config
        and recompute derived attributes."""
        self.system_type = self.scenario_config.system_type
        self.phi = self.scenario_config.phi
        self.surface_load = self.scenario_config.surface_load

        self._setup_scenario()
        self._calc_crack_height()

    def get_segment_idx(
        self, x: Union[float, Sequence[float], np.ndarray]
    ) -> Union[int, np.ndarray]:
        """
        Get the segment index for a given x-coordinate or coordinates.

        Parameters
        ----------
        x: Union[float, Sequence[float], np.ndarray]
            A single x-coordinate or a sequence of x-coordinates.

        Returns
        -------
        Union[int, np.ndarray]
            The segment index or an array of indices.
        """
        x_arr = np.asarray(x)
        indices = np.digitize(x_arr, self.cum_sum_li)

        if np.any(x_arr > self.L):
            raise ValueError(f"Coordinate {x_arr} is outside the slab length.")

        if x_arr.ndim == 0:
            return int(indices)

        return indices

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
        qs = self.surface_load

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
        qs = self.surface_load

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
        self.cum_sum_li = np.cumsum(self.li)

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
        # TODO: Is crack height the height of the collapsed weak layer or the height the height that is lost on collapse?
        collapsed_height = self.weak_layer.h - self.weak_layer.collapse_height
        self.crack_h = collapsed_height - self.qn / self.weak_layer.kn
