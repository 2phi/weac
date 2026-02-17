"""
This module defines the Scenario class, which encapsulates the physical setup of the model.
"""

import logging
from collections.abc import Sequence

import numpy as np

from weac.components import ScenarioConfig, Segment, SystemType, WeakLayer
from weac.core.slab import Slab
from weac.utils.misc import decompose_to_xyz
from weac.constants import G_MM_S2, LSKI_MM
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
    fi : List[np.arrays]
        force vectors at the boundaries of the segments masses (kg) on boundary of segment i and i+1 [N, N N, Nmm, Nmm, Nmm]
    gi : List[bool]
        booleans indicating loading for segment i
    load_vector_left : np.ndarray
        load vector on the left side of the configuration
    load_vector_right : np.ndarray
        load vector on the right side of the configuration

    

    system_type : SystemType
    phi : float
        Angle of slab in positive in counter-clockwise direction [deg]
    theta : float 
        Angle of slab rotation around its axis [deg] 
    L : float
        Length of the model [mm]
    crack_h: float
        Height of the crack [mm]
    """

    # Inputs
    scenario_config: ScenarioConfig
    segments: list[Segment]
    weak_layer: WeakLayer
    slab: Slab

    # Attributes
    li: np.ndarray  # length of segment i [mm]
    ki: np.ndarray  # booleans indicating foundation support for segment i
    gi: np.ndarray  # booleans indicating loading for segment i
    fi: np.ndarray  # load vectors on boundaries of segment i and i+1 
    load_vector_left: np.ndarray # load vector on the left side of the configuration
    load_vector_right: np.ndarray # load vector on the right side of the configuration

    cum_sum_li: np.ndarray  # cumulative sum of segment lengths [mm]

    system_type: SystemType
    phi: float  # Angle in [deg]
    theta: float # Angle in [deg]
    surface_load: float  # Surface Line-Load [N/mm]
    qw: float  # Weight Line-Load [N/mm]
    qx: float # Total Axial Line-Load [N/mm]
    qy: float # Total Transvers Line-Load [N/mm]
    qz: float # Total Vertical Line-Load [N/mm]
    L: float  # Length of the model [mm]
    crack_h: float  # Height of the crack [mm]
    cut_length: float  # Length of the cut [mm]

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        segments: list[Segment],
        weak_layer: WeakLayer,
        slab: Slab,
    ):
        self.scenario_config = scenario_config
        self.segments = segments
        self.weak_layer = weak_layer
        self.slab = slab

        self.system_type = scenario_config.system_type
        self.phi = scenario_config.phi
        self.theta = scenario_config.theta
        self.surface_load = scenario_config.surface_load
        self.cut_length = scenario_config.cut_length
        self.load_vector_left = scenario_config.load_vector_left
        self.load_vector_right = scenario_config.load_vector_right

        self._setup_scenario()
        self._calc_normal_load()
        self._calc_tangential_load()
        self._calc_crack_height()

    def refresh_from_config(self):
        """Pull changed values out of scenario_config
        and recompute derived attributes."""
        self.system_type = self.scenario_config.system_type
        self.phi = self.scenario_config.phi
        self.theta = self.scenario_config.theta
        self.surface_load = self.scenario_config.surface_load
        self.cut_length = self.scenario_config.cut_length
        self.load_vector_left = self.scenario_config.load_vector_left
        self.load_vector_right = self.scenario_config.load_vector_right

        self._setup_scenario()
        self._calc_normal_load()
        self._calc_tangential_load()
        self._calc_crack_height()

    def get_segment_idx(
        self, x: float | Sequence[float] | np.ndarray
    ) -> int | np.ndarray:
        """
        Get the segment index for a given x-coordinate or coordinates.

        Parameters
        ----------
        x: float | Sequence[float] | np.ndarray
            A single x-coordinate or a sequence of x-coordinates.

        Returns
        -------
        int | np.ndarray
            The segment index or an array of indices.
        """
        x_arr = np.asarray(x)
        indices = np.digitize(x_arr, self.cum_sum_li)

        if np.any(x_arr > self.L):
            raise ValueError(f"Coordinate {x_arr} exceeds the slab length.")

        if x_arr.ndim == 0:
            return int(indices)

        return indices

    def _setup_scenario(self):
        self.li = np.array([seg.length for seg in self.segments])
        self.ki = np.array([seg.has_foundation for seg in self.segments])
        self.gi = np.array([seg.is_loaded for seg in self.segments])
        # masses that act *between* segments: take all but the last one
        self.mi = np.array([seg.m for seg in self.segments[:-1]])
        # assume masses attack in the centerline of the upper side of the slab
        F_array = 1e-3 * self.mi * G_MM_S2 / LSKI_MM * self.slab.b
        Fx,Fy,Fz = decompose_to_xyz(F_array,self.phi,self.theta)
        self.fi = np.column_stack([
            Fx,
            Fy,
            Fz,
            Fy * self.slab.H/2,
            -Fx * self.slab.H/2,
            np.zeros_like(Fx),
            ])
        self.cum_sum_li = np.cumsum(self.li)

        # Add dummy segment if only one segment provided
        if len(self.li) == 1:
            self.li = np.append(self.li, 0)
            self.ki = np.append(self.ki, True)
            self.mi = np.append(self.mi, 0)
            self.fi = np.vstack([self.fi, np.zeros((1,6))])

        # Calculate the total slab length
        self.L = np.sum(self.li)

    def _calc_tangential_load(self):
        """
        Total Tangential Load (Surface Load + Weight Load)

        Returns:
        --------
        qx : float
            Axial Component of Load [N/mm]
        """
        # Surface Load & Weight Load
        qw = self.slab.qw
        qs = self.surface_load

        # Vertical components of forces
        phi = self.phi
        theta = self.theta
        qwx, _, _ = decompose_to_xyz(qw, phi, theta)
        qsx, _, _ = decompose_to_xyz(qs, phi, theta)
        qx = qwx + qsx
        self.qx = qx

    def _calc_normal_load(self):
        """
        Total Normal Load (Surface Load + Weight Load)

        Returns:
        --------
        qz: float
            Vertical Component of Load [N/mm]
        """
        # Surface Load & Weight Load
        qw = self.slab.qw
        qs = self.surface_load

        # Normal components of forces
        phi = self.phi
        theta = self.theta
        _, _, qwz = decompose_to_xyz(qw, phi, theta)
        _, _, qsz = decompose_to_xyz(qs, phi, theta)
        qz = qwz + qsz
        self.qz = qz

    def _calc_crack_height(self):
        """
        Crack Height: Difference between collapsed weak layer and
            Weak Layer (Winkler type) under slab load

        Example:
        if the collapse layer has a height of 5 and the non-collapsed layer
        has a height of 15 the collapse height is 10
        """
        self.crack_h = self.weak_layer.collapse_height - self.qz / self.weak_layer.kn
        if self.crack_h < 0:
            raise ValueError(
                f"Crack height is negative: {self.crack_h} decrease the surface load"
            )
