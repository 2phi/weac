import logging
from typing import Literal, Optional
from scipy.optimize import brentq

from weac.components.layer import WeakLayer
from weac.components.scenario_config import ScenarioConfig
from weac.components.segment import Segment
from weac.constants import STIFFNESS_COLLAPSE_FACTOR
from weac.core.eigensystem import Eigensystem
from weac.core.field_quantities import FieldQuantities
from weac.core.scenario import Scenario
from weac.core.unknown_constants_solver import UnknownConstantsSolver

logger = logging.getLogger(__name__)


class SlabTouchdown:
    """
    Handling the touchdown situation in a PST.
    Calculations follow paper Rosendahl et al. (2024)
        `The effect of slab touchdown on anticrack arrest in propagation saw tests`

    Types of Touchdown:
        `A_free_hanging` : Slab is free hanging (not in contact with the collapsed weak layer)
            touchdown_distance `=` cut_length -> the unsupported segment (touchdown_distance) equals the cut length
        `B_point_contact` : End of slab is in contact with the collapsed weak layer
            touchdown_distance `=` cut_length -> the unsupported segment (touchdown_distance) equals the cut length
        `C_in_contact` : more of the slab is in contact with the collapsed weak layer
            touchdown_distance `<` cut_length -> the unsupported segment (touchdown_distance) is strictly smaller than the cut length

    The Module does:
    1. Calculation of Zones of modes `[A_free_hanging, B_point_contact, C_in_contact]`::

        |+++++++++++++++++++|-------A-------|-------B-------|--------C-------- [...]
        | supported segment | free-hanging  | point contact |  in contact
                            0            `l_AB`           `l_BC`
        through calculation of boundary touchdown_distance `l_AB` and `l_BC`

    Parameters:
    -----------
    scenario: `Scenario`
    eigensystem: `Eigensystem`

    Attributes:
    -----------
    l_AB : float
        Length of the crack for transition of stage A to stage B [mm]
    l_BC : float
        Length of the crack for transition of stage B to stage C [mm]
    touchdown_mode : Literal["A_free_hanging", "B_point_contact", "C_in_contact"]
        Type of touchdown mode
    touchdown_distance : float
        Length of the touchdown segment [mm]
    collapsed_weak_layer_kR : Optional[float]
        Rotational spring stiffness of the collapsed weak layer segment
    """

    # Inputs
    scenario: Scenario
    eigensystem: Eigensystem

    # Attributes
    collapsed_weak_layer: WeakLayer  # WeakLayer with modified stiffness
    collapsed_eigensystem: Eigensystem
    straight_scenario: Scenario
    l_AB: float
    l_BC: float
    touchdown_mode: Literal[
        "A_free_hanging", "B_point_contact", "C_in_contact"
    ]  # Three types of contact with collapsed weak layer
    touchdown_distance: float
    collapsed_weak_layer_kR: Optional[float] = None

    def __init__(self, scenario: Scenario, eigensystem: Eigensystem):
        self.scenario = scenario
        self.eigensystem = eigensystem

        # Create a new scenario config with phi=0 (flat slab) while preserving other settings
        self.flat_config = ScenarioConfig(
            phi=0.0,  # Flat slab for collapsed scenario
            system_type=self.scenario.scenario_config.system_type,
            cut_length=self.scenario.scenario_config.cut_length,
            stiffness_ratio=self.scenario.scenario_config.stiffness_ratio,
            surface_load=self.scenario.scenario_config.surface_load,
        )

        self.collapsed_eigensystem = self._create_collapsed_eigensystem(
            qs=self.scenario.scenario_config.surface_load,
        )

        self._setup_touchdown_system()

    def _setup_touchdown_system(self):
        """Calculate touchdown"""
        self._calc_touchdown_mode()
        self._calc_touchdown_distance()

    def _calc_touchdown_mode(self):
        """Calculate touchdown-mode from thresholds"""
        # Calculate stage transitions
        try:
            self.l_AB = self._calc_l_AB()
        except ValueError:
            self.l_AB = self.scenario.L
        try:
            self.l_BC = self._calc_l_BC()
        except ValueError:
            self.l_BC = self.scenario.L
        # Assign stage
        if self.scenario.cut_length <= self.l_AB:
            touchdown_mode = "A_free_hanging"
        elif self.l_AB < self.scenario.cut_length <= self.l_BC:
            touchdown_mode = "B_point_contact"
        elif self.l_BC < self.scenario.cut_length:
            touchdown_mode = "C_in_contact"
        self.touchdown_mode = touchdown_mode

    def _calc_touchdown_distance(self):
        """Calculate touchdown distance"""
        if self.touchdown_mode in ["A_free_hanging"]:
            self.touchdown_distance = self.scenario.cut_length
        elif self.touchdown_mode in ["B_point_contact"]:
            self.touchdown_distance = self.scenario.cut_length
        elif self.touchdown_mode in ["C_in_contact"]:
            self.touchdown_distance = self._calc_touchdown_distance_in_mode_C()
            self.collapsed_weak_layer_kR = self._calc_collapsed_weak_layer_kR()

    def _calc_l_AB(self):
        """
        Calc transition lengths l_AB

        Returns
        -------
        l_AB : float
            Length of the crack for transition of stage A to stage B [mm]
        """
        # Unpack variables
        bs = -(self.eigensystem.B11**2 / self.eigensystem.A11 - self.eigensystem.D11)
        ss = self.eigensystem.kA55
        L = self.scenario.L
        crack_h = self.scenario.crack_h
        qn = self.scenario.qn

        # Create polynomial expression
        def polynomial(x: float) -> float:
            # Spring stiffness of uncollapsed eigensystem of length L - x
            straight_scenario = self._generate_straight_scenario(L - x)
            kRl = self._substitute_stiffness(
                straight_scenario, self.eigensystem, "rot"
            )  # rotational stiffness
            kNl = self._substitute_stiffness(
                straight_scenario, self.eigensystem, "trans"
            )  # pulling stiffness
            c1 = 1 / (8 * bs)
            c2 = 1 / (2 * kRl)
            c3 = 1 / (2 * ss)
            c4 = 1 / kNl
            c5 = -crack_h / qn
            return c1 * x**4 + c2 * x**3 + c3 * x**2 + c4 * x + c5

        # Find root
        l_AB = brentq(polynomial, L / 1000, 999 / 1000 * L)

        return l_AB

    def _calc_l_BC(self) -> float:
        """
        Calc transition lengths l_BC

        Returns
        -------
        l_BC : float
            Length of the crack for transition of stage B to stage C [mm]
        """
        # Unpack variables
        bs = -(self.eigensystem.B11**2 / self.eigensystem.A11 - self.eigensystem.D11)
        ss = self.eigensystem.kA55
        L = self.scenario.L
        crack_h = self.scenario.crack_h
        qn = self.scenario.qn

        # Create polynomial function
        def polynomial(x: float) -> float:
            # Spring stiffness of uncollapsed eigensystem of length L - x
            straight_scenario = self._generate_straight_scenario(L - x)
            kRl = self._substitute_stiffness(straight_scenario, self.eigensystem, "rot")
            kNl = self._substitute_stiffness(
                straight_scenario, self.eigensystem, "trans"
            )
            c1 = ss**2 * kRl * kNl * qn
            c2 = 6 * ss**2 * bs * kNl * qn
            c3 = 30 * bs * ss * kRl * kNl * qn
            c4 = 24 * bs * qn * (2 * ss**2 * kRl + 3 * bs * ss * kNl)
            c5 = 72 * bs * (bs * qn * (ss**2 + kRl * kNl) - ss**2 * kRl * kNl * crack_h)
            c6 = 144 * bs * ss * (bs * kRl * qn - bs * ss * kNl * crack_h)
            c7 = -144 * bs**2 * ss * kRl * kNl * crack_h
            return (
                c1 * x**6 + c2 * x**5 + c3 * x**4 + c4 * x**3 + c5 * x**2 + c6 * x + c7
            )

        # Find root
        l_BC = brentq(polynomial, L / 1000, 999 / 1000 * L)

        return l_BC

    def _create_collapsed_eigensystem(self, qs: float) -> Eigensystem:
        """
        Create the collapsed weak layer and eigensystem with modified stiffness values.
        This centralizes all collapsed-related logic within the SlabTouchdown class.
        """
        # Create collapsed weak layer with increased stiffness
        self.collapsed_weak_layer = self.scenario.weak_layer.model_copy(
            update={
                "kn": self.scenario.weak_layer.kn * STIFFNESS_COLLAPSE_FACTOR,
                "kt": self.scenario.weak_layer.kt * STIFFNESS_COLLAPSE_FACTOR,
            }
        )

        # Create eigensystem for the collapsed weak layer
        return Eigensystem(
            weak_layer=self.collapsed_weak_layer, slab=self.scenario.slab
        )

    def _calc_touchdown_distance_in_mode_C(self) -> float:
        """
        Calculate the length of the touchdown element in mode C
        when the slab is in contact.
        """
        # Unpack variables
        bs = -(self.eigensystem.B11**2 / self.eigensystem.A11 - self.eigensystem.D11)
        ss = self.eigensystem.kA55
        L = self.scenario.L
        cut_length = self.scenario.cut_length
        crack_h = self.scenario.crack_h
        qn = self.scenario.qn

        # Spring stiffness of uncollapsed eigensystem of length L - cut_length
        straight_scenario = self._generate_straight_scenario(L - cut_length)
        kRl = self._substitute_stiffness(straight_scenario, self.eigensystem, "rot")
        kNl = self._substitute_stiffness(straight_scenario, self.eigensystem, "trans")

        def polynomial(x: float) -> float:
            logger.info("Eval. Slab Geometry with Touchdown Distance x=%.2f mm", x)
            # Spring stiffness of collapsed eigensystem of length cut_length - x
            straight_scenario = self._generate_straight_scenario(cut_length - x)
            kRr = self._substitute_stiffness(
                straight_scenario, self.collapsed_eigensystem, "rot"
            )
            # define constants
            c1 = ss**2 * kRl * kNl * qn
            c2 = 6 * ss * kNl * qn * (bs * ss + kRl * kRr)
            c3 = 30 * bs * ss * kNl * qn * (kRl + kRr)
            c4 = (
                24
                * bs
                * qn
                * (2 * ss**2 * kRl + 3 * bs * ss * kNl + 3 * kRl * kRr * kNl)
            )
            c5 = (
                72
                * bs
                * (
                    bs * qn * (ss**2 + kNl * (kRl + kRr))
                    + ss * kRl * (2 * kRr * qn - ss * kNl * crack_h)
                )
            )
            c6 = (
                144
                * bs
                * ss
                * (bs * qn * (kRl + kRr) - kNl * crack_h * (bs * ss + kRl * kRr))
            )
            c7 = -144 * bs**2 * ss * kNl * crack_h * (kRl + kRr)
            return (
                c1 * x**6 + c2 * x**5 + c3 * x**4 + c4 * x**3 + c5 * x**2 + c6 * x + c7
            )

        # Find root
        touchdown_distance = brentq(
            polynomial, cut_length / 1000, 999 / 1000 * cut_length
        )

        return touchdown_distance

    def _calc_collapsed_weak_layer_kR(self) -> float:
        """
        Calculate the rotational stiffness of the collapsed weak layer
        """
        straight_scenario = self._generate_straight_scenario(
            self.scenario.cut_length - self.touchdown_distance
        )
        kR = self._substitute_stiffness(
            straight_scenario, self.collapsed_eigensystem, "rot"
        )
        return kR

    def _generate_straight_scenario(self, L: float) -> Scenario:
        """
        Generate a straight scenario with a given length.
        """
        segments = [Segment(length=L, has_foundation=True, m=0)]
        straight_scenario = Scenario(
            scenario_config=self.flat_config,
            segments=segments,
            weak_layer=self.scenario.weak_layer,
            slab=self.scenario.slab,
        )
        return straight_scenario

    def _substitute_stiffness(
        self,
        scenario: Scenario,
        eigensystem: Eigensystem,
        dof: Literal["rot", "trans"] = "rot",
    ) -> float:
        """
        Calc substitute stiffness for beam on elastic foundation.

        Arguments
        ---------
        dof : string
            Type of substitute spring, either 'rot' or 'trans'. Defaults to 'rot'.

        Returns
        -------
        has_foundation : stiffness of substitute spring.
        """

        unknown_constants = UnknownConstantsSolver.solve_for_unknown_constants(
            scenario=scenario, eigensystem=eigensystem, system_type=dof
        )

        # Calculate field quantities at x=0 (left end)
        Zh0 = eigensystem.zh(x=0, length=scenario.L, has_foundation=True)
        zp0 = eigensystem.zp(x=0, phi=0, has_foundation=True, qs=0)
        C_at_x0 = unknown_constants[:, 0].reshape(-1, 1)  # Ensure column vector
        z_at_x0 = Zh0 @ C_at_x0 + zp0

        # Calculate stiffness based on field quantities
        fq = FieldQuantities(eigensystem=eigensystem)

        if dof in ["rot"]:
            # For rotational stiffness: has_foundation = M / psi
            psi_val = fq.psi(z_at_x0)[0]  # Extract scalar value from the result
            has_foundation = abs(1 / psi_val) if abs(psi_val) > 1e-12 else 1e12
        elif dof in ["trans"]:
            # For translational stiffness: has_foundation = V / w
            w_val = fq.w(z_at_x0)[0]  # Extract scalar value from the result
            has_foundation = abs(1 / w_val) if abs(w_val) > 1e-12 else 1e12
        return has_foundation
