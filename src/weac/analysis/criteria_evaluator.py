"""
This module provides the CriteriaEvaluator class, which is used to evaluate various
fracture criteria based on the model results.
"""

# Standard library imports
import copy
import logging
import time
import warnings
from dataclasses import dataclass

# Third party imports
import numpy as np
from scipy.optimize import brentq, root_scalar

from weac.analysis.analyzer import Analyzer

# weac imports
from weac.components import (
    CriteriaConfig,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac.constants import RHO_ICE
from weac.core.system_model import SystemModel

logger = logging.getLogger(__name__)


@dataclass
class CoupledCriterionHistory:
    """Stores the history of the coupled criterion evaluation."""

    skier_weights: list[float]
    crack_lengths: list[float]
    incr_energies: list[np.ndarray]
    g_deltas: list[float]
    dist_maxs: list[float]
    dist_mins: list[float]


@dataclass
class CoupledCriterionResult:
    """
    Holds the results of the coupled criterion evaluation.

    Attributes:
    -----------
    converged : bool
        Whether the algorithm converged.
    message : str
        The message of the evaluation.
    self_collapse : bool
        Whether the system collapsed.
    pure_stress_criteria : bool
        Whether the pure stress criteria is satisfied.
    critical_skier_weight : float
        The critical skier weight.
    initial_critical_skier_weight : float
        The initial critical skier weight.
    crack_length : float
        The crack length.
    g_delta : float
        The g_delta value.
    dist_ERR_envelope : float
        The distance to the ERR envelope.
    iterations : int
        The number of iterations.
    history : CoupledCriterionHistory
        The history of the evaluation.
    final_system : SystemModel
        The final system model.
    max_dist_stress : float
        The maximum distance to failure.
    min_dist_stress : float
        The minimum distance to failure.
    """

    converged: bool
    message: str
    self_collapse: bool
    pure_stress_criteria: bool
    critical_skier_weight: float
    initial_critical_skier_weight: float
    crack_length: float
    g_delta: float
    dist_ERR_envelope: float
    iterations: int
    history: CoupledCriterionHistory | None
    final_system: SystemModel
    max_dist_stress: float
    min_dist_stress: float


@dataclass
class MaximalStressResult:
    """
    Holds the results of the maximal stress evaluation.

    Attributes:
    -----------
    principal_stress_kPa: np.ndarray
        The principal stress in kPa.
    Sxx_kPa: np.ndarray
        The axial normal stress in kPa.
    principal_stress_norm: np.ndarray
        The normalized principal stress to the tensile strength of the layers.
    Sxx_norm: np.ndarray
        The normalized axial normal stress to the tensile strength of the layers.
    max_principal_stress_norm: float
        The normalized maximum principal stress to the tensile strength of the layers.
    max_Sxx_norm: float
        The normalized maximum axial normal stress to the tensile strength of the layers.
    """

    principal_stress_kPa: np.ndarray
    Sxx_kPa: np.ndarray
    principal_stress_norm: np.ndarray
    Sxx_norm: np.ndarray
    max_principal_stress_norm: float
    max_Sxx_norm: float


@dataclass
class SteadyStateResult:
    """
    Holds the results of the Steady State evaluation.

    Attributes:
    -----------
    converged : bool
        Whether the algorithm converged.
    message : str
        The message of the evaluation.
    touchdown_distance : float
        The touchdown distance.
    energy_release_rate : float
        The steady-state energy release rate calculated with the
        touchdown distance from the differential energy release rate.
    maximal_stress_result: MaximalStressResult
        The maximal stresses in the system at the touchdown distance.
    system: SystemModel
        The modified system model used for the steady state evaluation.
    """

    converged: bool
    message: str
    touchdown_distance: float
    energy_release_rate: float
    maximal_stress_result: MaximalStressResult
    system: SystemModel


@dataclass
class FindMinimumForceResult:
    """
    Holds the results of the find_minimum_force evaluation.

    Attributes:
    -----------
    success : bool
        Whether the algorithm converged.
    critical_skier_weight : float
        The critical skier weight.
    new_segments : list[Segment]
        The new segments.
    old_segments : list[Segment]
        The old segments.
    iterations : int
        The number of iterations.
    max_dist_stress : float
        The maximum distance to failure.
    min_dist_stress : float
        The minimum distance to failure.
    """

    success: bool
    critical_skier_weight: float
    new_segments: list[Segment]
    old_segments: list[Segment]
    iterations: int | None
    max_dist_stress: float
    min_dist_stress: float


class CriteriaEvaluator:
    """
    Provides methods for stability analysis of layered slabs on compliant
    elastic foundations, based on the logic from criterion_check.py.
    """

    criteria_config: CriteriaConfig

    def __init__(self, criteria_config: CriteriaConfig):
        """
        Initializes the evaluator with global simulation and criteria configurations.

        Parameters:
        ----------
        criteria_config (CriteriaConfig): The configuration for failure criteria.
        """
        self.criteria_config = criteria_config

    def fracture_toughness_envelope(
        self, G_I: float | np.ndarray, G_II: float | np.ndarray, weak_layer: WeakLayer
    ) -> float | np.ndarray:
        """
        Evaluates the fracture toughness criterion for a given combination of
        Mode I (G_I) and Mode II (G_II) energy release rates.

        The criterion is defined as:
            g_delta = (|G_I| / G_Ic)^gn + (|G_II| / G_IIc)^gm

        A value of 1 indicates the boundary of the fracture toughness envelope is reached.

        Parameters:
        -----------
        G_I : float
            Mode I energy release rate (ERR) in J/m².
        G_II : float
            Mode II energy release rate (ERR) in J/m².
        weak_layer : WeakLayer
            The weak layer object containing G_Ic and G_IIc.

        Returns:
        -------
        g_delta : float
            Evaluation of the fracture toughness envelope.
        """
        g_delta = (np.abs(G_I) / weak_layer.G_Ic) ** self.criteria_config.gn + (
            np.abs(G_II) / weak_layer.G_IIc
        ) ** self.criteria_config.gm

        return g_delta

    def stress_envelope(
        self,
        sigma: float | np.ndarray,
        tau: float | np.ndarray,
        weak_layer: WeakLayer,
        method: str | None = None,
    ) -> np.ndarray:
        """
        Evaluate the stress envelope for given stress components.
        Weak Layer failure is defined as the stress envelope crossing 1.

        Parameters
        ----------
        sigma: ndarray
            Normal stress components (kPa).
        tau: ndarray
            Shear stress components (kPa).
        weak_layer: WeakLayer
            The weak layer object, used to get density.
        method: str, optional
            Method to use for the stress envelope. Defaults to None.

        Returns
        -------
        stress_envelope: ndarray
            Stress envelope evaluation values in [0, inf].
            Values > 1 indicate failure.

        Notes
        -----
        - Mede's envelopes ('mede_s-RG1', 'mede_s-RG2', 'mede_s-FCDH') are derived
            from the work of Mede et al. (2018), "Snow Failure Modes Under Mixed
            Loading," published in Geophysical Research Letters.
        - Schöttner's envelope ('schottner') is based on the preprint by Schöttner
            et al. (2025), "On the Compressive Strength of Weak Snow Layers of
            Depth Hoar".
        - The 'adam_unpublished' envelope scales with weak layer density linearly
            (compared to density baseline) by a 'scaling_factor'
            (weak layer density / density baseline), unless modified by
            'order_of_magnitude'.
        - Mede's criteria ('mede_s-RG1', 'mede_s-RG2', 'mede_s-FCDH') define
            failure based on a piecewise function of stress ranges.
        """
        sigma = np.abs(np.asarray(sigma))
        tau = np.abs(np.asarray(tau))
        results = np.zeros_like(sigma)

        envelope_method = (
            method
            if method is not None
            else self.criteria_config.stress_envelope_method
        )
        density = weak_layer.rho
        sigma_c = weak_layer.sigma_c
        tau_c = weak_layer.tau_c
        fn = self.criteria_config.fn
        fm = self.criteria_config.fm
        order_of_magnitude = self.criteria_config.order_of_magnitude
        scaling_factor = self.criteria_config.scaling_factor

        def mede_common_calculations(sigma, tau, p0, tau_T, p_T):
            results_local = np.zeros_like(sigma)
            in_first_range = (sigma >= (p_T - p0)) & (sigma <= p_T)
            in_second_range = sigma > p_T
            results_local[in_first_range] = (
                -tau[in_first_range] * (p0 / (tau_T * p_T))
                + sigma[in_first_range] * (1 / p_T)
                + p0 / p_T
            )
            results_local[in_second_range] = (tau[in_second_range] ** 2) + (
                (tau_T / p0) ** 2
            ) * ((sigma[in_second_range] - p_T) ** 2)
            return results_local

        if envelope_method == "adam_unpublished":
            if scaling_factor > 1:
                order_of_magnitude = 0.7
            scaling_factor = max(scaling_factor, 0.55)
            scaled_sigma_c = sigma_c * (scaling_factor**order_of_magnitude)
            scaled_tau_c = tau_c * (scaling_factor**order_of_magnitude)
            return (sigma / scaled_sigma_c) ** fn + (tau / scaled_tau_c) ** fm

        if envelope_method == "schottner":
            sigma_y = 2000
            scaled_sigma_c = sigma_y * 13 * (density / RHO_ICE) ** order_of_magnitude
            scaled_tau_c = tau_c * (scaled_sigma_c / sigma_c)
            return (sigma / scaled_sigma_c) ** fn + (tau / scaled_tau_c) ** fm

        if envelope_method == "mede_s-RG1":
            p0, tau_T, p_T = 7.00, 3.53, 1.49
            results = mede_common_calculations(sigma, tau, p0, tau_T, p_T)
            return results

        if envelope_method == "mede_s-RG2":
            p0, tau_T, p_T = 2.33, 1.22, 0.19
            results = mede_common_calculations(sigma, tau, p0, tau_T, p_T)
            return results

        if envelope_method == "mede_s-FCDH":
            p0, tau_T, p_T = 1.45, 0.61, 0.17
            results = mede_common_calculations(sigma, tau, p0, tau_T, p_T)
            return results

        raise ValueError(f"Invalid envelope type: {envelope_method}")

    def evaluate_coupled_criterion(
        self,
        system: SystemModel,
        max_iterations: int = 25,
        damping_ERR: float = 0.0,
        tolerance_ERR: float = 0.002,
        tolerance_stress: float = 0.005,
        print_call_stats: bool = False,
        _recursion_depth: int = 0,
    ) -> CoupledCriterionResult:
        """
        Evaluates the coupled criterion for anticrack nucleation, finding the
        critical combination of skier weight and anticrack length.

        Parameters:
        ----------
        system: SystemModel
            The system model.
        max_iterations: int
            Max iterations for the solver. Defaults to 25.
        damping_ERR: float
            damping factor for the ERR criterion. Defaults to 0.0.
        tolerance_ERR: float, optional
            Tolerance for g_delta convergence. Defaults to 0.002.
        tolerance_stress: float, optional
            Tolerance for stress envelope convergence. Defaults to 0.005.
        print_call_stats: bool
            Whether to print the call statistics. Defaults to False.
        _recursion_depth: int
            The depth of the recursion. Defaults to 0.

        Returns
        -------
        results: CoupledCriterionResult
            An object containing the results of the analysis, including
            critical skier weight, crack length, and convergence details.
        """
        logger.info("Starting coupled criterion evaluation.")
        L = system.scenario.L
        weak_layer = system.weak_layer

        logger.info("Finding minimum force...")
        force_finding_start = time.time()

        force_result = self.find_minimum_force(
            system, tolerance_stress=tolerance_stress, print_call_stats=print_call_stats
        )
        initial_critical_skier_weight = force_result.critical_skier_weight
        max_dist_stress = force_result.max_dist_stress
        min_dist_stress = force_result.min_dist_stress
        logger.info(
            "Minimum force finding took %.4f seconds.",
            time.time() - force_finding_start,
        )

        analyzer = Analyzer(system, printing_enabled=print_call_stats)
        # --- Failure: in finding the critical skier weight ---
        if not force_result.success:
            logger.warning("No critical skier weight found")
            analyzer.print_call_stats(
                message="evaluate_coupled_criterion Call Statistics"
            )
            return CoupledCriterionResult(
                converged=False,
                message="Failed to find critical skier weight.",
                self_collapse=False,
                pure_stress_criteria=False,
                critical_skier_weight=0,
                initial_critical_skier_weight=0,
                crack_length=0,
                g_delta=0,
                dist_ERR_envelope=1,
                iterations=0,
                history=None,
                final_system=system,
                max_dist_stress=0,
                min_dist_stress=0,
            )

        # --- Exception: the entire solution is cracked ---
        if min_dist_stress > 1:
            logger.info("The entire solution is cracked.")
            # --- Larger scenario to calculate the incremental ERR ---
            segments = copy.deepcopy(system.scenario.segments)
            for segment in segments:
                segment.has_foundation = False
            # Add 50m of padding to the left and right of the system
            segments.insert(0, Segment(length=50000, has_foundation=True, m=0))
            segments.append(Segment(length=50000, has_foundation=True, m=0))
            system.update_scenario(segments=segments)

            inc_energy = analyzer.incremental_ERR(unit="J/m^2")
            g_delta = self.fracture_toughness_envelope(
                inc_energy[1], inc_energy[2], system.weak_layer
            )

            history_data = CoupledCriterionHistory([], [], [], [], [], [])
            analyzer.print_call_stats(
                message="evaluate_coupled_criterion Call Statistics"
            )
            return CoupledCriterionResult(
                converged=True,
                message="System fails under its own weight (self-collapse).",
                self_collapse=True,
                pure_stress_criteria=False,
                critical_skier_weight=0,
                initial_critical_skier_weight=initial_critical_skier_weight,
                crack_length=L,
                g_delta=g_delta,
                dist_ERR_envelope=0,
                iterations=0,
                history=history_data,
                final_system=system,
                max_dist_stress=max_dist_stress,
                min_dist_stress=min_dist_stress,
            )

        # --- Main loop ---
        crack_length = 1.0
        dist_ERR_envelope = 1000
        g_delta = 0
        history = CoupledCriterionHistory([], [], [], [], [], [])
        iteration_count = 0
        skier_weight = initial_critical_skier_weight * 1.005
        min_skier_weight = 1e-6
        max_skier_weight = 200

        # Ensure Max Weight surpasses fracture toughness criterion
        max_weight_g_delta = 0
        while max_weight_g_delta < 1:
            max_skier_weight = max_skier_weight * 2

            segments = [
                Segment(length=L / 2 - crack_length / 2, has_foundation=True, m=0),
                Segment(
                    length=crack_length / 2,
                    has_foundation=False,
                    m=max_skier_weight,
                ),
                Segment(length=crack_length / 2, has_foundation=False, m=0),
                Segment(length=L / 2 - crack_length / 2, has_foundation=True, m=0),
            ]

            system.update_scenario(segments=segments)

            # Calculate fracture toughness criterion
            incr_energy = analyzer.incremental_ERR(unit="J/m^2")
            max_weight_g_delta = self.fracture_toughness_envelope(
                incr_energy[1], incr_energy[2], weak_layer
            )
            dist_ERR_envelope = abs(max_weight_g_delta - 1)

        logger.info("Max weight to look at: %.2f kg", max_skier_weight)
        segments = [
            Segment(
                length=L / 2 - crack_length / 2,
                has_foundation=True,
                m=0.0,
            ),
            Segment(length=crack_length / 2, has_foundation=False, m=skier_weight),
            Segment(length=crack_length / 2, has_foundation=False, m=0),
            Segment(length=L / 2 - crack_length / 2, has_foundation=True, m=0),
        ]

        while (
            abs(dist_ERR_envelope) > tolerance_ERR
            and iteration_count < max_iterations
            and any(s.has_foundation for s in segments)
        ):
            iteration_count += 1
            iter_start_time = time.time()
            logger.info(
                "Starting iteration %d of coupled criterion evaluation.",
                iteration_count,
            )

            system.update_scenario(segments=segments)
            _, z, _ = analyzer.rasterize_solution(mode="uncracked", num=2000)

            # Calculate stress envelope
            sigma_kPa = system.fq.sig(z, unit="kPa")
            tau_kPa = system.fq.tau(z, unit="kPa")
            stress_env = self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
            max_dist_stress = np.max(stress_env)
            min_dist_stress = np.min(stress_env)

            # Calculate fracture toughness criterion
            incr_energy = analyzer.incremental_ERR(unit="J/m^2")
            g_delta = self.fracture_toughness_envelope(
                incr_energy[1], incr_energy[2], weak_layer
            )
            dist_ERR_envelope = abs(g_delta - 1)

            # Update history
            history.skier_weights.append(skier_weight)
            history.crack_lengths.append(crack_length)
            history.incr_energies.append(incr_energy)
            history.g_deltas.append(g_delta)
            history.dist_maxs.append(max_dist_stress)
            history.dist_mins.append(min_dist_stress)

            # --- Exception: pure stress criterion ---
            # The fracture toughness is superseded for minimum critical skier weight
            if iteration_count == 1 and (g_delta > 1 or dist_ERR_envelope < 0.02):
                analyzer.print_call_stats(
                    message="evaluate_coupled_criterion Call Statistics"
                )
                return CoupledCriterionResult(
                    converged=True,
                    message="Fracture governed by pure stress criterion.",
                    self_collapse=False,
                    pure_stress_criteria=True,
                    critical_skier_weight=skier_weight,
                    initial_critical_skier_weight=initial_critical_skier_weight,
                    crack_length=crack_length,
                    g_delta=g_delta,
                    dist_ERR_envelope=dist_ERR_envelope,
                    iterations=iteration_count,
                    history=history,
                    final_system=system,
                    max_dist_stress=max_dist_stress,
                    min_dist_stress=min_dist_stress,
                )

            # Update skier weight boundaries
            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            # Update skier weight
            new_skier_weight = (min_skier_weight + max_skier_weight) / 2

            # Apply damping to avoid oscillation around goal
            if np.abs(dist_ERR_envelope) < 0.5 and damping_ERR > 0:
                scaling = (damping_ERR + 1 + (new_skier_weight / skier_weight)) / (
                    damping_ERR + 2
                )
            else:
                scaling = 1

            # Find new anticrack length
            if abs(dist_ERR_envelope) > tolerance_ERR:
                skier_weight = scaling * new_skier_weight
                crack_length, segments = self.find_crack_length_for_weight(
                    system, skier_weight
                )
            logger.info("New skier weight: %.2f kg", skier_weight)
            logger.info(
                "Iteration %d took %.4f seconds.",
                iteration_count,
                time.time() - iter_start_time,
            )

        if iteration_count < max_iterations and any(s.has_foundation for s in segments):
            logger.info("No Exception encountered - Converged successfully.")
            if crack_length > 0:
                analyzer.print_call_stats(
                    message="evaluate_coupled_criterion Call Statistics"
                )
                return CoupledCriterionResult(
                    converged=True,
                    message="No Exception encountered - Converged successfully.",
                    self_collapse=False,
                    pure_stress_criteria=False,
                    critical_skier_weight=skier_weight,
                    initial_critical_skier_weight=initial_critical_skier_weight,
                    crack_length=crack_length,
                    g_delta=g_delta,
                    dist_ERR_envelope=dist_ERR_envelope,
                    iterations=iteration_count,
                    history=history,
                    final_system=system,
                    max_dist_stress=max_dist_stress,
                    min_dist_stress=min_dist_stress,
                )
            if _recursion_depth < 5:
                logger.info("Reached max damping without converging.")
                analyzer.print_call_stats(
                    message="evaluate_coupled_criterion Call Statistics"
                )
                return self.evaluate_coupled_criterion(
                    system,
                    damping_ERR=damping_ERR + 1,
                    tolerance_ERR=tolerance_ERR,
                    tolerance_stress=tolerance_stress,
                    _recursion_depth=_recursion_depth + 1,
                )
            analyzer.print_call_stats(
                message="evaluate_coupled_criterion Call Statistics"
            )
            return CoupledCriterionResult(
                converged=False,
                message="Reached max damping without converging.",
                self_collapse=False,
                pure_stress_criteria=False,
                critical_skier_weight=0,
                initial_critical_skier_weight=initial_critical_skier_weight,
                crack_length=crack_length,
                g_delta=g_delta,
                dist_ERR_envelope=dist_ERR_envelope,
                iterations=iteration_count,
                history=history,
                final_system=system,
                max_dist_stress=max_dist_stress,
                min_dist_stress=min_dist_stress,
            )
        if not any(s.has_foundation for s in segments):
            analyzer.print_call_stats(
                message="evaluate_coupled_criterion Call Statistics"
            )
            return CoupledCriterionResult(
                converged=False,
                message="Reached max iterations without converging.",
                self_collapse=False,
                pure_stress_criteria=False,
                critical_skier_weight=0,
                initial_critical_skier_weight=initial_critical_skier_weight,
                crack_length=0,
                g_delta=0,
                dist_ERR_envelope=1,
                iterations=iteration_count,
                history=history,
                final_system=system,
                max_dist_stress=max_dist_stress,
                min_dist_stress=min_dist_stress,
            )
        analyzer.print_call_stats(message="evaluate_coupled_criterion Call Statistics")
        return self.evaluate_coupled_criterion(
            system,
            damping_ERR=damping_ERR + 1,
            tolerance_ERR=tolerance_ERR,
            tolerance_stress=tolerance_stress,
            _recursion_depth=_recursion_depth + 1,
        )

    def evaluate_SteadyState(
        self,
        system: SystemModel,
        vertical: bool = False,
        print_call_stats: bool = False,
    ) -> SteadyStateResult:
        """
        Evaluates the Touchdown Distance in the Steady State and the Steady State
        Energy Release Rate.

        Parameters:
        -----------
        system: SystemModel
            The system model.
        vertical: bool, optional
            Whether to evaluate the system in a vertical configuration.
            Defaults to False.
        print_call_stats: bool, optional
            Whether to print the call statistics. Defaults to False.

        IMPORTANT: There is a bug in vertical = True, so always slope normal,
        i.e. vertical=False should be used.
        """
        if vertical:
            warnings.warn(
                "vertical=True mode is currently buggy — results may be invalid. "
                "Please set vertical=False until this is fixed.",
                UserWarning,
            )
        system_copy = copy.deepcopy(system)
        system_copy.config.touchdown = True
        system_copy.update_scenario(scenario_config=ScenarioConfig(phi=0.0))
        l_BC = system.slab_touchdown.l_BC

        segments = [
            Segment(length=5e3, has_foundation=True, m=0.0),
            Segment(length=2 * l_BC, has_foundation=False, m=0.0),
        ]
        scenario_config = ScenarioConfig(
            system_type="vpst-" if vertical else "pst-",
            phi=0.0,  # Slab Touchdown works only for flat slab
            cut_length=2 * l_BC,
        )
        # system_copy.config.touchdown = True
        system_copy.update_scenario(segments=segments, scenario_config=scenario_config)
        touchdown_distance = system_copy.slab_touchdown.touchdown_distance
        analyzer = Analyzer(system_copy, printing_enabled=print_call_stats)
        energy_release_rate, _, _ = analyzer.differential_ERR(unit="J/m^2")
        maximal_stress_result = self._calculate_maximal_stresses(system_copy)
        return SteadyStateResult(
            converged=True,
            message="Steady State evaluation successful.",
            touchdown_distance=touchdown_distance,
            energy_release_rate=energy_release_rate,
            maximal_stress_result=maximal_stress_result,
            system=system_copy,
        )

    def find_minimum_force(
        self,
        system: SystemModel,
        tolerance_stress: float = 0.0005,
        print_call_stats: bool = False,
    ) -> FindMinimumForceResult:
        """
        Finds the minimum skier weight required to surpass the stress failure envelope.

        This method iteratively adjusts the skier weight until the maximum distance
        to the stress envelope converges to 1, indicating the critical state.

        Parameters:
        -----------
        system: SystemModel
            The system model.
        tolerance_stress: float, optional
            Tolerance for the stress envelope. Defaults to 0.005.
        print_call_stats: bool, optional
            Whether to print the call statistics. Defaults to False.

        Returns:
        --------
        results: FindMinimumForceResult
            An object containing the results of the analysis, including
            critical skier weight, and convergence details.
        """
        logger.info("Start: Find Minimum force to surpass Stress Env.")
        old_segments = copy.deepcopy(system.scenario.segments)
        total_length = system.scenario.L
        analyzer = Analyzer(system, printing_enabled=print_call_stats)

        # --- Initial uncracked configuration ---
        segments = [
            Segment(length=total_length / 2, has_foundation=True, m=0.0),
            Segment(length=total_length / 2, has_foundation=True, m=0.0),
        ]
        system.update_scenario(segments=segments)
        _, z_skier, _ = analyzer.rasterize_solution(mode="uncracked", num=2000)
        sigma_kPa = system.fq.sig(z_skier, unit="kPa")
        tau_kPa = system.fq.tau(z_skier, unit="kPa")
        max_dist_stress = np.max(
            self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
        )
        min_dist_stress = np.min(
            self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
        )

        # --- Early Exit: entire domain is cracked ---
        if min_dist_stress >= 1:
            analyzer.print_call_stats(
                message="min_dist_stress >= 1 in find_minimum_force Call Statistics"
            )
            return FindMinimumForceResult(
                success=True,
                critical_skier_weight=0.0,
                new_segments=segments,
                old_segments=old_segments,
                iterations=0,
                max_dist_stress=max_dist_stress,
                min_dist_stress=min_dist_stress,
            )

        def stress_envelope_residual(skier_weight: float, system: SystemModel) -> float:
            logger.info("Eval. Stress Envelope for weight %.2f kg.", skier_weight)
            segments = [
                Segment(length=total_length / 2, has_foundation=True, m=skier_weight),
                Segment(length=total_length / 2, has_foundation=True, m=0.0),
            ]
            system.update_scenario(segments=segments)
            _, z_skier, _ = analyzer.rasterize_solution(mode="cracked", num=2000)
            sigma_kPa = system.fq.sig(z_skier, unit="kPa")
            tau_kPa = system.fq.tau(z_skier, unit="kPa")
            max_dist = np.max(
                self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
            )
            return max_dist - 1

        # Now do root finding with brentq
        def root_fn(weight):
            return stress_envelope_residual(weight, system)

        # Search interval
        w_min = 0.0
        w_max = 300.0
        while True:
            try:
                critical_weight = brentq(root_fn, w_min, w_max, xtol=tolerance_stress)
                break
            except ValueError as exc:
                w_max = w_max * 2
                if w_max > 10000:
                    raise ValueError(
                        "No sign change found in [w_min, w_max]. Cannot use brentq."
                    ) from exc

        # Final evaluation
        logger.info("Final evaluation for skier weight %.2f kg.", critical_weight)
        system.update_scenario(
            segments=[
                Segment(
                    length=total_length / 2, has_foundation=True, m=critical_weight
                ),
                Segment(length=total_length / 2, has_foundation=True, m=0.0),
            ]
        )
        _, z_skier, _ = analyzer.rasterize_solution(mode="cracked", num=2000)
        sigma_kPa = system.fq.sig(z_skier, unit="kPa")
        tau_kPa = system.fq.tau(z_skier, unit="kPa")
        max_dist_stress = np.max(
            self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
        )
        min_dist_stress = np.min(
            self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
        )

        analyzer.print_call_stats(message="find_minimum_force Call Statistics")
        return FindMinimumForceResult(
            success=True,
            critical_skier_weight=critical_weight,
            new_segments=copy.deepcopy(system.scenario.segments),
            old_segments=old_segments,
            iterations=None,
            max_dist_stress=max_dist_stress,
            min_dist_stress=min_dist_stress,
        )

    def find_minimum_crack_length(
        self,
        system: SystemModel,
        search_interval: tuple[float, float] | None = None,
        target: float = 1,
    ) -> tuple[float, list[Segment]]:
        """
        Finds the minimum crack length required to surpass the energy release rate envelope.

        Parameters:
        -----------
        system: SystemModel
            The system model.

        Returns:
        --------
        minimum_crack_length: float
            The minimum crack length required to surpass the energy release rate envelope [mm]
        new_segments: list[Segment]
            The updated list of segments
        """
        old_segments = copy.deepcopy(system.scenario.segments)

        if search_interval is None:
            a = 0
            b = system.scenario.L / 2
        else:
            a, b = search_interval
        logger.info("Interval for crack length search: %s, %s", a, b)
        logger.info(
            "Calculation of fracture toughness envelope: %s, %s",
            self._fracture_toughness_exceedance(a, system),
            self._fracture_toughness_exceedance(b, system),
        )

        # Use root_scalar to find the root
        result = root_scalar(
            self._fracture_toughness_exceedance,
            args=(system, target),
            bracket=[a, b],  # Interval where the root is expected
            method="brentq",  # Brent's method
        )

        new_segments = system.scenario.segments

        system.update_scenario(segments=old_segments)

        if result.converged:
            return result.root, new_segments
        logger.error("Root search did not converge.")
        return 0.0, new_segments

    def check_crack_self_propagation(
        self,
        system: SystemModel,
        rm_skier_weight: bool = False,
    ) -> tuple[float, bool]:
        """
        Evaluates whether a crack will propagate without any additional load.
        This method determines if a pre-existing crack will propagate without any
        additional load.

        Parameters:
        ----------
        system: SystemModel

        Returns
        -------
        g_delta_diff: float
            The evaluation of the fracture toughness envelope.
        can_propagate: bool
            True if the criterion is met (g_delta_diff >= 1).
        """
        logger.info("Checking for self-propagation of pre-existing crack.")
        new_system = copy.deepcopy(system)
        logger.debug("Segments: %s", new_system.scenario.segments)

        start_time = time.time()
        # No skier weight is applied for self-propagation check
        if rm_skier_weight:
            for seg in new_system.scenario.segments:
                seg.m = 0
        new_system.update_scenario(segments=new_system.scenario.segments)

        analyzer = Analyzer(new_system)
        diff_energy = analyzer.differential_ERR(unit="J/m^2")
        G_I = diff_energy[1]
        G_II = diff_energy[2]

        # Evaluate the fracture toughness criterion
        g_delta_diff = self.fracture_toughness_envelope(
            G_I, G_II, new_system.weak_layer
        )
        can_propagate = g_delta_diff >= 1
        logger.info(
            "Self-propagation check finished in %.4f seconds. Result: "
            "g_delta_diff=%.4f, can_propagate=%s",
            time.time() - start_time,
            g_delta_diff,
            can_propagate,
        )

        return g_delta_diff, bool(can_propagate)

    def find_crack_length_for_weight(
        self,
        system: SystemModel,
        skier_weight: float,
    ) -> tuple[float, list[Segment]]:
        """
        Finds the resulting anticrack length and updated segment configurations
        for a given skier weight.

        Parameters:
        -----------
        system: SystemModel
            The system model.
        skier_weight: float
            The weight of the skier [kg]

        Returns
        -------
        new_crack_length: float
            The total length of the new cracked segments [mm]
        new_segments: list[Segment]
            The updated list of segments
        """
        logger.info(
            "Finding new anticrack length for skier weight %.2f kg.", skier_weight
        )
        start_time = time.time()
        total_length = system.scenario.L
        weak_layer = system.weak_layer

        old_segments = copy.deepcopy(system.scenario.segments)

        initial_segments = [
            Segment(length=total_length / 2, has_foundation=True, m=skier_weight),
            Segment(length=total_length / 2, has_foundation=True, m=0),
        ]
        system.update_scenario(segments=initial_segments)

        analyzer = Analyzer(system)
        _, z, _ = analyzer.rasterize_solution(mode="cracked", num=2000)
        sigma_kPa = system.fq.sig(z, unit="kPa")
        tau_kPa = system.fq.tau(z, unit="kPa")
        min_dist_stress = np.min(self.stress_envelope(sigma_kPa, tau_kPa, weak_layer))

        # Find all points where the stress envelope is crossed
        crossings_start_time = time.time()
        roots = self._find_stress_envelope_crossings(system, weak_layer)
        logger.info(
            "Finding stress envelope crossings took %.4f seconds.",
            time.time() - crossings_start_time,
        )

        # --- Standard case: if roots exist ---
        if len(roots) > 0:
            # Reconstruct segments based on the roots
            midpoint_load_application = total_length / 2
            segment_boundaries = sorted(
                list(set([0] + roots + [midpoint_load_application] + [total_length]))
            )
            new_segments = []

            for i in range(len(segment_boundaries) - 1):
                start = segment_boundaries[i]
                end = segment_boundaries[i + 1]
                midpoint = (start + end) / 2

                # Check stress at the midpoint of the new potential segment
                # to determine if it's cracked (has_foundation=False)
                mid_sigma, mid_tau = self._calculate_sigma_tau_at_x(midpoint, system)
                stress_check = self.stress_envelope(
                    np.array([mid_sigma]), np.array([mid_tau]), weak_layer
                )[0]

                has_foundation = stress_check <= 1

                # Re-apply the skier weight to the correct new segment
                m = skier_weight if i == 1 else 0

                new_segments.append(
                    Segment(length=end - start, has_foundation=has_foundation, m=m)
                )

            # Consolidate mass onto one segment if it was split
            mass_segments = [s for s in new_segments if s.m > 0]
            if len(mass_segments) > 1:
                for s in mass_segments[1:]:
                    s.m = 0

            new_crack_length = sum(
                seg.length for seg in new_segments if not seg.has_foundation
            )

            logger.info(
                "Finished finding new anticrack length in %.4f seconds. New length: %.2f mm.",
                time.time() - start_time,
                new_crack_length,
            )

        # --- Exception: the entire domain is cracked ---
        elif min_dist_stress > 1:
            # The entire domain is cracked
            new_segments = [
                Segment(length=total_length / 2, has_foundation=False, m=skier_weight),
                Segment(length=total_length / 2, has_foundation=False, m=0),
            ]
            new_crack_length = total_length

        elif not roots:
            # No part of the slab is cracked
            new_crack_length = 0
            new_segments = initial_segments

        system.update_scenario(segments=old_segments)

        return new_crack_length, new_segments

    def _calculate_sigma_tau_at_x(
        self, x_value: float, system: SystemModel
    ) -> tuple[float, float]:
        """Calculate normal and shear stresses at a given horizontal x-coordinate."""
        # Get the segment index and coordinate within the segment
        segment_index = system.scenario.get_segment_idx(x_value)

        start_of_segment = (
            system.scenario.cum_sum_li[segment_index - 1] if segment_index > 0 else 0
        )
        coordinate_in_segment = x_value - start_of_segment

        # Get the constants for the segment
        C = system.unknown_constants[:, [segment_index]]
        li_segment = system.scenario.li[segment_index]
        phi = system.scenario.phi
        has_foundation = system.scenario.ki[segment_index]

        # Calculate the displacement field
        Z = system.z(
            coordinate_in_segment, C, li_segment, phi, has_foundation=has_foundation
        )

        # Calculate the stresses
        tau = -system.fq.tau(Z, unit="kPa")  # Negated to match sign convention
        sigma = system.fq.sig(Z, unit="kPa")

        return sigma, tau

    def _get_stress_envelope_exceedance(
        self, x_value: float, system: SystemModel, weak_layer: WeakLayer
    ) -> float:
        """
        Objective function for the root finder.
        Returns the stress envelope evaluation minus 1.
        """
        sigma, tau = self._calculate_sigma_tau_at_x(x_value, system)
        return (
            self.stress_envelope(
                np.array([sigma]), np.array([tau]), weak_layer=weak_layer
            )[0]
            - 1
        )

    def _find_stress_envelope_crossings(
        self, system: SystemModel, weak_layer: WeakLayer
    ) -> list[float]:
        """
        Finds the exact x-coordinates where the stress envelope is crossed.
        """
        logger.debug("Finding stress envelope crossings.")
        start_time = time.time()
        analyzer = Analyzer(system)
        x_coords, z, _ = analyzer.rasterize_solution(mode="cracked", num=2000)

        sigma_kPa = system.fq.sig(z, unit="kPa")
        tau_kPa = system.fq.tau(z, unit="kPa")

        # Calculate the discrete distance to failure
        dist_to_stress_envelope = (
            self.stress_envelope(sigma_kPa, tau_kPa, weak_layer=weak_layer) - 1
        )

        # Find indices where the envelope function transitions
        transition_indices = np.where(np.diff(np.sign(dist_to_stress_envelope)))[0]

        # Find root candidates from transitions
        root_candidates = []
        for idx in transition_indices:
            x_left = x_coords[idx]
            x_right = x_coords[idx + 1]
            root_candidates.append((x_left, x_right))

        # Search for roots within the identified candidates
        roots = []
        logger.debug(
            "Found %d potential crossing regions. Finding exact roots.",
            len(root_candidates),
        )
        roots_start_time = time.time()
        for x_left, x_right in root_candidates:
            try:
                root_result = root_scalar(
                    self._get_stress_envelope_exceedance,
                    args=(system, weak_layer),
                    bracket=[x_left, x_right],
                    method="brentq",
                )
                if root_result.converged:
                    roots.append(root_result.root)
            except ValueError:
                # This can happen if the signs at the bracket edges are not opposite.
                # It's safe to ignore in this context.
                pass
        logger.debug("Root finding took %.4f seconds.", time.time() - roots_start_time)
        logger.info(
            "Found %d stress envelope crossings in %.4f seconds.",
            len(roots),
            time.time() - start_time,
        )
        return roots

    def _fracture_toughness_exceedance(
        self, crack_length: float, system: SystemModel, target: float = 1
    ) -> float:
        """
        Objective function to evaluate the fracture toughness function.
        """
        length = system.scenario.L
        segments = [
            Segment(length=length / 2 - crack_length / 2, has_foundation=True, m=0),
            Segment(length=crack_length / 2, has_foundation=False, m=0),
            Segment(length=crack_length / 2, has_foundation=False, m=0),
            Segment(length=length / 2 - crack_length / 2, has_foundation=True, m=0),
        ]
        system.update_scenario(segments=segments)

        analyzer = Analyzer(system)
        diff_energy = analyzer.differential_ERR(unit="J/m^2")
        G_I = diff_energy[1]
        G_II = diff_energy[2]

        # Evaluate the fracture toughness function (boundary is equal to 1)
        g_delta_diff = self.fracture_toughness_envelope(G_I, G_II, system.weak_layer)

        # Return the difference from the target
        return g_delta_diff - target

    def _calculate_maximal_stresses(
        self,
        system: SystemModel,
        print_call_stats: bool = False,
    ) -> MaximalStressResult:
        """
        Calculates the maximal stresses in the system.
        """
        analyzer = Analyzer(system, printing_enabled=print_call_stats)
        _, Z, _ = analyzer.rasterize_solution(num=4000)
        Sxx_kPa = analyzer.Sxx(Z=Z, phi=system.scenario.phi, dz=5, unit="kPa")
        principal_stress_kPa = analyzer.principal_stress_slab(
            Z=Z, phi=system.scenario.phi, dz=5, unit="kPa"
        )
        Sxx_norm = analyzer.Sxx(
            Z=Z, phi=system.scenario.phi, dz=5, unit="kPa", normalize=True
        )
        principal_stress_norm = analyzer.principal_stress_slab(
            Z=Z, phi=system.scenario.phi, dz=5, unit="kPa", normalize=True
        )
        max_principal_stress_norm = np.max(principal_stress_norm)
        max_Sxx_norm = np.max(Sxx_norm)
        return MaximalStressResult(
            principal_stress_kPa=principal_stress_kPa,
            Sxx_kPa=Sxx_kPa,
            principal_stress_norm=principal_stress_norm,
            Sxx_norm=Sxx_norm,
            max_principal_stress_norm=max_principal_stress_norm,
            max_Sxx_norm=max_Sxx_norm,
        )
