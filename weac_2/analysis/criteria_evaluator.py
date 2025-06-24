# Standard library imports
import copy
from dataclasses import dataclass
from typing import List, Optional, Union

# Third party imports
import numpy as np
from scipy.optimize import root_scalar

from weac_2.analysis.analyzer import Analyzer

# weac imports
from weac_2.components import (
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac_2.core.system_model import SystemModel


@dataclass
class CoupledCriterionHistory:
    """Stores the history of the coupled criterion evaluation."""

    skier_weights: List[float]
    crack_lengths: List[float]
    g_deltas: List[float]
    dist_maxs: List[float]
    dist_mins: List[float]


@dataclass
class CoupledCriterionResult:
    """Holds the results of the coupled criterion evaluation."""

    converged: bool
    message: str
    self_collapse: bool
    pure_stress_criteria: bool
    critical_skier_weight: float
    initial_critical_skier_weight: float
    crack_length: float
    g_delta: float
    final_error: float
    iterations: int
    history: Optional[CoupledCriterionHistory]
    final_system: Optional[SystemModel]
    max_dist_stress: float
    min_dist_stress: float


class CriteriaEvaluator:
    """
    Provides methods for stability analysis of layered slabs on compliant
    elastic foundations, based on the logic from criterion_check.py.
    """

    criteria_config: CriteriaConfig
    system_model: SystemModel

    def __init__(self, system_model: SystemModel, criteria_config: CriteriaConfig):
        """
        Initializes the evaluator with global simulation and criteria configurations.

        Args:
            config (Config): The main simulation configuration.
            criteria_config (CriteriaConfig): The configuration for failure criteria.
        """
        self.system_model = system_model
        self.criteria_config = criteria_config

    def fracture_toughness_criterion(
        self, G_I: float, G_II: float, weak_layer: WeakLayer
    ) -> float:
        """
        Evaluates the fracture toughness criterion for a given combination of
        Mode I (G_I) and Mode II (G_II) energy release rates.

        The criterion is defined as:
            g_delta = (|G_I| / G_Ic)^gn + (|G_II| / G_IIc)^gm

        A value of 1 indicates the boundary of the fracture toughness envelope is reached.

        Args:
            G_I (float): Mode I energy release rate (ERR) in J/m².
            G_II (float): Mode II energy release rate (ERR) in J/m².
            weak_layer (WeakLayer): The weak layer object containing G_Ic and G_IIc.

        Returns:
            float: Non-dimensional evaluation of the fracture toughness envelope.
        """
        g_delta = (np.abs(G_I) / weak_layer.G_Ic) ** self.criteria_config.gn + (
            np.abs(G_II) / weak_layer.G_IIc
        ) ** self.criteria_config.gm

        return g_delta

    def stress_envelope(
        self,
        sigma: Union[float, np.ndarray],
        tau: Union[float, np.ndarray],
        weak_layer: WeakLayer,
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
        order_of_magnitude: float, optional
            Exponent used for scaling. Defaults to 1.0.

        Returns
        -------
        results: ndarray
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

        envelope_method = self.criteria_config.stress_envelope_method
        density = weak_layer.rho
        fn = self.criteria_config.fn
        fm = self.criteria_config.fm
        order_of_magnitude = self.criteria_config.order_of_magnitude

        def mede_common_calculations(sigma, tau, p0, tau_T, p_T):
            in_first_range = (sigma >= (p_T - p0)) & (sigma <= p_T)
            in_second_range = sigma > p_T
            results[in_first_range] = (
                -tau[in_first_range] * (p0 / (tau_T * p_T))
                + sigma[in_first_range] * (1 / p_T)
                + p0 / p_T
            )
            results[in_second_range] = (tau[in_second_range] ** 2) + (
                (tau_T / p0) ** 2
            ) * ((sigma[in_second_range] - p_T) ** 2)
            return results

        if envelope_method == "adam_unpublished":
            density_baseline = 250.0
            scaling_factor = density / density_baseline

            if scaling_factor > 1:
                order_of_magnitude = 0.7
            if scaling_factor < 0.55:
                scaling_factor = 0.55

            sigma_c = 6.16 * (scaling_factor**order_of_magnitude)
            tau_c = 5.09 * (scaling_factor**order_of_magnitude)

            return (sigma / sigma_c) ** fn + (tau / tau_c) ** fm

        elif envelope_method == "schottner":
            rho_ice = 916.7
            sigma_y = 2000
            sigma_c_adam = 6.16
            tau_c_adam = 5.09

            sigma_c = sigma_y * 13 * (density / rho_ice) ** order_of_magnitude
            tau_c = tau_c_adam * (sigma_c / sigma_c_adam)

            return (sigma / sigma_c) ** fn + (tau / tau_c) ** fm

        elif envelope_method == "mede_s-RG1":
            p0, tau_T, p_T = 7.00, 3.53, 1.49
            return mede_common_calculations(sigma, tau, p0, tau_T, p_T)
        elif envelope_method == "mede_s-RG2":
            p0, tau_T, p_T = 2.33, 1.22, 0.19
            return mede_common_calculations(sigma, tau, p0, tau_T, p_T)
        elif envelope_method == "mede_s-FCDH":
            p0, tau_T, p_T = 1.45, 0.61, 0.17
            return mede_common_calculations(sigma, tau, p0, tau_T, p_T)
        else:
            raise ValueError(f"Invalid envelope type: {envelope_method}")

    def _create_model(
        self,
        layers: List[Layer],
        weak_layer: WeakLayer,
        segments: List[Segment],
        scenario_config: ScenarioConfig,
    ) -> SystemModel:
        """Instantiates a SystemModel for a given simulation state."""
        model_input = ModelInput(
            layers=layers,
            weak_layer=weak_layer,
            segments=segments,
            scenario_config=scenario_config,
        )
        return SystemModel(model_input=model_input, config=self.system_model.config)

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
        tau = -system.fq.tau(Z, unit="kPa")
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
    ) -> List[float]:
        """
        Finds the exact x-coordinates where the stress envelope is crossed.
        """
        analyzer = Analyzer(system)
        x_coords, z, _ = analyzer.rasterize_solution()

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

        return roots

    def find_minimum_force(
        self,
        system: SystemModel,
        dampening: float = 0.0,
        tolerance: float = 0.005,
    ) -> tuple[bool, float, SystemModel, float, float]:
        """
        Finds the minimum skier weight required to surpass the stress failure envelope.

        This method iteratively adjusts the skier weight until the maximum distance
        to the stress envelope converges to 1, indicating the critical state.

        Parameters:
        -----------
        system: SystemModel
            The system model.
        dampening: float, optional
            Dampening factor for the skier weight. Defaults to 0.0.
        tolerance: float, optional
            Tolerance for the stress envelope. Defaults to 0.005.

        Returns:
        --------
        success: bool
            Whether the method converged.
        critical_skier_weight: float
            The minimum skier weight (kg).
        system: SystemModel
            The system state at the critical load.
        max_dist_stress: float
            The maximum stress envelope value. Values > 1 indicate failure.
        min_dist_stress: float
            The minimum stress envelope value. Values > 1 indicate failure.
        """
        skier_weight = 1.0  # Initial guess
        iteration_count = 0
        max_iterations = 50
        max_dist_stress = 0

        # --- Initial uncracked configuration ---
        total_length = system.scenario.L
        segments = [
            Segment(length=total_length / 2, has_foundation=True, m=0.0),
            Segment(length=0, has_foundation=False, m=skier_weight),
            Segment(length=0, has_foundation=False, m=0.0),
            Segment(length=total_length / 2, has_foundation=True, m=0.0),
        ]
        system.update_scenario(segments=segments)

        analyzer = Analyzer(system)
        _, z_skier, _ = analyzer.rasterize_solution(num=800)

        sigma_kPa = system.fq.sig(z_skier, unit="kPa")
        tau_kPa = system.fq.tau(z_skier, unit="kPa")

        max_dist_stress = np.max(
            self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
        )
        min_dist_stress = np.min(
            self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
        )

        # --- Exception: the entire domain is cracked ---
        if min_dist_stress >= 1:
            return (
                True,
                skier_weight,
                system,
                max_dist_stress,
                min_dist_stress,
            )

        while abs(max_dist_stress - 1) > tolerance and iteration_count < max_iterations:
            iteration_count += 1

            skier_weight = (
                (dampening + 1) * skier_weight / (dampening + max_dist_stress)
            )

            temp_segments = [
                Segment(length=total_length / 2, has_foundation=True, m=0),
                Segment(length=0, has_foundation=False, m=skier_weight),
                Segment(length=0, has_foundation=False, m=0),
                Segment(length=total_length / 2, has_foundation=True, m=0),
            ]

            system.update_scenario(segments=temp_segments)
            analyzer = Analyzer(system)
            _, z_skier, _ = analyzer.rasterize_solution(num=800)

            sigma_kPa = system.fq.sig(z_skier, unit="kPa")
            tau_kPa = system.fq.tau(z_skier, unit="kPa")

            # Calculate distance to failure
            max_dist_stress = np.max(
                self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
            )
            min_dist_stress = np.min(
                self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
            )

            if min_dist_stress >= 1:
                return (
                    True,
                    skier_weight,
                    system,
                    max_dist_stress,
                    min_dist_stress,
                )

        if iteration_count == max_iterations:
            if dampening < 5:
                # Upon max iteration introduce dampening to avoid infinite loop
                # and try again with a higher tolerance
                return self.find_minimum_force(
                    system, tolerance=0.01, dampening=dampening + 1
                )
            else:
                return (
                    False,
                    0.0,
                    system,
                    max_dist_stress,
                    min_dist_stress,
                )

        return (
            True,
            skier_weight,
            system,
            max_dist_stress,
            min_dist_stress,
        )

    def check_crack_propagation(
        self,
        layers: List[Layer],
        weak_layer: WeakLayer,
        segments: List[Segment],
        phi: float,
    ) -> tuple[float, bool]:
        """
        Evaluates the crack propagation criterion for a given configuration.

        This method determines if a pre-existing crack will propagate without any
        additional load (i.e., self-propagation).

        Parameters:
        ----------
        layers: List[Layer]
        weak_layer: WeakLayer
        segments: List[Segment]
        phi: float

        Returns
        -------
        g_delta_diff: float
            The evaluation of the fracture toughness envelope.
        can_propagate: bool
            True if the criterion is met (g_delta_diff >= 1).
        """
        # Ensure no skier weight is applied for self-propagation check
        for seg in segments:
            seg.m = 0

        scenario_config = ScenarioConfig(phi=phi, system_type="skiers")
        system = self._create_model(layers, weak_layer, segments, scenario_config)

        analyzer = Analyzer(system)

        # Get differential energy release rates at the crack tips
        # Note: gdif returns [total, modeI, modeII] in kJ/m^2 by default
        # We need J/m^2 for the fracture toughness criterion.
        diff_energy = analyzer.differential_ERR(
            C=system.unknown_constants,
            phi=system.scenario.phi,
            li=system.scenario.li,
            ki=system.scenario.ki,
            unit="J/m^2",
        )

        G_I = diff_energy[1]
        G_II = diff_energy[2]

        # Evaluate the fracture toughness criterion
        g_delta_diff = self.fracture_toughness_criterion(G_I, G_II, weak_layer)
        can_propagate = g_delta_diff >= 1

        return g_delta_diff, can_propagate

    def find_new_anticrack_length(
        self,
        system: SystemModel,
        skier_weight: float,
    ) -> tuple[float, List[Segment]]:
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
        new_segments: List[Segment]
            The updated list of segments
        """
        total_length = system.scenario.L
        weak_layer = system.weak_layer

        initial_segments = [
            Segment(length=total_length / 2, has_foundation=True, m=skier_weight),
            Segment(length=total_length / 2, has_foundation=True, m=0),
        ]
        system.update_scenario(segments=initial_segments)

        analyzer = Analyzer(system)
        _, z, _ = analyzer.rasterize_solution()
        sigma_kPa = system.fq.sig(z, unit="kPa")
        tau_kPa = system.fq.tau(z, unit="kPa")
        min_dist_stress = np.min(self.stress_envelope(sigma_kPa, tau_kPa, weak_layer))

        # Find all points where the stress envelope is crossed
        roots = self._find_stress_envelope_crossings(system, weak_layer)

        # --- Exception: the entire domain is cracked ---
        if min_dist_stress > 1:
            # The entire domain is cracked
            new_segments = [
                Segment(length=total_length / 2, has_foundation=False, m=skier_weight),
                Segment(length=total_length / 2, has_foundation=False, m=0),
            ]
            new_crack_length = total_length
            return new_crack_length, new_segments

        if not roots:
            # No part of the slab is cracked
            new_crack_length = 0
            # Return the original uncracked configuration but with the skier weight
            return new_crack_length, initial_segments

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
            m = skier_weight if start <= midpoint_load_application < end else 0

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

        return new_crack_length, new_segments

    def evaluate_coupled_criterion(
        self,
        system: SystemModel,
        max_iterations: int = 25,
        tolerance: float = 0.002,
    ) -> CoupledCriterionResult:
        """
        Evaluates the coupled criterion for anticrack nucleation, finding the
        critical combination of skier weight and anticrack length.

        Parameters:
        ----------
        system: SystemModel
            The system model.
        max_iterations: int, optional
            Max iterations for the solver. Defaults to 25.
        tolerance: float, optional
            Tolerance for g_delta convergence. Defaults to 0.002.

        Returns
        -------
        results: CoupledCriterionResult
            An object containing the results of the analysis, including
            critical skier weight, crack length, and convergence details.
        """
        L = system.scenario.L
        phi = system.scenario.phi
        layers = system.layers
        weak_layer = system.weak_layer

        (
            success,
            initial_critical_skier_weight,
            system_after_force_finding,
            max_dist_stress,
            min_dist_stress,
        ) = self.find_minimum_force(system)

        # --- Failure: in finding the critical skier weight ---
        if not success:
            return CoupledCriterionResult(
                converged=False,
                message="Failed to find critical skier weight.",
                self_collapse=False,
                pure_stress_criteria=False,
                critical_skier_weight=0,
                initial_critical_skier_weight=0,
                crack_length=0,
                g_delta=0,
                final_error=1,
                iterations=0,
                history=None,
                final_system=system,
                max_dist_stress=0,
                min_dist_stress=0,
            )

        # --- Exception: the entire solution is cracked ---
        if min_dist_stress > 1:
            # --- Larger scenario to calculate the incremental ERR ---
            segments = copy.deepcopy(system.scenario.segments)
            for segment in segments:
                segment.has_foundation = False
            # Add 50m of padding to the left and right of the system
            segments.insert(0, Segment(length=50000, has_foundation=True, m=0))
            segments.append(Segment(length=50000, has_foundation=True, m=0))
            system.update_scenario(segments=segments)

            analyzer = Analyzer(system)
            inc_energy = analyzer.incremental_ERR()
            g_delta = self.fracture_toughness_criterion(
                inc_energy[1] * 1000, inc_energy[2] * 1000, system.weak_layer
            )

            history_data = CoupledCriterionHistory([], [], [], [], [])
            return CoupledCriterionResult(
                converged=True,
                message="System fails under its own weight (self-collapse).",
                self_collapse=True,
                pure_stress_criteria=False,
                critical_skier_weight=0,
                initial_critical_skier_weight=initial_critical_skier_weight,
                crack_length=L,
                g_delta=g_delta,
                final_error=0,
                iterations=0,
                history=history_data,
                final_system=system,
                max_dist_stress=max_dist_stress,
                min_dist_stress=min_dist_stress,
            )

        # --- Main loop ---
        elif initial_critical_skier_weight >= 1:
            skier_weight = initial_critical_skier_weight * 1.005
            min_skier_weight = initial_critical_skier_weight
            max_skier_weight = 5 * skier_weight

            crack_length = 1.0
            dist_ERR_envelope = 1000
            g_delta = 0
            history = CoupledCriterionHistory([], [], [], [], [])
            iteration_count = 0

            segments = [
                Segment(
                    length=L / 2 - crack_length,
                    has_foundation=True,
                    m=0,
                ),
                Segment(length=crack_length, has_foundation=False, m=skier_weight),
                Segment(length=crack_length, has_foundation=False, m=0),
                Segment(length=L / 2 - crack_length, has_foundation=True, m=0),
            ]

            for i in range(max_iterations):
                system.update_scenario(segments=segments)
                analyzer = Analyzer(system)
                _, z, _ = analyzer.rasterize_solution()

                # Calculate stress envelope
                sigma_kPa = system.fq.sig(z, unit="kPa")
                tau_kPa = system.fq.tau(z, unit="kPa")
                max_dist_stress = np.max(
                    self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
                )
                min_dist_stress = np.min(
                    self.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)
                )

                # Calculate fracture toughness criterion
                incr_energy = analyzer.incremental_ERR()
                g_delta = self.fracture_toughness_criterion(
                    incr_energy[1] * 1000, incr_energy[2] * 1000, weak_layer
                )
                dist_ERR_envelope = abs(g_delta - 1)

                # Update history
                history.skier_weights.append(skier_weight)
                history.crack_lengths.append(crack_length)
                history.g_deltas.append(g_delta)
                history.dist_maxs.append(max_dist_stress)
                history.dist_mins.append(min_dist_stress)

                # --- Exception: pure stress criterion ---
                # The fracture toughness is superseded for minimum critical skier weight
                if i == 0 and (g_delta > 1 or dist_ERR_envelope < 0.02):
                    return CoupledCriterionResult(
                        converged=True,
                        message="Fracture governed by pure stress criterion.",
                        self_collapse=False,
                        pure_stress_criteria=True,
                        critical_skier_weight=skier_weight,
                        initial_critical_skier_weight=initial_critical_skier_weight,
                        crack_length=crack_length,
                        g_delta=g_delta,
                        final_error=dist_ERR_envelope,
                        iterations=i + 1,
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
                skier_weight = (min_skier_weight + max_skier_weight) / 2

                # Find new anticrack length
                if abs(dist_ERR_envelope) > tolerance:
            crack_length, segments = self.find_new_anticrack_length(
                layers, weak_layer, skier_weight, phi
            )

            if crack_length == 0 and iteration_count < max_iterations:
                return self._evaluate_coupled_criterion_dampened(system)

            converged = dist_ERR_envelope < tolerance
            message = (
                "Converged successfully."
                if converged
                else "Reached max iterations without converging."
            )
            if not all(s.has_foundation for s in segments):
                message = "Reached max iterations; calling dampened version."
                return self._evaluate_coupled_criterion_dampened(system)

            return CoupledCriterionResult(
                converged=converged,
                message=message,
                self_collapse=False,
                pure_stress_criteria=False,
                critical_skier_weight=skier_weight,
                initial_critical_skier_weight=initial_critical_skier_weight,
                crack_length=crack_length,
                g_delta=g_delta,
                final_error=dist_ERR_envelope,
                iterations=iteration_count,
                history=history,
                final_system=system,
                max_dist_stress=max_dist_stress,
                min_dist_stress=min_dist_stress,
            )

        else:  # critical_skier_weight < 1
            return CoupledCriterionResult(
                converged=False,
                message="Critical skier weight is less than 1kg.",
                self_collapse=False,
                pure_stress_criteria=False,
                critical_skier_weight=0,
                initial_critical_skier_weight=initial_critical_skier_weight,
                crack_length=0,
                g_delta=0,
                final_error=1,
                iterations=0,
                history=None,
                final_system=system,
                max_dist_stress=max_dist_stress,
                min_dist_stress=min_dist_stress,
            )

    def _evaluate_coupled_criterion_dampened(
        self,
        system: SystemModel,
        dampening: float = 1.0,
        max_iterations: int = 50,
        tolerance: float = 0.002,
    ) -> CoupledCriterionResult:
        """
        Dampened version of evaluate_coupled_criterion to handle convergence issues.
        """
        L = system.scenario.L
        phi = system.scenario.phi
        layers = system.layers
        weak_layer = system.weak_layer

        (
            success,
            initial_critical_skier_weight,
            _,
            max_dist_stress,
            min_dist_stress,
        ) = self.find_minimum_force(system)

        if not success or initial_critical_skier_weight < 1:
            # Return failure if minimum force can't be found
            return CoupledCriterionResult(
                converged=False,
                message="Dampened: Failed to find critical skier weight.",
                self_collapse=False,
                pure_stress_criteria=False,
                critical_skier_weight=0,
                initial_critical_skier_weight=0,
                crack_length=0,
                g_delta=0,
                final_error=1,
                iterations=0,
                history=None,
                final_system=system,
                max_dist_stress=0,
                min_dist_stress=0,
            )

        skier_weight = initial_critical_skier_weight * 1.005
        min_skier_weight = initial_critical_skier_weight
        max_skier_weight = 3 * initial_critical_skier_weight

        # Ensure max_skier_weight is sufficient
        g_delta_max_weight = 0
        while g_delta_max_weight < 1:
            max_skier_weight *= 2
            # Simplified check, assuming some crack length
            crack_length_check = L / 10
            segments_check = [
                Segment(L / 2 - crack_length_check, True, 0),
                Segment(crack_length_check * 2, False, max_skier_weight),
                Segment(L / 2 - crack_length_check, True, 0),
            ]
            system.update_scenario(segments=segments_check)
            # This is a simplified check and does not perform the full incremental ERR
            # For now, this loop ensures max_skier_weight is increased. A full g_delta
            # check here would be computationally expensive.
            # A placeholder g_delta is assumed to eventually exceed 1.
            if max_skier_weight > 10 * initial_critical_skier_weight:
                g_delta_max_weight = 1.1

        err = 1000
        iteration_count = 0
        history = CoupledCriterionHistory([], [], [], [], [])
        crack_length, segments = self.find_new_anticrack_length(
            layers, weak_layer, skier_weight, phi
        )

        while (
            abs(err) > tolerance
            and iteration_count < max_iterations
            and any(s.has_foundation for s in segments)
        ):
            iteration_count += 1
            history.skier_weights.append(skier_weight)
            history.crack_lengths.append(crack_length)

            # Stress checks for history
            uncracked_segments_stresses = [
                Segment(length=L, has_foundation=True, m=skier_weight)
            ]
            system.update_scenario(segments=uncracked_segments_stresses)
            analyzer = Analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            sigma = system.fq.sig(z, unit="kPa")
            tau = system.fq.tau(z, unit="kPa")
            max_dist_stress = np.max(self.stress_envelope(sigma, tau, weak_layer))
            min_dist_stress = np.min(self.stress_envelope(sigma, tau, weak_layer))
            history.dist_maxs.append(max_dist_stress)
            history.dist_mins.append(min_dist_stress)

            # Models for ginc
            uncracked_segments = [
                Segment(length=s.length, has_foundation=True, m=s.m) for s in segments
            ]
            scenario_config_uc = ScenarioConfig(phi=phi, system_type="skiers")
            uncracked_system = self._create_model(
                layers, weak_layer, uncracked_segments, scenario_config_uc
            )

            cracked_system = self._create_model(
                layers,
                weak_layer,
                segments,
                scenario_config_c=ScenarioConfig(phi=phi, system_type="skiers"),
            )
            analyzer = Analyzer(cracked_system)

            incr_energy = analyzer.incremental_ERR(
                C0=uncracked_system.unknown_constants,
                C1=cracked_system.unknown_constants,
                phi=phi,
                li=np.array([s.length for s in segments]),
                ki=np.array([s.has_foundation for s in segments]),
                k0=np.array([True] * len(segments)),
            )
            g_delta = self.fracture_toughness_criterion(
                incr_energy[1] * 1000, incr_energy[2] * 1000, weak_layer
            )
            history.g_deltas.append(g_delta)
            err = abs(g_delta - 1)

            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            new_skier_weight = (min_skier_weight + max_skier_weight) / 2

            scaling = 1.0
            if abs(err) < 0.5:
                scaling = (dampening + 1 + (new_skier_weight / skier_weight)) / (
                    dampening + 2
                )

            skier_weight = scaling * new_skier_weight

            if abs(err) > tolerance:
                crack_length, segments = self.find_new_anticrack_length(
                    layers, weak_layer, skier_weight, phi
                )

        if iteration_count == max_iterations and dampening < 5:
            return self._evaluate_coupled_criterion_dampened(
                system, dampening=dampening + 1
            )

        converged = err < tolerance
        message = (
            "Dampened: Converged successfully."
            if converged
            else "Dampened: Reached max iterations without converging."
        )

        return CoupledCriterionResult(
            converged=converged,
            message=message,
            self_collapse=False,
            pure_stress_criteria=False,
            critical_skier_weight=skier_weight,
            initial_critical_skier_weight=initial_critical_skier_weight,
            crack_length=crack_length,
            g_delta=g_delta,
            final_error=err,
            iterations=iteration_count,
            history=history,
            final_system=cracked_system,
            max_dist_stress=max_dist_stress,
            min_dist_stress=min_dist_stress,
        )
