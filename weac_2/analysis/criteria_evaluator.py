# Standard library imports
from typing import List

# Third party imports
import numpy as np
from scipy.optimize import root_scalar

from weac_2.analysis.analyzer import Analyzer

# weac imports
from weac_2.components import (
    Config,
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac_2.core.system_model import SystemModel


class CriteriaEvaluator:
    """
    Provides methods for stability analysis of layered slabs on compliant
    elastic foundations, based on the logic from criterion_check.py.
    """

    config: Config
    criteria_config: CriteriaConfig

    def __init__(self, config: Config, criteria_config: CriteriaConfig):
        """
        Initializes the evaluator with global simulation and criteria configurations.

        Args:
            config (Config): The main simulation configuration.
            criteria_config (CriteriaConfig): The configuration for failure criteria.
        """
        self.config = config
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
        sigma: np.ndarray,
        tau: np.ndarray,
        weak_layer: WeakLayer,
        order_of_magnitude: float = 1.0,
    ) -> np.ndarray:
        """
        Evaluate the stress envelope for given stress components.

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
            Non-dimensional stress evaluation values. Values > 1 indicate failure.

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

        envelope_method = self.config.stress_envelope_method
        density = weak_layer.rho
        fn = self.criteria_config.fn
        fm = self.criteria_config.fm

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
        return SystemModel(model_input=model_input, config=self.config)

    def _calculate_sigma_tau_at_x(
        self, x_value: float, system: SystemModel
    ) -> tuple[float, float]:
        """Calculate normal and shear stresses at a given horizontal x-coordinate."""

        # Find segment index and coordinate within the segment
        total_length = 0
        segment_index = -1
        coordinate_in_segment = -1

        for i, length in enumerate(system.scenario.li):
            total_length += length
            if x_value <= total_length:
                segment_index = i
                coordinate_in_segment = x_value - (total_length - length)
                break

        if segment_index == -1:
            raise ValueError(f"Coordinate {x_value} is outside the slab length.")

        C = system.unknown_constants[:, [segment_index]]
        li_segment = system.scenario.li[segment_index]
        phi = system.scenario.phi
        has_foundation = system.scenario.ki[segment_index]

        Z = system.z(
            coordinate_in_segment, C, li_segment, phi, has_foundation=has_foundation
        )

        tau = -system.fq.tau(Z, unit="kPa")[0]  # Switched sign to match convention
        sigma = system.fq.sig(Z, unit="kPa")[0]

        return sigma, tau

    def _root_function(
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

        # Define the lambda function for the root function
        func = lambda x: self._root_function(x, system=system, weak_layer=weak_layer)

        # Calculate the discrete distance to failure
        discrete_dist_to_fail = (
            self.stress_envelope(sigma_kPa, tau_kPa, weak_layer=weak_layer) - 1
        )

        # Find indices where the envelope function transitions
        transition_indices = np.where(np.diff(np.sign(discrete_dist_to_fail)))[0]

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
                    func, bracket=[x_left, x_right], method="brentq"
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
        layers: List[Layer],
        weak_layer: WeakLayer,
        phi: float,
        order_of_magnitude: float = 1.0,
    ):
        """
        Finds the minimum skier weight required to surpass the stress failure envelope.

        This method iteratively adjusts the skier weight until the maximum distance
        to the stress envelope converges to 1, indicating the critical state.

        Args:
            layers (List[Layer]): The slab layers.
            weak_layer (WeakLayer): The weak layer properties.
            phi (float): The slope angle in degrees.
            order_of_magnitude (float, optional): Scaling exponent for some envelopes. Defaults to 1.0.

        Returns:
            tuple: A tuple containing:
                - critical_skier_weight (float): The minimum skier weight (kg).
                - system (SystemModel): The system state at the critical load.
                - dist_max (float): The maximum distance to the stress envelope.
                - dist_min (float): The minimum distance to the stress envelope.
        """
        skier_weight = 1.0  # Initial guess
        iteration_count = 0
        max_iterations = 50
        dist_max = 0

        # Initial uncracked configuration
        total_length = sum(layer.h for layer in layers) + weak_layer.h
        segments = [Segment(length=total_length, has_foundation=True, m=0.0)]

        while abs(dist_max - 1) > 0.005 and iteration_count < max_iterations:
            iteration_count += 1

            # Set skier weight on the middle segment (or only segment)
            segments[-1].m = skier_weight

            # Create a temporary scenario for this iteration
            # Note: For find_minimum_force, we start with a simple, uncracked setup.
            # The skier load is applied as a point load via the segment's 'm' attribute.
            # We assume a single segment representing the whole domain.

            temp_segments = [
                Segment(length=total_length / 2, has_foundation=True, m=skier_weight),
                Segment(length=total_length / 2, has_foundation=True, m=0),
            ]

            scenario_config = ScenarioConfig(phi=phi, system_type="skiers")
            system = self._create_model(
                layers, weak_layer, temp_segments, scenario_config
            )

            # Rasterize and get stresses
            analyzer = Analyzer(system)
            x, z, _ = analyzer.rasterize_solution()
            sigma = system.fq.sig(z, unit="kPa")
            tau = system.fq.tau(z, unit="kPa")

            # Calculate distance to failure
            distance_to_failure = self.stress_envelope(
                sigma, tau, weak_layer, order_of_magnitude
            )
            dist_max = np.max(distance_to_failure)
            dist_min = np.min(distance_to_failure)

            if dist_min >= 1 and skier_weight == 1.0:
                # Failure occurs even with minimal load
                return 0.0, system, dist_max, dist_min

            # Update skier weight
            if dist_max > 0:
                skier_weight = skier_weight / dist_max
            else:
                # Should not happen, but as a fallback
                skier_weight *= 2

        if iteration_count == max_iterations:
            # TODO: Implement dampened version or raise warning
            print("Warning: find_minimum_force did not converge within max iterations.")

        return skier_weight, system, dist_max, dist_min

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
        layers: List[Layer],
        weak_layer: WeakLayer,
        skier_weight: float,
        phi: float,
        order_of_magnitude: float = 1.0,
    ) -> tuple[float, List[Segment]]:
        """
        Finds the resulting anticrack length and updated segment configurations
        for a given skier weight.

        Args:
            layers (List[Layer]): The slab layers.
            weak_layer (WeakLayer): The weak layer properties.
            skier_weight (float): The weight of the skier (kg).
            phi (float): The slope angle (degrees).
            order_of_magnitude (float, optional): Scaling exponent for envelopes. Defaults to 1.0.

        Returns:
            tuple: A tuple containing:
                - new_crack_length (float): The total length of the new cracked segments (mm).
                - new_segments (List[Segment]): The updated list of segments.
        """
        # Start with a single, uncracked segment
        total_length = sum(layer.h for layer in layers) + weak_layer.h

        # The skier load is applied as a point load, so we split the domain
        # into two segments with the load at the midpoint.
        initial_segments = [
            Segment(length=total_length / 2, has_foundation=True, m=skier_weight),
            Segment(length=total_length / 2, has_foundation=True, m=0),
        ]
        scenario_config = ScenarioConfig(phi=phi, system_type="skiers")

        system = self._create_model(
            layers, weak_layer, initial_segments, scenario_config
        )

        # Find all points where the stress envelope is crossed
        roots = self._find_stress_envelope_crossings(system, weak_layer)

        # Check if all points are outside the envelope
        analyzer = Analyzer(system)
        x_coords, z, _ = analyzer.rasterize_solution()
        sigma = system.fq.sig(z, unit="kPa")
        tau = system.fq.tau(z, unit="kPa")
        dist_min = np.min(self.stress_envelope(sigma, tau, weak_layer))

        if dist_min > 1:
            # The entire domain is cracked
            new_segments = [Segment(length=total_length, has_foundation=False, m=0)]
            new_crack_length = total_length
            return new_crack_length, new_segments

        if not roots:
            # No part of the slab is cracked
            new_crack_length = 0
            # Return the original uncracked configuration but with the skier weight
            return new_crack_length, initial_segments

        # Reconstruct segments based on the roots
        segment_boundaries = sorted(list(set([0] + roots + [total_length])))
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
            m = skier_weight if start <= total_length / 2 < end else 0

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
        layers: List[Layer],
        weak_layer: WeakLayer,
        phi: float,
        max_iterations: int = 25,
    ) -> dict:
        """
        Evaluates the coupled criterion for anticrack nucleation, finding the
        critical combination of skier weight and anticrack length.

        Parameters:
        ----------
        layers: List[Layer]
            The slab layers.
        weak_layer: WeakLayer
            The weak layer properties.
        phi: float
            The slope angle in degrees.
        max_iterations: int, optional
            Max iterations for the solver. Defaults to 25.

        Returns
        -------
        results: dict
            A dictionary containing the results of the analysis, including
            critical skier weight, crack length, and convergence details.
        """
        # --- 1. Initialization ---
        (
            critical_skier_weight,
            system,
            dist_max,
            dist_min,
        ) = self.find_minimum_force(layers, weak_layer, phi)

        total_length = sum(layer.h for layer in layers) + weak_layer.h

        # --- 2. Self-collapse check ---
        if dist_min > 1:
            return {
                "result": True,
                "self_collapse": True,
                "critical_skier_weight": 0,
                "crack_length": total_length,
                "message": "System fails under its own weight (self-collapse).",
            }

        if critical_skier_weight < 1:
            return {
                "result": False,
                "self_collapse": False,
                "critical_skier_weight": critical_skier_weight,
                "message": "System is stable; critical skier weight is less than 1kg.",
            }

        # --- 3. Main Iteration Loop ---
        skier_weight = critical_skier_weight * 1.005
        min_skier_weight = critical_skier_weight
        max_skier_weight = 5 * skier_weight

        crack_length = 1.0
        err = 1000
        g_delta = 0

        # History trackers
        history = {
            "skier_weights": [],
            "crack_lengths": [],
            "g_deltas": [],
            "dist_maxs": [],
        }

        for i in range(max_iterations):
            # Find the new crack geometry for the current skier weight
            crack_length, segments = self.find_new_anticrack_length(
                layers, weak_layer, skier_weight, phi
            )

            # --- Create two models: one for the cracked state, one for uncracked ---
            # Uncracked model (k0)
            uncracked_segments = [
                Segment(length=total_length / 2, has_foundation=True, m=skier_weight),
                Segment(length=total_length / 2, has_foundation=True, m=0),
            ]
            scenario_config_uc = ScenarioConfig(phi=phi, system_type="skiers")
            uncracked_system = self._create_model(
                layers, weak_layer, uncracked_segments, scenario_config_uc
            )

            # Cracked model (ki)
            scenario_config_c = ScenarioConfig(phi=phi, system_type="skiers")
            cracked_system = self._create_model(
                layers, weak_layer, segments, scenario_config_c
            )

            # Calculate incremental energy release rate
            analyzer = Analyzer(cracked_system)
            k0_bools = [s.has_foundation for s in uncracked_segments]

            # The ginc function requires careful setup of li, ki, and k0
            # to compare the two states correctly.
            # This part is complex and may need refinement. For now, a placeholder logic:

            # We need a common segment definition to compare. Let's use the cracked segments geometry.
            li_ginc = [s.length for s in segments]
            ki_ginc = [s.has_foundation for s in segments]

            # For the uncracked state, all corresponding segments are on a foundation.
            k0_ginc = [True] * len(ki_ginc)

            # We need to re-solve the uncracked system on the *same mesh* as the cracked one.
            uncracked_segments_ginc = [
                Segment(length=l, has_foundation=True, m=0) for l in li_ginc
            ]
            # Place mass correctly
            mass_placed = False
            cumulative_l = 0
            mid_point = total_length / 2
            for j, seg in enumerate(uncracked_segments_ginc):
                cumulative_l += seg.length
                if not mass_placed and cumulative_l >= mid_point:
                    seg.m = skier_weight
                    mass_placed = True

            uncracked_system_ginc = self._create_model(
                layers, weak_layer, uncracked_segments_ginc, scenario_config_uc
            )

            incr_energy = analyzer.incremental_ERR(
                C0=uncracked_system_ginc.unknown_constants,
                C1=cracked_system.unknown_constants,
                phi=phi,
                li=np.array(li_ginc),
                ki=np.array(ki_ginc),
                k0=np.array(k0_ginc),
            )

            # Ginc returns [total, G_I, G_II] in kJ/m^2. Convert to J/m^2.
            g_delta = self.fracture_toughness_criterion(
                incr_energy[1] * 1000, incr_energy[2] * 1000, weak_layer
            )

            # Update history
            history["skier_weights"].append(skier_weight)
            history["crack_lengths"].append(crack_length)
            history["g_deltas"].append(g_delta)

            # Update error and check for convergence
            err = abs(g_delta - 1)
            if err < 0.002:
                break

            # Binary search for skier weight
            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            skier_weight = (min_skier_weight + max_skier_weight) / 2

        # --- 4. Finalization and Return ---
        converged = err < 0.002
        message = (
            "Converged successfully."
            if converged
            else "Reached max iterations without converging."
        )

        return {
            "result": converged,
            "message": message,
            "converged": converged,
            "self_collapse": False,
            "critical_skier_weight": skier_weight,
            "crack_length": crack_length,
            "g_delta": g_delta,
            "final_error": err,
            "iterations": i + 1,
            "history": history,
            "final_system": cracked_system,
        }
