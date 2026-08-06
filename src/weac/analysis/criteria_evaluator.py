"""
CriteriaEvaluator facade — public entry for fracture-criteria evaluations.

Envelope, coupled-criterion, and steady-state logic live in sibling modules;
this class holds ``CriteriaConfig`` and delegates.
"""

from __future__ import annotations

import numpy as np

from weac.analysis.coupled_criterion import (
    CoupledCriterionEngine,
    CoupledCriterionHistory,
    CoupledCriterionResult,
    FindMinimumForceResult,
    MaximalStressResult,
)
from weac.analysis.envelopes import (
    fracture_toughness_envelope as _fracture_toughness_envelope,
    stress_envelope as _stress_envelope,
)
from weac.analysis.steady_state import SteadyStateResult, evaluate_steady_state
from weac.components import CriteriaConfig, Segment, WeakLayer
from weac.core.system_model import SystemModel

__all__ = [
    "CoupledCriterionHistory",
    "CoupledCriterionResult",
    "CriteriaEvaluator",
    "FindMinimumForceResult",
    "MaximalStressResult",
    "SteadyStateResult",
]


class CriteriaEvaluator:
    """
    Public facade for stability analysis of layered slabs on compliant
    elastic foundations.
    """

    criteria_config: CriteriaConfig

    def __init__(self, criteria_config: CriteriaConfig):
        """
        Initialize the evaluator with criteria configuration.

        Parameters
        ----------
        criteria_config : CriteriaConfig
            Configuration for failure criteria.
        """
        self.criteria_config = criteria_config
        self._coupled = CoupledCriterionEngine(criteria_config)

    def fracture_toughness_envelope(
        self, G_I: float | np.ndarray, G_II: float | np.ndarray, weak_layer: WeakLayer
    ) -> float | np.ndarray:
        """
        Evaluate the fracture toughness criterion for Mode I / Mode II ERRs.

        The criterion is defined as:
            g_delta = (|G_I| / G_Ic)^gn + (|G_II| / G_IIc)^gm

        A value of 1 indicates the boundary of the fracture toughness envelope.
        """
        return _fracture_toughness_envelope(self.criteria_config, G_I, G_II, weak_layer)

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
        """
        return _stress_envelope(
            self.criteria_config, sigma, tau, weak_layer, method=method
        )

    def evaluate_coupled_criterion(
        self,
        system: SystemModel,
        max_iterations: int = 25,
        damping_ERR: float = 0.0,
        tolerance_ERR: float = 0.002,
        tolerance_stress: float = 0.005,
        print_call_stats: bool = False,
        _recursion_depth: int = 0,
        _force_result: FindMinimumForceResult | None = None,
    ) -> CoupledCriterionResult:
        """
        Evaluate the coupled criterion for anticrack nucleation.
        """
        return self._coupled.evaluate_coupled_criterion(
            system,
            max_iterations=max_iterations,
            damping_ERR=damping_ERR,
            tolerance_ERR=tolerance_ERR,
            tolerance_stress=tolerance_stress,
            print_call_stats=print_call_stats,
            _recursion_depth=_recursion_depth,
            _force_result=_force_result,
        )

    def evaluate_SteadyState(
        self,
        system: SystemModel,
        print_call_stats: bool = False,
    ) -> SteadyStateResult:
        """
        Evaluate hybrid steady state from ``system``.

        Extracts layers, weak layer, and inclination φ from ``SystemModel``.
        Returns a structured result with independent ``tensile`` and ``err``
        blocks. Does not accept touchdown mode and does not force φ→0.

        Per-leg ``elapsed_s`` / ``n_cut_samples`` are always recorded in
        ``result.diagnostics``; set ``print_call_stats=True`` to print them.

        Breaking change
        ---------------
        Former flat-touchdown modes (``TouchdownMode`` / ``mode=``) and the
        old flat ``SteadyStateResult`` fields (``touchdown_distance``, top-level
        ``energy_release_rate``, single ``maximal_stress_result``) are no longer
        part of this API. Use ``result.tensile.critical_cut_length`` and
        ``result.err.energy_release_rate`` instead.
        """
        return evaluate_steady_state(system, print_call_stats=print_call_stats)

    def find_minimum_force(
        self,
        system: SystemModel,
        tolerance_stress: float = 0.0005,
        print_call_stats: bool = False,
    ) -> FindMinimumForceResult:
        """Find the minimum skier weight to surpass the stress failure envelope."""
        return self._coupled.find_minimum_force(
            system,
            tolerance_stress=tolerance_stress,
            print_call_stats=print_call_stats,
        )

    def find_minimum_crack_length(
        self,
        system: SystemModel,
        search_interval: tuple[float, float] | None = None,
        target: float = 1,
    ) -> tuple[float, list[Segment]]:
        """Find the minimum crack length to surpass the ERR envelope."""
        return self._coupled.find_minimum_crack_length(
            system, search_interval=search_interval, target=target
        )

    def check_crack_self_propagation(
        self,
        system: SystemModel,
        rm_skier_weight: bool = False,
    ) -> tuple[float, bool]:
        """Evaluate whether a crack will propagate without additional load."""
        return self._coupled.check_crack_self_propagation(
            system, rm_skier_weight=rm_skier_weight
        )

    def find_crack_length_for_weight(
        self,
        system: SystemModel,
        skier_weight: float,
    ) -> tuple[float, list[Segment]]:
        """Find anticrack length and segments for a given skier weight."""
        return self._coupled.find_crack_length_for_weight(system, skier_weight)

    def _calculate_maximal_stresses(
        self,
        system: SystemModel,
        print_call_stats: bool = False,
    ) -> MaximalStressResult:
        """Calculate maximal stresses in the system (facade delegate)."""
        return self._coupled._calculate_maximal_stresses(
            system, print_call_stats=print_call_stats
        )
