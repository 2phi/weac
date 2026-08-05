"""Baseline adapter wrapping production ``evaluate_SteadyState``."""

from __future__ import annotations

from weac.analysis.criteria_evaluator import CriteriaEvaluator, MaximalStressResult
from weac.analysis.experimental.util.metrics import experimental_tensile_metrics
from weac.analysis.experimental.util.result import ExperimentalSteadyStateResult
from weac.components import CriteriaConfig
from weac.core.system_model import SystemModel


def _with_experimental_tensile(
    stress: MaximalStressResult,
) -> tuple[MaximalStressResult, float]:
    """
    Replace production ``slab_tensile_criterion`` with the no-ρ-gate fraction.

    Returns the rewritten stress result and the original production criterion.
    """
    tensile = experimental_tensile_metrics(stress.Sxx_norm)
    rewritten = MaximalStressResult(
        principal_stress_kPa=stress.principal_stress_kPa,
        Sxx_kPa=stress.Sxx_kPa,
        principal_stress_norm=stress.principal_stress_norm,
        Sxx_norm=stress.Sxx_norm,
        max_principal_stress_norm=stress.max_principal_stress_norm,
        max_Sxx_norm=tensile.max_Sxx_norm,
        slab_tensile_criterion=tensile.thickness_fraction_without_density_gate,
    )
    return rewritten, float(stress.slab_tensile_criterion)


def evaluate_baseline_steady_state(
    system: SystemModel,
    *,
    evaluator: CriteriaEvaluator | None = None,
    vertical: bool = False,
    print_call_stats: bool = False,
) -> ExperimentalSteadyStateResult:
    """
    Wrap ``CriteriaEvaluator.evaluate_SteadyState(..., mode="B_point_contact")``.

    Core tensile metric uses thickness fraction without the ρ ≤ 100 exclusion;
    the production criterion is retained in ``diagnostics``.
    """
    if evaluator is None:
        evaluator = CriteriaEvaluator(CriteriaConfig())
    ss = evaluator.evaluate_SteadyState(
        system,
        mode="B_point_contact",
        vertical=vertical,
        print_call_stats=print_call_stats,
    )
    maximal_stress, production_criterion = _with_experimental_tensile(
        ss.maximal_stress_result
    )
    return ExperimentalSteadyStateResult(
        converged=ss.converged,
        message=ss.message,
        characteristic_length=float(ss.touchdown_distance),
        energy_release_rate=float(ss.energy_release_rate),
        maximal_stress_result=maximal_stress,
        system=ss.system,
        diagnostics={
            "method": "baseline_B_point_contact",
            "touchdown_distance": float(ss.touchdown_distance),
            "production_slab_tensile_criterion": production_criterion,
        },
    )
