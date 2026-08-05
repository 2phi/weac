"""
Shared experimental helpers for alternative steady-state evaluations.

Not model logic — approach modules own their system construction and evaluate
entrypoints. Comparison runners load the saved ``baseline_ss`` snapshot via
``util.compare`` and must not recompute baseline.
"""

from __future__ import annotations

from weac.analysis.experimental.util.baseline import evaluate_baseline_steady_state
from weac.analysis.experimental.util.compare import (
    ApproachCompareConfig,
    MetricSpec,
    MissingBaselineError,
    RatioMetricSpec,
    load_baseline_results,
    orientation_metric_value,
    run_approach_comparison,
)
from weac.analysis.experimental.util.ease import (
    EaseSelection,
    is_usable_orientation,
    select_ease_orientation,
)
from weac.analysis.experimental.util.metrics import (
    ExperimentalTensileMetrics,
    evaluate_energy_release_rate,
    evaluate_stresses,
    experimental_tensile_metrics,
    thickness_fraction_without_density_gate,
)
from weac.analysis.experimental.util.result import ExperimentalSteadyStateResult

__all__ = [
    "ApproachCompareConfig",
    "EaseSelection",
    "ExperimentalSteadyStateResult",
    "ExperimentalTensileMetrics",
    "MetricSpec",
    "MissingBaselineError",
    "RatioMetricSpec",
    "evaluate_baseline_steady_state",
    "evaluate_energy_release_rate",
    "evaluate_stresses",
    "experimental_tensile_metrics",
    "is_usable_orientation",
    "load_baseline_results",
    "orientation_metric_value",
    "run_approach_comparison",
    "select_ease_orientation",
    "thickness_fraction_without_density_gate",
]
