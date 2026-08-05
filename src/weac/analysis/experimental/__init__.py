"""
Experimental analysis helpers for alternative steady-state evaluations.

This package is intentionally quarantined. Import it via
``weac.analysis.experimental`` only — it is not part of the public
``weac.analysis`` ``__all__`` surface and must not be treated as a
stable production API until an approach is selected.
"""

from __future__ import annotations

from typing import Any

from weac.analysis.experimental.util.baseline import evaluate_baseline_steady_state
from weac.analysis.experimental.util.result import ExperimentalSteadyStateResult

__all__ = [
    "ExperimentalSteadyStateResult",
    "evaluate_baseline_steady_state",
    "evaluate_pst_critical_cut",
    "evaluate_pst_critical_mass",
    "evaluate_pst_fixed_cut",
    "evaluate_pst_touchdown_cut",
]


def __getattr__(name: str) -> Any:
    """Lazy-load approach modules so ``python -m …`` smoke entries stay clean."""
    if name == "evaluate_pst_fixed_cut":
        from weac.analysis.experimental.pst_fixed_cut import (
            evaluate_pst_fixed_cut,
        )

        return evaluate_pst_fixed_cut
    if name == "evaluate_pst_critical_mass":
        from weac.analysis.experimental.pst_critical_mass import (
            evaluate_pst_critical_mass,
        )

        return evaluate_pst_critical_mass
    if name == "evaluate_pst_critical_cut":
        from weac.analysis.experimental.pst_critical_cut import (
            evaluate_pst_critical_cut,
        )

        return evaluate_pst_critical_cut
    if name == "evaluate_pst_touchdown_cut":
        from weac.analysis.experimental.pst_touchdown_cut import (
            evaluate_pst_touchdown_cut,
        )

        return evaluate_pst_touchdown_cut
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
