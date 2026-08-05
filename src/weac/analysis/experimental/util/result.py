"""Shared result wrapper for experimental steady-state methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from weac.analysis.criteria_evaluator import MaximalStressResult, SteadyStateResult
from weac.core.system_model import SystemModel


@dataclass
class ExperimentalSteadyStateResult:
    """
    Quarantined steady-state result with SS-compatible core fields.

    ``characteristic_length`` is the neutral length field: touchdown distance
    for the production baseline, cut length for PST-based alternatives.
    Method-specific extras live in ``diagnostics``.
    """

    converged: bool
    message: str
    characteristic_length: float
    energy_release_rate: float
    maximal_stress_result: MaximalStressResult
    system: SystemModel
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_steady_state_result(self) -> SteadyStateResult:
        """Return a ``SteadyStateResult``-shaped view for compatibility."""
        return SteadyStateResult(
            converged=self.converged,
            message=self.message,
            touchdown_distance=self.characteristic_length,
            energy_release_rate=self.energy_release_rate,
            maximal_stress_result=self.maximal_stress_result,
            system=self.system,
        )

    def core_scalars(self) -> dict[str, float | bool | str]:
        """JSON-friendly core scalars used by smoke / comparison runners."""
        stress = self.maximal_stress_result
        return {
            "converged": self.converged,
            "message": self.message,
            "characteristic_length": float(self.characteristic_length),
            "energy_release_rate": float(self.energy_release_rate),
            "max_Sxx_norm": float(stress.max_Sxx_norm),
            "thickness_fraction_without_density_gate": float(
                stress.slab_tensile_criterion
            ),
        }
