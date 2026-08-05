"""Shared stress, ERR, and tensile metrics for experimental evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from weac.analysis.analyzer import Analyzer
from weac.analysis.criteria_evaluator import MaximalStressResult
from weac.core.system_model import SystemModel


@dataclass(frozen=True)
class ExperimentalTensileMetrics:
    """Primary tensile scalars for alternative steady-state methods."""

    thickness_fraction_without_density_gate: float
    max_Sxx_norm: float


def thickness_fraction_without_density_gate(
    Sxx_norm: NDArray[np.floating],
) -> float:
    """
    Thickness fraction of height levels with ``max(Sxx_norm) > 1``.

    Unlike production ``slab_tensile_criterion``, every height level is included
    in the denominator — there is no ρ ≤ 100 kg/m³ exclusion.
    """
    if Sxx_norm.size == 0:
        return 0.0
    tensile_exceeds = np.max(Sxx_norm, axis=1) > 1
    return float(np.mean(tensile_exceeds))


def experimental_tensile_metrics(
    Sxx_norm: NDArray[np.floating],
) -> ExperimentalTensileMetrics:
    """Return thickness fraction (no ρ-gate) and ``max_Sxx_norm``."""
    return ExperimentalTensileMetrics(
        thickness_fraction_without_density_gate=thickness_fraction_without_density_gate(
            Sxx_norm
        ),
        max_Sxx_norm=float(np.max(Sxx_norm)) if Sxx_norm.size else 0.0,
    )


def evaluate_energy_release_rate(
    system: SystemModel,
    *,
    print_call_stats: bool = False,
) -> float:
    """Differential energy release rate [J/m²] for a configured system."""
    analyzer = Analyzer(system, printing_enabled=print_call_stats)
    energy_release_rate, _, _, _ = analyzer.differential_ERR(unit="J/m^2")
    return float(energy_release_rate)


def evaluate_stresses(
    system: SystemModel,
    *,
    num: int = 4000,
    print_call_stats: bool = False,
) -> MaximalStressResult:
    """
    Stress fields for a configured system.

    ``slab_tensile_criterion`` is the experimental thickness fraction without
    the production low-density (ρ ≤ 100 kg/m³) exclusion.
    """
    analyzer = Analyzer(system, printing_enabled=print_call_stats)
    _, Z, _ = analyzer.rasterize_solution(num=num, mode="cracked")
    Sxx_kPa = analyzer.Sxx(Z=Z, phi=system.scenario.phi, dz=1, unit="kPa")
    principal_stress_kPa = analyzer.principal_stress_slab(
        Z=Z, phi=system.scenario.phi, dz=1, unit="kPa"
    )
    Sxx_norm = analyzer.Sxx(Z=Z, phi=system.scenario.phi, dz=1, normalize=True)
    principal_stress_norm = analyzer.principal_stress_slab(
        Z=Z, phi=system.scenario.phi, dz=1, normalize=True
    )
    tensile = experimental_tensile_metrics(Sxx_norm)
    if print_call_stats:
        analyzer.print_call_stats(
            message="experimental.evaluate_stresses Call Statistics"
        )
    return MaximalStressResult(
        principal_stress_kPa=principal_stress_kPa,
        Sxx_kPa=Sxx_kPa,
        principal_stress_norm=principal_stress_norm,
        Sxx_norm=Sxx_norm,
        max_principal_stress_norm=float(np.max(principal_stress_norm)),
        max_Sxx_norm=tensile.max_Sxx_norm,
        slab_tensile_criterion=tensile.thickness_fraction_without_density_gate,
    )
