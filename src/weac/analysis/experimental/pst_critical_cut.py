"""
Approach 3 — critical PST cut length to first tensile crack.

Places the slab on slope with no end mass, searches cut length until
``max_Sxx_norm`` first reaches 1.0, evaluates both upslope/downslope
orientations at ``+φ₀``, and selects the shorter critical cut among
usable sides (ERR tie-break; upslope default).

Owns PST system construction in-file (``touchdown=False``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from scipy.optimize import brentq

from weac.analysis.criteria_evaluator import MaximalStressResult
from weac.analysis.experimental.util.ease import select_ease_orientation
from weac.analysis.experimental.util.metrics import (
    evaluate_energy_release_rate,
    evaluate_stresses,
)
from weac.analysis.experimental.util.result import ExperimentalSteadyStateResult
from weac.components import (
    Config,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac.components.scenario_config import SystemType
from weac.core.system_model import SystemModel

CUT_MIN_MM = 1.0
CUT_MAX_MM = 5000.0
CUT_XTOL_MM = 0.5

PST_SYSTEM_TYPES: tuple[SystemType, ...] = ("pst-", "-pst")

OrientationName = Literal["upslope", "downslope"]

ORIENTATIONS: tuple[tuple[OrientationName, SystemType], ...] = (
    ("upslope", "-pst"),
    ("downslope", "pst-"),
)


@dataclass
class CriticalCutSearchResult:
    """Outcome of a single-orientation critical-cut search."""

    critical_cut_length: float
    already_cracked: bool
    no_crack: bool
    converged: bool
    message: str
    energy_release_rate: float
    maximal_stress_result: MaximalStressResult
    system: SystemModel
    system_type: SystemType
    phi: float

    def diagnostics(self) -> dict[str, Any]:
        """JSON-friendly per-orientation diagnostics."""
        stress = self.maximal_stress_result
        return {
            "system_type": self.system_type,
            "phi": float(self.phi),
            "critical_cut_length": float(self.critical_cut_length),
            "energy_release_rate": float(self.energy_release_rate),
            "max_Sxx_norm": float(stress.max_Sxx_norm),
            "thickness_fraction_without_density_gate": float(
                stress.slab_tensile_criterion
            ),
            "already_cracked": self.already_cracked,
            "no_crack": self.no_crack,
            "converged": self.converged,
            "message": self.message,
        }


def build_pst_segments(
    *,
    system_type: SystemType,
    cut_length: float,
    bedded_length: float = 5e3,
    end_mass: float = 0.0,
) -> list[Segment]:
    """
    Build bedded + free segments for a PST cut.

    Optional ``end_mass`` is applied on the free segment's right edge.
    """
    if system_type not in PST_SYSTEM_TYPES:
        raise ValueError(
            f"system_type must be one of {PST_SYSTEM_TYPES}, got {system_type!r}"
        )
    bedded = Segment(length=bedded_length, has_foundation=True, m=0.0)
    free = Segment(length=cut_length, has_foundation=False, m=end_mass)
    if system_type == "pst-":
        return [bedded, free]
    return [free, bedded]


def build_pst_system(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_length: float,
    bedded_length: float = 5e3,
    end_mass: float = 0.0,
    config: Config | None = None,
) -> SystemModel:
    """
    Build a PST ``SystemModel`` with ``touchdown=False``.

    With touchdown disabled, the free-segment length equals ``cut_length``
    even when ``phi != 0``.
    """
    if system_type not in PST_SYSTEM_TYPES:
        raise ValueError(
            f"system_type must be one of {PST_SYSTEM_TYPES}, got {system_type!r}"
        )
    segments = build_pst_segments(
        system_type=system_type,
        cut_length=cut_length,
        bedded_length=bedded_length,
        end_mass=end_mass,
    )
    scenario_config = ScenarioConfig(
        system_type=system_type,
        phi=phi,
        cut_length=cut_length,
    )
    model_input = ModelInput(
        layers=layers,
        weak_layer=weak_layer,
        segments=segments,
        scenario_config=scenario_config,
    )
    if config is None:
        config = Config(touchdown=False)
    elif config.touchdown:
        raise ValueError(
            "build_pst_system requires touchdown=False so the requested cut "
            "length is preserved"
        )
    return SystemModel(model_input=model_input, config=config)


def _evaluate_at_cut(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_length: float,
    bedded_length: float,
) -> tuple[SystemModel, MaximalStressResult, float]:
    """Build PST system at ``cut_length`` and return stresses + ERR."""
    system = build_pst_system(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=cut_length,
        bedded_length=bedded_length,
        end_mass=0.0,
    )
    stress = evaluate_stresses(system)
    err = evaluate_energy_release_rate(system)
    return system, stress, err


def search_critical_cut_length(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_min: float = CUT_MIN_MM,
    cut_max: float = CUT_MAX_MM,
    xtol: float = CUT_XTOL_MM,
    bedded_length: float = 5e3,
) -> CriticalCutSearchResult:
    """
    Search the smallest cut length in ``[cut_min, cut_max]`` with
    ``max_Sxx_norm >= 1``.

    Flags:
    - ``already_cracked``: cracked at ``cut_min`` → critical ≈ min cut
    - ``no_crack``: never reaches 1 up to ``cut_max`` → ``converged=False``;
      diagnostics still report the max-cut state
    """
    if cut_min <= 0 or cut_max <= cut_min:
        raise ValueError(f"Need 0 < cut_min < cut_max, got [{cut_min}, {cut_max}]")

    system_lo, stress_lo, err_lo = _evaluate_at_cut(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=cut_min,
        bedded_length=bedded_length,
    )
    if stress_lo.max_Sxx_norm >= 1.0:
        return CriticalCutSearchResult(
            critical_cut_length=float(cut_min),
            already_cracked=True,
            no_crack=False,
            converged=True,
            message=(
                f"already_cracked at cut_min={cut_min:.3g} mm "
                f"(max_Sxx_norm={stress_lo.max_Sxx_norm:.4g})"
            ),
            energy_release_rate=err_lo,
            maximal_stress_result=stress_lo,
            system=system_lo,
            system_type=system_type,
            phi=phi,
        )

    system_hi, stress_hi, err_hi = _evaluate_at_cut(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=cut_max,
        bedded_length=bedded_length,
    )
    if stress_hi.max_Sxx_norm < 1.0:
        return CriticalCutSearchResult(
            critical_cut_length=float(cut_max),
            already_cracked=False,
            no_crack=True,
            converged=False,
            message=(
                f"no_crack up to cut_max={cut_max:.3g} mm "
                f"(max_Sxx_norm={stress_hi.max_Sxx_norm:.4g})"
            ),
            energy_release_rate=err_hi,
            maximal_stress_result=stress_hi,
            system=system_hi,
            system_type=system_type,
            phi=phi,
        )

    def residual(cut_length: float) -> float:
        _, stress, _ = _evaluate_at_cut(
            layers=layers,
            weak_layer=weak_layer,
            system_type=system_type,
            phi=phi,
            cut_length=cut_length,
            bedded_length=bedded_length,
        )
        return float(stress.max_Sxx_norm) - 1.0

    critical = float(brentq(residual, cut_min, cut_max, xtol=xtol))
    system, stress, err = _evaluate_at_cut(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=critical,
        bedded_length=bedded_length,
    )
    return CriticalCutSearchResult(
        critical_cut_length=critical,
        already_cracked=False,
        no_crack=False,
        converged=True,
        message=(
            f"critical cut={critical:.4g} mm (max_Sxx_norm={stress.max_Sxx_norm:.4g})"
        ),
        energy_release_rate=err,
        maximal_stress_result=stress,
        system=system,
        system_type=system_type,
        phi=phi,
    )


def evaluate_pst_critical_cut(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    phi: float,
    cut_min: float = CUT_MIN_MM,
    cut_max: float = CUT_MAX_MM,
    xtol: float = CUT_XTOL_MM,
    bedded_length: float = 5e3,
) -> ExperimentalSteadyStateResult:
    """
    Dual-orientation critical-cut evaluator.

    Both orientations use ``phi = +φ₀`` (no sign flip) and no end mass.
    Production orientation is the shorter ``critical_cut_length`` among
    usable sides (``converged`` and not ``no_crack``); ERR tie-break with
    upslope default. Core fields come from the ease-winning orientation.
    """
    side_results: dict[OrientationName, CriticalCutSearchResult] = {}
    for name, system_type in ORIENTATIONS:
        side_results[name] = search_critical_cut_length(
            layers=layers,
            weak_layer=weak_layer,
            system_type=system_type,
            phi=phi,
            cut_min=cut_min,
            cut_max=cut_max,
            xtol=xtol,
            bedded_length=bedded_length,
        )

    side_diagnostics = {
        name: result.diagnostics() for name, result in side_results.items()
    }
    selection = select_ease_orientation(
        side_diagnostics["upslope"],
        side_diagnostics["downslope"],
        ease_key="critical_cut_length",
        higher_is_easier=False,
    )
    winner_name = selection.winner
    winner = side_results[winner_name]
    diagnostics: dict[str, Any] = {
        "method": "pst_critical_cut",
        "winner": winner_name,
        "err_winner": selection.err_winner,
        "selection_rule": selection.selection_rule,
        "cut_min_mm": float(cut_min),
        "cut_max_mm": float(cut_max),
        "phi": float(phi),
        "touchdown": False,
        "end_mass": 0.0,
        "upslope": side_diagnostics["upslope"],
        "downslope": side_diagnostics["downslope"],
    }

    return ExperimentalSteadyStateResult(
        converged=winner.converged,
        message=f"{winner_name}: {winner.message}",
        characteristic_length=float(winner.critical_cut_length),
        energy_release_rate=float(winner.energy_release_rate),
        maximal_stress_result=winner.maximal_stress_result,
        system=winner.system,
        diagnostics=diagnostics,
    )
