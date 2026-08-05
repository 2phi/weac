"""
Approach 1 — PST fixed 50 cm cut (both orientations, ease wins).

Evaluates upslope (``-pst``) and downslope (``pst-``) at ``phi = +φ₀`` with a
fixed unbedded cut, ``touchdown=False``, and no end mass. Core fields come from
the ease-winning orientation (higher ``max_Sxx_norm``); both runs land in
``diagnostics``.

System construction for fixed-cut PST lives in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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

FIXED_CUT_LENGTH_MM = 500.0
PST_SYSTEM_TYPES: tuple[SystemType, ...] = ("pst-", "-pst")
OrientationName = Literal["upslope", "downslope"]

_ORIENTATIONS: tuple[tuple[OrientationName, SystemType], ...] = (
    ("upslope", "-pst"),
    ("downslope", "pst-"),
)


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


def free_segment_length(system: SystemModel) -> float:
    """Return the free (unbedded) segment length of a PST system."""
    free = [seg.length for seg in system.scenario.segments if not seg.has_foundation]
    if len(free) != 1:
        raise ValueError(
            f"Expected exactly one free segment, found {len(free)} in "
            f"{[seg.has_foundation for seg in system.scenario.segments]}"
        )
    return float(free[0])


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


@dataclass(frozen=True)
class FixedCutOrientationRun:
    """One fixed-cut PST orientation evaluation."""

    orientation: OrientationName
    system_type: SystemType
    phi: float
    energy_release_rate: float
    max_Sxx_norm: float
    thickness_fraction_without_density_gate: float
    system: SystemModel
    maximal_stress_result: MaximalStressResult

    def diagnostics_block(self) -> dict[str, float | str]:
        """JSON-friendly scalars for this orientation."""
        return {
            "orientation": self.orientation,
            "system_type": self.system_type,
            "phi": float(self.phi),
            "energy_release_rate": float(self.energy_release_rate),
            "max_Sxx_norm": float(self.max_Sxx_norm),
            "thickness_fraction_without_density_gate": float(
                self.thickness_fraction_without_density_gate
            ),
            "free_segment_length": free_segment_length(self.system),
        }


def _evaluate_orientation(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    orientation: OrientationName,
    system_type: SystemType,
    phi: float,
    cut_length: float,
    bedded_length: float,
    config: Config | None,
    print_call_stats: bool,
) -> FixedCutOrientationRun:
    system = build_pst_system(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=cut_length,
        bedded_length=bedded_length,
        end_mass=0.0,
        config=config,
    )
    err = evaluate_energy_release_rate(system, print_call_stats=print_call_stats)
    stress = evaluate_stresses(system, print_call_stats=print_call_stats)
    return FixedCutOrientationRun(
        orientation=orientation,
        system_type=system_type,
        phi=float(phi),
        energy_release_rate=err,
        max_Sxx_norm=float(stress.max_Sxx_norm),
        thickness_fraction_without_density_gate=float(stress.slab_tensile_criterion),
        system=system,
        maximal_stress_result=stress,
    )


def evaluate_pst_fixed_cut(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    phi: float,
    cut_length: float = FIXED_CUT_LENGTH_MM,
    bedded_length: float = 5e3,
    config: Config | None = None,
    print_call_stats: bool = False,
) -> ExperimentalSteadyStateResult:
    """
    Fixed-cut PST evaluator: both orientations, ease populates core fields.

    Ease = higher ``max_Sxx_norm`` at the fixed cut. Upslope uses
    ``system_type="-pst"``; downslope uses ``"pst-"``. Both keep ``phi = +φ₀``
    (no sign flip) and ``end_mass=0``. ``touchdown`` must stay ``False`` so the
    requested cut length is preserved. When evaluation succeeds both
    orientations are usable (no crack-search flags).
    """
    if cut_length <= 0:
        raise ValueError(f"cut_length must be positive, got {cut_length}")

    runs = {
        name: _evaluate_orientation(
            layers=layers,
            weak_layer=weak_layer,
            orientation=name,
            system_type=system_type,
            phi=phi,
            cut_length=cut_length,
            bedded_length=bedded_length,
            config=config,
            print_call_stats=print_call_stats,
        )
        for name, system_type in _ORIENTATIONS
    }
    upslope = runs["upslope"]
    downslope = runs["downslope"]
    selection = select_ease_orientation(
        upslope.diagnostics_block(),
        downslope.diagnostics_block(),
        ease_key="max_Sxx_norm",
        higher_is_easier=True,
        err_key="energy_release_rate",
        unusable_if_true=(),
    )
    winner = runs[selection.winner]

    diagnostics: dict[str, Any] = {
        "method": "pst_fixed_cut",
        "cut_length": float(cut_length),
        "phi": float(phi),
        "winner": selection.winner,
        "err_winner": selection.err_winner,
        "selection_rule": selection.selection_rule,
        "upslope": upslope.diagnostics_block(),
        "downslope": downslope.diagnostics_block(),
    }
    return ExperimentalSteadyStateResult(
        converged=True,
        message=(
            f"PST fixed cut {cut_length:g} mm; "
            f"{selection.winner} orientation won on ease (max_Sxx_norm)"
        ),
        characteristic_length=float(cut_length),
        energy_release_rate=float(winner.energy_release_rate),
        maximal_stress_result=winner.maximal_stress_result,
        system=winner.system,
        diagnostics=diagnostics,
    )
