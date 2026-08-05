"""
Approach 4 — PST cut length at first tip contact (deformation search).

For each upslope/downslope orientation at ``+φ₀``, search the free-cut length
where the free tip just about touches the collapsed weak layer:

    ``w_tip(cut) = crack_h``

using the cracked, tilted PST solution (``touchdown=False``). Then evaluate
slab tensile metrics and ERR at that configuration. Production orientation is
the ease winner (higher thickness fraction).

Does **not** use flat ``SlabTouchdown.l_AB`` — contact is found from the
deformed shape on the slope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq

from weac.analysis.analyzer import Analyzer
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
RASTER_NUM = 2000

PST_SYSTEM_TYPES: tuple[SystemType, ...] = ("pst-", "-pst")

OrientationName = Literal["upslope", "downslope"]

ORIENTATIONS: tuple[tuple[OrientationName, SystemType], ...] = (
    ("upslope", "-pst"),
    ("downslope", "pst-"),
)


@dataclass
class TouchdownCutOrientationResult:
    """Outcome of one orientation at the deformation-based contact cut."""

    cut_length: float
    crack_h: float
    w_tip: float
    energy_release_rate: float
    maximal_stress_result: MaximalStressResult
    system: SystemModel
    system_type: SystemType
    phi: float
    already_touching: bool
    never_touches: bool
    converged: bool
    message: str

    def diagnostics(self) -> dict[str, Any]:
        """JSON-friendly per-orientation diagnostics."""
        stress = self.maximal_stress_result
        return {
            "system_type": self.system_type,
            "phi": float(self.phi),
            "cut_length": float(self.cut_length),
            "crack_h": float(self.crack_h),
            "w_tip": float(self.w_tip),
            "energy_release_rate": float(self.energy_release_rate),
            "max_Sxx_norm": float(stress.max_Sxx_norm),
            "thickness_fraction_without_density_gate": float(
                stress.slab_tensile_criterion
            ),
            "already_touching": self.already_touching,
            "never_touches": self.never_touches,
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
    """Build bedded + free segments for a PST cut."""
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

    Touchdown stays off so the free-segment length equals ``cut_length`` and
    contact is diagnosed from the deformed solution, not from ``l_AB``.
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
            "build_pst_system requires touchdown=False so contact is found "
            "from the deformed slope PST, not from flat l_AB"
        )
    return SystemModel(model_input=model_input, config=config)


def free_tip_deflection(
    system: SystemModel,
    *,
    num: int = RASTER_NUM,
) -> tuple[float, float]:
    """
    Return ``(w_tip, crack_h)`` [mm] for the free tip of a PST system.

    ``w`` is mid-plane deflection from the cracked solution. The free tip is
    the unsupported end (right end for ``pst-``, left end for ``-pst``).
    """
    analyzer = Analyzer(system, printing_enabled=False)
    x, z, _ = analyzer.rasterize_solution(mode="cracked", num=num)
    w = np.asarray(system.fq.w(z, unit="mm"), dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    free_indices = [
        i for i, seg in enumerate(system.scenario.segments) if not seg.has_foundation
    ]
    if len(free_indices) != 1:
        raise ValueError(
            f"Expected exactly one free segment, found {len(free_indices)}"
        )
    free_i = free_indices[0]
    lengths = [float(seg.length) for seg in system.scenario.segments]
    starts = np.cumsum([0.0, *lengths[:-1]])
    ends = np.cumsum(lengths)
    mask = (x >= starts[free_i] - 1e-9) & (x <= ends[free_i] + 1e-9)
    if not np.any(mask):
        raise RuntimeError("No raster points found on the free segment")

    x_free = x[mask]
    w_free = w[mask]
    tip_i = int(
        np.argmax(x_free)
        if system.scenario.system_type == "pst-"
        else np.argmin(x_free)
    )
    return float(w_free[tip_i]), float(system.scenario.crack_h)


def tip_contact_residual(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_length: float,
    bedded_length: float,
) -> tuple[float, SystemModel, float, float]:
    """
    Build system at ``cut_length`` and return ``(w_tip - crack_h, system, w_tip, crack_h)``.

    Zero crossing = free tip just touches the collapsed weak layer.
    """
    system = build_pst_system(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=cut_length,
        bedded_length=bedded_length,
        end_mass=0.0,
    )
    w_tip, crack_h = free_tip_deflection(system)
    return w_tip - crack_h, system, w_tip, crack_h


def search_touchdown_cut_length(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_min: float = CUT_MIN_MM,
    cut_max: float = CUT_MAX_MM,
    xtol: float = CUT_XTOL_MM,
    bedded_length: float = 5e3,
) -> TouchdownCutOrientationResult:
    """
    Search the smallest cut in ``[cut_min, cut_max]`` with ``w_tip >= crack_h``.

    Flags:
    - ``already_touching``: tip already past contact at ``cut_min``
    - ``never_touches``: tip still short of contact at ``cut_max``
      (``converged=False``; diagnostics report the max-cut state)
    """
    if cut_min <= 0 or cut_max <= cut_min:
        raise ValueError(f"Need 0 < cut_min < cut_max, got [{cut_min}, {cut_max}]")

    residual_lo, system_lo, w_lo, crack_h_lo = tip_contact_residual(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=cut_min,
        bedded_length=bedded_length,
    )
    if residual_lo >= 0.0:
        stress = evaluate_stresses(system_lo)
        err = evaluate_energy_release_rate(system_lo)
        return TouchdownCutOrientationResult(
            cut_length=float(cut_min),
            crack_h=crack_h_lo,
            w_tip=w_lo,
            energy_release_rate=err,
            maximal_stress_result=stress,
            system=system_lo,
            system_type=system_type,
            phi=float(phi),
            already_touching=True,
            never_touches=False,
            converged=True,
            message=(
                f"already_touching at cut_min={cut_min:.3g} mm "
                f"(w_tip={w_lo:.4g}, crack_h={crack_h_lo:.4g})"
            ),
        )

    residual_hi, system_hi, w_hi, crack_h_hi = tip_contact_residual(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=cut_max,
        bedded_length=bedded_length,
    )
    if residual_hi < 0.0:
        stress = evaluate_stresses(system_hi)
        err = evaluate_energy_release_rate(system_hi)
        return TouchdownCutOrientationResult(
            cut_length=float(cut_max),
            crack_h=crack_h_hi,
            w_tip=w_hi,
            energy_release_rate=err,
            maximal_stress_result=stress,
            system=system_hi,
            system_type=system_type,
            phi=float(phi),
            already_touching=False,
            never_touches=True,
            converged=False,
            message=(
                f"never_touches up to cut_max={cut_max:.3g} mm "
                f"(w_tip={w_hi:.4g}, crack_h={crack_h_hi:.4g})"
            ),
        )

    def residual(cut_length: float) -> float:
        value, _, _, _ = tip_contact_residual(
            layers=layers,
            weak_layer=weak_layer,
            system_type=system_type,
            phi=phi,
            cut_length=cut_length,
            bedded_length=bedded_length,
        )
        return float(value)

    critical = float(brentq(residual, cut_min, cut_max, xtol=xtol))
    _, system, w_tip, crack_h = tip_contact_residual(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=critical,
        bedded_length=bedded_length,
    )
    stress = evaluate_stresses(system)
    err = evaluate_energy_release_rate(system)
    return TouchdownCutOrientationResult(
        cut_length=critical,
        crack_h=crack_h,
        w_tip=w_tip,
        energy_release_rate=err,
        maximal_stress_result=stress,
        system=system,
        system_type=system_type,
        phi=float(phi),
        already_touching=False,
        never_touches=False,
        converged=True,
        message=(
            f"touchdown cut={critical:.4g} mm "
            f"(w_tip={w_tip:.4g}, crack_h={crack_h:.4g}, "
            f"max_Sxx_norm={stress.max_Sxx_norm:.4g})"
        ),
    )


def evaluate_pst_touchdown_cut(
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
    Dual-orientation deformation-based first-contact evaluator.

    Both orientations use ``phi = +φ₀`` (no sign flip) and no end mass.
    Production orientation is the higher thickness fraction among usable sides;
    ERR tie-break with upslope default. Core fields come from the ease winner.
    """
    side_results: dict[OrientationName, TouchdownCutOrientationResult] = {}
    for name, system_type in ORIENTATIONS:
        side_results[name] = search_touchdown_cut_length(
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
        ease_key="thickness_fraction_without_density_gate",
        higher_is_easier=True,
        unusable_if_true=("never_touches",),
    )
    winner_name = selection.winner
    winner = side_results[winner_name]
    diagnostics: dict[str, Any] = {
        "method": "pst_touchdown_cut",
        "winner": winner_name,
        "err_winner": selection.err_winner,
        "selection_rule": selection.selection_rule,
        "cut_min_mm": float(cut_min),
        "cut_max_mm": float(cut_max),
        "phi": float(phi),
        "touchdown": False,
        "contact_residual": "w_tip - crack_h",
        "end_mass": 0.0,
        "upslope": side_diagnostics["upslope"],
        "downslope": side_diagnostics["downslope"],
    }

    return ExperimentalSteadyStateResult(
        converged=winner.converged,
        message=f"{winner_name}: {winner.message}",
        characteristic_length=float(winner.cut_length),
        energy_release_rate=float(winner.energy_release_rate),
        maximal_stress_result=winner.maximal_stress_result,
        system=winner.system,
        diagnostics=diagnostics,
    )
