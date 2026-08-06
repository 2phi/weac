"""
Hybrid steady-state evaluation — independent critical-cut + tip-contact legs.

Tensile ease comes from the critical-cut dual-orientation search (shorter
``critical_cut_length`` wins). ERR comes from the tip-contact (touchdown-cut)
dual-orientation search (higher thickness fraction; public scalar is
``energy_release_rate``). Orientation winners are chosen independently.

PST builders, Brent cut search, tip-contact residual, ease selection, and
hybrid orchestration all live in this module. Helpers are importable for
tests but are not part of ``__all__``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from weac.analysis.analyzer import (
    BOUNDARY_DX_MM,
    BOUNDARY_WINDOW_MM,
    RASTER_NUM,
    Analyzer,
)
from weac.analysis.coupled_criterion import MaximalStressResult
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

__all__ = [
    "SteadyStateErrBlock",
    "SteadyStateResult",
    "SteadyStateTensileBlock",
    "evaluate_steady_state",
    "evaluate_steady_state_from_layers",
]

# ---------------------------------------------------------------------------
# Constants / orientation
# ---------------------------------------------------------------------------

CUT_MIN_MM = 1.0
CUT_MAX_MM = 5000.0
CUT_XTOL_MM = 0.5
BEDDED_LENGTH_DEFAULT = 5e3

PST_SYSTEM_TYPES: tuple[SystemType, ...] = ("pst-", "-pst")

OrientationName = Literal["upslope", "downslope"]

ORIENTATIONS: tuple[tuple[OrientationName, SystemType], ...] = (
    ("upslope", "-pst"),
    ("downslope", "pst-"),
)

_DEFAULT_UNUSABLE_FLAGS: tuple[str, ...] = ("never_cracked", "no_crack")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Ease selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EaseSelection:
    """Result of comparing upslope vs downslope on an ease metric."""

    winner: OrientationName
    err_winner: OrientationName
    selection_rule: str


def is_usable_orientation(
    block: Mapping[str, Any],
    *,
    unusable_if_true: Sequence[str] = _DEFAULT_UNUSABLE_FLAGS,
) -> bool:
    """
    Return whether an orientation block may enter the ease comparison.

    Missing ``converged`` is treated as usable. Truthy values of any name in
    ``unusable_if_true`` exclude the side. ``already_cracked`` remains usable.
    """
    if block.get("converged") is False:
        return False
    for flag in unusable_if_true:
        if block.get(flag) is True:
            return False
    return True


def _numeric(block: Mapping[str, Any], key: str) -> float | None:
    value = block.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _higher_err_winner(
    upslope: Mapping[str, Any],
    downslope: Mapping[str, Any],
    *,
    err_key: str,
) -> OrientationName:
    err_up = _numeric(upslope, err_key)
    err_down = _numeric(downslope, err_key)
    if err_up is None and err_down is None:
        return "upslope"
    if err_up is None:
        return "downslope"
    if err_down is None:
        return "upslope"
    if err_up >= err_down:
        return "upslope"
    return "downslope"


def select_ease_orientation(
    upslope: Mapping[str, Any],
    downslope: Mapping[str, Any],
    *,
    ease_key: str,
    higher_is_easier: bool,
    err_key: str = "energy_release_rate",
    unusable_if_true: Sequence[str] = _DEFAULT_UNUSABLE_FLAGS,
) -> EaseSelection:
    """
    Pick the production orientation by ease among usable sides.

    Order: usable-side filter → ease compare → ERR tie-break → upslope default.
    ``selection_rule`` is always ``ease:<ease_key>``.
    """
    selection_rule = f"ease:{ease_key}"
    err_winner = _higher_err_winner(upslope, downslope, err_key=err_key)

    sides: dict[OrientationName, Mapping[str, Any]] = {
        "upslope": upslope,
        "downslope": downslope,
    }
    usable: list[OrientationName] = [
        name
        for name, block in sides.items()
        if is_usable_orientation(block, unusable_if_true=unusable_if_true)
        and _numeric(block, ease_key) is not None
    ]

    if len(usable) == 0:
        return EaseSelection(
            winner=err_winner,
            err_winner=err_winner,
            selection_rule=selection_rule,
        )
    if len(usable) == 1:
        return EaseSelection(
            winner=usable[0],
            err_winner=err_winner,
            selection_rule=selection_rule,
        )

    ease_up = _numeric(upslope, ease_key)
    ease_down = _numeric(downslope, ease_key)
    assert ease_up is not None and ease_down is not None

    if ease_up == ease_down:
        winner: OrientationName = err_winner
    elif higher_is_easier:
        winner = "upslope" if ease_up > ease_down else "downslope"
    else:
        winner = "upslope" if ease_up < ease_down else "downslope"

    return EaseSelection(
        winner=winner,
        err_winner=err_winner,
        selection_rule=selection_rule,
    )


# ---------------------------------------------------------------------------
# Stress / ERR metrics
# ---------------------------------------------------------------------------


def thickness_fraction_without_density_gate(
    Sxx_norm: NDArray[np.floating],
) -> float:
    """
    Thickness fraction of height levels with ``max(Sxx_norm) > 1``.

    Unlike production ``slab_tensile_criterion`` historically used elsewhere,
    every height level is included — there is no ρ ≤ 100 kg/m³ exclusion.
    """
    if Sxx_norm.size == 0:
        return 0.0
    tensile_exceeds = np.max(Sxx_norm, axis=1) > 1
    return float(np.mean(tensile_exceeds))


def evaluate_energy_release_rate(system: SystemModel) -> float:
    """Differential energy release rate [J/m²] for a configured system."""
    analyzer = Analyzer(system, printing_enabled=False)
    energy_release_rate, _, _, _ = analyzer.differential_ERR(unit="J/m^2")
    return float(energy_release_rate)


def evaluate_stresses(
    system: SystemModel,
    *,
    num: int = RASTER_NUM,
    boundary_window: float | None = BOUNDARY_WINDOW_MM,
    boundary_dx: float | None = BOUNDARY_DX_MM,
) -> MaximalStressResult:
    """
    Stress fields for a configured system.

    ``slab_tensile_criterion`` is the thickness fraction without the
    low-density (ρ ≤ 100 kg/m³) exclusion.

    Defaults use boundary-refined rasterization (fine ``boundary_dx`` near
    segment ends / joints, coarse interior) so near-tip ``Sxx`` peaks are
    resolved without a globally dense grid.
    """
    analyzer = Analyzer(system, printing_enabled=False)
    _, Z, _ = analyzer.rasterize_solution(
        num=num,
        mode="cracked",
        boundary_window=boundary_window,
        boundary_dx=boundary_dx,
    )
    Sxx_kPa = analyzer.Sxx(Z=Z, phi=system.scenario.phi, dz=1, unit="kPa")
    principal_stress_kPa = analyzer.principal_stress_slab(
        Z=Z, phi=system.scenario.phi, dz=1, unit="kPa"
    )
    Sxx_norm = analyzer.Sxx(Z=Z, phi=system.scenario.phi, dz=1, normalize=True)
    principal_stress_norm = analyzer.principal_stress_slab(
        Z=Z, phi=system.scenario.phi, dz=1, normalize=True
    )
    max_sxx = float(np.max(Sxx_norm)) if Sxx_norm.size else 0.0
    return MaximalStressResult(
        principal_stress_kPa=principal_stress_kPa,
        Sxx_kPa=Sxx_kPa,
        principal_stress_norm=principal_stress_norm,
        Sxx_norm=Sxx_norm,
        max_principal_stress_norm=float(np.max(principal_stress_norm)),
        max_Sxx_norm=max_sxx,
        slab_tensile_criterion=thickness_fraction_without_density_gate(Sxx_norm),
    )


# ---------------------------------------------------------------------------
# PST builders and cut-length search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CutSearchResult(Generic[T]):
    """Outcome of a single-orientation Brent cut search."""

    cut_length: float
    already_at_min: bool
    never_reached: bool
    converged: bool
    sample: T


def build_pst_segments(
    *,
    system_type: SystemType,
    cut_length: float,
    bedded_length: float = BEDDED_LENGTH_DEFAULT,
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
    bedded_length: float = BEDDED_LENGTH_DEFAULT,
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


def evaluate_pst_at_cut(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_length: float,
    bedded_length: float = BEDDED_LENGTH_DEFAULT,
) -> tuple[SystemModel, MaximalStressResult, float]:
    """Build sloping PST at ``cut_length`` and return stresses + differential ERR."""
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


def free_tip_deflection(
    system: SystemModel,
    *,
    num: int = RASTER_NUM,
    boundary_window: float | None = BOUNDARY_WINDOW_MM,
    boundary_dx: float | None = BOUNDARY_DX_MM,
) -> tuple[float, float]:
    """
    Return ``(w_tip, crack_h)`` [mm] for the free tip of a PST system.

    ``w`` is mid-plane deflection from the cracked solution. The free tip is
    the unsupported end (right end for ``pst-``, left end for ``-pst``).
    """
    analyzer = Analyzer(system, printing_enabled=False)
    x, z, _ = analyzer.rasterize_solution(
        mode="cracked",
        num=num,
        boundary_window=boundary_window,
        boundary_dx=boundary_dx,
    )
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
    bedded_length: float = BEDDED_LENGTH_DEFAULT,
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


def search_cut_by_residual(
    sample_at_cut: Callable[[float], tuple[float, T]],
    *,
    cut_min: float = CUT_MIN_MM,
    cut_max: float = CUT_MAX_MM,
    xtol: float = CUT_XTOL_MM,
) -> CutSearchResult[T]:
    """
    Search the smallest cut in ``[cut_min, cut_max]`` where residual crosses ≥ 0.

    ``sample_at_cut(cut)`` returns ``(residual, sample)``. Flags:
    - ``already_at_min``: residual ≥ 0 at ``cut_min``
    - ``never_reached``: residual still < 0 at ``cut_max`` (``converged=False``)
    """
    if cut_min <= 0 or cut_max <= cut_min:
        raise ValueError(f"Need 0 < cut_min < cut_max, got [{cut_min}, {cut_max}]")

    residual_lo, sample_lo = sample_at_cut(cut_min)
    if residual_lo >= 0.0:
        return CutSearchResult(
            cut_length=float(cut_min),
            already_at_min=True,
            never_reached=False,
            converged=True,
            sample=sample_lo,
        )

    residual_hi, sample_hi = sample_at_cut(cut_max)
    if residual_hi < 0.0:
        return CutSearchResult(
            cut_length=float(cut_max),
            already_at_min=False,
            never_reached=True,
            converged=False,
            sample=sample_hi,
        )

    def residual(cut_length: float) -> float:
        value, _ = sample_at_cut(cut_length)
        return float(value)

    critical = float(brentq(residual, cut_min, cut_max, xtol=xtol))
    _, sample = sample_at_cut(critical)
    return CutSearchResult(
        cut_length=critical,
        already_at_min=False,
        never_reached=False,
        converged=True,
        sample=sample,
    )


def search_both_orientations(
    search_one: Callable[[SystemType], T],
) -> dict[OrientationName, T]:
    """Run a single-orientation search for upslope (``-pst``) and downslope (``pst-``)."""
    return {name: search_one(system_type) for name, system_type in ORIENTATIONS}


# ---------------------------------------------------------------------------
# Critical-cut (tensile) search kernel
# ---------------------------------------------------------------------------


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


def search_critical_cut_length(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_min: float = CUT_MIN_MM,
    cut_max: float = CUT_MAX_MM,
    xtol: float = CUT_XTOL_MM,
    bedded_length: float = BEDDED_LENGTH_DEFAULT,
    sample_counter: list[int] | None = None,
) -> CriticalCutSearchResult:
    """
    Search the smallest cut length in ``[cut_min, cut_max]`` with
    ``max_Sxx_norm >= 1``.

    One Brent search only. After a tensile root is found, check tip contact
    at that cut once (``w_tip`` vs ``crack_h``); if already touching, treat
    as no tensile crack.

    Flags:
    - ``already_cracked``: cracked at ``cut_min`` and tip not yet touching
    - ``no_crack``: never reaches 1 up to ``cut_max``, **or** tensile root
      found but tip already touching there → report ``cut_max`` (full PST
      length) with ``converged=False``
    """

    def sample_at_cut(
        cut_length: float,
    ) -> tuple[float, tuple[SystemModel, MaximalStressResult, float]]:
        if sample_counter is not None:
            sample_counter.append(1)
        system, stress, err = evaluate_pst_at_cut(
            layers=layers,
            weak_layer=weak_layer,
            system_type=system_type,
            phi=phi,
            cut_length=cut_length,
            bedded_length=bedded_length,
        )
        return float(stress.max_Sxx_norm) - 1.0, (system, stress, err)

    found = search_cut_by_residual(
        sample_at_cut,
        cut_min=cut_min,
        cut_max=cut_max,
        xtol=xtol,
    )
    system, stress, err = found.sample

    if found.never_reached:
        message = (
            f"no_crack up to cut_max={cut_max:.3g} mm "
            f"(max_Sxx_norm={stress.max_Sxx_norm:.4g}); "
            f"reporting full PST length"
        )
        return CriticalCutSearchResult(
            critical_cut_length=float(cut_max),
            already_cracked=False,
            no_crack=True,
            converged=False,
            message=message,
            energy_release_rate=err,
            maximal_stress_result=stress,
            system=system,
            system_type=system_type,
            phi=phi,
        )

    w_tip, crack_h = free_tip_deflection(system)
    if w_tip >= crack_h:
        # Tip already touching at the tensile root → not a valid free crack.
        if sample_counter is not None:
            sample_counter.append(1)
        system, stress, err = evaluate_pst_at_cut(
            layers=layers,
            weak_layer=weak_layer,
            system_type=system_type,
            phi=phi,
            cut_length=cut_max,
            bedded_length=bedded_length,
        )
        message = (
            f"no_crack: tip already touching at tensile cut="
            f"{found.cut_length:.4g} mm "
            f"(w_tip={w_tip:.4g} >= crack_h={crack_h:.4g}); "
            f"reporting cut_max={cut_max:.3g} mm"
        )
        return CriticalCutSearchResult(
            critical_cut_length=float(cut_max),
            already_cracked=False,
            no_crack=True,
            converged=False,
            message=message,
            energy_release_rate=err,
            maximal_stress_result=stress,
            system=system,
            system_type=system_type,
            phi=phi,
        )

    if found.already_at_min:
        message = (
            f"already_cracked at cut_min={cut_min:.3g} mm "
            f"(max_Sxx_norm={stress.max_Sxx_norm:.4g})"
        )
    else:
        message = (
            f"critical cut={found.cut_length:.4g} mm "
            f"(max_Sxx_norm={stress.max_Sxx_norm:.4g})"
        )

    return CriticalCutSearchResult(
        critical_cut_length=float(found.cut_length),
        already_cracked=found.already_at_min,
        no_crack=False,
        converged=True,
        message=message,
        energy_release_rate=err,
        maximal_stress_result=stress,
        system=system,
        system_type=system_type,
        phi=phi,
    )


# ---------------------------------------------------------------------------
# Tip-contact (touchdown-cut) search kernel
# ---------------------------------------------------------------------------


@dataclass
class TipContactSearchResult:
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


def search_touchdown_cut_length(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_min: float = CUT_MIN_MM,
    cut_max: float = CUT_MAX_MM,
    xtol: float = CUT_XTOL_MM,
    bedded_length: float = BEDDED_LENGTH_DEFAULT,
    sample_counter: list[int] | None = None,
) -> TipContactSearchResult:
    """
    Search the smallest cut in ``[cut_min, cut_max]`` with ``w_tip >= crack_h``.

    Flags:
    - ``already_touching``: tip already past contact at ``cut_min``
    - ``never_touches``: tip still short of contact at ``cut_max``
      (``converged=False``; diagnostics report the max-cut state)
    """

    def sample_at_cut(
        cut_length: float,
    ) -> tuple[float, tuple[SystemModel, float, float]]:
        if sample_counter is not None:
            sample_counter.append(1)
        residual, system, w_tip, crack_h = tip_contact_residual(
            layers=layers,
            weak_layer=weak_layer,
            system_type=system_type,
            phi=phi,
            cut_length=cut_length,
            bedded_length=bedded_length,
        )
        return float(residual), (system, w_tip, crack_h)

    found = search_cut_by_residual(
        sample_at_cut,
        cut_min=cut_min,
        cut_max=cut_max,
        xtol=xtol,
    )
    system, w_tip, crack_h = found.sample
    stress = evaluate_stresses(system)
    err = evaluate_energy_release_rate(system)

    if found.already_at_min:
        message = (
            f"already_touching at cut_min={cut_min:.3g} mm "
            f"(w_tip={w_tip:.4g}, crack_h={crack_h:.4g})"
        )
    elif found.never_reached:
        message = (
            f"never_touches up to cut_max={cut_max:.3g} mm "
            f"(w_tip={w_tip:.4g}, crack_h={crack_h:.4g})"
        )
    else:
        message = (
            f"touchdown cut={found.cut_length:.4g} mm "
            f"(w_tip={w_tip:.4g}, crack_h={crack_h:.4g}, "
            f"max_Sxx_norm={stress.max_Sxx_norm:.4g})"
        )

    return TipContactSearchResult(
        cut_length=float(found.cut_length),
        crack_h=crack_h,
        w_tip=w_tip,
        energy_release_rate=err,
        maximal_stress_result=stress,
        system=system,
        system_type=system_type,
        phi=float(phi),
        already_touching=found.already_at_min,
        never_touches=found.never_reached,
        converged=found.converged,
        message=message,
    )


# ---------------------------------------------------------------------------
# Hybrid result types + orchestration
# ---------------------------------------------------------------------------


@dataclass
class SteadyStateTensileBlock:
    """Tensile-ease block from the critical-cut search."""

    critical_cut_length: float
    cut_direction_winner: OrientationName
    converged: bool
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    maximal_stress_result: MaximalStressResult | None = None
    system: SystemModel | None = None


@dataclass
class SteadyStateErrBlock:
    """ERR block from the tip-contact (touchdown-cut) search."""

    energy_release_rate: float
    cut_length: float
    cut_direction_winner: OrientationName
    converged: bool
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    maximal_stress_result: MaximalStressResult | None = None
    system: SystemModel | None = None


@dataclass
class SteadyStateResult:
    """
    Structured hybrid steady-state result with independent tensile / ERR legs.

    Exposes ``core_scalars()`` / ``diagnostics`` for comparison harnesses
    (``characteristic_length`` = tensile ``L_crit``,
    ``energy_release_rate`` = ERR-leg winner).
    """

    tensile: SteadyStateTensileBlock
    err: SteadyStateErrBlock
    phi: float

    @property
    def converged(self) -> bool:
        return bool(self.tensile.converged and self.err.converged)

    @property
    def message(self) -> str:
        return (
            f"tensile[{self.tensile.cut_direction_winner}]: {self.tensile.message}; "
            f"err[{self.err.cut_direction_winner}]: {self.err.message}"
        )

    @property
    def diagnostics(self) -> dict[str, Any]:
        tensile_elapsed = float(self.tensile.diagnostics.get("elapsed_s", 0.0))
        err_elapsed = float(self.err.diagnostics.get("elapsed_s", 0.0))
        tensile_samples = int(self.tensile.diagnostics.get("n_cut_samples", 0))
        err_samples = int(self.err.diagnostics.get("n_cut_samples", 0))
        return {
            "method": "hybrid_steady_state",
            "phi": float(self.phi),
            "touchdown": False,
            "end_mass": 0.0,
            "elapsed_s": tensile_elapsed + err_elapsed,
            "n_cut_samples": tensile_samples + err_samples,
            "tensile": dict(self.tensile.diagnostics),
            "err": dict(self.err.diagnostics),
        }

    def core_scalars(self) -> dict[str, float | bool | str]:
        """JSON-friendly top-level scalars for comparison runners."""
        stress = self.tensile.maximal_stress_result
        max_sxx = float(stress.max_Sxx_norm) if stress is not None else float("nan")
        thickness = (
            float(stress.slab_tensile_criterion) if stress is not None else float("nan")
        )
        return {
            "converged": self.converged,
            "message": self.message,
            "characteristic_length": float(self.tensile.critical_cut_length),
            "energy_release_rate": float(self.err.energy_release_rate),
            "max_Sxx_norm": max_sxx,
            "thickness_fraction_without_density_gate": thickness,
        }


def _print_ss_call_stats(result: SteadyStateResult) -> None:
    """Print per-leg timing / cut-sample counts when ``print_call_stats=True``."""
    diag = result.diagnostics
    print("--- evaluate_steady_state Call Statistics ---")
    for leg_name in ("tensile", "err"):
        leg = diag.get(leg_name, {})
        elapsed = float(leg.get("elapsed_s", 0.0))
        n_samples = int(leg.get("n_cut_samples", 0))
        print(f"- {leg_name}: {elapsed:.4f}s wall, {n_samples} cut samples")
    print(
        f"- total: {float(diag.get('elapsed_s', 0.0)):.4f}s wall, "
        f"{int(diag.get('n_cut_samples', 0))} cut samples"
    )
    print("---------------------------------------------")


def evaluate_steady_state_from_layers(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    phi: float,
    cut_min: float = CUT_MIN_MM,
    cut_max: float = CUT_MAX_MM,
    xtol: float = CUT_XTOL_MM,
    bedded_length: float = BEDDED_LENGTH_DEFAULT,
    print_call_stats: bool = False,
) -> SteadyStateResult:
    """
    Dual-leg hybrid evaluator (independent orientation selection per block).

    - Tensile: critical-cut search; ease = shorter ``critical_cut_length``.
    - ERR: tip-contact search; ease = higher thickness fraction; public scalar
      is ``energy_release_rate``.

    Inclination ``phi`` is taken as given — no flat-slab (φ→0) override.
    Per-leg ``elapsed_s`` / ``n_cut_samples`` are always written to
    ``diagnostics``; printing occurs only when ``print_call_stats`` is True.
    """
    tensile_samples: list[int] = []
    t_tensile = time.perf_counter()

    def search_tensile(system_type: SystemType):
        return search_critical_cut_length(
            layers=layers,
            weak_layer=weak_layer,
            system_type=system_type,
            phi=phi,
            cut_min=cut_min,
            cut_max=cut_max,
            xtol=xtol,
            bedded_length=bedded_length,
            sample_counter=tensile_samples,
        )

    tensile_sides = search_both_orientations(search_tensile)
    tensile_side_diag = {
        name: result.diagnostics() for name, result in tensile_sides.items()
    }
    tensile_selection = select_ease_orientation(
        tensile_side_diag["upslope"],
        tensile_side_diag["downslope"],
        ease_key="critical_cut_length",
        higher_is_easier=False,
    )
    tensile_winner_name = tensile_selection.winner
    tensile_winner = tensile_sides[tensile_winner_name]
    tensile_elapsed = time.perf_counter() - t_tensile
    tensile_diagnostics: dict[str, Any] = {
        "critical_cut_length": float(tensile_winner.critical_cut_length),
        "cut_direction_winner": tensile_winner_name,
        "winner": tensile_winner_name,
        "err_winner": tensile_selection.err_winner,
        "selection_rule": tensile_selection.selection_rule,
        "converged": tensile_winner.converged,
        "message": tensile_winner.message,
        "energy_release_rate": float(tensile_winner.energy_release_rate),
        "max_Sxx_norm": float(tensile_winner.maximal_stress_result.max_Sxx_norm),
        "thickness_fraction_without_density_gate": float(
            tensile_winner.maximal_stress_result.slab_tensile_criterion
        ),
        "cut_min_mm": float(cut_min),
        "cut_max_mm": float(cut_max),
        "phi": float(phi),
        "elapsed_s": float(tensile_elapsed),
        "n_cut_samples": len(tensile_samples),
        "upslope": tensile_side_diag["upslope"],
        "downslope": tensile_side_diag["downslope"],
    }

    err_samples: list[int] = []
    t_err = time.perf_counter()

    def search_err(system_type: SystemType):
        return search_touchdown_cut_length(
            layers=layers,
            weak_layer=weak_layer,
            system_type=system_type,
            phi=phi,
            cut_min=cut_min,
            cut_max=cut_max,
            xtol=xtol,
            bedded_length=bedded_length,
            sample_counter=err_samples,
        )

    err_sides = search_both_orientations(search_err)
    err_side_diag = {name: result.diagnostics() for name, result in err_sides.items()}
    err_selection = select_ease_orientation(
        err_side_diag["upslope"],
        err_side_diag["downslope"],
        ease_key="thickness_fraction_without_density_gate",
        higher_is_easier=True,
        unusable_if_true=("never_touches",),
    )
    err_winner_name = err_selection.winner
    err_winner = err_sides[err_winner_name]
    err_elapsed = time.perf_counter() - t_err
    err_diagnostics: dict[str, Any] = {
        "energy_release_rate": float(err_winner.energy_release_rate),
        "cut_length": float(err_winner.cut_length),
        "cut_direction_winner": err_winner_name,
        "winner": err_winner_name,
        "err_winner": err_selection.err_winner,
        "selection_rule": err_selection.selection_rule,
        "converged": err_winner.converged,
        "message": err_winner.message,
        "max_Sxx_norm": float(err_winner.maximal_stress_result.max_Sxx_norm),
        "thickness_fraction_without_density_gate": float(
            err_winner.maximal_stress_result.slab_tensile_criterion
        ),
        "cut_min_mm": float(cut_min),
        "cut_max_mm": float(cut_max),
        "phi": float(phi),
        "contact_residual": "w_tip - crack_h",
        "elapsed_s": float(err_elapsed),
        "n_cut_samples": len(err_samples),
        "upslope": err_side_diag["upslope"],
        "downslope": err_side_diag["downslope"],
    }

    result = SteadyStateResult(
        tensile=SteadyStateTensileBlock(
            critical_cut_length=float(tensile_winner.critical_cut_length),
            cut_direction_winner=tensile_winner_name,
            converged=tensile_winner.converged,
            message=f"{tensile_winner_name}: {tensile_winner.message}",
            diagnostics=tensile_diagnostics,
            maximal_stress_result=tensile_winner.maximal_stress_result,
            system=tensile_winner.system,
        ),
        err=SteadyStateErrBlock(
            energy_release_rate=float(err_winner.energy_release_rate),
            cut_length=float(err_winner.cut_length),
            cut_direction_winner=err_winner_name,
            converged=err_winner.converged,
            message=f"{err_winner_name}: {err_winner.message}",
            diagnostics=err_diagnostics,
            maximal_stress_result=err_winner.maximal_stress_result,
            system=err_winner.system,
        ),
        phi=float(phi),
    )
    if print_call_stats:
        _print_ss_call_stats(result)
    return result


def evaluate_steady_state(
    system: SystemModel,
    *,
    print_call_stats: bool = False,
) -> SteadyStateResult:
    """
    Evaluate hybrid steady state from a ``SystemModel``.

    Extracts layers, weak layer, and inclination φ from ``system``.
    Does not accept touchdown mode and does not force φ→0.
    """
    return evaluate_steady_state_from_layers(
        layers=list(system.slab.layers),
        weak_layer=system.weak_layer,
        phi=float(system.scenario.phi),
        print_call_stats=print_call_stats,
    )
