"""
Approach 2 — PST fixed cut with critical right-edge end mass.

Keeps a 100 mm free cut (``touchdown=False``) and searches the free-segment
right-edge mass until ``max_Sxx_norm`` first reaches 1.0.

Both orientations use ``system_type="pst-"`` so the free face (and end mass) sit
on the right edge. Labels match the other approaches / WEAC naming:

- downslope: ``pst-`` at ``+φ₀`` (cut from the top / uphill end)
- upslope: ``pst-`` at ``−φ₀`` (mirror of ``-pst`` at ``+φ₀``)

Core fields come from the ease-winning orientation (lower critical mass).

System / segment construction lives in this module (not ``experimental.pst``).
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
from weac.constants import LSKI_MM
from weac.core.system_model import SystemModel

CUT_LENGTH_MM = 100.0
DEFAULT_M_MAX_KG = 2000.0
DEFAULT_MASS_TOL_KG = 1e-3
DEFAULT_STRESS_TOL = 5e-4
CRACK_THRESHOLD = 1.0

PST_SYSTEM_TYPES: tuple[SystemType, ...] = ("pst-", "-pst")

OrientationName = Literal["upslope", "downslope"]


@dataclass(frozen=True)
class _OrientationSpec:
    name: OrientationName
    system_type: SystemType
    phi: float


@dataclass
class CriticalMassSearchResult:
    """Outcome of a single-orientation critical end-mass search."""

    critical_mass_kg: float
    already_cracked: bool
    never_cracked: bool
    converged: bool
    message: str
    max_Sxx_norm: float
    energy_release_rate: float
    thickness_fraction_without_density_gate: float
    maximal_stress_result: MaximalStressResult
    system: SystemModel
    phi: float
    system_type: SystemType
    iterations: int


def _build_pst_segments(
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


def _build_pst_system_with_end_mass(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_length: float,
    bedded_length: float,
    end_mass: float,
    config: Config | None,
) -> SystemModel:
    """
    Build a PST system with mass on the free segment's right edge.

    ``ModelInput`` requires the last segment's ``m`` to be 0 (masses act at
    interfaces between segments). For ``pst-`` the free segment is last, so a
    zero-length trailing segment is appended when ``end_mass > 0``.
    """
    segments = _build_pst_segments(
        system_type=system_type,
        cut_length=cut_length,
        bedded_length=bedded_length,
        end_mass=end_mass,
    )
    if segments[-1].m != 0:
        segments = [
            *segments,
            Segment(length=0.0, has_foundation=False, m=0.0),
        ]
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
            "pst critical-mass requires touchdown=False so the requested cut "
            "length is preserved"
        )
    return SystemModel(model_input=model_input, config=config)


def _evaluate_at_mass(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_length: float,
    bedded_length: float,
    end_mass: float,
    config: Config | None,
    num: int,
) -> tuple[SystemModel, MaximalStressResult, float]:
    """Build a PST system at ``end_mass`` and return stresses + ERR."""
    system = _build_pst_system_with_end_mass(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=cut_length,
        bedded_length=bedded_length,
        end_mass=end_mass,
        config=config,
    )
    stress = evaluate_stresses(system, num=num)
    err = evaluate_energy_release_rate(system)
    return system, stress, err


def search_critical_end_mass(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    system_type: SystemType,
    phi: float,
    cut_length: float = CUT_LENGTH_MM,
    bedded_length: float = 5e3,
    m_max: float = DEFAULT_M_MAX_KG,
    mass_tol: float = DEFAULT_MASS_TOL_KG,
    stress_tol: float = DEFAULT_STRESS_TOL,
    config: Config | None = None,
    num: int = 4000,
) -> CriticalMassSearchResult:
    """
    Bracket/bisect the free-segment right-edge mass until first tensile crack.

    First crack means the smallest ``m ≥ 0`` with ``max_Sxx_norm ≥ 1``.
    Flags ``already_cracked`` (at ``m=0``) and ``never_cracked`` (up to ``m_max``).
    """
    if m_max < 0:
        raise ValueError(f"m_max must be >= 0, got {m_max}")

    system0, stress0, err0 = _evaluate_at_mass(
        layers=layers,
        weak_layer=weak_layer,
        system_type=system_type,
        phi=phi,
        cut_length=cut_length,
        bedded_length=bedded_length,
        end_mass=0.0,
        config=config,
        num=num,
    )
    sxx0 = float(stress0.max_Sxx_norm)
    if sxx0 >= CRACK_THRESHOLD - stress_tol:
        return CriticalMassSearchResult(
            critical_mass_kg=0.0,
            already_cracked=True,
            never_cracked=False,
            converged=True,
            message="already cracked at m=0",
            max_Sxx_norm=sxx0,
            energy_release_rate=err0,
            thickness_fraction_without_density_gate=float(
                stress0.slab_tensile_criterion
            ),
            maximal_stress_result=stress0,
            system=system0,
            phi=phi,
            system_type=system_type,
            iterations=0,
        )

    # Expand upper bound until cracked or m_max is reached.
    m_lo = 0.0
    stress_lo = stress0
    err_lo = err0
    system_lo = system0
    m_hi: float | None = None
    stress_hi: MaximalStressResult | None = None
    err_hi: float | None = None
    system_hi: SystemModel | None = None
    iterations = 0

    probe = min(m_max, 50.0) if m_max > 0 else 0.0
    while probe <= m_max + 1e-15:
        iterations += 1
        system_p, stress_p, err_p = _evaluate_at_mass(
            layers=layers,
            weak_layer=weak_layer,
            system_type=system_type,
            phi=phi,
            cut_length=cut_length,
            bedded_length=bedded_length,
            end_mass=probe,
            config=config,
            num=num,
        )
        if float(stress_p.max_Sxx_norm) >= CRACK_THRESHOLD:
            m_hi = probe
            stress_hi = stress_p
            err_hi = err_p
            system_hi = system_p
            break
        m_lo = probe
        stress_lo = stress_p
        err_lo = err_p
        system_lo = system_p
        if probe >= m_max:
            break
        # Geometric expansion, capped at m_max.
        next_probe = probe * 2.0 if probe > 0 else min(50.0, m_max)
        probe = min(m_max, next_probe) if next_probe < m_max else m_max
        if probe <= m_lo + 1e-15:
            break

    if m_hi is None or stress_hi is None or err_hi is None or system_hi is None:
        return CriticalMassSearchResult(
            critical_mass_kg=m_max,
            already_cracked=False,
            never_cracked=True,
            converged=False,
            message=f"no crack up to m_max={m_max}",
            max_Sxx_norm=float(stress_lo.max_Sxx_norm),
            energy_release_rate=err_lo,
            thickness_fraction_without_density_gate=float(
                stress_lo.slab_tensile_criterion
            ),
            maximal_stress_result=stress_lo,
            system=system_lo,
            phi=phi,
            system_type=system_type,
            iterations=iterations,
        )

    # Bisection on mass until stress residual or mass bracket is tight.
    best_system = system_hi
    best_stress = stress_hi
    best_err = err_hi
    best_m = m_hi
    while (m_hi - m_lo) > mass_tol and abs(
        float(best_stress.max_Sxx_norm) - CRACK_THRESHOLD
    ) > stress_tol:
        iterations += 1
        mid = 0.5 * (m_lo + m_hi)
        system_m, stress_m, err_m = _evaluate_at_mass(
            layers=layers,
            weak_layer=weak_layer,
            system_type=system_type,
            phi=phi,
            cut_length=cut_length,
            bedded_length=bedded_length,
            end_mass=mid,
            config=config,
            num=num,
        )
        if float(stress_m.max_Sxx_norm) >= CRACK_THRESHOLD:
            m_hi = mid
            stress_hi = stress_m
            err_hi = err_m
            system_hi = system_m
            best_system = system_m
            best_stress = stress_m
            best_err = err_m
            best_m = mid
        else:
            m_lo = mid
            stress_lo = stress_m
            err_lo = err_m
            system_lo = system_m

    return CriticalMassSearchResult(
        critical_mass_kg=float(best_m),
        already_cracked=False,
        never_cracked=False,
        converged=True,
        message="critical end mass found",
        max_Sxx_norm=float(best_stress.max_Sxx_norm),
        energy_release_rate=float(best_err),
        thickness_fraction_without_density_gate=float(
            best_stress.slab_tensile_criterion
        ),
        maximal_stress_result=best_stress,
        system=best_system,
        phi=phi,
        system_type=system_type,
        iterations=iterations,
    )


def _orientation_specs(phi0: float) -> tuple[_OrientationSpec, _OrientationSpec]:
    """
    Upslope / downslope specs aligned with fixed-cut and critical-cut.

    Other approaches use ``-pst`` (upslope) and ``pst-`` (downslope) at ``+φ₀``.
    End mass can only sit on a segment's right edge, so both sides stay on
    ``pst-`` (free face + weight on the right) and upslope is the φ-mirror of
    ``-pst`` at ``+φ₀``:

    - downslope: ``pst-``, ``+φ₀`` — same as the other approaches
    - upslope: ``pst-``, ``−φ₀`` — mirror of ``-pst`` / ``+φ₀``
    """
    return (
        _OrientationSpec(name="upslope", system_type="pst-", phi=-float(phi0)),
        _OrientationSpec(name="downslope", system_type="pst-", phi=+float(phi0)),
    )


def _side_diagnostics(result: CriticalMassSearchResult) -> dict[str, Any]:
    """JSON-friendly per-orientation block."""
    return {
        "system_type": result.system_type,
        "phi": float(result.phi),
        "critical_mass_kg": float(result.critical_mass_kg),
        "already_cracked": result.already_cracked,
        "never_cracked": result.never_cracked,
        "converged": result.converged,
        "message": result.message,
        "energy_release_rate": float(result.energy_release_rate),
        "max_Sxx_norm": float(result.max_Sxx_norm),
        "thickness_fraction_without_density_gate": float(
            result.thickness_fraction_without_density_gate
        ),
        "iterations": result.iterations,
    }


def evaluate_pst_critical_mass(
    *,
    layers: list[Layer],
    weak_layer: WeakLayer,
    phi: float,
    cut_length: float = CUT_LENGTH_MM,
    bedded_length: float = 5e3,
    m_max: float = DEFAULT_M_MAX_KG,
    mass_tol: float = DEFAULT_MASS_TOL_KG,
    stress_tol: float = DEFAULT_STRESS_TOL,
    config: Config | None = None,
    num: int = 4000,
) -> ExperimentalSteadyStateResult:
    """
    Dual-orientation PST critical end-mass evaluator.

    Weight is always on the free face (both sides ``pst-``). Downslope matches
    the other approaches (``pst-`` at ``+φ₀``); upslope is the φ-mirror
    (``pst-`` at ``−φ₀``). Core fields come from the ease-winning orientation
    (lower ``critical_mass_kg`` among usable sides); ``characteristic_length``
    is the fixed cut.
    """
    if config is None:
        config = Config(touchdown=False)
    elif config.touchdown:
        raise ValueError(
            "evaluate_pst_critical_mass requires touchdown=False so the "
            "requested cut length is preserved"
        )

    sides: dict[OrientationName, CriticalMassSearchResult] = {}
    for spec in _orientation_specs(phi):
        sides[spec.name] = search_critical_end_mass(
            layers=layers,
            weak_layer=weak_layer,
            system_type=spec.system_type,
            phi=spec.phi,
            cut_length=cut_length,
            bedded_length=bedded_length,
            m_max=m_max,
            mass_tol=mass_tol,
            stress_tol=stress_tol,
            config=config,
            num=num,
        )

    upslope = sides["upslope"]
    downslope = sides["downslope"]
    up_diag = _side_diagnostics(upslope)
    down_diag = _side_diagnostics(downslope)
    selection = select_ease_orientation(
        up_diag,
        down_diag,
        ease_key="critical_mass_kg",
        higher_is_easier=False,
    )
    winner_name = selection.winner
    winner = sides[winner_name]

    diagnostics: dict[str, Any] = {
        "method": "pst_critical_mass",
        "cut_length_mm": float(cut_length),
        "m_max_kg": float(m_max),
        "winner": winner_name,
        "err_winner": selection.err_winner,
        "selection_rule": selection.selection_rule,
        "phi0": float(phi),
        "critical_mass_kg": {
            "upslope": float(upslope.critical_mass_kg),
            "downslope": float(downslope.critical_mass_kg),
            "winner": float(winner.critical_mass_kg),
        },
        "upslope": up_diag,
        "downslope": down_diag,
        "LSKI_MM": float(LSKI_MM),
        "l_eff_note": (
            "End mass is converted to a line load using the legacy global "
            f"LSKI_MM={LSKI_MM} mm out-of-plane ski length; per-segment "
            "l_eff is not implemented yet."
        ),
    }

    if winner.already_cracked:
        message = f"{winner_name}: already cracked at m=0"
    elif winner.never_cracked:
        message = f"{winner_name}: no crack up to m_max={m_max}"
    else:
        message = (
            f"{winner_name}: critical end mass "
            f"{winner.critical_mass_kg:.6g} kg (lower critical mass)"
        )

    return ExperimentalSteadyStateResult(
        converged=winner.converged,
        message=message,
        characteristic_length=float(cut_length),
        energy_release_rate=float(winner.energy_release_rate),
        maximal_stress_result=winner.maximal_stress_result,
        system=winner.system,
        diagnostics=diagnostics,
    )
