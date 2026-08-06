"""
Data-driven slab A/B comparison tests (hybrid steady state).

Populate ``COMPARISON_CASES`` with A/B setup pairs. Both gates use production
``CriteriaEvaluator.evaluate_SteadyState`` (hybrid).

Tensile ease (shorter ``critical_cut_length`` = easier) follows Steph's tensile
map: cases 1–11 & 21–23 → A easier; 12–15 → B easier. Cases 16–20 are
smoke-only for tensile (structured positive fields; no A/B ordering).

ERR yardstick (higher ``err.energy_release_rate`` wins) follows Steph §2 with
**no** smoke exclusions: cases 1–11 & 16–20 → B higher ERR; 12–15 & 21–23 → A
higher ERR.
"""

from dataclasses import dataclass, field

import pytest

from weac.analysis.criteria_evaluator import CriteriaEvaluator
from weac.components import (
    Config,
    CriteriaConfig,
    Layer,
    ScenarioConfig,
    Segment,
)
from weac.components.presets import WEAK_LAYER
from weac.components.model_input import ModelInput
from weac.core.system_model import SystemModel


@dataclass(frozen=True)
class LayerDefinition:
    """Minimal slab layer definition for comparison cases."""

    rho: float
    h: float


@dataclass
class SetupDefinition:
    """A comparison setup with at most two slab layers."""

    layers: tuple[LayerDefinition, ...]
    weak_layer_kwargs: dict[str, float] = field(default_factory=dict)
    scenario_kwargs: dict[str, float | str] = field(default_factory=dict)
    config_kwargs: dict[str, bool | str] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonCase:
    """Two setups compared for tensile ease."""

    name: str
    setup_a: SetupDefinition
    setup_b: SetupDefinition


DEFAULT_SCENARIO_KWARGS: dict[str, float | str] = {"phi": 35.0}
DEFAULT_CONFIG_KWARGS: dict[str, bool | str] = {
    "touchdown": False,
}
DEFAULT_SEGMENTS: tuple[tuple[float, bool, float], ...] = (
    (10000.0, True, 0.0),
    (10000.0, True, 0.0),
)

# Steph tensile map: A easier (shorter L_crit) except 12–15 (B easier).
# Cases 16–20 are smoke-only (no tensile A/B ordering).
A_EASIER_CASES = frozenset(
    {f"case_{i}" for i in list(range(1, 12)) + list(range(21, 24))}
)
B_EASIER_CASES = frozenset({f"case_{i}" for i in range(12, 16)})
SMOKE_ONLY_CASES = frozenset({f"case_{i}" for i in range(16, 21)})

# Steph ERR map (§2): higher energy_release_rate wins; no smoke exclusions.
A_HIGHER_ERR_CASES = frozenset(
    {f"case_{i}" for i in list(range(12, 16)) + list(range(21, 24))}
)
B_HIGHER_ERR_CASES = frozenset(
    {f"case_{i}" for i in list(range(1, 12)) + list(range(16, 21))}
)


def _layer_cm(thickness_cm: float, density: float) -> LayerDefinition:
    """Create a layer definition from thickness in centimeters."""
    return LayerDefinition(rho=density, h=thickness_cm * 10.0)


def _setup_from_cm(*layers: tuple[float, float]) -> SetupDefinition:
    """Create a setup from top-to-bottom ``(thickness_cm, density)`` tuples."""
    return SetupDefinition(
        layers=tuple(
            _layer_cm(thickness_cm, density) for thickness_cm, density in layers
        )
    )


COMPARISON_CASES: tuple[ComparisonCase, ...] = (
    # Left slab thinner than right slab; same density
    ComparisonCase(
        name="case_1",
        setup_a=_setup_from_cm((20, 75)),
        setup_b=_setup_from_cm((40, 75)),
    ),
    ComparisonCase(
        name="case_2",
        setup_a=_setup_from_cm((20, 125)),
        setup_b=_setup_from_cm((40, 125)),
    ),
    ComparisonCase(
        name="case_3",
        setup_a=_setup_from_cm((20, 175)),
        setup_b=_setup_from_cm((40, 175)),
    ),
    ComparisonCase(
        name="case_4",
        setup_a=_setup_from_cm((20, 275)),
        setup_b=_setup_from_cm((40, 275)),
    ),
    # Same height of slabs; A slab has lower density than B slab
    ComparisonCase(
        name="case_5",
        setup_a=_setup_from_cm((50, 75)),
        setup_b=_setup_from_cm((50, 125)),
    ),
    ComparisonCase(
        name="case_6",
        setup_a=_setup_from_cm((50, 125)),
        setup_b=_setup_from_cm((50, 175)),
    ),
    ComparisonCase(
        name="case_7",
        setup_a=_setup_from_cm((50, 175)),
        setup_b=_setup_from_cm((50, 275)),
    ),
    # A slab is B slab plus a thin slab of lower density on top
    ComparisonCase(
        name="case_8",
        setup_a=_setup_from_cm((20, 75), (30, 175)),
        setup_b=_setup_from_cm((50, 175)),
    ),
    ComparisonCase(
        name="case_9",
        setup_a=_setup_from_cm((20, 75), (30, 275)),
        setup_b=_setup_from_cm((50, 275)),
    ),
    ComparisonCase(
        name="case_10",
        setup_a=_setup_from_cm((20, 125), (30, 275)),
        setup_b=_setup_from_cm((50, 275)),
    ),
    ComparisonCase(
        name="case_11",
        setup_a=_setup_from_cm((20, 175), (30, 275)),
        setup_b=_setup_from_cm((50, 275)),
    ),
    # A slab is B slab plus a thin slab of higher density on top
    ComparisonCase(
        name="case_12",
        setup_a=_setup_from_cm((20, 175), (30, 75)),
        setup_b=_setup_from_cm((50, 75)),
    ),
    ComparisonCase(
        name="case_13",
        setup_a=_setup_from_cm((20, 275), (30, 75)),
        setup_b=_setup_from_cm((50, 75)),
    ),
    ComparisonCase(
        name="case_14",
        setup_a=_setup_from_cm((20, 175), (30, 125)),
        setup_b=_setup_from_cm((50, 125)),
    ),
    ComparisonCase(
        name="case_15",
        setup_a=_setup_from_cm((20, 275), (30, 125)),
        setup_b=_setup_from_cm((50, 125)),
    ),
    # Both slabs two layers; thin and thicker, but A slab's thin slab is thinner
    ComparisonCase(
        name="case_16",
        setup_a=_setup_from_cm((30, 75), (20, 125)),
        setup_b=_setup_from_cm((50, 75), (20, 125)),
    ),
    ComparisonCase(
        name="case_17",
        setup_a=_setup_from_cm((30, 75), (20, 225)),
        setup_b=_setup_from_cm((50, 75), (20, 225)),
    ),
    ComparisonCase(
        name="case_18",
        setup_a=_setup_from_cm((30, 75), (20, 275)),
        setup_b=_setup_from_cm((50, 75), (20, 275)),
    ),
    ComparisonCase(
        name="case_19",
        setup_a=_setup_from_cm((30, 125), (20, 225)),
        setup_b=_setup_from_cm((50, 125), (20, 225)),
    ),
    ComparisonCase(
        name="case_20",
        setup_a=_setup_from_cm((30, 125), (20, 275)),
        setup_b=_setup_from_cm((50, 125), (20, 275)),
    ),
    # A has higher density slab at bottom and lower density slab at top, B vice versa
    ComparisonCase(
        name="case_21",
        setup_a=_setup_from_cm((40, 125), (5, 350)),
        setup_b=_setup_from_cm((5, 350), (40, 125)),
    ),
    ComparisonCase(
        name="case_22",
        setup_a=_setup_from_cm((40, 75), (15, 275)),
        setup_b=_setup_from_cm((15, 275), (40, 75)),
    ),
    ComparisonCase(
        name="case_23",
        setup_a=_setup_from_cm((40, 175), (15, 275)),
        setup_b=_setup_from_cm((15, 275), (40, 175)),
    ),
)


def _build_layers(layer_defs: tuple[LayerDefinition, ...]) -> list[Layer]:
    """Convert lightweight layer definitions into WEAC layers."""
    if not 1 <= len(layer_defs) <= 2:
        raise ValueError("Each setup must define one or two slab layers.")
    return [Layer(rho=layer_def.rho, h=layer_def.h) for layer_def in layer_defs]


def _build_segments() -> list[Segment]:
    """Create stable steady-state segments for each comparison."""
    return [
        Segment(length=length, has_foundation=has_foundation, m=mass)
        for length, has_foundation, mass in DEFAULT_SEGMENTS
    ]


def _build_system(setup: SetupDefinition) -> SystemModel:
    """Create a WEAC system model from a compact setup definition."""
    scenario_kwargs = {**DEFAULT_SCENARIO_KWARGS, **setup.scenario_kwargs}
    config_kwargs = {**DEFAULT_CONFIG_KWARGS, **setup.config_kwargs}
    weak_layer = WEAK_LAYER.model_copy(update=setup.weak_layer_kwargs)

    model_input = ModelInput(
        layers=_build_layers(setup.layers),
        weak_layer=weak_layer,
        segments=_build_segments(),
        scenario_config=ScenarioConfig(**scenario_kwargs),
    )
    return SystemModel(model_input=model_input, config=Config(**config_kwargs))


def _evaluate_hybrid_ss(evaluator: CriteriaEvaluator, setup: SetupDefinition):
    """Run production hybrid steady state for a setup."""
    return evaluator.evaluate_SteadyState(_build_system(setup))


@pytest.fixture(scope="module")
def evaluator():
    """Shared CriteriaEvaluator for the comparison matrix."""
    return CriteriaEvaluator(CriteriaConfig())


@pytest.fixture(scope="module")
def hybrid_results(evaluator):
    """Cache hybrid A/B steady-state results per comparison case."""
    return {
        case.name: (
            _evaluate_hybrid_ss(evaluator, case.setup_a),
            _evaluate_hybrid_ss(evaluator, case.setup_b),
        )
        for case in COMPARISON_CASES
    }


class TestSlabTensileComparisons:
    """Regression checks for hybrid tensile ease and ERR ordering."""

    def test_hybrid_steady_state_structured_fields(self, hybrid_results):
        """Smoke: hybrid SS exposes tensile L_crit and ERR scalars."""
        if not COMPARISON_CASES:
            pytest.skip("Populate COMPARISON_CASES A/B setup pairs.")
        result_a, _ = hybrid_results[COMPARISON_CASES[0].name]
        assert result_a.tensile.critical_cut_length > 0
        assert hasattr(result_a.err, "energy_release_rate")
        assert result_a.err.energy_release_rate > 0

    @pytest.mark.parametrize(
        "case",
        COMPARISON_CASES,
        ids=[case.name for case in COMPARISON_CASES],
    )
    def test_slab_tensile_ease_ordering(self, hybrid_results, case):
        """Steph tensile map: A/B ease via shorter critical_cut_length."""
        result_a, result_b = hybrid_results[case.name]
        l_a = result_a.tensile.critical_cut_length
        l_b = result_b.tensile.critical_cut_length
        # Always require structured ERR fields.
        assert result_a.err.energy_release_rate > 0
        assert result_b.err.energy_release_rate > 0

        if case.name in SMOKE_ONLY_CASES:
            assert l_a > 0
            assert l_b > 0
            return
        if case.name in A_EASIER_CASES:
            assert l_a <= l_b, (
                f"{case.name}: expected A easier (L_crit A <= B), got "
                f"A={l_a:.6f}, B={l_b:.6f}"
            )
        elif case.name in B_EASIER_CASES:
            assert l_b <= l_a, (
                f"{case.name}: expected B easier (L_crit B <= A), got "
                f"A={l_a:.6f}, B={l_b:.6f}"
            )
        else:
            assert l_a > 0
            assert l_b > 0

    @pytest.mark.parametrize(
        "case",
        COMPARISON_CASES,
        ids=[case.name for case in COMPARISON_CASES],
    )
    def test_slab_err_ordering(self, hybrid_results, case):
        """Steph ERR map: higher energy_release_rate for all cases (no smoke)."""
        result_a, result_b = hybrid_results[case.name]
        err_a = result_a.err.energy_release_rate
        err_b = result_b.err.energy_release_rate
        assert err_a > 0
        assert err_b > 0

        if case.name in A_HIGHER_ERR_CASES:
            assert err_a >= err_b, (
                f"{case.name}: expected A higher ERR (A >= B), got "
                f"A={err_a:.6f}, B={err_b:.6f}"
            )
        elif case.name in B_HIGHER_ERR_CASES:
            assert err_b >= err_a, (
                f"{case.name}: expected B higher ERR (B >= A), got "
                f"A={err_a:.6f}, B={err_b:.6f}"
            )
        else:
            pytest.fail(f"{case.name}: missing from ERR expectation frozensets")
