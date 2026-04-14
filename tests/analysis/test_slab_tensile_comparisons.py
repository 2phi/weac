"""
Data-driven slab tensile criterion comparison tests.

Populate ``COMPARISON_CASES``defines A/B setup pairs to assert that
``A.slab_tensile_criterion > B.slab_tensile_criterion`` for each case.
"""

from dataclasses import dataclass, field
import unittest

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
    """Two setups where setup A must exceed setup B."""

    name: str
    setup_a: SetupDefinition
    setup_b: SetupDefinition


DEFAULT_SCENARIO_KWARGS: dict[str, float | str] = {"phi": 35.0}
DEFAULT_CONFIG_KWARGS: dict[str, bool | str] = {
    "touchdown": True,
}
DEFAULT_SEGMENTS: tuple[tuple[float, bool, float], ...] = (
    (10000.0, True, 0.0),
    (10000.0, True, 0.0),
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
    ComparisonCase(
        name="case_1",
        setup_a=_setup_from_cm((20, 175), (30, 275)),
        setup_b=_setup_from_cm((30, 275)),
    ),
    ComparisonCase(
        name="case_2",
        setup_a=_setup_from_cm((50, 75), (20, 225)),
        setup_b=_setup_from_cm((30, 75), (20, 225)),
    ),
    ComparisonCase(
        name="case_3",
        setup_a=_setup_from_cm((50, 75), (20, 125)),
        setup_b=_setup_from_cm((30, 75), (20, 125)),
    ),
    ComparisonCase(
        name="case_4",
        setup_a=_setup_from_cm((50, 75)),
        setup_b=_setup_from_cm((50, 125)),
    ),
    ComparisonCase(
        name="case_5",
        setup_a=_setup_from_cm((50, 175)),
        setup_b=_setup_from_cm((50, 125)),
    ),
    ComparisonCase(
        name="case_6",
        setup_a=_setup_from_cm((20, 275)),
        setup_b=_setup_from_cm((40, 275)),
    ),
    ComparisonCase(
        name="case_7",
        setup_a=_setup_from_cm((40, 125), (5, 350)),
        setup_b=_setup_from_cm((5, 350), (40, 125)),
    ),
    ComparisonCase(
        name="case_8",
        setup_a=_setup_from_cm((40, 175), (15, 275)),
        setup_b=_setup_from_cm((15, 275), (40, 175)),
    ),
    ComparisonCase(
        name="case_9",
        setup_a=_setup_from_cm((15, 275), (40, 75)),
        setup_b=_setup_from_cm((40, 75), (15, 275)),
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


def _evaluate_slab_tensile_criterion(
    evaluator: CriteriaEvaluator, setup: SetupDefinition
) -> float:
    """Run the steady-state evaluator and return the slab tensile criterion."""
    result = evaluator.evaluate_SteadyState(
        _build_system(setup), mode="B_point_contact"
    )
    return result.maximal_stress_result.slab_tensile_criterion


class TestSlabTensileComparisons(unittest.TestCase):
    """Regression checks for slab tensile setup ordering."""

    @classmethod
    def setUpClass(cls):
        """Create a shared evaluator for the comparison matrix."""
        cls.evaluator = CriteriaEvaluator(CriteriaConfig())

    def test_slab_tensile_criterion_ordering(self):
        """Each case asserts that setup A exceeds setup B."""
        if not COMPARISON_CASES:
            self.skipTest("Populate COMPARISON_CASES with the seven A/B setup pairs.")

        for case in COMPARISON_CASES:
            with self.subTest(case=case.name):
                criterion_a = _evaluate_slab_tensile_criterion(
                    self.evaluator, case.setup_a
                )
                criterion_b = _evaluate_slab_tensile_criterion(
                    self.evaluator, case.setup_b
                )
                # print(f"{case.name}: A={criterion_a:.6f}, B={criterion_b:.6f}")
                self.assertGreaterEqual(
                    criterion_a,
                    criterion_b,
                    msg=(
                        f"{case.name}: expected A > B, got "
                        f"A={criterion_a:.6f}, B={criterion_b:.6f}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
