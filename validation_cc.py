"""
This script demonstrates the basic usage of the WEAC package to run a simulation.
"""

import logging

from weac.analysis import criteria_evaluator
from weac.analysis.plotter import Plotter
from weac.components import (
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac.components.config import Config
from weac.core.system_model import SystemModel
from weac.logging_config import setup_logging

from weac.components.criteria_config import CriteriaConfig
from weac.analysis.criteria_evaluator import CriteriaEvaluator, CoupledCriterionResult

setup_logging()

# Suppress matplotlib debug logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("weac.core").setLevel(logging.WARNING)
logging.getLogger("weac.analysis").setLevel(logging.WARNING)

# Define thinner snow profile (standard snow profile A), with higher weak layer Young's Modulus
layers = [
    Layer(rho=350, h=120),
    Layer(rho=270, h=120),
    Layer(rho=180, h=120),
]
scenario_config = ScenarioConfig(
    system_type="skier",
    phi=30,
)
segments = [
    Segment(length=18000, has_foundation=True, m=0),
    Segment(length=0, has_foundation=False, m=75),
    Segment(length=0, has_foundation=False, m=0),
    Segment(length=18000, has_foundation=False, m=0),
]
weak_layer = WeakLayer(
    rho=150,
    h=30,
    E=1,
)
criteria_config = CriteriaConfig(
    stress_envelope_method="adam_unpublished",
    scaling_factor=1,
    order_of_magnitude=1,
)
model_input = ModelInput(
    scenario_config=scenario_config,
    layers=layers,
    segments=segments,
    weak_layer=weak_layer,
)

sys_model = SystemModel(
    model_input=model_input,
)

crit_eval = CriteriaEvaluator(
    criteria_config=criteria_config,
)

results: CoupledCriterionResult = crit_eval.evaluate_coupled_criterion(system=sys_model)

print("Algorithm convergence:", results.converged)
print("Message:", results.message)
print("Self-collapse:", results.self_collapse)
print("Pure stress criteria:", results.pure_stress_criteria)
print("Critical skier weight:", results.critical_skier_weight)
print("Initial critical skier weight:", results.initial_critical_skier_weight)
print("Crack length:", results.crack_length)
print("G delta:", results.g_delta)
print("Final error:", results.dist_ERR_envelope)
print("Iterations:", results.iterations)
