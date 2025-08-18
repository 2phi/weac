"""
This script demonstrates the basic usage of the WEAC package to run a simulation.
"""

import logging

from weac.analysis.criteria_evaluator import (
    CoupledCriterionResult,
    CriteriaEvaluator,
)
from weac.analysis.analyzer import Analyzer
from weac.analysis.plotter import Plotter
from weac.components import (
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
    Config,
)
from weac.core.system_model import SystemModel
from weac.logging_config import setup_logging

setup_logging(level="INFO")

# Suppress matplotlib debug logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# === SYSTEM 1: Basic Configuration ===
config1 = Config(
    touchdown=True,
)
scenario_config1 = ScenarioConfig(phi=5, system_type="skier")  # Gentle slope
criteria_config1 = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)

weak_layer1 = WeakLayer(rho=80, h=25, E=0.25, G_Ic=1)
layers1 = [
    Layer(rho=170, h=100),  # Top Layer
    Layer(rho=280, h=100),  # Bottom Layer
]
segments1 = [
    Segment(length=3000, has_foundation=True, m=70),
    Segment(length=4000, has_foundation=True, m=0),
]

model_input1 = ModelInput(
    scenario_config=scenario_config1,
    weak_layer=weak_layer1,
    layers=layers1,
    segments=segments1,
)

system1 = SystemModel(config=config1, model_input=model_input1)

# === SYSTEM 2: Different Slope Angle ===
config2 = Config(
    touchdown=False,
)
scenario_config2 = ScenarioConfig(phi=30, system_type="skier")  # Steeper slope
weak_layer2 = WeakLayer(rho=80, h=25, E=0.25, G_Ic=1)
layers2 = [
    Layer(rho=170, h=100),  # Top Layer
    Layer(rho=280, h=100),  # Bottom Layer
]
segments2 = [
    Segment(length=3000, has_foundation=True, m=70),
    Segment(length=4000, has_foundation=True, m=0),
]

model_input2 = ModelInput(
    scenario_config=scenario_config2,
    weak_layer=weak_layer2,
    layers=layers2,
    segments=segments2,
)

system2 = SystemModel(config=config2, model_input=model_input2)

# === SYSTEM 3: Different Layer Configuration ===
config3 = Config(
    touchdown=False,
)
scenario_config3 = ScenarioConfig(phi=15, system_type="skier")  # Medium slope
weak_layer3 = WeakLayer(rho=80, h=25, E=0.3, G_Ic=1.2)  # Different weak layer
layers3 = [
    Layer(rho=150, h=80),  # Lighter top layer
    Layer(rho=200, h=60),  # Medium layer
    Layer(rho=320, h=120),  # Heavier bottom layer
]
segments3 = [
    Segment(length=3500, has_foundation=True, m=60),  # Different skier mass
    Segment(length=3500, has_foundation=True, m=0),
]

model_input3 = ModelInput(
    scenario_config=scenario_config3,
    weak_layer=weak_layer3,
    layers=layers3,
    segments=segments3,
)

system3 = SystemModel(config=config3, model_input=model_input3)

# === SYSTEM 4: Advanced Configuration ===
config4 = Config(
    touchdown=False,
)
scenario_config4 = ScenarioConfig(phi=38, system_type="skier")
weak_layer4 = WeakLayer(rho=80, h=25, E=0.25, G_Ic=1)
layers4 = [
    Layer(rho=170, h=100),  # (1) Top Layer
    Layer(rho=190, h=40),  # (2)
    Layer(rho=230, h=130),
    Layer(rho=250, h=20),
    Layer(rho=210, h=70),
    Layer(rho=380, h=20),
    Layer(rho=280, h=100),  # (N) Bottom Layer
]
segments4 = [
    Segment(length=5000, has_foundation=True, m=80),
    Segment(length=3000, has_foundation=True, m=0),
    Segment(length=3000, has_foundation=False, m=0),
    Segment(length=4000, has_foundation=True, m=70),
    Segment(length=3000, has_foundation=True, m=0),
]
criteria_config4 = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)
model_input4 = ModelInput(
    scenario_config=scenario_config4,
    weak_layer=weak_layer4,
    layers=layers4,
    segments=segments4,
)

system4 = SystemModel(config=config4, model_input=model_input4)

# === DEMONSTRATION OF PLOTTING CAPABILITIES ===

print("=== WEAC Plotting Demonstration ===")

# Single system plotting
print("\n1. Single System Analysis:")
print(f"   System 1 - φ={system1.scenario.phi}°, H={system1.slab.H}mm")

plotter_single = Plotter()
analyzer1 = Analyzer(system1)
xsl, z, xwl = analyzer1.rasterize_solution()

# Generate individual plots
print("   - Generating slab profile...")
plotter_single.plot_slab_profile(
    weak_layers=system1.weak_layer,
    slabs=system1.slab,
    labels=["φ=5° System"],
    filename="single_slab_profile",
)

print("   - Generating displacement plot...")
plotter_single.plot_displacements(
    analyzer=analyzer1, x=xsl, z=z, filename="single_displacements"
)

print("   - Generating section forces plot...")
plotter_single.plot_section_forces(
    system_model=system1, filename="single_section_forces"
)

print("   - Generating stress plot...")
plotter_single.plot_stresses(analyzer=analyzer1, x=xwl, z=z, filename="single_stresses")

print("   - Generating deformed contour plot...")
plotter_single.plot_deformed(
    xsl, xwl, z, analyzer1, field="w", filename="single_deformed_w"
)
plotter_single.plot_deformed(
    xsl, xwl, z, analyzer1, field="principal", filename="single_deformed_principal"
)

print("   - Generating stress envelope...")
plotter_single.plot_stress_envelope(
    system_model=system1,
    criteria_evaluator=CriteriaEvaluator(criteria_config1),
    all_envelopes=False,
    filename="single_stress_envelope",
)

# === CRITERIA ANALYSIS DEMONSTRATION ===
print("\n2. Coupled Criterion Analysis Example:")
print("   This example is from the demo notebook and shows a more advanced analysis.")

# Define thinner snow profile (standard snow profile A), with higher weak layer Young's Modulus
layers_analysis = [
    Layer(rho=350, h=120),
    Layer(rho=270, h=120),
    Layer(rho=180, h=120),
]
scenario_config_analysis = ScenarioConfig(
    system_type="skier",
    phi=30,
)
segments_analysis = [
    Segment(length=18000, has_foundation=True, m=0),
    Segment(length=0, has_foundation=False, m=75),
    Segment(length=0, has_foundation=False, m=0),
    Segment(length=18000, has_foundation=False, m=0),
]
weak_layer_analysis = WeakLayer(
    rho=150,
    h=30,
    E=1,
)
criteria_config_analysis = CriteriaConfig(
    stress_envelope_method="adam_unpublished",
    scaling_factor=1,
    order_of_magnitude=1,
)
model_input_analysis = ModelInput(
    scenario_config=scenario_config_analysis,
    layers=layers_analysis,
    segments=segments_analysis,
    weak_layer=weak_layer_analysis,
)

sys_model_analysis = SystemModel(
    model_input=model_input_analysis,
)

criteria_evaluator = CriteriaEvaluator(
    criteria_config=criteria_config_analysis,
)

results: CoupledCriterionResult = criteria_evaluator.evaluate_coupled_criterion(
    system=sys_model_analysis
)

print("\n--- Coupled Criterion Analysis Results ---")
print(
    "The thinner snow profile, with adjusted weak layer Young's Modulus, is governed by a coupled criterion for anticrack nucleation."
)
print(
    f"The critical skier weight is {results.critical_skier_weight:.1f} kg and the associated crack length is {results.crack_length:.1f} mm."
)
print("\nDetailed results:")
print(f"  Algorithm convergence: {results.converged}")
print(f"  Message: {results.message}")
print(f"  Self-collapse: {results.self_collapse}")
print(f"  Pure stress criteria: {results.pure_stress_criteria}")
print(
    f"  Initial critical skier weight: {results.initial_critical_skier_weight:.1f} kg"
)
print(f"  G delta: {results.g_delta:.4f}")
print(f"  Final error: {results.dist_ERR_envelope:.4f}")
print(f"  Max distance to failure: {results.max_dist_stress:.4f}")
print(f"  Iterations: {results.iterations}")


# Check for crack self-propagation
system = results.final_system
propagation_results = criteria_evaluator.check_crack_self_propagation(system)
print("\n--- Crack Self-Propagation Check ---")
print(
    f"Results of crack propagation criterion: G_delta = {propagation_results[0]:.4f}, Propagation expected: {propagation_results[1]}"
)
print(
    "As the crack propagation criterion is not met, we investigate the minimum self-propagation crack boundary."
)


# Find minimum crack length for self-propagation
initial_interval = (1, 3000)  # Interval for the crack length search (mm)
min_crack_length, new_segments = criteria_evaluator.find_minimum_crack_length(
    system, search_interval=initial_interval
)

print("\n--- Minimum Self-Propagation Crack Length ---")
if min_crack_length is not None:
    print(f"Minimum Crack Length for Self-Propagation: {min_crack_length:.1f} mm")
else:
    print("The search for the minimum crack length did not converge.")

print(
    "\nThe anticrack created is not sufficiently long to surpass the self-propagation boundary. The propensity of the generated anticrack to propagate is low."
)


print("\n=== Analysis Complete ===")
print("Check the 'plots/' directory for generated visualizations.")
print("\nPlot files generated:")
print("  - single_*.png")
