"""
This script demonstrates the basic usage of the WEAC package to run a simulation.
"""

import logging

from weac_2.analysis.plotter import Plotter
from weac_2.components import (
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac_2.components.config import Config
from weac_2.core.system_model import SystemModel
from weac_2.logging_config import setup_logging

setup_logging()

# Suppress matplotlib debug logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# === SYSTEM 1: Basic Configuration ===
config1 = Config(
    touchdown=False,
)
scenario_config1 = ScenarioConfig(phi=5, system_type="skier")  # Steeper slope
weak_layer1 = WeakLayer(rho=10, h=25, E=0.25, G_Ic=1)
layers1 = [
    Layer(rho=170, h=100),  # Top Layer
    Layer(rho=280, h=100),  # Bottom Layer
]
segments1 = [
    Segment(length=3000, has_foundation=True, m=0),
    Segment(length=4000, has_foundation=True, m=0),
]
criteria_config1 = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)

model_input1 = ModelInput(
    scenario_config=scenario_config1,
    weak_layer=weak_layer1,
    layers=layers1,
    segments=segments1,
    criteria_config=criteria_config1,
)

system1 = SystemModel(config=config1, model_input=model_input1)
unknown_constants1 = system1.unknown_constants

# === DEMO 1: Single System Analysis ===

print("=== WEAC Plotting Demonstration ===")

# Single system plotting
print("\n1. Single System Analysis:")
print(f"   System 1 - φ={system1.scenario.phi}°, H={system1.slab.H}mm")

plotter_single = Plotter(system1, labels=["φ=5° System"])

# Generate individual plots
print("   - Generating slab profile...")
plotter_single.plot_slab_profile(filename="single_slab_profile")

print("   - Generating displacement plot...")
plotter_single.plot_displacements(filename="single_displacements")

print("   - Generating section forces plot...")
plotter_single.plot_section_forces(filename="single_section_forces")

print("   - Generating stress plot...")
plotter_single.plot_stresses(filename="single_stresses")

print("   - Generating deformed contour plot...")
plotter_single.plot_deformed(field="w", filename="single_deformed_w")
plotter_single.plot_deformed(field="principal", filename="single_deformed_principal")

print("   - Generating stress envelope...")
plotter_single.plot_stress_envelope(filename="single_stress_envelope")
