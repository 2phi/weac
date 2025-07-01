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

logger = logging.getLogger(__name__)

# Suppress matplotlib debug logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


config1 = Config(
    touchdown=True,
    youngs_modulus_method="bergfeld",
    stress_envelope_method="adam_unpublished",
)
scenario_config1 = ScenarioConfig(
    phi=5, system_type="pst-", crack_length=1000
)  # Steeper slope
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
logger.info("Validated model input 1")

model_input1 = ModelInput(
    scenario_config=scenario_config1,
    weak_layer=weak_layer1,
    layers=layers1,
    segments=segments1,
    criteria_config=criteria_config1,
)

system1 = SystemModel(model_input=model_input1, config=config1)
logger.info("System 1 setup")
unknown_constants = system1.get_unknown_constants()
logger.info("Unknown constants: %s", unknown_constants)


# Equivalent setup in new system
layers = [
    Layer(rho=200, h=150),
    Layer(rho=300, h=100),
]

# For touchdown=True, the segmentation will be different
# Need to match the segments that would be created by calc_segments with touchdown=True
segments = [
    Segment(length=6000, has_foundation=True, m=0),
    Segment(length=1000, has_foundation=False, m=75),
    Segment(length=1000, has_foundation=False, m=0),
    Segment(length=6000, has_foundation=True, m=0),
]

scenario_config = ScenarioConfig(phi=30.0, system_type="skier", crack_length=2000)
weak_layer = WeakLayer(rho=10, h=30, E=0.25, G_Ic=1)  # Default weak layer properties
criteria_config = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)
config = Config()  # Use default configuration

model_input = ModelInput(
    scenario_config=scenario_config,
    weak_layer=weak_layer,
    layers=layers,
    segments=segments,
    criteria_config=criteria_config,
)

new_system = SystemModel(config=config, model_input=model_input)
new_constants = new_system.unknown_constants
print(new_system.scenario.crack_h)
print(new_system.scenario.phi)

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
