'''
This script demonstrates the basic usage of the WEAC package to run a simulation.
'''
from weac_2.logging_config import setup_logging
from weac_2.components import ModelInput, Layer, Segment, CriteriaConfig, WeakLayer, ScenarioConfig
from weac_2.components.config import Config
from weac_2.core.system_model import SystemModel
from weac_2.analysis.analyzer import Analyzer
from weac_2.analysis.plotter import Plotter
from weac_2.analysis.criteria_evaluator import CriteriaEvaluator
import numpy as np
import logging

setup_logging()

# Suppress matplotlib debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# === SYSTEM 1: Basic Configuration ===
config1 = Config(touchdown=True, youngs_modulus_method='bergfeld', stress_failure_envelope_method='adam_unpublished')
scenario_config1 = ScenarioConfig(phi=5, system_type='skier')  # Steeper slope
criteria_config1 = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)

weak_layer1 = WeakLayer(rho=10, h=25, E=0.25, G_Ic=1)
layers1 = [
    Layer(rho=170, h=100), # Top Layer
    Layer(rho=280, h=100), # Bottom Layer
]
segments1 = [
    Segment(l=3000, has_foundation=True, m=70),
    Segment(l=4000, has_foundation=True, m=0)
]

model_input1 = ModelInput(
    scenario_config=scenario_config1, 
    weak_layer=weak_layer1, 
    layers=layers1, 
    segments=segments1, 
    criteria_config=criteria_config1
)

system1 = SystemModel(config=config1, model_input=model_input1)

# === SYSTEM 2: Different Slope Angle ===
config2 = Config(touchdown=False, youngs_modulus_method='bergfeld', stress_failure_envelope_method='adam_unpublished')
scenario_config2 = ScenarioConfig(phi=30, system_type='skier')  # Steeper slope
weak_layer2 = WeakLayer(rho=10, h=25, E=0.25, G_Ic=1)
layers2 = [
    Layer(rho=170, h=100), # Top Layer
    Layer(rho=280, h=100), # Bottom Layer
]
segments2 = [
    Segment(l=3000, has_foundation=True, m=70),
    Segment(l=4000, has_foundation=True, m=0)
]
criteria_config2 = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)

model_input2 = ModelInput(
    scenario_config=scenario_config2, 
    weak_layer=weak_layer2, 
    layers=layers2, 
    segments=segments2, 
    criteria_config=criteria_config2
)

system2 = SystemModel(config=config2, model_input=model_input2)

# === SYSTEM 3: Different Layer Configuration ===
config3 = Config(touchdown=False, youngs_modulus_method='bergfeld', stress_failure_envelope_method='adam_unpublished')
scenario_config3 = ScenarioConfig(phi=15, system_type='skier')  # Medium slope
weak_layer3 = WeakLayer(rho=15, h=25, E=0.3, G_Ic=1.2)  # Different weak layer
layers3 = [
    Layer(rho=150, h=80),  # Lighter top layer
    Layer(rho=200, h=60),  # Medium layer
    Layer(rho=320, h=120), # Heavier bottom layer
]
segments3 = [
    Segment(l=3500, has_foundation=True, m=60),  # Different skier mass
    Segment(l=3500, has_foundation=True, m=0)
]
criteria_config3 = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)

model_input3 = ModelInput(
    scenario_config=scenario_config3, 
    weak_layer=weak_layer3, 
    layers=layers3, 
    segments=segments3, 
    criteria_config=criteria_config3
)

system3 = SystemModel(config=config3, model_input=model_input3)

# === SYSTEM 4: Advanced Configuration ===
config4 = Config(touchdown=False, youngs_modulus_method='bergfeld', stress_failure_envelope_method='adam_unpublished')
scenario_config4 = ScenarioConfig(phi=38, system_type='skier')
weak_layer4 = WeakLayer(rho=10, h=25, E=0.25, G_Ic=1)
layers4 = [
    Layer(rho=170, h=100), # (1) Top Layer
    Layer(rho=190, h=40),  # (2)
    Layer(rho=230, h=130),
    Layer(rho=250, h=20),
    Layer(rho=210, h=70),
    Layer(rho=380, h=20),  
    Layer(rho=280, h=100), # (N) Bottom Layer
]
segments4 = [
    Segment(l=5000, has_foundation=True, m=80),
    Segment(l=3000, has_foundation=True, m=0),
    Segment(l=3000, has_foundation=False, m=0),
    Segment(l=4000, has_foundation=True, m=70),
    Segment(l=3000, has_foundation=True, m=0)
]
criteria_config4 = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)
model_input4 = ModelInput(
    scenario_config=scenario_config4, 
    weak_layer=weak_layer4, 
    layers=layers4, 
    segments=segments4, 
    criteria_config=criteria_config4
)

system4 = SystemModel(config=config4, model_input=model_input4)

# === DEMONSTRATION OF PLOTTING CAPABILITIES ===

print("=== WEAC Plotting Demonstration ===")

# Single system plotting
print("\n1. Single System Analysis:")
print(f"   System 1 - φ={system1.scenario.phi}°, H={system1.slab.H}mm")

plotter_single = Plotter(system1, labels=["φ=5° System"])

# Generate individual plots
print("   - Generating slab profile...")
plotter_single.plot_slab_profile(filename='single_slab_profile')

print("   - Generating displacement plot...")
plotter_single.plot_displacements(filename='single_displacements')

print("   - Generating section forces plot...")
plotter_single.plot_section_forces(filename='single_section_forces')

print("   - Generating stress plot...")
plotter_single.plot_stresses(filename='single_stresses')

print("   - Generating deformed contour plot...")
plotter_single.plot_deformed(field='w', filename='single_deformed_w')
plotter_single.plot_deformed(field='principal', filename='single_deformed_principal')

print("   - Generating stress envelope...")
plotter_single.plot_stress_envelope(filename='single_stress_envelope')

# # Multi-system comparison
# print("\n2. Multi-System Comparison:")
# print(f"   System 1: φ={system1.scenario.phi}°, H={system1.slab.H}mm")
# print(f"   System 2: φ={system2.scenario.phi}°, H={system2.slab.H}mm") 
# print(f"   System 3: φ={system3.scenario.phi}°, H={system3.slab.H}mm")

# plotter_multi = Plotter(
#     systems=[system1, system2, system3],
#     labels=[f"φ={system1.scenario.phi}° (Light)", f"φ={system2.scenario.phi}° (Steep)", f"φ={system3.scenario.phi}° (Multi-layer)"],
#     colors=['#5D85C3', '#E6001A', '#009D81']  # Blue, Red, Teal
# )

# print("   - Generating comparison plots...")
# plotter_multi.plot_slab_profile(filename='comparison_slab_profiles')
# plotter_multi.plot_displacements(filename='comparison_displacements')
# plotter_multi.plot_section_forces(filename='comparison_section_forces')
# plotter_multi.plot_stresses(filename='comparison_stresses')
# plotter_multi.plot_energy_release_rates(filename='comparison_energy_release_rates')

# print("   - Generating comprehensive dashboard...")
# plotter_multi.create_comparison_dashboard(filename='comparison_dashboard')

# # Demonstrate system override functionality
# print("\n3. System Override Examples:")
# print("   - Plotting only systems 1 and 3 for displacement comparison...")
# plotter_multi.plot_displacements(
#     system_models=[system1, system3], 
#     filename='override_displacements_1_3'
# )

# print("   - Plotting system 2 deformed shape...")
# plotter_multi.plot_deformed(
#     system_model=system2, 
#     field='principal', 
#     filename='override_deformed_system2'
# )

# # Print system information
# print("\n=== System Information ===")
# for i, system in enumerate([system1, system2, system3], 1):
#     print(f"\nSystem {i}:")
#     print(f"  Slope angle: {system.scenario.phi}°")
#     print(f"  Total slab thickness: {system.slab.H} mm")
#     print(f"  Number of layers: {len(system.slab.layers)}")
#     print(f"  Weak layer thickness: {system.weak_layer.h} mm")
#     print(f"  Weak layer density: {system.weak_layer.rho} kg/m³")
    
#     # Calculate some basic results
#     analyzer = Analyzer(system=system)
#     x, z, _ = analyzer.rasterize_solution()
#     fq = system.fq
    
#     max_deflection = np.max(np.abs(fq.w(z)))
#     max_stress = np.max(np.abs(fq.tau(z, unit='kPa')))
    
#     print(f"  Max vertical deflection: {max_deflection:.3f} mm")
#     print(f"  Max shear stress: {max_stress:.3f} kPa")

# print("\n=== Plotting Complete ===")
# print("Check the 'plots/' directory for generated visualizations.")
# print("\nPlot files generated:")
# print("  Single system: single_*.png")
# print("  Comparisons: comparison_*.png") 
# print("  Overrides: override_*.png")
# print("  Dashboard: comparison_dashboard.png")
