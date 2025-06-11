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

setup_logging()

# config = Config(density_method='adam_unpublished', stress_failure_envelope_method='adam_unpublished')
# scenario_config = ScenarioConfig(phi=38, touchdown=True, system='skiers')
# weak_layer = WeakLayer(rho=10, h=1000, E=0.25, G_Ic=1)
# layers = [
#     Layer(rho=170, h=100), # (1) Top Layer
#     Layer(rho=190, h=40),  # (2)
#     Layer(rho=230, h=130),
#     Layer(rho=250, h=20),
#     Layer(rho=210, h=70),
#     Layer(rho=380, h=20),  
#     Layer(rho=280, h=100), # (N) Bottom Layer
# ]
# segments = [
#     Segment(l=5000, k=True, m=80),
#     Segment(l=3000, k=False, m=0),
#     Segment(l=4000, k=True, m=70),
#     Segment(l=3000, k=True, m=0)
# ]
# criteria_config = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)

config = Config(youngs_modulus_method='bergfeld', stress_failure_envelope_method='adam_unpublished')
scenario_config = ScenarioConfig(phi=5, touchdown=True, system='skier')
weak_layer = WeakLayer(rho=10, h=30, E=0.25, G_Ic=1)
layers = [
    Layer(rho=170, h=100), # (1) Top Layer
    Layer(rho=280, h=100), # (N) Bottom Layer
]
segments = [
    Segment(l=3000, k=True, m=70),
    Segment(l=4000, k=True, m=0)
]
criteria_config = CriteriaConfig(fn=1, fm=1, gn=1, gm=1)

model_input = ModelInput(scenario_config=scenario_config, weak_layer=weak_layer, layers=layers, segments=segments, criteria_config=criteria_config)

system = SystemModel(config=config, model_input=model_input)
unknown_constants = system.unknown_constants
print(unknown_constants)

system.update_scenario(phi=20.0)
unknown_constants = system.unknown_constants
print(unknown_constants)

Analyzer(system=system)
plotter = Plotter(system=system)
CriteriaEvaluator(system=system, criteria_config=criteria_config)
