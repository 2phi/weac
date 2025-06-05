'''
This script demonstrates the basic usage of the WEAC package to run a simulation.
'''
from weac_2.logging_config import setup_logging
from weac_2.components import ModelInput, Layer, Segment, CriteriaOverrides, WeakLayer, Scenario
from weac_2.components.config import Config
from weac_2.core.system_model import SystemModel

setup_logging()

config = Config(density_method='adam_unpublished', stress_failure_envelope_method='adam_unpublished')
scenario = Scenario(phi=38, touchdown=True, system='skiers')
weak_layer = WeakLayer(rho=10, h=1000, E=0.25, G_Ic=1)
layers = [
    Layer(rho=170, h=100), # (1) Top Layer
    Layer(rho=190, h=40),  # (2)
    Layer(rho=230, h=130),
    Layer(rho=250, h=20),
    Layer(rho=210, h=70),
    Layer(rho=380, h=20),  
    Layer(rho=280, h=100), # (N) Bottom Layer
]
segments = [
    Segment(length=5000, fractured=True, skier_weight=80, surface_load=0),
    Segment(length=3000, fractured=False, skier_weight=0, surface_load=0),
    Segment(length=4000, fractured=True, skier_weight=70, surface_load=0),
    Segment(length=3000, fractured=True, skier_weight=0, surface_load=0)
]
criteria_overrides = CriteriaOverrides(fn=1, fm=1, gn=1, gm=1)

model_input = ModelInput(scenario=scenario, weak_layer=weak_layer, layers=layers, segments=segments, criteria_overrides=criteria_overrides)

system = SystemModel(config=config, model_input=model_input)
