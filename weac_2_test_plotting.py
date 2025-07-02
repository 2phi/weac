from weac_2.components import (
    Layer,
    WeakLayer,
    Segment,
    CriteriaConfig,
    ModelInput,
    ScenarioConfig,
)
from weac_2.core import SystemModel, Scenario, Slab
from weac_2.analysis import (
    CriteriaEvaluator,
    Plotter,
    CoupledCriterionResult,
    CoupledCriterionHistory,
)


layers = [
    Layer(rho=350, h=120),
    Layer(rho=270, h=120),
    Layer(rho=180, h=120),
]
scenario_config = ScenarioConfig(
    system_type="skier",
    phi=-35,
)
segments = [
    Segment(length=180000, has_foundation=True, m=0),
    Segment(length=0, has_foundation=False, m=75),
    Segment(length=0, has_foundation=False, m=0),
    Segment(length=180000, has_foundation=False, m=0),
]
weak_layer = WeakLayer(
    rho=125,
    h=30,
    E=1,
)
criteria_config = CriteriaConfig(
    stress_envelope_method="adam_unpublished",
    scaling_factor=125 / 250,
    order_of_magnitude=3,
)
model_input = ModelInput(
    scenario_config=scenario_config,
    layers=layers,
    segments=segments,
    weak_layer=weak_layer,
    criteria_config=criteria_config,
)

system = SystemModel(model_input=model_input)
criteria_evaluator = CriteriaEvaluator(criteria_config=criteria_config)
results: CoupledCriterionResult = criteria_evaluator.evaluate_coupled_criterion(system)


print("Algorithm convergence:", results.converged)
print("Message:", results.message)
print("Critical skier weight:", results.critical_skier_weight)
print("Crack length:", results.crack_length)
print("G delta:", results.g_delta)
print("Iterations:", results.iterations)
print("dist_ERR_envelope:", results.dist_ERR_envelope)
print("History:", results.history.incr_energies[-1])

system = results.final_system
g_delta, propagation_status = criteria_evaluator.check_crack_self_propagation(system)
print("Results of crack propagation criterion: ", propagation_status)
print("G delta: ", g_delta)

print("   - Generating stress envelope...")
plotter = Plotter()
fig1 = plotter.plot_stress_envelope(
    system_model=system,
    criteria_evaluator=criteria_evaluator,
    all_envelopes=False,
    filename="stress_envelope",
)

print("   - Generating fracture toughness envelope...")
plotter = Plotter()
fig2 = plotter.plot_err_envelope(
    system_model=system,
    criteria_evaluator=criteria_evaluator,
    filename="err_envelope",
)

fig1.savefig("stress_envelope.png")
fig2.savefig("err_envelope.png")
