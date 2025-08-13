# TODOs

## Major

- [ ] Use Classes for Boundary Types
- [ ] Automatically figure out type of system
- [ ] Automatically set boundary conditions based on system

## Minor

- [ ] resolve fracture criterion also when lower than strength criterion
- [ ] Florian CriterionEvaluator: clarify and fix dampening behavior (find_minimum_force / evaluate_coupled_criterion)
  - Expected behavior
    - find_minimum_force: compute the critical skier weight w* [kg] such that max(stress_envelope) == 1 within tolerance_stress. This solver should not apply dampening; it must return the numerically precise root of residual(weight) = max(stress_envelope) - 1 using a bracketed method and finite tolerances.
    - evaluate_coupled_criterion: iterate on skier_weight and crack_length to satisfy both stress and fracture toughness criteria (g_delta ≈ 1). Apply a dampening factor only to the weight update to avoid oscillations near the ERR envelope; dampening must not alter the physical evaluations (sigma, tau, G_I, G_II).
  - Algorithm
    - Names/units: `skier_weight` [kg] ≥ 0; `g_delta` [-]; `dist_ERR_envelope` = |g_delta - 1| [-]; `tolerance_ERR` ∈ [1e-4, 5e-2]; `tolerance_stress` ∈ [1e-4, 5e-3]; `dampening_ERR` ∈ [0, 5].
    - Clamp inputs: clamp `skier_weight` to [0, W_MAX]; clamp `dampening_ERR` to [0, 5]; if any intermediate is non-finite (NaN/inf), abort with a clear failure message.
    - Maintain a weight bracket [w_min, w_max] around the ERR envelope crossing: set w_min if g_delta < 1, w_max if g_delta ≥ 1; compute mid = 0.5 · (w_min + w_max).
    - Dampened update step (weight only):
      - λ = 1 / (1 + dampening_ERR)
      - new_weight = skier_weight + λ · (mid - skier_weight)
      - Interpretation: dampening_ERR=0 → pure bisection step (λ=1); dampening_ERR=1 → half-step (λ=0.5); larger dampening slows updates and reduces oscillations.
    - After updating `new_weight`, recompute crack length via `find_crack_length_for_weight(system, new_weight)`.
    - Stop when `dist_ERR_envelope ≤ tolerance_ERR` or `max_iterations` reached. With dampening_ERR=0 the behavior should match undampened bisection; with dampening_ERR>0 the path changes but the converged weight is the same within tolerance.
  - Failure modes to handle
    - Negative/zero weights: never propose negative weights; allow zero only when self-collapse is detected.
    - Divergence/oscillation: dampening reduces step size near convergence; ensure [w_min, w_max] shrinks monotonically.
    - Coupled scaling: dampening only scales the update step; do not alter the evaluation of stresses or ERRs.
    - Idempotence: same inputs produce the same final result; dampening may change iterations, not the target value (within tolerance).
    - Non-finite numbers: detect and fail fast with an informative message.
    - Entire domain cracked: keep the existing short-circuit to self-collapse.
  - Parameters and expected ranges
    - `dampening_ERR`: float in [0, 5], default 0.0. Recommended 0–2 for stability without excessive slowdown.
    - `tolerance_ERR`: float in [1e-4, 5e-2], default 2e-3.
    - `tolerance_stress`: float in [1e-4, 5e-3], default 5e-3.
    - `max_iterations`: int in [10, 200], default 25.
    - `W_MAX`: safety cap for weight search, default 2000 kg.
  - Formulae (document in docstrings)
    - dist_ERR_envelope = |g_delta - 1|
    - λ = 1 / (1 + dampening_ERR)
    - new_weight = skier_weight + λ · (mid - skier_weight)
    - Units: weights in kg, stresses in kPa, ERR in J/m^2, lengths in mm.
  - Unit tests to add (demonstrate intended outcomes)
    1) Independent criterion (pure stress governed; idempotent with dampening)
       - Setup: create a stable weak layer where fracture toughness is not limiting at the critical stress weight. Compute w0 via `find_minimum_force`. Run `evaluate_coupled_criterion` twice with `dampening_ERR=0.0` and `dampening_ERR=3.0` on fresh copies of the same system.
       - Expect:
         - `pure_stress_criteria == True`
         - Returned `critical_skier_weight ≈ w0` (within 1%) for both runs
         - All `history.skier_weights` ≥ 0; no negative or NaN values
       - Example:

          ```python
          def test_dampening_idempotent_under_pure_stress():
              config = Config()
              criteria = CriteriaConfig()
              evaluator = CriteriaEvaluator(criteria)
              layers = [Layer(rho=170, h=100), Layer(rho=230, h=130)]
              wl = WeakLayer(rho=180, h=10, G_Ic=5.0, G_IIc=8.0, kn=100, kt=100)  # strong toughness
              seg_len = 10000
              base_segments = [
                  Segment(length=seg_len, has_foundation=True, m=0),
                  Segment(length=0, has_foundation=False, m=0),
                  Segment(length=0, has_foundation=False, m=0),
                  Segment(length=seg_len, has_foundation=True, m=0),
              ]
              def make_system():
                  return SystemModel(
                      model_input=ModelInput(
                          layers=layers, weak_layer=wl, segments=copy.deepcopy(base_segments),
                          scenario_config=ScenarioConfig(phi=30.0)
                      ),
                      config=config,
                  )
              w0 = evaluator.find_minimum_force(system=make_system()).critical_skier_weight
              res0 = evaluator.evaluate_coupled_criterion(system=make_system(), dampening_ERR=0.0)
              res3 = evaluator.evaluate_coupled_criterion(system=make_system(), dampening_ERR=3.0)
              assert res0.pure_stress_criteria and res3.pure_stress_criteria
              assert abs(res0.critical_skier_weight - w0) / w0 < 0.01
              assert abs(res3.critical_skier_weight - w0) / w0 < 0.01
              assert all(w >= 0 for w in res0.history.skier_weights)
              assert all(w >= 0 for w in res3.history.skier_weights)
          ```

    2) Strongly coupled criteria (ERR governed; dampening reduces oscillations, same target)
       - Setup: choose a very weak layer (small G_Ic/G_IIc) so ERR governs. Run `evaluate_coupled_criterion` with `dampening_ERR=0` and with `dampening_ERR=2` on fresh systems and the same tolerances.
       - Expect:
         - Both runs converge with `dist_ERR_envelope ≤ tolerance_ERR`
         - The two `critical_skier_weight` values differ by ≤ 2%
         - The dampened run shows fewer overshoot/flip events (e.g., fewer changes of the w_min/w_max assignment or monotone shrinking bracket) and never proposes negative weight
       - Example:

          ```python
          def test_dampening_stabilizes_coupled_err():
              config = Config()
              criteria = CriteriaConfig()
              evaluator = CriteriaEvaluator(criteria)
              layers = [Layer(rho=170, h=100), Layer(rho=230, h=130)]
              wl = WeakLayer(rho=180, h=10, G_Ic=0.02, G_IIc=0.02, kn=100, kt=100)  # weak toughness
              seg_len = 10000
              segments = [
                  Segment(length=seg_len, has_foundation=True, m=0),
                  Segment(length=0, has_foundation=False, m=0),
                  Segment(length=0, has_foundation=False, m=0),
                  Segment(length=seg_len, has_foundation=True, m=0),
              ]
              def make_system():
                  return SystemModel(
                      model_input=ModelInput(
                          layers=layers, weak_layer=wl, segments=copy.deepcopy(segments),
                          scenario_config=ScenarioConfig(phi=30.0)
                      ),
                      config=config,
                  )
              res_undamped = evaluator.evaluate_coupled_criterion(
                  system=make_system(), dampening_ERR=0.0, tolerance_ERR=0.002
              )
              res_damped = evaluator.evaluate_coupled_criterion(
                  system=make_system(), dampening_ERR=2.0, tolerance_ERR=0.002
              )
              assert res_undamped.converged and res_damped.converged
              assert res_undamped.dist_ERR_envelope <= 0.002
              assert res_damped.dist_ERR_envelope <= 0.002
              w_u = res_undamped.critical_skier_weight
              w_d = res_damped.critical_skier_weight
              assert abs(w_u - w_d) / max(w_u, 1e-9) <= 0.02
              assert all(w >= 0 for w in res_damped.history.skier_weights)
          ```

- [ ] Make rasterize_solution smarter (iterative convergence)
- [ ] SNOWPACK Parser
- [ ] SMP Parser
- [ ] Build Tests: Integration -> Pure

## Patch

- [ ] (Add Patch items as needed)
