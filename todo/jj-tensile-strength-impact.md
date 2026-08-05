# Impact of Jamieson–Johnson as default tensile strength

Changing the default `Layer.tensile_strength_method` from `hybrid` to `jamieson_johnson` does **not** change the mechanical stress solve. It changes the **strength denominator** used when stresses are normalized. That alone explains the jump in Max Sxx norm and the shift in slab tensile criterion / ordering.

![Tensile strength laws vs density](dev/tensile_strength_laws.png)

Over typical slab densities, Jamieson–Johnson (green) sits **below** Sigrist and Adam. Hybrid used Sigrist below 250 kg/m³ and Adam above, so the old default was almost always a stronger denominator than JJ. Weaker strength → larger normalized stress → more height levels counted as tensile-prone.

Relevant default: `Layer.tensile_strength_method` in `src/weac/components/layer.py`.

---

## 1. Why Max Sxx norm is higher

**Pipeline**

1. Solve the cracked system and rasterize the solution.
2. Compute absolute axial stress `Sxx` from strain, modulus, and self-weight.
3. Optionally normalize: each height row is divided by that layer’s tensile strength.
4. `max_Sxx_norm` is simply the maximum of that normalized field.

Absolute `Sxx` (kPa) depends on `E`, `ν`, geometry, and loading — **not** on the tensile-strength law. The law only enters when `normalize=True`.

```text
Sxx_norm = Sxx / tensile_strength(layer)
max_Sxx_norm = max(Sxx_norm)
```

With JJ as default, the same absolute stress is divided by a smaller strength, so the dimensionless field scales up. Regression cases that asserted Max Sxx norm therefore fail in the “norm too high” direction, while touchdown distance and energy release rate (which do not use slab tensile strength) stay put.

**Code path**

- Absolute vs normalized `Sxx`: `Analyzer.Sxx` in `src/weac/analysis/analyzer.py` (normalize branch divides by `zmesh["tensile_strength"]`).
- Taking the max: `CriteriaEvaluator._calculate_maximal_stresses` in `src/weac/analysis/criteria_evaluator.py` (`max_Sxx_norm = np.max(Sxx_norm)`).

Same story for normalized principal stress: identical absolute field, weaker denominator.

---

## 2. How that feeds the slab tensile criterion

The slab tensile criterion is **not** Max Sxx norm. It is a **thickness fraction**: among load-bearing height levels, what share is “prone to fail.”

Rough sequence in `_calculate_maximal_stresses`:

1. Build `Sxx_norm` as above.
2. Per height level: `tensile_exceeds = max(Sxx_norm along that row) > 1`.
3. Mark low-density levels (`ρ ≤ low_density_threshold`, default 100 kg/m³) specially: they only count as prone if everything above them is already prone (downward growth from failures above). When they are prone, they are **excluded from the denominator**.
4. Criterion = mean of “prone” over the remaining **load-bearing** levels (or 1.0 if none remain).

Trend under JJ:

- More levels cross `Sxx_norm > 1` because the threshold is the same (1) but the normalized values are larger.
- That can increase the numerator (more prone load-bearing levels).
- It can also change which low-density stacks are treated as failed-from-above and dropped from the denominator — so the criterion can move a lot even when the ranking of setups is subtle.

So Max Sxx norm rising is the direct, local effect of a weaker law; the criterion is the **downstream, thresholded, geometry-aware** effect of that same scaling.

---

## 3. case_3 ordering (qualitative)

`tests/analysis/test_slab_tensile_comparisons.py` asserts setup A’s criterion ≥ setup B’s.

**case_3**

| | Soft top (low density) | Base |
|---|---|---|
| **A** | thicker, ρ = 75 | ρ = 125 |
| **B** | thinner, ρ = 75 | same ρ = 125 |

Both tops are below the low-density threshold; both bases are load-bearing. A and B differ only by how much soft snow sits above the same denser layer.

Under the old hybrid default (Sigrist for these densities), A stayed above B: more soft overburden tended to produce a higher (or equal) tensile-prone fraction on the load-bearing part.

Under JJ:

- Both layers get a weaker strength, so `Sxx_norm` rises in soft top and base.
- The soft top is still governed by the low-density rule: it does not simply add 1:1 into the criterion percentage; it mainly modulates whether failure is considered to propagate downward and whether those levels are dropped from the denominator.
- With a lower strength scale, the **base** of B can cross (or stay over) the `> 1` line over a larger share of its thickness, while A’s thicker soft stack changes how load-bearing levels are counted after low-density exclusion.
- Net effect: B’s criterion can overtake A’s, so the assumed ordering `A ≥ B` flips even though the structural contrast (thicker vs thinner soft top) is unchanged.

In short: case_3 is sensitive because it sits on the **low-density / load-bearing boundary**. JJ does not invent a new comparison rule; it moves who clears `Sxx_norm > 1` and how the exclusion logic weights A vs B.

**Code path**

- Comparison setups: `COMPARISON_CASES` / `case_3` in `tests/analysis/test_slab_tensile_comparisons.py`.
- Criterion definition: low-density loop and `slab_tensile_criterion = mean(...)` in `CriteriaEvaluator._calculate_maximal_stresses`.

---

## Takeaway

| Quantity | Depends on tensile law? | Trend under JJ |
|---|---|---|
| Absolute `Sxx` | No | Unchanged |
| `Sxx_norm` / Max Sxx norm | Yes (denominator) | Higher |
| Slab tensile criterion | Yes (via `> 1` and low-density filtering) | More levels prone; ordering can change |
| case_3 A ≥ B | Yes | Can reverse when soft-top thickness interacts with the threshold |

The plot’s message is the whole story at root: Jamieson–Johnson is a systematically weaker strength law for most densities, and every normalized tensile metric in this pipeline inherits that.
