# Slab tensile comparison failures

Source: `tests/analysis/test_slab_tensile_comparisons.py`  
Expectation for every case: `A.slab_tensile_criterion ≥ B.slab_tensile_criterion`.

Evaluated under four tensile-strength methods (`sigrist`, `adam`, `jamieson_johnson`, `hybrid`) in two regimes:

1. **B-contact** — current `evaluate_SteadyState(..., mode="B_point_contact")` (flat, per-setup touchdown length).
2. **Fixed length** — same cracked geometry for A and B (`L ∈ {800, 1200, 1600, 2000}` mm), at `φ = 0°` and `φ = 35°`, no touchdown.

---

## Failed cases (union)

| Case | A (cm, kg/m³) | B (cm, kg/m³) | Intent |
|---|---|---|---|
| **6** | (50, 125) | (50, 175) | Same height; lower density more tensile-prone |
| **7** | (50, 175) | (50, 275) | Same height; lower density more tensile-prone |
| **10** | (20, 125)+(30, 275) | (50, 275) | Soft overburden on dense base vs uniform dense |
| **12** | (20, 175)+(30, 75) | (50, 75) | Dense crust on soft vs all soft |
| **13** | (20, 275)+(30, 75) | (50, 75) | Dense crust on soft vs all soft |
| **14** | (20, 175)+(30, 125) | (50, 125) | Dense crust on mid vs uniform mid |
| **15** | (20, 275)+(30, 125) | (50, 125) | Dense crust on mid vs uniform mid |
| **18** | (30, 75)+(20, 275) | (50, 75)+(20, 275) | Thinner soft top more tensile-prone |
| **19** | (30, 125)+(20, 225) | (50, 125)+(20, 225) | Thinner mid top more tensile-prone |
| **20** | (30, 125)+(20, 275) | (50, 125)+(20, 275) | Thinner mid top more tensile-prone |
| **22** | (40, 75)+(15, 275) | (15, 275)+(40, 75) | Soft over dense vs dense over soft |

Cases **1–5, 8–9, 11, 16–17, 21, 23** never failed in this matrix.

---

## Where each case fails

| Case | B-contact fails | Fixed-L fails (any L / φ) | Rescued by fixed L? |
|---|---|---|---|
| 6 | Sigrist, JJ, Hybrid | — | Yes |
| 7 | Sigrist, JJ | — | Yes |
| 10 | Sigrist, Hybrid | — | Yes |
| 12 | All four methods | All four methods | No |
| 13 | All four methods | All four methods | No |
| 14 | Sigrist, Hybrid | All four methods | No (worse under fixed L for Adam/JJ) |
| 15 | Sigrist, Hybrid | All four methods | No (worse under fixed L for Adam/JJ) |
| 18 | — | All four methods (long L) | — (fixed-L only) |
| 19 | — | All four methods (long L) | — (fixed-L only) |
| 20 | — | All four methods (long L) | — (fixed-L only) |
| 22 | — | Sigrist, JJ (some L / φ) | — (fixed-L only) |

### B-contact fail counts

| Method | Fails | Cases |
|---|---|---|
| Sigrist | 7 | 6, 7, 10, 12, 13, 14, 15 |
| Adam | 2 | 12, 13 |
| Jamieson–Johnson | 4 | 6, 7, 12, 13 |
| Hybrid | 6 | 6, 10, 12, 13, 14, 15 |

### Fixed-L fail counts (union over L and φ)

| Method | Fails | Cases |
|---|---|---|
| Sigrist | 8 | 12, 13, 14, 15, 18, 19, 20, 22 |
| Adam | 7 | 12, 13, 14, 15, 18, 19, 20 |
| Jamieson–Johnson | 8 | 12, 13, 14, 15, 18, 19, 20, 22 |
| Hybrid | 7 | 12, 13, 14, 15, 18, 19, 20 |

---

## One explanation — B-contact / strength method

Under B-contact, each setup gets its **own** unsupported length from touchdown. Denser / stiffer slabs touch down later, so they develop a longer free span and larger absolute `Sxx`. Strength laws with a moderate density exponent (Sigrist, Jamieson–Johnson, Hybrid below 250 kg/m³) do not rise as fast as that stress growth, so same-height lower-density slabs (6, 7) and some soft-overburden rankings (10) can look *less* tensile-prone than denser counterparts — an artifact of **comparing different geometries**, not of σ/σ_t alone. Adam’s steeper strength law often keeps up with that stress growth, which is why it mostly survives B-contact except on the dense-crust-on-soft cases (12, 13).

---

## One explanation — fixed cut length

At fixed `L` (flat or 35°), the density-at-same-height reversals disappear: absolute stress still rises with density, but slower than σ_t, so 6 / 7 / 10 pass for every method. What remains are cases where the **expected ordering fights the criterion definition**. In 12–15, setup B is a soft (or mid-density) uniform slab that saturates as fully tensile-prone (`criterion → 1`), while A’s dense crust is strong, stays below threshold, and — with low-density exclusion — **reduces** the tensile-prone load-bearing fraction, so A < B by construction. Cases 18–20 (and sometimes 22) are milder: at long fixed spans the thicker soft/mid overburden in B can drive a higher prone fraction than A’s thinner top, so the “thinner top should rank higher” intuition fails even without touchdown. Slope angle does not remove this cluster; it only shifts a few L-sensitive borders.
