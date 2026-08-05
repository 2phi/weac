# Alternative steady-state study (snapshot)

Research snapshot of PST-style proxies for slab tensile capping, compared against
Steph’s A/B intuition on `COMPARISON_CASES`. This tree (plus
`src/weac/analysis/experimental/` and the notes under `todo/`) is tracked on
`feat/alternative-steady-state` so the runners and scored results stay findable.

**Yardstick (tensile ease):** cases **1–11 & 21–23** → A easier than B; cases
**12–15** → B easier than A. Cases **16–20** are noisy for A/B judgment.

---

## What changed (code)

| Piece | Role |
|-------|------|
| `src/weac/analysis/experimental/` | Quarantined package: four PST evaluators + shared util (baseline, ease selection, A/B+UD compare, plots) |
| `tests/analysis/test_experimental_util_ease_ud_compare.py` | Unit tests for ease selection and UD table/plot writers |
| This folder | One `run.py` per approach, shared batch scripts, frozen `results.json` / tables / plots |
| `todo/alternative_steady_state_approaches.md` | Draft write-up of the three main proxies (EN/DE) |
| `todo/slab-tensile-comparison-*.md` | Scored A/B tables vs Steph (tensile law + φ=15°) |
| `todo/jj-tensile-strength-impact.md` | Why JJ as default tensile law shifts normalized stress / criterion |

Shared setup across approaches: PST geometry, `touchdown=False`, both orientations
evaluated, production side chosen by **ease**, crack onset when
`max(S_xx / σ_tensile) ≥ 1`. ERR is reported at that configuration as a secondary metric.

| Approach | Ease metric | Easier means |
|----------|-------------|--------------|
| `pst_fixed_cut` | `max_Sxx_norm` (tables also use thickness fraction) | higher |
| `pst_critical_mass` | `critical_mass_kg` at L = 100 mm | lower |
| `pst_critical_cut` | `critical_cut_length` / `L_crit` | lower |
| `pst_touchdown_cut` | `thickness_fraction_without_density_gate` at tip touchdown cut | higher |

---

## Key results (frozen in this tree)

Defaults unless noted: **φ = 35°**, tensile law **Jamieson–Johnson (JJ)**.
Marks are vs Steph (✅ match / ❌ miss). Full case tables:
[`todo/slab-tensile-comparison-results.md`](../../todo/slab-tensile-comparison-results.md),
[`todo/slab-tensile-comparison-phi15.md`](../../todo/slab-tensile-comparison-phi15.md).

### Tensile ease vs Steph (✅ / 23)

| Condition | `pst_critical_cut` | `pst_critical_mass` | `pst_fixed_cut` | `pst_touchdown_cut` |
|-----------|:------------------:|:-------------------:|:---------------:|:-------------------:|
| JJ @ 35° | 21 | 21 | 21 | 15 |
| JJ @ 15° | 22 | 22 | 21 | 16 |
| hybrid @ 35° | 21 | 22 | 13 | 13 |
| adam @ 35° | 20 | 21 | 14 | 21 |

### ERR vs Steph (✅ / 23)

| Condition | `pst_critical_cut` | `pst_critical_mass` | `pst_fixed_cut` | `pst_touchdown_cut` |
|-----------|:------------------:|:-------------------:|:---------------:|:-------------------:|
| JJ @ 35° | 15 | 14 | 11 | **23** |
| JJ @ 15° | 15 | 16 | 9 | **23** |
| hybrid @ 35° | 15 | 16 | 11 | **23** |
| adam @ 35° | 16 | 17 | 11 | **23** |

### Takeaways

- **Tensile A/B:** `pst_critical_cut`, `pst_critical_mass`, and `pst_fixed_cut` (JJ)
  largely recover Steph’s yardstick (~21/23). Dropping the ρ ≤ 100 kg/m³ overwrite
  is viable for those proxies.
- **`pst_touchdown_cut`:** perfect **ERR** agreement with Steph (23/23) across laws/φ,
  but weaker on tensile ease under JJ (~15–16/23).
- **Tensile law:** switching default to JJ weakens the strength denominator (higher
  `Sxx_norm`); hybrid/adam change some ease tallies, especially for fixed-cut and
  touchdown-cut. See `todo/jj-tensile-strength-impact.md`.
- **Inclination:** φ = 15° vs 35° A/B sides mostly agree (tensile ~22–23/23,
  ERR ~21–23/23 depending on approach). Monotonic φ-sweep (0→45°) is mixed for
  ease/ERR on the probed cases — see `phi_sweep/phi_sweep.md`.
- **No production pick yet** — package stays experimental; `pst_fixed_cut` was the
  simplest tensile proxy in the draft note, while touchdown-cut is the ERR standout.

### Artifact layout in this folder

| Path | Contents |
|------|----------|
| `baseline_ss/`, `pst_*/` | Default JJ @ 35° runners + results/plots |
| `phi_15/` | Same approaches re-run at φ = 15° |
| `ts_hybrid/`, `ts_adam/` | Tensile-law variants @ 35° |
| `phi_sweep/` | Inclination sweep tables (`phi_sweep.py`) |
| `run_at_phi.py`, `run_by_tensile_law.py` | Batch reproducers |
| `score_vs_steph.py`, `score_phi_vs_35.py` | Rebuild scored markdown tables |

---

## How to reproduce

Each method has its own folder with `run.py` and artifacts. Shared helpers live under
`weac.analysis.experimental.util`.

**1. Baseline first** (approaches hard-fail if this snapshot is missing):

```bash
uv run python dev/alt_steady_state/baseline_ss/run.py
```

**2. Then any approach:**

```bash
uv run python dev/alt_steady_state/pst_fixed_cut/run.py
uv run python dev/alt_steady_state/pst_critical_mass/run.py
uv run python dev/alt_steady_state/pst_critical_cut/run.py
uv run python dev/alt_steady_state/pst_touchdown_cut/run.py
```

Optional flags:

```bash
uv run python dev/alt_steady_state/pst_fixed_cut/run.py --cases case_1 case_5
uv run python dev/alt_steady_state/pst_fixed_cut/run.py --skip-plots
```

| Folder | Role |
|--------|------|
| `baseline_ss/` | Production `B_point_contact` snapshot |
| `pst_fixed_cut/` | Fixed 50 cm cut |
| `pst_critical_mass/` | Critical right-edge end mass |
| `pst_critical_cut/` | Critical cut length to first tensile crack |
| `pst_touchdown_cut/` | Cut where free tip touches (`w_tip = crack_h`) |

### Ease-based orientation selection

Both upslope and downslope are evaluated; production fields come from the ease
winner. Diagnostics always retain `winner`, `err_winner`, and
`selection_rule` (`ease:<ease_key>`). Ties / unusable sides fall back to ERR, then
upslope (`weac.analysis.experimental.util.ease.select_ease_orientation`).

### Per-approach artifacts

| File | Contents |
|------|----------|
| `results.json` | Nested payloads (case → setup) |
| `comparison_ab.csv` / `.md` | Paired A/B judgment table |
| `comparison_ud.csv` / `.md` | Upslope vs downslope (`ease_winner`, `err_winner`, Δ) |
| `plots/*.png` | vs baseline, A/B, and UD grouped bars |

`baseline_ss/` writes only `results.json`. Failed cells record an error; runners do
not abort the matrix.

| Approach | `ud_*` plots |
|----------|--------------|
| `pst_fixed_cut` | `ud_ERR`, `ud_max_Sxx_norm`, `ud_thickness_fraction` |
| `pst_critical_mass` | `ud_ERR`, `ud_critical_mass` |
| `pst_critical_cut` | `ud_ERR`, `ud_L_crit` |
| `pst_touchdown_cut` | `ud_ERR`, `ud_thickness_fraction`, `ud_cut_length` |
