# Alternative A — English draft

**To:** Valle, Philipp  
**From:** Yannik  
**Subject:** Update on PST proxies for slab tensile capping in LayerWise

---

Hey Valle, hey Philipp,

short update on the three PST-style proxies we sketched for capping propagation by slab tensile failure. Context from last week: with touchdown mode B, stiffer/denser slabs get longer free lengths, so `S_xx` scales faster than tensile strength and the A/B ordering vs Steph’s intuition goes the wrong way. We also had the ρ ≤ 100 kg/m³ overwrite as a hack.

Steph and I went through the comparison library on the [slab generator](https://beta.whiterisk.ch/weac-slab-generator). Yardstick for tensile ease:

- cases **1–11 & 21–23**: A easier than B  
- cases **12–15**: B easier than A  
- cases **16–20**: dropped for A/B judgment — too many overlapping effects, not transparent enough

All three approaches below drop the ρ ≤ 100 hack. On the yardstick cases they all recover the expected tensile ordering. ERR is a different story (see below).

## Shared setup

For all three:

- PST geometry, **`touchdown=False`** (cut length is what we ask for; slope angle enters the loading directly)
- evaluate **upslope and downslope**, pick a production side by **ease**
- crack onset: first time **`max(S_xx / σ_tensile) ≥ 1`** somewhere in the slab
- report **ERR** at that configuration as a secondary metric

Orientation labeling:

- fixed-cut & critical-cut: upslope = `-pst`, downslope = `pst-`, both at **`+φ₀`**
- critical-mass: end mass can only sit on a segment’s **right** edge, so both sides use **`pst-`**; downslope = `+φ₀`, upslope = **`−φ₀`** (φ-mirror of `-pst` at `+φ₀`)

---

## 1) `pst_fixed_cut`

**Idea / math**  
Fixed free cut **L = 50 cm**, no end mass. Both orientations at `+φ₀`. Ease = higher `max S_xx_norm` (equivalently we look at the **thickness fraction** of height levels with `S_xx/σ_t > 1`). Same continuous tensile metric as today’s LayerWise criterion, just without the density gate and without touchdown-length scaling.

**Results**  
Thickness fraction matches the Steph yardstick (A higher for 1–11 & 21–23; B higher for 12–15). Fast to compute.

ERR at fixed L is harder to trust: e.g. case 1 (little new snow) shows a very high ERR, likely because 50 cm is close to its old touchdown-B length, so it looks more “energetic” than thicker/stiffer layerings where that does not make sense. So for this approach the tensile metric is the useful one; ERR probably needs a different pairing or control values.

**Plots:** `ab_thickness_fraction.png`, `ab_ERR.png`, `ud_max_Sxx_norm.png` (or `ud_thickness_fraction.png`)

---

## 2) `pst_critical_mass`

**Idea / math**  
Fixed short free cut **L = 100 mm**, search the free-tip end mass `m` until first crack (`max S_xx_norm = 1`). Ease = **lower critical mass**. ERR evaluated at that `(L, m)`.

**Results**  
Critical-mass A/B trends match the yardstick (lower mass = easier). After correcting the orientation (both sides `pst-`, φ-mirror for upslope), **upslope is always the easier side** on our matrix. ERR vs ease can still disagree on individual cases.

**Plots:** `ab_critical_mass.png`, `ud_critical_mass.png`

---

## 3) `pst_critical_cut`

**Idea / math**  
No end mass. Search the smallest cut length **L_crit ∈ [1 mm, 5 m]** with `max S_xx_norm = 1` (Brent). Ease = **shorter L_crit**. ERR at that critical cut.

**Results**  
L_crit matches the yardstick (shorter = easier). Same UD pattern as fixed-cut: upslope usually easier, with flips at **12a, 13a, 15a, 21b, 22b**. ERR again not something we can rank confidently yet.

**Plots:** `ab_L_crit.png` (plus ERR plots if useful)

---

## Cross-cutting notes

- **Tensile A/B:** all three approaches agree with Steph on the yardstick; the ρ ≤ 100 overwrite is no longer needed.
- **ERR:** varies a lot across approaches and often does not line up with tensile ease. We probably need intuitive / expert control values before using ERR for ranking or for the LayerWise energy side.
- **Up / downslope:** fixed-cut & critical-cut → mostly upslope easier (flips listed above); critical-mass → always upslope after the orientation fix. Slope is now in the setup directly (no flat-propagation workaround).

## Lean + availability

I’m not pushing a production pick — you two will have a better feel — but from my side **`pst_fixed_cut` feels the most straightforward**: we stay with a thickness-fraction-style tensile metric at a fixed cut, which is close to what we already interpret, and it is cheap.

Happy to jump on a short call if there are remaining questions or to discuss how to continue. I will not be working until next Tuesday (holiday), but I can answer questions or hop on a short call over WhatsApp or Teams.

Cheers  
Yannik

---

## Attachments (suggested)

From `dev/alt_steady_state/`:

1. `pst_fixed_cut/plots/ab_thickness_fraction.png`
2. `pst_fixed_cut/plots/ab_ERR.png`
3. `pst_fixed_cut/plots/ud_max_Sxx_norm.png`
4. `pst_critical_cut/plots/ab_L_crit.png`
5. `pst_critical_mass/plots/ab_critical_mass.png`
6. `pst_critical_mass/plots/ud_critical_mass.png`

---
---

# Alternative B — deutscher Entwurf

**An:** Valle, Philipp  
**Von:** Yannik  
**Betreff:** Re: Question regarding our tensile failure approach in LayerWise

---

Hey all,

kurzes Update zu den drei PST-Proxys, die wir diskutiert hatten (fixed Cut-Length (pst_fixed_cut), kritische Masse bis zum ersten Versagen bei 10cm Schnitt (pst_critical_mass), und kritische Cut-Length bis zum ersten Versagen (pst_critical_cut)).

**Kurzer Refresher:** Mit Touchdown-Mode B steigt der normalisierte Stress (`max_S_xx / tensile_strength`) bei den dichteren/steiferen Slabs schneller — weil die Touchdown Distanz mit der Steifigkeit mitgeht. Wenn man bei allen die gleiche Länge nimmt, ist die Hoffnung, dass die Intuition wieder korrekt ist (weniger dichter Slab bricht eher). Momentaner Hack war: Slab mit weniger wie 100er Dichte bricht immer, bzw. wenn der Slab von oben oder unten bis zur wenig dichten Schicht gebrochen ist.

Vorteil vom Proxy: Winkel können wir direkt reinbringen und müssen keinen Teil der Propagation-Auswertung im Flachen machen.

**Test-Cases:**
Steph und ich sind die Cases im [Slab Generator](https://beta.whiterisk.ch/weac-slab-generator) durchgegangen. Ich habe die ganzen Cases dort mal implementiert, dabei sind 18 Cases übriggeblieben, welche wir jetzt als Ground-Truth nutzen. Yardstick für tensile ease:

- Cases **1–11 & 21–23**: A leichter als B  
- Cases **12–15**: B leichter als A  
- Cases **16–20**: für A/B-Urteil raus — zu viele überlagerte Effekte, nicht transparent genug

**Drei Ansätze:**
Für alle drei Ansätze unten (ohne den ρ ≤ 100 Hack) kommt das erwartete 'Tensile Ease'-Ordering raus. Bei ERR fühlt es sich so an, als wären die Proxies weniger aussagekräftig wie bisher.

Alle drei benutzen das folgende Setup:

- PST Geometrie, **`touchdown=False`**
- upslope und downslope wir ausgewertet (upslope = `-pst`, downslope = `pst-`, beide bei **`+φ₀`**)
- Crack onset: erstes Mal **`max(S_xx / σ_tensile) ≥ 1`** irgendwo im Slab

Caveat:
Die Endmasse kann nur am **rechten** Segmentrand sitzen. "critical_mass" benutzt deshalb zweimal **`pst-`**, für downslope = `+φ₀` und upslope = **`−φ₀`**.

1) `pst_fixed_cut`

Feste free cut length **L = 500 m**; Ease = höhere broken thickness fraction (`S_xx/σ_t > 1`). Schnell zu rechnen. Verteilung zwischen Up-/Downslope: upslope bricht meist leichter, downslope nur bei **12a, 13a, 15a, 21b, 22b**. ERR bei fixed L ist schwierig zu trauen — Case 1 (wenig Neuschnee) hat sehr hohen ERR, vermutlich weil 50 cm nah an der alten Touchdown-B-Länge liegt und dadurch energetischer wirkt als dickere/steifere Layerings.

**Plots:** `ab_thickness_fraction.png`, `ab_ERR.png`, `ud_thickness_fraction.png`

2) `pst_critical_mass`

Feste kurze cut length  **L = 100 mm**, Endmasse `m` am free tip hochdrehen bis Crack onset; Ease = **niedrigere critical mass**. Verteilung Up-/Downslope: Upslope bricht immer schneller.

**Plots:** `ab_critical_mass.png`, `ud_critical_mass.png`

3) `pst_critical_cut`

Keine Endmasse; kleinste cut length **L_crit ∈ [1 mm, 5 m]** mit Crack onset suchen (Brent). Ease = **kürzeres L_crit**. Gleiche Verteilung zwischen Up-/Downslope wie fixed-cut: upslope meist leichter, Flips bei **12a, 13a, 15a, 21b, 22b**.

**Plots:** `ab_L_crit.png`

**Cross-cutting**
- Tensile A/B: alle drei matchen den Yardstick; der ρ ≤ 100 Overwrite ist nicht mehr nötig.
- ERR: schwankt stark zwischen den Ansätzen und läuft oft nicht mit tensile ease mit — bevor wir es für Ranking oder die LayerWise Energy Seite nutzen, bräuchten wir Control Values.

**Tendenz + Verfügbarkeit**
Für mich fühlt sich **`pst_fixed_cut` am geradlinigsten** an: wir bleiben bei einer thickness-fraction-artigen tensile metric bei fester cut length, nah an dem was wir schon interpretieren, und es ist günstig. Das einzige was eventuell für z.b. die Masse sprechen würde, wäre, dass man dann gut ein Experiment implementieren könnte, bis z.b. der ganze Block bricht oder Vglb. aber ich denke davon ist die momentane Idee trotzdem noch weit entfernt.

Ich wäre gerne für einen kurzen Call zu haben um das Ganze noch kurz zu dritt/zweit durchzugehen und um das weitere Vorgehen abzusprechen.
Bis nächsten Dienstag bin ich im Urlaub, aber für Fragen oder einen kurzen Call über WhatsApp/Teams bin ich verfügbar!

LG  
Yannik

PS:
Der Entstehungsprozess dieser Email war: Selber herausschreiben was mir aufgefallen ist, dem AI Modell meine Codebase geben und eine Email generieren lassen und dann nochmals alles komplett umformulieren, weil es unleserlich war. Der ganze Prozess war nicht wirklich schnell, deshalb würde es mir helfen, wenn ihr mir kurz sagt, ob es wenigstens leserlich war, oder ob der Prozess ein Reinfall war.
