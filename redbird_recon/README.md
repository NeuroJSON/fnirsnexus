# Redbird port of the Cedalion fNIRS image-reconstruction challenge

This folder ports the image-reconstruction stage of Cedalion's
`hmstandardization_challenge_prep.ipynb` to the MATLAB/FEM **redbird-m** toolbox, and
explores how to make redbird's diffuse-optical-tomography (DOT) reconstruction match
Cedalion's.

The input data (`../synthetic_hrf.snirf`, `../spatial_act.csv`, `../hm_icbm152/`) is a
synthetic whole-head fNIRS dataset: real NinjaNIRS-22 resting-state OD with two
synthetic cortical activations (Gaussian ΔHbO/ΔHbR blobs under landmarks **C3** (left)
and **C4** (right)), block-averaged into an HRF ΔOD.

---

## TL;DR — what works best

Reconstruct on a **two-surface unknown space** (cortex + scalp, like Cedalion's
`TwoSurfaceHeadModel`) with **Rytov** sensitivity, **`c_meas` whitening**, and **depth
weighting** → `s4_recon_surface.m`. At `alpha=0.003, beta=0.01` it localizes
**C3 ≈ 6.7 mm, C4 ≈ 19 mm** with scalp leakage suppressed (`leak<1`); `beta=0.03` gives
the sharpest **C3 ≈ 3.7 mm**. See `recon_surface.png` / `recon_scalp.png`.

---

## Pipeline

| stage | script | output | notes |
|---|---|---|---|
| 0. export | `export_inputs.py` (Python) | `groundtruth.bgii`, `hrf_measurement.jdb` | needs `jdata, h5py, pandas` |
| 1. mesh | `s1_build_mesh.m` (iso2mesh) | `head_mesh.bmsh` | 5-tissue tet mesh from ICBM masks, RAS mm |
| 2. linear recon (GM volume) | `s2_recon_gm.m` | `recon_gm.bgii`, `recon_gm_op.mat`(cache) | single Rytov step on gray-matter nodes |
| 3. iterative recon (GM volume) | `s3_recon_gm_iter.m` | `figures/recon_gm_iter_C4.png` | official `rbrun`+`recon.isratio`, N Gauss-Newton iters |
| 4. two-surface recon (conservative SUM) | `s4_recon_surface.m` | **`recon.bgii`**, `figures/recon_surface.png`, `s4_op.mat`(cache) | **cortex+scalp unknowns; best localization** |
| 4b. two-surface recon (GATHER) | `s4_recon_gather.m` | `recon_gather.bgii`, `recon_gather.png`, `s4_op_gather.mat`(cache) | same, but non-conservative gather → **higher recovered contrast** |

Run (MATLAB launches use the `run_*.m` driver wrappers to avoid inline-newline issues):
```bash
python3 export_inputs.py                       # stage 0
matlab -nodisplay -r "s1_build_mesh; exit"     # stage 1: build the FEM mesh
matlab -nodisplay -r "run_s4b"                 # stage 4 (best): two-surface recon -> recon.bgii + figures
python3 score_recon.py                         # score recon.bgii vs groundtruth.bgii
# variants: run_s2 (GM-volume linear), run_s3 (iterative Gauss-Newton), run_s4 (s4 without re-render),
#           run_s4_gather (non-conservative gather -> recon_gather.bgii, higher contrast)
```
Tuning (instant — uses the cached operator): `s2_recon_gm([],alpha,beta)`,
`s4_recon_surface(alpha,beta)`, `s4_sweep.m`.

---

## Data-exchange files (NeuroJSON / JData) — shareable to a third solver

| file | format | content |
|---|---|---|
| `head_mesh.bmsh` | binary **JMesh** | tet head mesh, RAS mm; tissue label 1=scalp 2=bone 3=CSF 4=gray 5=white in `MeshTet4.Properties.Value` |
| `hrf_measurement.jdb` | binary **JData** | block-averaged HRF ΔOD `[pair×wavelength×time×trial]`, `srcpos`, `detpos`, SD `pairs`, `wavelengths`, `time` |
| `groundtruth.bgii` | binary **JGIfTI** | cortex surface (25k verts/faces) + true per-vertex `HbO_C3/HbR_C3/HbO_C4/HbR_C4` (µM) |
| `recon.bgii` | binary **JGIfTI** | reconstructed per-vertex ΔHbO/ΔHbR on the cortex (currently the two-surface `s4` result) |

Load with `loadjmesh`/`loadbj`/`loadjd` (MATLAB) or `jdata.load` (Python). To replicate
in TOAST++/MMC, share the first three + the baseline-property/recon-basis spec (below).

---

## Cedalion ↔ redbird correspondence

| Cedalion | redbird |
|---|---|
| `TwoSurfaceHeadModel` (cortex+scalp surfaces) | `s4` surface unknowns; or `head_mesh` GM nodes (`s2`/`s3`) |
| precomputed MC `Adot` (`∂OD/∂μₐ`) | FEM Jacobian (`rbfemmatrix`) ÷ baseline model = Rytov OD sensitivity |
| `recon_mode="mua2conc"` | `rbjacchrome` (Adot + extinction → HbO/HbR Jacobian `W`) |
| `ImageRecon.reconstruct` (`M=D(F+λC)⁻¹`, applied once to all time cols) | `rbreginv` (solves `(WRWᵀ+λC)z=Δy`, returns `RWᵀz`) — applied per snapshot |
| `apply_c_meas` / `alpha_meas` / spatial prior `R` (`alpha_spatial`) | `c_meas` row-whitening + depth column-weighting `beta` + Tikhonov `alpha` |
| ΔOD input (Born/Rytov OD space) | `reform='logphase'` (Rytov) or manual `Anorm = W./ymodel`; `recon.isratio` for ratio input |

---

## Key findings

1. **Recon/parameter mesh is the #1 difference.** Cedalion reconstructs on **thin 2-D
   surfaces** (cortex 25k + scalp 10k); my early redbird used a **GM volume band**
   (62.6k nodes), which has depth ambiguity → diffuse. Switching to the two-surface
   space (`s4`) was the single biggest improvement (C3 47.5 mm → 6.7 mm).
2. **Scalp must be in the unknowns *and* depth-weighted.** With scalp added but weak
   weighting (`beta=1`) the scalp **dominates** (`leak>1`) — reproducing Cedalion's
   `brain_only=False` scalp leakage (`recon_scalp.png`). Strong depth weighting
   (`beta≈0.01–0.03`) makes the scalp a **common-mode sink** so the cortex localizes.
3. **The data is Rytov-linear**: the activation was injected as `ΔOD = Adot·Δμₐ` added
   in OD/log space, so a **single linear Rytov step is the matched estimator**; extra
   Gauss-Newton iterations (`s3`) don't reduce the structural residual (≈0.70 floor =
   un-fittable resting-state common-mode, not nonlinearity).
4. **The two conditions are ~96.5% common-mode** (`corr(dOD_C3,dOD_C4)=0.999`); only
   ~3.5% is discriminative (correctly lateralized). The **C3−C4 contrast** cleanly
   recovers the truth dipole (3rd column of `figures/recon_surface.png`).
5. **`OD = −ln(I/I₀)` with `I₀` = temporal mean** (`int2od`); ratio `I/I₀ = exp(−ΔOD)`.
   `Cedalion C_meas` = per-channel **temporal variance** of the resting OD
   (`quality.measurement_variance`, diagonal; bad channels filled with 1e6×max).

### Engineering fixes that were required
- `removeisolatednode` before saving the mesh (isolated nodes → singular FEM matrix).
- voxel(IJK)→RAS transform on **both** the mesh and the ground-truth cortex (a frame
  mismatch otherwise flips L/R and gives ~110 mm errors).
- keep scattering in `cfg.prop`, **not** `cfg.param` (else `rbupdateprop:102` fails when
  concentrations go node-wise during recon).
- `reform='logphase'` (Rytov), **not** the default `'real'` (Born).
- λ scaled **relative** to `max(diag(A·Aᵀ))` (`rbreginv` adds λ absolutely).
- set `isratio` as a **`recon.isratio`** struct field (read inside `rbrunrecon`), not an
  `rbrun` name/value option.

---

## New redbird feature added: `recon.isratio`

To let `rbrunrecon` ingest **differential/ratiometric** fNIRS data, a flag was added in
`redbird-m/matlab/rbrunrecon.m` (after the forward solve): when `recon.isratio=1`, on
iteration 1 the measured **ratio `I/I₀`** is multiplied by the baseline forward model to
form the effective absolute measurement, then frozen. Combine with
`reform='real'` for **Born** or `reform='logphase'` for **Rytov** difference imaging.
Validated to machine precision (ratio-mode == absolute-mode) by the test
`redbird-m/test/test_isratio.m` (moved into the redbird-m repo alongside the patch).

---

## Layout
- `figures/` — `recon_surface.png` (**two-surface cortex recon vs truth + C3−C4 contrast, best**),
  `recon_lateral.png` (lateral views), `recon_scalp.png` (scalp common-mode leakage),
  `recon_gm_iter_C4.png` (iterative Gauss-Newton result).
- `logs/` — stdout captures from each MATLAB run (disposable).

## Caches (regenerable)
- `s4_op.mat` (~530 MB): cached two-surface linear operator for instant retuning via
  `s4_recon_surface(alpha,beta)` / `s4_sweep.m` — **safe to delete**, rebuilt on next run.
  (`s2`'s `recon_gm_op.mat` cache is likewise rebuilt by `s2_recon_gm.m` if needed.)

## Next steps
- Add a **surface Laplacian prior** to tighten the cortical blobs (corr is still ~0.3–0.4).
- Use **Cedalion-matched `C_meas`** = temporal variance of the raw resting-state OD
  (export per-channel from `../nn22_resting_state/.../*.snirf`).
- For a true numeric head-to-head, run Cedalion's recon to emit `recon_img` and score
  both on the same cortex vertices.
- Regenerate the data with **MMC** to get genuine (non-inverse-crime, nonlinear) forward
  measurements; then the iterative `recon.isratio` Gauss-Newton path becomes the right tool.
