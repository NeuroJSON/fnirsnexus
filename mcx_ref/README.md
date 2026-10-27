# MCX reference data for the ICBM-152 / NinjaNIRS-22 image-reconstruction challenge

This folder provides a **Monte-Carlo-generated fNIRS dataset** that any reconstruction pipeline
(Cedalion, NeuroDOT, redbird/redbirdpy, …) can reconstruct. The truth is known, and the data
is free of the *inverse crime* in Cedalion's original challenge data.

Cedalion's original data (`../cedalion_ref`) was synthesized with the same precomputed
sensitivity matrix `Adot` that Cedalion then reconstructs with. Here the identical experiment
(same head, montage, activation, resting-state physiology and temporal workflow) is regenerated
with an independent forward model. The model uses **voxel-based MCX** on the segmented
ICBM-152 head with physically consistent detectors, and it is validated against detected
photons. Only the channel-space activation changes. Everything else is bit-identical to the
original.

---

## 1. Quick start

**Files you need** (the first two are about 3.4 MB together):

| file | what |
|---|---|
| `mcx_ref/hrf_measurement.jdb` | **the measurement**: block-averaged HRF ΔOD + optode geometry (§5.1) |
| `redbird_recon/groundtruth.bgii` | **the truth**: cortex surface + true ΔHbO/ΔHbR per vertex (§5.3) |
| `hm_icbm152/` | the ICBM-152 head model (masks, surfaces, voxel→vertex maps; 170 MB). Not in git: get it from Cedalion with `cedalion.data.get_icbm152_headmodel_files()` (downloads `hm_icbm152.zip`) |

`hrf_measurement.jdb` has the same schema as `redbird_recon/hrf_measurement.jdb` (Cedalion's
original data). A pipeline can therefore switch datasets by changing only the file name.

**Load it:**

```matlab
% MATLAB (JSONLab, https://github.com/NeuroJSON/jsonlab)
m  = loadjd('mcx_ref/hrf_measurement.jdb');
gt = loadjd('redbird_recon/groundtruth.bgii');
[~, it] = min(abs(m.time - 10));          % t_rel = 10 s snapshot
y = m.dOD(:, :, it, 1);                   % [497 pairs x 2 wavelengths], condition "Stim C3"
```

```python
# Python (jdata >= 0.9.5: pip install -U jdata bjdata)
import jdata as jd, numpy as np
m  = jd.load("mcx_ref/hrf_measurement.jdb")
gt = jd.load("redbird_recon/groundtruth.bgii")
dOD = np.asarray(m["dOD"])                # (497, 2, 190, 2): pair, wavelength, time, trial
it  = np.argmin(np.abs(np.asarray(m["time"]).ravel() - 10.0))
y   = dOD[:, :, it, 0]                    # Stim C3 at t_rel = 10 s
```

Score a reconstruction on the 25,000 cortex vertices of `groundtruth.bgii`, for example by peak
localization error, spatial correlation and recovered amplitude. `compare_recons.py` and
`test_redbird.m` do this.

---

## 2. The experiment being reproduced

From Cedalion's `hmstandardization_challenge_prep.ipynb`:

* **Head**: ICBM-152, 5 tissues, 1 mm voxels (`hm_icbm152/`).
* **Montage**: NinjaNIRS-22 whole-head cap, 56 sources and 144 detectors, **497 channels** (pairs)
  after Cedalion's quality pruning; positions registered to the scalp (`geo3d_relaxed`, RAS mm).
* **Wavelengths**: 760 and 850 nm.
* **Activation**: two conditions, **Stim C3** (left) and **Stim C4** (right). Each is a Gaussian
  blob on the cortex (σ = 2 cm geodesic, 5 µM peak ΔHbO, ΔHbR = −0.4 ΔHbO) centred under the
  10-10 landmarks C3 and C4 (`cedalion_ref/spatial_act.csv`).
* **Timing**: 14 events, 7 per condition, 10 s blocks at random onsets in a 6-min real
  resting-state recording (nn22, sub-08).
* **Output**: the block-averaged HRF ΔOD, t_rel ∈ [−1, 20] s.

---

## 3. How the data were generated

### 3.1 Head model and optical properties

The label volume uses Cedalion's convention (`cedalion.io.read_segmentation_masks`), one label
per mask in the order of its `mask_files`. Masks are 193 × 239 × 263 voxels, 1 mm isotropic.
The NIfTI affine has an identity rotation and offset (−96, −132, −148) mm.

| label | tissue | mask | μa (mm⁻¹) | μs (mm⁻¹) | g | n |
|---|---|---|---|---|---|---|
| 0 | background | — | 0 | 0 | 1 | 1 |
| 1 | CSF | `mask_csf.nii` | 0.0026 | 0.01 | 0.001 | 1 |
| 2 | gray matter | `mask_gray.nii` | 0.0186 | 1.10 | 0.001 | 1 |
| 3 | scalp | `mask_skin.nii` | 0.0191 | 0.66 | 0.001 | 1 |
| 4 | skull | `mask_bone.nii` | 0.0136 | 0.86 | 0.001 | 1 |
| 5 | white matter | `mask_white.nii` | 0.0186 | 1.10 | 0.001 | 1 |

* The optical properties are Cedalion's own (`cedalion/dot/tissue_properties.py`). They are the
  **same at both wavelengths**, so one set of simulations serves both. Refractive index n = 1
  everywhere, so the boundary is matched (no Fresnel reflection).
* CSF is physiological (nearly clear). That is fine for Monte Carlo; diffusion-based solvers may
  want an equivalent CSF (e.g. μs′ = 0.15 mm⁻¹).
* For the simulations the grid is cropped to the head bounding box plus one air voxel:
  **189 × 222 × 253 voxels**, starting at full-grid index `lo = [3 12 1]` (1-based).

### 3.2 Optode positions and directions

1. `srcpos`/`detpos` (RAS mm, from Cedalion's `geo3d_relaxed`) are mapped to voxel coordinates,
   `ijk = A⁻¹ · RAS`. These are 0-based voxel centres, as in nibabel.
2. **As Cedalion**: snap to the nearest vertex of the scalp mesh `mask_scalp.obj`, then to the
   nearest scalp voxel. The **direction** is the inward normal of that mesh vertex (iso2mesh
   `nodesurfnorm`, flipped towards the head centroid if needed).
3. **Difference from Cedalion**: the optode is moved to the nearest scalp voxel that **touches
   exterior air** (6-neighbourhood; exterior = complement of the hole-filled head). Median shift
   1 voxel, maximum 4.
   * Cedalion's snapping leaves 26 sources and 67 detectors 2–5 voxels below the skin, almost all
     on the left. The decimated scalp mesh sits about 2.2 voxels inside the voxel skin on the
     left but only about 1.0 on the right.
   * A physical optode sits on the skin, so the reference uses skin positions.
4. MCX coordinates (`issrcfrom0 = 1`): voxel centre = `ijk + 0.5`, shifted by the crop offset.

### 3.3 MCX simulations: one fluence per optode

All 200 optodes are simulated separately with mcxlab:

| setting | value |
|---|---|
| photons | **1 × 10⁹ per optode** |
| output | `outputtype='fluence'`, `isnormalized=1`: fluence Φ (mm⁻² per launched photon) |
| time gate | single gate 0–5 ns (`tstart=0, tend=tstep=5e-9`), as Cedalion |
| boundary | `isreflect=0` (n = 1 everywhere) |
| sources (56) | **pencil beam** at the skin-voxel centre, along the inward normal |
| detectors (144), as the adjoint source | **Lambertian disk**, radius **1.5 mm** (3 mm diameter), `srctype='disk'`, `srcdir(4) = −Inf`, launched from the skin-surface point above the detector |
| seeds | `20261027 + optode index` |
| GPU | RTX 5090: about 7 s per optode at 10⁹ photons |

The detector's launch point `b₀` is found by marching outward from the detector voxel along
−(inward normal) in 0.025-voxel steps until exterior air is reached.

### 3.4 Sensitivity (Jacobian) and normalization

For channel *p* = (source *s*, detector *d*), the (Born) fluence Jacobian for an absorption change
in voxel *v* is negative — raising μa lowers the signal — and is what MCX's adjoint mode returns
as `jmua`:

```
∂Φ_p/∂μa(v) = −φ_s(v) · φ_d(v) · V_vox                                   (V_vox = 1 mm³)
```

Since OD = −ln Φ, the Rytov (OD) sensitivity is positive:

```
∂OD_p/∂μa(v) = −(1/Φ_p) ∂Φ_p/∂μa(v) = J_p(v) / Φ_p ,      J_p(v) ≡ φ_s(v) · φ_d(v) · V_vox ≥ 0
```

where φ_s is the source fluence and φ_d is the fluence of the detector's adjoint (Lambertian)
source. **Throughout this document and in all saved arrays, J_p denotes this positive magnitude**
(the code computes it directly as `phi_s .* phi_d * Vvox`), so `Lvox`, `Ltotal` and `maxOD` are
positive OD sensitivities / OD changes; the physical fluence Jacobian is −J_p.

**Why Lambertian, and the factor 4.** A detector that counts every photon leaving through its
disk (area A) has, by reciprocity, the adjoint source "radiance-uniform (cosine-weighted) inward
emission over the disk". The probability that a photon at *v* (isotropic radiance 1/4π) is
detected is then

```
P_det(v) = (A/4) · φ_Lambert(v)      (1/4π × π from the cosine-weighted source)
```

and the detected signal is M_p = A · R̄_p, with R̄_p the mean diffuse reflectance under the disk.
Hence

```
δOD_p = −δM_p / M_p = Σ_v φ_s(v) φ_d(v) δμa(v) V_vox / (4 R̄_p)    →    Φ_p = 4 R̄_p
```

**Diffuse reflectance from the interior fluence** (Kienle & Patterson, JOSA A 14:246, 1997, Eq. 7).
With a matched boundary (R_Fresnel = 0), ∫cosθ dΩ = π and ∫cos²θ dΩ = 2π/3 give

```
R = Φ(0)/4 + (D/2) · ∂Φ/∂z ,      D = 1 / (3 (μa + μs(1−g)))    (scalp)
```

with z along the inward normal and Φ(0) the fluence on the skin surface. Both are taken from a
least-squares fit over the tissue voxels under the disk (lateral distance ≤ 1.5 mm, enlarged to
2.25 or 3 mm if fewer than 8 voxels; 0 < z ≤ 3.5 mm):

```
ln φ_s(v) = a + b·z + c·u + d·w      →   Φ(0) = eᵃ,   ∂Φ/∂z = b·eᵃ
```

(u, w are lateral axes.) Because many voxels are averaged, this readout is far less noisy than
counting the escaping photons, and it works wherever the optode is.

**Validation against detected photons** (`mcx_ppath.m`: separate baseline MCX runs, 10⁹
photons/source, 1.5 mm detectors at the same positions, per-photon partial paths):

| quantity | fluence-only model / detected photons |
|---|---|
| intensity (R̄ × disk area vs detected weight) | 0.967 median, r(log) = 0.988 |
| total path Σ_v J/Φ vs ⟨L_total⟩ | **1.050** (DPF 4.68 vs 4.88) |
| C3 / C4 region partial path | 1.101 / 0.927 (r = 0.994 / 0.990) |

A collimated detector adjoint (MCX's adjoint mode, and Cedalion's pencil beams) overstates
the sensitivity by 1.6–1.8×.

### 3.5 Activation → channel ΔOD

1. **Vertex → voxel**: Cedalion's own `hm_icbm152/voxel_to_vertex_brain.mtx.gz` assigns each of
   the 1,619,514 gray/white-matter voxels to its nearest cortex vertex. Each voxel takes its
   vertex's ΔHbO/ΔHbR from `spatial_act.csv` (µM).
2. **Absorption change** (Prahl spectrum, converted exactly as in Cedalion):

   ```
   Δμa(v, λ, c) = ε_HbO(λ) ΔHbO(v, c) + ε_HbR(λ) ΔHbR(v, c)      [mm⁻¹]
   ε [mm⁻¹ M⁻¹] = ln(10)/10 × ε_Prahl [cm⁻¹ M⁻¹] ;  ΔHb in M (= µM × 10⁻⁶)
   ε_Prahl(760) = 586 (HbO), 1548.52 (HbR);   ε_Prahl(850) = 1058 (HbO), 691.32 (HbR)
   ```
3. **Peak channel ΔOD** (first-order Rytov, linear in the activation amplitude):

   ```
   maxOD(p, λ, c) = Σ_v [J_p(v)/Φ_p] · Δμa(v, λ, c)
   ```

   This is stored as `maxOD`, and `dOD_amp(a) = a · maxOD` for a = 0, 0.05, …, 1.

ΔOD = −ln(I/I₀) uses the natural log. Positive values mean absorption increased.

### 3.6 Temporal augmentation (Cedalion's workflow, ported to MATLAB)

Cedalion adds the activation to the real resting-state OD and block-averages it. In its notebook
`od_augm` carries a `trial_type` dimension, so **both conditions average all 14 epochs**, and
the original data has the form

```
hrf_c(p, λ, t) = R(p, λ, t) + K_c(t) · maxOD_c(p, λ)      (c = C3, C4)
```

with a common resting-state term **R** and a noise-free kernel `K_c = pipeline(reg_c)`. The
pipeline is ported exactly:

| step | definition |
|---|---|
| time axis | nn22 sub-08 run-02, 3309 samples, f_s = 1/mean(Δt) = **8.9888 Hz** |
| events | `cedalion_ref/synthetic_hrf.snirf` stims. C3: 16.39, 59.49, 84.54, 151.60, 251.16, 274.05, 296.83 s; C4: 38.62, 109.99, 130.05, 172.57, 199.87, 225.39, 318.93 s; all 10 s, value 1 |
| basis | Gamma(τ=0, σ=3 s, T=3 s): h(t) = x² e^(−x²), x = (t−τ)/σ, n = ⌈(4.1σ+τ+T) f_s⌉ = 138 samples, convolved with a boxcar of ⌊T f_s⌋ = 26 samples, truncated, max-normalized |
| regressor | reg_c(t) = (boxcar_c ⊛ h)(t), max-normalized (boxcar: `searchsorted` onset to onset + duration) |
| filter | Butterworth order 4, band-pass 0.02–0.5 Hz, zero-phase (`filtfilt`; Cedalion uses `sosfiltfilt`) |
| epochs | t_rel = (−9:180) × 0.111 s (−0.999 … 19.98 s, 190 samples), linear interpolation at onset + t_rel |
| baseline | subtract the mean over t_rel < 0, then average over the 14 epochs |

**Recovering R exactly.** `hrf_C3 − hrf_C4 = K_C3·m_C3 − K_C4·m_C4` is exactly rank 2. A
per-(pair, λ) least-squares fit gives Cedalion's `maxOD_cedalion` and R, with residual
**1.6 × 10⁻⁴**, which validates the port. Then
`R = ½[(hrf_C3 − K_C3 m_C3) + (hrf_C4 − K_C4 m_C4)]`.

**The new data:**

```
dOD_c(p, λ, t)       = R(p, λ, t) + pipeline( maxOD_mcx(p, λ, c) · reg_c(t) )      → dOD
dOD_clean_c(p, λ, t) =              pipeline( maxOD_mcx(p, λ, c) · reg_c(t) )      → dOD_clean
```

The physiology, the noise and every channel's pre-stimulus variance (`c_meas`) are therefore
identical to Cedalion's file. Only the activation forward model differs.

### 3.7 How it compares with Cedalion's data

* **Channel pattern**: MCX peak ΔOD vs Cedalion's `Adot` peak ΔOD, r = 0.905 over all channels.
* **Magnitude**: weaker, median ratio 0.315 on strong channels. Slope 0.56 on C3 and 0.24 on C4.
  * `mcx_cedalion_check.m` reruns MCX with **Cedalion's exact conventions**: buried voxel-corner
    optodes, pencil-beam detectors, point-fluence normalization. It reproduces the data's `Adot`
    (slope 1.045 / 0.977, r = 0.998 / 0.999). So the difference comes from Cedalion's method,
    not its MC engine.
  * Collimated detectors inflate the sensitivity everywhere. The single-voxel point-fluence
    normalization cancels part of that only for buried optodes, so the inflation depends on
    optode depth. Because of the mesh offset in §3.2, depth differs by hemisphere: Cedalion's
    data overstates **C3 by 1.8× and C4 by 4.1×**, a left/right bias that a recon using the
    same `Adot` cannot see.
* **SNR**: lower than in the original, at the same resting-state noise. Median over the 20
  strongest 850 nm channels at t = 10 s: C3 11.2 (original 17.3), C4 8.9 (original 35.1).

---

## 4. Reconstructions of this data (for reference)

At t_rel = 10 s, with the peak localization error vs truth:

| solver | C3 | C4 |
|---|---|---|
| Cedalion `ImageRecon` (`mua2conc`, `alpha_meas=0.5`, `brain_only=False`, `c_meas`), original `Adot` | 3.6 mm | 18.9 mm |
| redbird `s4_recon_surface` (voxel aggregation, own optics), α = 0.003, β = 0.01 | 114 mm | 42 mm |
| redbird Jacobian through Cedalion's inverse (`inverse_match.m`, S4) | 3.6 mm | 18.9 mm |

On this data the **inverse** (regularization) matters more than the forward model.
`inverse_match.m` and `figures/inverse_match_*.png` show the conversion of redbird's inverse
into Cedalion's step by step. With Cedalion's inverse and per-channel scale, redbird reproduces
Cedalion's maps at r = 0.93–0.96.

Notes for running Cedalion on this data:
* The copy of `sensitivity_nn22_resting_icbm152.nc` now on Cedalion's server **has been
  replaced**: its hash differs, and it reproduces the original data only at r = 0.70 / 0.83.
  `adot_cedalion.mat` is the original `Adot`, reproduced with MCX (§3.7).
* `cmeas_cedalion.mat` holds Cedalion's `c_meas` (resting-state OD variance) in pair order.
* Recent Cedalion SCI/PSP code prunes 93 channels instead of the notebook's 70. Use the 497
  pairs in the file.

---

## 5. Data file structure

All `.jdb` / `.bgii` files are binary JData (BJData, zlib-compressed). MATLAB reads them with
JSONLab `loadjd`/`loadbj`; Python with `jdata` ≥ 0.9.5 (`jd.load`). Array shapes are the same
in both. Indices stored in the files are **1-based**.

### 5.1 `hrf_measurement.jdb`: the measurement (schema `redbird-hrf-dod`)

| field | type, shape | content |
|---|---|---|
| `format` | string | `"redbird-hrf-dod"` |
| `wavelengths` | uint16 [2] | 760, 850 nm |
| `srcpos` | double [56 × 3] | source positions, RAS mm (Cedalion `geo3d_relaxed`, before any snapping) |
| `detpos` | double [144 × 3] | detector positions, RAS mm |
| `pairs` | int32 [497 × 2] | channel list `[source, detector]`, 1-based rows of `srcpos`/`detpos` |
| `trialtype` | string [2] | `"Stim C3"`, `"Stim C4"` (last index of `dOD`) |
| `time` | double [190] | t_rel (s), −0.999 … 19.98, Δt = 0.111 s |
| **`dOD`** | float32 [497 × 2 × 190 × 2] | **block-averaged ΔOD** [pair, wavelength, t_rel, trial] = R + MCX activation. Use this. |
| `dOD_clean` | float32 [497 × 2 × 190 × 2] | MCX activation only (no resting-state background) |
| `maxOD` | double [497 × 2 × 2] | MCX peak ΔOD [pair, wavelength, trial] (activation amplitude 1) |
| `maxOD_cedalion` | double [497 × 2 × 2] | Cedalion `Adot` peak ΔOD that synthesized the original data (fitted) |
| `maxOD_se` | double [497 × 2 × 2] | Monte Carlo standard error (NaN for this fluence-based model) |
| `Unit`, `Space` | string | `"dimensionless"`, `"scanner_anat_RAS_mm"` |
| `Description`, `Generator` | string | provenance |
| `ForwardSettings` | struct | MCX settings (`nphoton`, `detradius`, `detlaunch`, `norm`, `seed`, …) |
| `TemporalSettings` | struct | temporal settings (`fband`, `butter_order`, `epoch`, resting-state file) |

Notes:
* Pre-stimulus variance: `var(dOD(:, w, time<0, :))` across t_rel < 0 and both trials is what
  `s4_recon_surface` uses as `c_meas`. Cedalion uses the 6-min resting-state OD variance
  (`cmeas_cedalion.mat`).
* The source and detector index order follows the SNIRF probe of `cedalion_ref/synthetic_hrf.snirf`.
  The channel label is `sourceLabels[s] + detectorLabels[d]`, e.g. `S10D87`.

### 5.2 `mcx_jacobian.jdb`: the forward-model summary (`mcx-jacobian-dod`)

| field | shape | content |
|---|---|---|
| `pairs`, `wavelengths`, `trialtype` | | as in §5.1 |
| `maxOD` | [497 × 2 × 2] | peak ΔOD (§3.5), copied into `hrf_measurement.jdb` |
| `amp`, `dOD_amp` | [21], [497 × 2 × 2 × 21] | ΔOD at activation amplitude a ∈ [0, 1] (linear: a · maxOD) |
| `Phi` | [497] | Φ_p used for normalization (= `Phi_kienle`), mm⁻² per photon |
| `Phi_kienle`, `Phi_footprint` | [497] | 4 R̄_p (Kienle) / mean fluence under the disk (alternative) |
| `Ltotal` | [497] | Σ_v J_p/Φ_p = mean total path (mm) |
| `Lregion` | [497 × 2] | sensitivity summed over the C3 / C4 region (Δμa ≥ 1% of peak), mm |
| `sds` | [497] | source–detector distance (mm), 15.6–43.1 |
| `maxOD_cedalion` | [497 × 2 × 2] | as in §5.1 |
| `settings` | struct | `nphoton` 1e9, `detradius` 1.5, `detlaunch` 'lambertian', `norm` 'kienle', `seed` 20261027, … |

### 5.3 `../redbird_recon/groundtruth.bgii`: the truth (JGIfTI)

* `GIFTIData.MeshVertex3.Data`: single [25000 × 3], cortex vertices (RAS mm). This is
  `mask_brain.obj` after the voxel→RAS mapping, the same vertex order as Cedalion's brain surface.
* `GIFTIData.MeshTri3.Data`: int32 [49992 × 3], triangles (1-based).
* `GIFTIData.MeshVertex3.Properties.HbO_C3 / HbR_C3 / HbO_C4 / HbR_C4`: true ΔHbO/ΔHbR (µM),
  peak 5 / −2 µM.

### 5.4 Large regenerable files (git-ignored)

| path | content |
|---|---|
| `fluence/optodeNNN.bin` | MCX fluence of optode NNN (001–056 sources, 057–200 detectors): raw little-endian **float32**, MATLAB column-major, cropped grid **189 × 222 × 253** (§3.1). Voxel (i, j, k) of the crop is full-grid voxel (i+2, j+11, k) (1-based); its RAS centre is A · [i+1, j+10, k−1]ᵀ. Units mm⁻² per launched photon, time-integrated over 0–5 ns. |
| `fluence/optodeNNN.sig` | ASCII settings signature `[nphoton detradius dims(3) lo(3) seed (1 = Lambertian detector)]`; matching fluences are reused |
| `fluence/jac_brain.mat` | `Lvox` single [1,619,514 × 497] = J_p(v)/Φ_p (mm) on the gray/white voxels (`voxc`, linear index in the crop): reweight with any brain perturbation, Σ_v Lvox(v, p) Δμa(v) |
| `fluence_cedalion/` | the same with Cedalion's exact conventions (`mcx_cedalion_check.m`) |
| `cache/ppath_srcNN.mat` | detected-photon partial paths of the validation run (`mcx_ppath.m`) |

### 5.5 Other small files

| file | content |
|---|---|
| `adot_cedalion.mat` | `Adot` single [497 × 35018] (mm): Cedalion's original sensitivity reproduced with MCX, 25,000 cortex + 10,018 scalp vertices (`is_brain`), pair order of §5.1 |
| `cmeas_cedalion.mat` | `cmeas` [497 × 2]: Cedalion's `c_meas` (resting-state OD variance) per pair and wavelength |
| `mcx_ppath.jdb` | per-pair detected-photon statistics (validation, §3.4) |
| `mcx_cedalion_check.jdb` | peak ΔOD with Cedalion's conventions (§3.7) |
| `recon_cedalion_{orig,mcx}.bgii`, `recon_mcx.bgii` | example reconstructions (cortex, µM) |

---

## 6. Reproducing the data

**Requirements**:
* MATLAB R2019b+ with the Signal Processing Toolbox (`butter`, `filtfilt`) and the Image
  Processing Toolbox (`imfill`, `imdilate`, `bwdist`).
* **mcxlab** (MCX 2026; uses the disk source with `srcdir(4) = −Inf` Lambertian launch) and a CUDA GPU.
* iso2mesh, JSONLab.

**Inputs** (relative to the repo root): `hm_icbm152/`; `cedalion_ref/spatial_act.csv` and
`synthetic_hrf.snirf`; `redbird_recon/hrf_measurement.jdb` (geometry and Cedalion's original
data, needed to recover R); and `nn22_resting_state/sub-08/nirs/sub-08_task-RS_run-02_nirs.snirf`
(time axis only).

```bash
cd mcx_ref
matlab -nodisplay -batch "mcx_jacobian"     # 200 MCX runs (~20-25 min on an RTX 5090) + Jacobian -> mcx_jacobian.jdb
matlab -nodisplay -batch "mcx_build_hrf"    # temporal stage (~1 min) -> hrf_measurement.jdb
# or both: matlab -nodisplay -r "run_mcx_ref"
```

Options: `mcx_jacobian('nphoton', 3e9, 'gpuid', '1')`, `mcx_build_hrf('fitonly', true)`
(temporal-port check only).

The default GPU is `gpuid='10'` (the first device in MCX's order). Fluences are cached in
`fluence/` and reused when the settings match. Shared helpers are in `private/`: `headmodel.m`,
`snapoptode.m`, `drefprobe.m`/`drefread.m`, `readvox2vert.m`, `readspatialact.m`, `prahl.m`.

**Validation and comparison scripts**:

| script | purpose |
|---|---|
| `mcx_ppath.m` | detected-photon validation run (§3.4) |
| `mcx_cedalion_check.m` | MCX with Cedalion's conventions; exports `adot_cedalion.mat` |
| `cedalion_recon.py` | Cedalion `ImageRecon` on both datasets (needs a Cedalion environment, Python ≥ 3.11) |
| `test_redbird.m`, `inverse_swap.m`, `inverse_match.m` | redbird recon, Jacobian × inverse swaps, stepwise inverse matching |
| `compare_jacobians.m`, `compare_recons.py`, `plot_recon_compare.m` | Jacobian and recon comparisons, figures in `figures/` |
| `mcx_forward.m` | an alternative nonlinear binned perturbation-MC forward model (not used for the data) |

---

## 7. Caveats

* The activation enters to **first order** (Rytov, linear in amplitude). With the peak ΔOD
  ≤ 0.011, a nonlinear perturbation-MC check gave a deviation of 5 × 10⁻⁵.
* Detectors are ideal: they count every exiting photon over a 1.5 mm radius disk with a
  Lambertian adjoint. No angular acceptance (NA) or coupling losses are modelled.
* Optodes sit on the voxel skin surface (§3.2). The montage itself is Cedalion's registration.
* The CSF is physiological (μs = 0.01 mm⁻¹), which suits Monte Carlo. Diffusion-based solvers
  should use a diffusion-equivalent CSF.
* `maxOD_se` is NaN: the fluence-based model has no per-channel Monte Carlo error estimate.
  For scale, the detected-photon validation run (also 10⁹ photons per source) had a standard
  error of about 1.3% on the region partial paths of the strong channels.
