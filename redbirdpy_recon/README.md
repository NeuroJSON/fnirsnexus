# redbirdpy port of the Cedalion fNIRS image-reconstruction challenge

Python mirror of [`../redbird_recon`](../redbird_recon) (the MATLAB/redbird-m port).
Every reconstruction script here is a faithful translation of its `*.m` counterpart,
re-implemented on the Python toolchain:

| MATLAB | Python |
|---|---|
| **redbird-m** (FEM forward / Jacobian / inverse) | **redbirdpy** (`import redbirdpy as rb`) |
| **iso2mesh** (mesh ops) | **pyiso2mesh** (`import iso2mesh as i2m`) |
| **jsonlab / jmesh / jnifti** (I/O) | **jdata** / pyjdata (`import jdata as jd`) |

The recon **data files stay in `../redbird_recon`** (`head_mesh.bmsh`,
`hrf_measurement.jdb`, `groundtruth.bgii`); these scripts only read them. The ICBM-152
tissue masks / surfaces live in `../hm_icbm152`. No local toolbox paths are hard-coded —
`redbirdpy`, `iso2mesh` and `jdata` are assumed importable (on `PYTHONPATH`).

---

## Pipeline

| stage | script | output | notes |
|---|---|---|---|
| 1. mesh | `s1_build_mesh.py` | `../redbird_recon/head_mesh.bmsh` | 5-tissue tet mesh from ICBM masks (CGAL `v2m`), RAS mm |
| 2. linear recon (GM volume) | `s2_recon_gm.py` | `recon_gm.bgii`, `recon_gm_op.jdb`(cache) | single Rytov step on gray-matter nodes |
| 3. iterative recon (GM volume) | `s3_recon_gm_iter.py` | `figures/recon_gm_iter_C4.png` | official `rb.run`+`recon["isratio"]`, N Gauss-Newton iters |
| 4. two-surface recon (conservative SUM) | `s4_recon_surface.py` | **`recon.bgii`**, `figures/recon_surface.png`, `s4_op.jdb`(cache) | **cortex+scalp unknowns; best localization** |
| 4b. two-surface recon (GATHER) | `s4_recon_gather.py` | `recon_gather.bgii`, `figures/recon_gather.png`, `s4_op_gather.jdb`(cache) | same, but non-conservative gather → **higher recovered contrast** |
| score | `score_recon.py` | stdout | peak-err / corr vs ground truth |
| render | `render_lateral.py`, `render_scalp.py` | `figures/recon_lateral.png`, `figures/recon_scalp.png` | lateral cortex + scalp-leakage figures |
| tune | `s4_sweep.py` | stdout | sweep alpha/beta on the cached operator (no forward) |

Run (with the three toolboxes on `PYTHONPATH`):
```bash
python3 s1_build_mesh.py                 # stage 1 (only to regenerate the mesh; already shipped)
python3 s4_recon_surface.py              # stage 4 (best): two-surface recon -> recon.bgii + figure
python3 score_recon.py                   # score recon.bgii vs groundtruth.bgii
python3 render_lateral.py                # lateral comparison figure
# variants: s2_recon_gm.py (GM-volume linear), s3_recon_gm_iter.py (iterative GN)
```
Tuning (instant — uses the cached operator): `python3 s4_recon_surface.py --alpha A --beta B`,
`python3 s4_sweep.py`.

Results match the MATLAB port: `s4` at `alpha=0.003, beta=0.003` gives
**C3 peak-err 4.3 mm (corr 0.27), C4 21.6 mm (corr 0.40)**, scalp leakage suppressed;
the **C3−C4 contrast** cleanly recovers the truth dipole (`figures/recon_surface.png`).
All rendered figures are written to `figures/` (matching the MATLAB `redbird_recon/figures/`).

### Aggregation variants (`s4_recon_surface.py` vs `s4_recon_gather.py`)

Both map the nodal forward Jacobian onto the surface vertices; they differ only in
how:

* **conservative SUM** (`s4_recon_surface.py`): each forward node → its nearest surface
  vertex, columns **summed** (total sensitivity conserved). Smoother, well-localized,
  lower recovered peak (~0.6–0.9 µM at `α=β=0.003`).
* **non-conservative GATHER** (`s4_recon_gather.py`): each surface vertex ← its single
  nearest forward node (`J[:, snode]`). Not conserved / aliasing-prone, but yields a
  **higher recovered contrast** (peaks ~4.2 µM, closer to the 5 µM truth amplitude) at
  the cost of looser peak localization. This was the original aggregation before the
  conservative version.

---

## What's shared (`nirs_common.py`)

Data loaders (`load_mesh`, `load_meas`, `load_truth`, `load_scalp`), optode→scalp
projection (`snap2surf` + `cKDTree` nearest-node), the redbirdpy cfg builder
(`make_cfg`: 5-tissue baseline → `rb.meshprep` + `rb.updateprop`), the measured-pair
`sd` table / Rytov `ymodel` stackers, and a matplotlib renderer with the RdBu diverging
colormap. The per-wavelength Jacobian is **streamed and aggregated immediately**
(`s2`/`s4`), keeping peak memory to one `(npair × Nnode)` block at a time.

### MATLAB → redbirdpy primitive map
`rbmeshprep`→`rb.meshprep`, `rbrunforward`→`rb.runforward`, `rbfemmatrix`(nodal
Jacobian)→`rb.jac`, `rbjacchrome`→`rb.jacchrome`, `rbmatflat`→`rb.matflat`,
`rbreginv`→`rb.reginv`, `rbsdmap`→`rb.sdmap`, `rbupdateprop`→`rb.updateprop`,
`rbrun`/`rbrunrecon`→`rb.run`/`rb.runrecon`. `knnsearch`→`scipy.spatial.cKDTree`,
`tsearchn`→`iso2mesh.tsearchn`, `volface`/`nodesurfnorm`/`readobjmesh`/`v2m`→`iso2mesh`.
Indices are 1-based throughout (iso2mesh/JMesh convention); the redbirdpy `sd` table
uses 0-based phi-column indices.

---

## Toolbox changes made for this port

* **`redbirdpy/recon.py` (`runrecon`)** — added `recon["isratio"]` support (mirrors the
  redbird-m `rbrunrecon` change): when set, on iteration 1 the measured ratio `I/I₀`
  (= `exp(−ΔOD)`) is scaled by the baseline forward model to form the effective absolute
  measurement. Used by `s3`.
* **`pyjdata/jdata/jnifti.py` (`nii2jnii` / `loadjnifti`)** — fixed `.nii` reading: the
  header-parse branch called `mmap.mmap(path, …, format=…)` (invalid); replaced with
  `memmapstream(bytes, niftiheader)` like the `.gz` branch; cast `imgbytenum` to `int`;
  fixed `loadjnifti` returning an unset `jnii` for `.nii` and accepted `.nii.gz`.
  `loadjnifti(...)["NIFTIHeader"]["Affine"]` now matches nibabel exactly.

---

## Caches (regenerable, git-ignored)

* `s4_op.jdb` (~250 MB, binary JData/zlib): cached two-surface Rytov operator for instant
  retuning via `s4_recon_surface.py --alpha .. --beta ..` / `s4_sweep.py` — safe to delete,
  rebuilt on next run. `s2`'s `recon_gm_op.jdb` is likewise rebuilt by `s2_recon_gm.py`.

## Notes / caveats

* **`s3` needs RAM (~16–24 GB free).** The official `runrecon` path materializes the full
  dense nodal Jacobian on the 144k-node forward mesh (both wavelengths) before remapping
  to the GM submesh. `s2`/`s4` are the memory-light equivalents (stream + aggregate per
  wavelength). On a small/shared host `s3` may be OOM-killed mid-`runrecon`.
* matplotlib 3-D `view_init` angles approximate MATLAB `plotmesh`'s `view([az el])`; the
  scalar fields and color scales match, the camera framing is close but not identical.

See [`../redbird_recon/README.md`](../redbird_recon/README.md) for the full
Cedalion↔redbird correspondence, key findings, and the data-exchange file spec.
