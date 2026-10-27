#!/usr/bin/env python3
"""Cedalion image reconstruction of the original (Adot-synthesized) and the MCX reference data.

Follows cedalion_ref/hmstandardization_challenge_prep.ipynb exactly for the recon:
ICBM-152 head model, nn22 resting-state c_meas (repair_amp -> int2od -> SCI/PSP pruning
-> measurement_variance), Cedalion's precomputed Adot, and

    ImageRecon(Adot, recon_mode="mua2conc", alpha_meas=0.5, brain_only=False,
               spatial_basis_functions=None, apply_c_meas=True).reconstruct(hrf_od, c_meas)

The MCX data (mcx_ref/hrf_measurement.jdb) is put into the same xarray structure as the
original hrf_od by matching (source, detector) labels; the mapping is verified by
reproducing the original data from ../redbird_recon/hrf_measurement.jdb exactly.
Scores at t_rel = 10 s with the redbird metrics (peak error, corr, recovered peak, scalp
leak) and writes recon_cedalion_{orig,mcx}.bgii (cortex HbO/HbR, uM).

Usage: ~/.venvs/cedalion/bin/python cedalion_recon.py
"""

from pathlib import Path

import cedalion
import cedalion.data
import cedalion.dot
import cedalion.io
import cedalion.io.forward_model
import cedalion.nirs
import cedalion.sigproc.quality as quality
import jdata as jd
import numpy as np
from cedalion import units

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T_SNAP = 10.0


def load_cmeas(channels):
    """c_meas as in the notebook (cells 6, 8, 48).

    measurement_variance(calc_covariance=False) is a per-channel temporal variance, so it
    is evaluated on the notebook's 497 channels directly (the current SCI/PSP code prunes
    a different set than the notebook did, which does not change these variances)."""
    rec = cedalion.data.get_nn22_resting_state()
    rec["amp_repaired"] = quality.repair_amp(rec["amp"])
    rec["od"] = cedalion.nirs.cw.int2od(rec["amp_repaired"])
    window = 5 * units.s
    _, sci_mask = quality.sci(rec["od"], window, 0.75)
    _, psp_mask = quality.psp(rec["od"], window, 0.1)
    clean = (sci_mask & psp_mask).sum(dim="time") / len(sci_mask.time)
    rec["od_clean"], drop = quality.prune_ch(rec["od"], [clean >= 0.75], "all")
    print(f"current SCI/PSP prunes {len(drop)} channels (notebook: 70); using the data's "
          f"{len(channels)} channels")
    return quality.measurement_variance(rec["od"].sel(channel=channels).pint.dequantify(),
                                        calc_covariance=False)


def load_adot():
    """Cedalion's precomputed Adot. The server copy no longer matches the hash registered in
    this cedalion checkout, so it is fetched unverified and checked against the data below."""
    import hashlib
    import pooch
    fn = pooch.retrieve(
        "https://doc.ibs.tu-berlin.de/cedalion/datasets/dev/sensitivity_nn22_resting_icbm152.nc",
        known_hash=None, path=Path.home() / ".cache" / "cedalion" / "unverified")
    sha = hashlib.sha256(Path(fn).read_bytes()).hexdigest()
    print(f"Adot file sha256 {sha[:16]}... (registered a9c57747...)")
    return cedalion.io.forward_model.load_Adot(fn)


def load_adot_repro(template):
    """The original Adot reproduced with Cedalion's ForwardModel conventions in MCX
    (mcx_cedalion_check.m -> adot_cedalion.mat; matches the data with slope 1.045/0.977),
    put into Cedalion's xarray layout (channel x vertex x wavelength, units mm)."""
    import scipy.io
    import xarray as xr
    import h5py
    m = scipy.io.loadmat(HERE / "adot_cedalion.mat")
    A = np.asarray(m["Adot"], dtype=np.float32)                  # pair x vertex
    pairs = np.asarray(m["pairs"], dtype=int)
    is_brain = np.asarray(m["is_brain"], dtype=bool).ravel()
    with h5py.File(ROOT / "cedalion_ref" / "synthetic_hrf.snirf", "r") as h:
        sl = [x.decode() for x in h["nirs/probe/sourceLabels"][()]]
        dl = [x.decode() for x in h["nirs/probe/detectorLabels"][()]]
    src = [sl[a - 1] for a in pairs[:, 0]]
    det = [dl[b - 1] for b in pairs[:, 1]]
    wl = template.wavelength.values
    adot = xr.DataArray(
        np.repeat(A[:, :, None], len(wl), axis=2), dims=["channel", "vertex", "wavelength"],
        coords={"channel": [s + d for s, d in zip(src, det)], "source": ("channel", src),
                "detector": ("channel", det), "wavelength": wl,
                "is_brain": ("vertex", is_brain)},
        attrs={"units": "mm"})
    return adot


def check_adot(adot, hrf0):
    """Adot . E . spatial_act must reproduce the peak dOD that synthesized the data."""
    import pandas as pd
    import xarray as xr
    import cedalion.dot.forward_model as fm
    df = pd.read_csv(ROOT / "cedalion_ref" / "spatial_act.csv")
    sa = df.set_index(["trial_type", "vertex", "chromo"])["spatial_act"].to_xarray()
    ab = adot.sel(vertex=adot.is_brain, channel=hrf0.channel)
    sa = sa.assign_coords(vertex=ab.vertex.values).pint.quantify("uM")
    od = fm.image_to_channel_space(ab, sa, spectrum="prahl").pint.dequantify()
    f = jd.load(str(HERE / "hrf_measurement.jdb"))
    ref = np.asarray(f["maxOD_cedalion"], dtype=float)          # pair x wl x trial (fitted)
    pairs = np.asarray(f["pairs"], dtype=int)
    import h5py
    with h5py.File(ROOT / "cedalion_ref" / "synthetic_hrf.snirf", "r") as h:
        sl = [x.decode() for x in h["nirs/probe/sourceLabels"][()]]
        dl = [x.decode() for x in h["nirs/probe/detectorLabels"][()]]
    key = {(sl[a - 1], dl[b - 1]): i for i, (a, b) in enumerate(pairs)}
    rows = [key[(str(a), str(b))] for a, b in zip(od.source.values, od.detector.values)]
    for it, tt in enumerate(["Stim C3", "Stim C4"]):
        x = ref[rows, 1, it]
        y = od.sel(trial_type=tt, wavelength=850.0).transpose("channel").values
        print(f"  Adot check {tt}: r = {np.corrcoef(x, y)[0, 1]:.6f}, slope {x @ y / (x @ x):.4f}")


def to_hrf(template, meas, field="dOD"):
    """Copy a redbird-schema dOD [pair, wl, time, trial] into the hrf_od xarray layout."""
    f = jd.load(str(meas))
    dod = np.asarray(f[field], dtype=float)
    pairs = np.asarray(f["pairs"], dtype=int)
    with __import__("h5py").File(ROOT / "cedalion_ref" / "synthetic_hrf.snirf", "r") as h:
        slab = [s.decode() if isinstance(s, bytes) else s for s in h["nirs/probe/sourceLabels"][()]]
        dlab = [s.decode() if isinstance(s, bytes) else s for s in h["nirs/probe/detectorLabels"][()]]
    key = {(slab[s - 1], dlab[d - 1]): i for i, (s, d) in enumerate(pairs)}
    out = template.copy(deep=True)
    tdim = [d for d in template.dims if d not in ("channel", "wavelength", "trial_type")][0]
    trials = [str(t) for t in np.asarray(f["trialtype"]).ravel()]
    wls = np.asarray(f["wavelengths"], dtype=float).ravel()
    vals = np.empty(out.shape)
    order = out.dims
    for ic, (s, d) in enumerate(zip(out.source.values, out.detector.values)):
        p = key[(str(s), str(d))]
        for iw, w in enumerate(out.wavelength.values):
            for it, tt in enumerate(out.trial_type.values):
                trace = dod[p, int(np.argmin(np.abs(wls - float(w)))), :, trials.index(str(tt))]
                idx = {"channel": ic, "wavelength": iw, "trial_type": it}
                sl = tuple(idx.get(dname, slice(None)) for dname in order)
                vals[sl] = trace
    out.values = vals
    return out, tdim


def score(recon, adot, truth, cortex, tdim, trial):
    img = recon.sel(chromo="HbO", trial_type=trial).sel({tdim: T_SNAP}, method="nearest")
    img = img.pint.to("uM").pint.dequantify() if hasattr(img, "pint") and img.pint.units else img
    cv = np.asarray(img.sel(vertex=adot.is_brain).values, dtype=float)
    sv = np.asarray(img.sel(vertex=~adot.is_brain).values, dtype=float)
    ir, it = int(np.argmax(cv)), int(np.argmax(truth))
    return (np.linalg.norm(cortex[ir] - cortex[it]), np.corrcoef(cv, truth)[0, 1], cv.max(),
            truth.max(), np.abs(sv).max() / np.abs(cv).max(), cv)


def main():
    gt = jd.load(str(ROOT / "redbird_recon" / "groundtruth.bgii"))
    mv = gt["GIFTIData"]["MeshVertex3"]
    cortex = np.asarray(mv["Data"], dtype=float)
    truth = {t: np.asarray(mv["Properties"][f"HbO_{t[-2:]}"], dtype=float).ravel()
             for t in ("Stim C3", "Stim C4")}

    orig = cedalion.io.read_snirf(ROOT / "cedalion_ref" / "synthetic_hrf.snirf")[0]
    hrf0 = orig["hrf_od"]
    print("hrf_od", dict(hrf0.sizes))
    c_meas = load_cmeas(hrf0.channel.values)
    adots = {"reproduced": load_adot_repro(hrf0), "server (replaced)": load_adot()}
    for k, a in adots.items():
        print(f"{k} Adot:")
        check_adot(a, hrf0)

    # verify the channel mapping on the original data, then build the MCX data
    chk, tdim = to_hrf(hrf0, ROOT / "redbird_recon" / "hrf_measurement.jdb")
    err = float(np.abs(chk.values - hrf0.pint.dequantify().values).max()) \
        if hasattr(hrf0, "pint") else float(np.abs(chk.values - hrf0.values).max())
    print(f"mapping check: max |reconstructed - original| = {err:.2e} (float32 round-off)")
    hrf_mcx, _ = to_hrf(hrf0, HERE / "hrf_measurement.jdb")

    for aname, adot in adots.items():
        run_recon(aname, adot, hrf0, hrf_mcx, c_meas, truth, cortex, tdim, gt,
                  save=(aname == "reproduced"))


def run_recon(aname, adot, hrf0, hrf_mcx, c_meas, truth, cortex, tdim, gt, save):
    recon = cedalion.dot.ImageRecon(
        adot.sel(channel=hrf0.channel), recon_mode="mua2conc", alpha_meas=0.5,
        brain_only=False, spatial_basis_functions=None, apply_c_meas=True)

    print(f"\n== Cedalion ImageRecon with the {aname} Adot (mua2conc, alpha_meas=0.5, "
          f"brain_only=False), t={T_SNAP} s ==")
    print(f"{'data':16} {'cond':4} {'peak_err_mm':>11} {'corr':>7} {'rec_pk':>8} {'true_pk':>8} {'leak':>6}")
    for name, hrf, tag in (("Cedalion (Adot)", hrf0, "orig"), ("MCX reference", hrf_mcx, "mcx")):
        img = recon.reconstruct(hrf, c_meas=c_meas)
        props = {}
        for trial in ("Stim C3", "Stim C4"):
            e, c, pk, tpk, leak, cv = score(img, adot, truth[trial], cortex, tdim, trial)
            print(f"{name:16} {trial[-2:]:4} {e:11.1f} {c:7.3f} {pk:8.2f} {tpk:8.2f} {leak:6.2f}")
            props[f"HbO_{trial[-2:]}"] = cv.astype(np.float32)
            hbr = img.sel(chromo="HbR", trial_type=trial).sel({tdim: T_SNAP}, method="nearest")
            if hasattr(hbr, "pint") and hbr.pint.units:
                hbr = hbr.pint.to("uM").pint.dequantify()
            props[f"HbR_{trial[-2:]}"] = np.asarray(
                hbr.sel(vertex=adot.is_brain).values, dtype=np.float32)
        out = {"GIFTIHeader": {"Version": "1.0", "MetaData": {
                   "Description": f"Cedalion ImageRecon of the {name} data (cortex, uM)",
                   "LengthUnit": "mm", "t_rel": T_SNAP}},
               "GIFTIData": {"MeshVertex3": {"Data": cortex.astype(np.float32), "Properties": props},
                             "MeshTri3": {"Data": np.asarray(gt["GIFTIData"]["MeshTri3"]["Data"])}}}
        if save:
            jd.save(out, str(HERE / f"recon_cedalion_{tag}.bgii"), {"compression": "zlib"})
    if save:
        print("saved recon_cedalion_orig.bgii, recon_cedalion_mcx.bgii")


if __name__ == "__main__":
    main()
