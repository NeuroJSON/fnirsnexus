#!/usr/bin/env python3
"""Score the redbird reconstruction (recon.bgii) against the ground truth.

Both files are JGIfTI on the same 25k ICBM cortex surface. For each condition
(C3, C4) we compare the reconstructed per-vertex dHbO against the true activation:

  - peak-localization error: distance (mm) between the true and reconstructed
    HbO maxima on the cortex
  - spatial correlation of the HbO maps
  - recovered vs. true peak amplitude

Usage: python3 score_recon.py
"""

from pathlib import Path
import jdata as jd
import numpy as np

HERE = Path(__file__).resolve().parent


def load_props(fname):
    g = jd.load(str(HERE / fname))
    mv = g["GIFTIData"]["MeshVertex3"]
    verts = np.asarray(mv["Data"], dtype=float)
    props = {k: np.asarray(v, dtype=float).ravel() for k, v in mv["Properties"].items()}
    return verts, props


def main():
    tv, tp = load_props("groundtruth.bgii")
    rv, rp = load_props("recon.bgii")
    assert tv.shape == rv.shape, "vertex mismatch between truth and recon"

    print(f"{'cond':6} {'peak_err_mm':>11} {'corr':>7} {'true_pk':>8} {'rec_pk':>8}")
    for tag in ("C3", "C4"):
        true = tp[f"HbO_{tag}"]
        rec = rp.get(f"HbO_{tag}")
        if rec is None:
            print(f"{tag}: missing in recon")
            continue
        it, ir = np.argmax(true), np.argmax(rec)
        derr = np.linalg.norm(tv[it] - rv[ir])
        # correlation over the cortex
        corr = np.corrcoef(true, rec)[0, 1]
        print(f"{tag:6} {derr:11.1f} {corr:7.3f} {true[it]:8.2f} {rec[ir]:8.2f}")


if __name__ == "__main__":
    main()
