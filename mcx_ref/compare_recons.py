#!/usr/bin/env python3
"""Compare reconstructed cortex HbO maps (t_rel = 10 s) between solvers, per dataset.

Reference: Cedalion's ImageRecon (cedalion_recon.py: recon_cedalion_{orig,mcx}.bgii).
For each redbird recon: vertex-wise Pearson r vs Cedalion, distance between the two
peaks, peak-amplitude ratio, and each map's peak error vs the ground truth.

Usage: python3 compare_recons.py
"""

from pathlib import Path

import jdata as jd
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

RECONS = {
    "orig": [
        # written by whichever of s4_recon_surface.m / .py ran last (same scores)
        ("redbird(py) s4 (voxel agg)", ROOT / "redbird_recon" / "recon.bgii"),
        ("redbird matched (Cedalion optics)", ROOT / "redbird_recon" / "recon_matched_redbird_recon.bgii"),
    ],
    "mcx": [
        ("redbird matched + Cedalion row scale", HERE / "recon_mcx.bgii"),
        ("redbird matched (Cedalion optics)", ROOT / "redbird_recon" / "recon_matched_mcx_ref.bgii"),
    ],
}


def load(fname):
    g = jd.load(str(fname))
    mv = g["GIFTIData"]["MeshVertex3"]
    return np.asarray(mv["Data"], float), {k: np.asarray(v, float).ravel()
                                           for k, v in mv["Properties"].items()}


def main():
    cortex, truth = load(ROOT / "redbird_recon" / "groundtruth.bgii")
    for data, items in RECONS.items():
        _, ced = load(HERE / f"recon_cedalion_{data}.bgii")
        print(f"\n== {data} data: vs Cedalion ImageRecon (HbO, t=10 s) ==")
        print(f"{'recon':36} {'cond':4} {'r_vs_ced':>8} {'peak_dist_mm':>12} {'amp_ratio':>9} "
              f"{'err_truth_mm':>12} {'ced_err_mm':>10}")
        for name, fn in items:
            if not fn.exists():
                print(f"{name:36} (missing {fn.name})")
                continue
            _, rec = load(fn)
            for c in ("C3", "C4"):
                a, b, t = rec[f"HbO_{c}"], ced[f"HbO_{c}"], truth[f"HbO_{c}"]
                r = np.corrcoef(a, b)[0, 1]
                pa, pb, pt = int(np.argmax(a)), int(np.argmax(b)), int(np.argmax(t))
                print(f"{name:36} {c:4} {r:8.3f} {np.linalg.norm(cortex[pa] - cortex[pb]):12.1f} "
                      f"{a.max() / b.max():9.2f} {np.linalg.norm(cortex[pa] - cortex[pt]):12.1f} "
                      f"{np.linalg.norm(cortex[pb] - cortex[pt]):10.1f}")


if __name__ == "__main__":
    main()
