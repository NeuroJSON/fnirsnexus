#!/usr/bin/env python3
"""render_lateral.py -- redbirdpy port of redbird_recon/render_lateral.m.

Render the redbird cortical dHbO recon in the same lateral views and diverging
color scale Cedalion uses, for side-by-side comparison: truth vs redbird, C3
(left lateral) and C4 (right lateral). Reads groundtruth.bgii + recon.bgii from
../redbird_recon.

Usage:  python3 render_lateral.py
"""

import jdata as jd
import numpy as np

import nirs_common as nc


def main():
    gt = jd.loadjd(str(nc.DATA / "groundtruth.bgii"))
    rc = jd.loadjd(str(nc.DATA / "recon.bgii"))
    node = np.asarray(gt["GIFTIData"]["MeshVertex3"]["Data"], dtype=float)
    faces = np.asarray(gt["GIFTIData"]["MeshTri3"]["Data"], dtype=int)
    G = gt["GIFTIData"]["MeshVertex3"]["Properties"]
    R = rc["GIFTIData"]["MeshVertex3"]["Properties"]

    # panels: (value, title, view([az el]), fixed clim or None=autoscale)
    panels = [
        (np.asarray(G["HbO_C3"], float).ravel(), "truth C3", (-90, 0), 5.0),
        (np.asarray(R["HbO_C3"], float).ravel(), "redbird C3", (-90, 0), None),
        (np.asarray(G["HbO_C4"], float).ravel(), "truth C4", (90, 0), 5.0),
        (np.asarray(R["HbO_C4"], float).ravel(), "redbird C4", (90, 0), None),
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = nc.diverging_cmap()
    fig = plt.figure(figsize=(12, 9.5))
    for i, (val, ttl, view, cm) in enumerate(panels):
        clim = (-cm, cm) if cm else None
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        nc.plot_surface(
            ax,
            node,
            faces,
            val,
            view,
            clim=clim,
            cmap=cmap,
            title="%s  (pk %.2f uM)" % (ttl, val.max()),
        )
    fig.tight_layout()
    fig.savefig(nc.FIG / "recon_lateral.png", dpi=110)
    print("wrote recon_lateral.png")


if __name__ == "__main__":
    main()
