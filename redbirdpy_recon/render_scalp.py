#!/usr/bin/env python3
"""render_scalp.py -- redbirdpy port of redbird_recon/render_scalp.m.

Render the SCALP-surface portion of the two-surface recon (the superficial
leakage) for C3, C4 and the C3-C4 contrast, using the cached operator
(s4_op.jdb) at alpha=0.003, beta=0.01.

Usage:  python3 render_scalp.py
"""

import jdata as jd
import numpy as np
import redbirdpy as rb

import nirs_common as nc


def main(alpha=0.003, beta=0.01):
    cachef = nc.HERE / "s4_op.jdb"
    if not cachef.exists():
        raise SystemExit("missing %s; run s4_recon_surface.py first" % cachef.name)
    S = jd.loadjd(str(cachef))
    Anorm, cmeas = np.asarray(S["Anorm"]), np.asarray(S["cmeas"])
    ncortex, nscalp = int(S["ncortex"]), int(S["nscalp"])

    meas = nc.load_meas()
    scalp, sf = nc.load_scalp(withfaces=True)
    pairs, wlkey, dOD, tidx = meas["pairs"], meas["wlkey"], meas["dOD"], meas["tidx"]
    npair, nwl = pairs.shape[0], len(wlkey)

    wrow = 1.0 / np.sqrt(cmeas + 1e-6 * cmeas.max())
    Aw = wrow[:, None] * Anorm
    cn = np.sqrt(np.sum(Aw**2, axis=0))
    Wc = 1.0 / (cn + beta * cn.max())
    Aw = Aw * Wc[None, :]
    lam = alpha * np.max(np.sum(Aw**2, axis=1))

    targets = [
        ("C3", dOD[:, :, tidx, 0]),
        ("C4", dOD[:, :, tidx, 1]),
        ("C3-C4", dOD[:, :, tidx, 0] - dOD[:, :, tidx, 1]),
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = nc.diverging_cmap()
    fig = plt.figure(figsize=(15, 5.2))
    for j, (tag, d) in enumerate(targets):
        rhs = np.zeros(npair * nwl)
        for w in range(nwl):
            rhs[w * npair : (w + 1) * npair] = -d[:, w]
        dconc = rb.reginv(Aw, wrow * rhs, lam) * Wc
        scalp_h = dconc[ncortex : ncortex + nscalp]
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        nc.plot_surface(
            ax,
            scalp,
            sf[:, :3],
            scalp_h,
            (0, 90),
            cmap=cmap,
            title="scalp leakage %s (pk %.2f uM)" % (tag, scalp_h.max()),
        )
    fig.tight_layout()
    fig.savefig(nc.FIG / "recon_scalp.png", dpi=110)
    print("wrote recon_scalp.png")


if __name__ == "__main__":
    main()
