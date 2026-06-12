#!/usr/bin/env python3
"""s4_recon_gather.py -- two-surface DOT recon with NON-conservative GATHER.

Companion to s4_recon_surface.py. Same two-surface (cortex + scalp) unknown
space, same Rytov / c_meas / depth-weighting / Tikhonov inversion -- the ONLY
difference is how the nodal forward Jacobian is mapped onto the surface vertices:

  * s4_recon_surface.py (conservative SUM): each forward node -> its nearest
    surface vertex, columns SUMMED (total sensitivity conserved). Smoother,
    lower recovered peak amplitude.

  * THIS unit (non-conservative GATHER): each surface vertex <- its single
    nearest forward node (``J[:, snode]``). One node sampled per vertex (not
    conserved, aliasing-prone), but it yields a HIGHER recovered contrast /
    sharper peaks -- the original aggregation before the conservative version.

Writes recon_gather.bgii + recon_gather.png and caches the operator in
s4_op_gather.jdb (so it does not clobber the conservative s4 outputs).

Usage:  python3 s4_recon_gather.py [--alpha 0.003] [--beta 0.003]
"""

import argparse

import iso2mesh as i2m
import jdata as jd
import numpy as np
import redbirdpy as rb

import nirs_common as nc


def build_operator(meas, Vsurf, ncortex, nscalp, cachef):
    """Build (or load) the cached two-surface Rytov operator (GATHER mapping)."""
    if cachef.exists():
        print("loading cached surface operator", cachef.name)
        S = jd.loadjd(str(cachef))
        return np.asarray(S["Anorm"]), np.asarray(S["cmeas"])

    pairs, wlkey = meas["pairs"], meas["wlkey"]
    npair, nwl = pairs.shape[0], len(wlkey)

    node, elem = nc.load_mesh()
    face, _ = i2m.volface(elem[:, :4].astype(int))
    nv = i2m.nodesurfnorm(node, face)
    srcpos, srcdir = nc.snap2surf(node, face, nv, meas["srcpos"])
    detpos, detdir = nc.snap2surf(node, face, nv, meas["detpos"])

    cfg, _ = nc.make_cfg(node, elem, srcpos, srcdir, detpos, detdir, wlkey)
    ns = srcpos.shape[0]
    sd = rb.sdmap(cfg)
    for key in sd:  # activate every pair
        sd[key][:, 2] = 1
    detphi, phi = nc.baseline_forward(cfg, sd)

    # non-conservative GATHER: each surface vertex takes its single NEAREST
    # forward node's nodal-Jacobian column (one node sampled per vertex). Sharper
    # peaks than the conservative SUM, but total sensitivity is not conserved.
    snode = nc.nearestnodes(node, Vsurf)  # nearest forward node per surf vertex
    sdmeas = nc.measured_sd(pairs, ns)
    Js = {}
    for key in wlkey:
        Jw, _ = rb.jac(sdmeas, phi[key], cfg["deldotdel"], cfg["elem"], cfg["evol"])
        Js[key] = Jw[:, snode]  # [npair x Nsurf] (gathered)
        del Jw

    Amat = rb.matflat(rb.jacchrome(Js, ["hbo", "hbr"]))  # [npair*nwl x 2*Nsurf]
    ymodel = nc.stack_ymodel(detphi, pairs, wlkey, npair)
    Anorm = Amat / ymodel[:, None]  # Rytov (OD) operator

    # c_meas: per-(pair,wl) noise variance from the pre-stimulus baseline (reltime<0)
    pre = meas["taxis"] < 0
    cmeas = np.zeros(npair * nwl)
    for w in range(nwl):
        v = np.var(meas["dOD"][:, w, pre, :].reshape(npair, -1), axis=1, ddof=0)
        cmeas[w * npair : (w + 1) * npair] = v

    jd.savejd(
        {"Anorm": Anorm, "cmeas": cmeas, "ncortex": ncortex, "nscalp": nscalp},
        str(cachef),
        compression="zlib",
    )
    print("cached surface operator", cachef.name)
    return Anorm, cmeas


def main(alpha=0.003, beta=0.003):
    meas = nc.load_meas()
    cortex, faces, props, gt = nc.load_truth()
    ncortex = cortex.shape[0]
    scalp = nc.load_scalp()
    nscalp = scalp.shape[0]
    Vsurf = np.vstack([cortex, scalp])
    Nsurf = ncortex + nscalp
    print(
        "two-surface unknowns (GATHER): %d cortex + %d scalp = %d"
        % (ncortex, nscalp, Nsurf)
    )

    Anorm, cmeas = build_operator(
        meas, Vsurf, ncortex, nscalp, nc.HERE / "s4_op_gather.jdb"
    )

    # weighting + Tikhonov inverse operator (identical to s4_recon_surface.py)
    wrow = 1.0 / np.sqrt(cmeas + 1e-6 * cmeas.max())  # c_meas whitening
    Aw = wrow[:, None] * Anorm
    cn = np.sqrt(np.sum(Aw**2, axis=0))
    Wc = 1.0 / (cn + beta * cn.max())  # depth (column) weighting
    Aw = Aw * Wc[None, :]
    lam = alpha * np.max(np.sum(Aw**2, axis=1))
    print("alpha=%.3g beta=%.3g lambda=%.3e" % (alpha, beta, lam))

    pairs, wlkey, dOD, tidx = meas["pairs"], meas["wlkey"], meas["dOD"], meas["tidx"]
    npair, nwl = pairs.shape[0], len(wlkey)
    targets = [
        ("C3", dOD[:, :, tidx, 0]),
        ("C4", dOD[:, :, tidx, 1]),
        ("C3-C4", dOD[:, :, tidx, 0] - dOD[:, :, tidx, 1]),
    ]
    hbo, hbr = {}, {}
    for tag, d in targets:
        rhs = np.zeros(npair * nwl)
        for w in range(nwl):
            rhs[w * npair : (w + 1) * npair] = -d[:, w]
        z = rb.reginv(Aw, wrow * rhs, lam)
        dconc = z * Wc
        h = dconc[:Nsurf]
        hbo[tag], hbr[tag] = h, dconc[Nsurf:]
        cv, sv = h[:ncortex], h[ncortex:]
        print(
            "%-6s cortex dHbO[%.2f %.2f] scalp dHbO[%.2f %.2f]  leak=%.2f"
            % (
                tag,
                cv.min(),
                cv.max(),
                sv.min(),
                sv.max(),
                np.max(np.abs(sv)) / np.max(np.abs(cv)),
            )
        )

    # save the CORTEX result to recon_gather.bgii
    out = {
        "GIFTIHeader": {
            "Version": "1.0",
            "MetaData": {
                "Description": "redbirdpy two-surface DOT recon, GATHER aggregation (cortex dHbO/dHbR, uM)",
                "LengthUnit": "mm",
                "alpha": alpha,
                "beta": beta,
            },
        },
        "GIFTIData": {
            "MeshVertex3": {
                "Data": cortex.astype(np.float32),
                "Properties": {
                    "HbO_C3": hbo["C3"][:ncortex].astype(np.float32),
                    "HbR_C3": hbr["C3"][:ncortex].astype(np.float32),
                    "HbO_C4": hbo["C4"][:ncortex].astype(np.float32),
                    "HbR_C4": hbr["C4"][:ncortex].astype(np.float32),
                },
            },
            "MeshTri3": {"Data": faces.astype(np.int32)},
        },
    }
    jd.savejd(out, str(nc.DATA / "recon_gather.bgii"), compression="zlib")
    print("saved recon_gather.bgii (two-surface cortex result, GATHER)")

    # score cortex vs truth
    for tag in ("C3", "C4"):
        cv = hbo[tag][:ncortex]
        tr = props["HbO_%s" % tag]
        ir, it = np.argmax(cv), np.argmax(tr)
        cc = np.corrcoef(cv, tr)[0, 1]
        print(
            "score %s: peak-err %.1f mm, corr %.3f, pk %.2f (truth %.2f)"
            % (tag, np.linalg.norm(cortex[ir] - cortex[it]), cc, cv.max(), tr.max())
        )

    render(cortex, faces, props, hbo, ncortex)


def render(cortex, faces, props, hbo, ncortex):
    """Cortex truth vs redbird-gather, 6-panel comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = nc.diverging_cmap()
    panels = [
        (props["HbO_C3"], "truth C3", (-90, 0)),
        (props["HbO_C4"], "truth C4", (90, 0)),
        (props["HbO_C3"] - props["HbO_C4"], "truth C3-C4", (0, 90)),
        (hbo["C3"][:ncortex], "redbirdpy C3 (gather)", (-90, 0)),
        (hbo["C4"][:ncortex], "redbirdpy C4 (gather)", (90, 0)),
        (hbo["C3-C4"][:ncortex], "redbirdpy C3-C4 (gather)", (0, 90)),
    ]
    fig = plt.figure(figsize=(15, 9))
    for i, (val, ttl, view) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        nc.plot_surface(
            ax,
            cortex,
            faces,
            val,
            view,
            cmap=cmap,
            title="%s (pk %.2f)" % (ttl, np.max(val)),
        )
    fig.tight_layout()
    fig.savefig(nc.FIG / "recon_gather.png", dpi=110)
    print("wrote recon_gather.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.003)
    ap.add_argument("--beta", type=float, default=0.003)
    args = ap.parse_args()
    main(args.alpha, args.beta)
