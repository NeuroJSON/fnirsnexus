#!/usr/bin/env python3
"""s3_recon_gm_iter.py -- redbirdpy port of redbird_recon/s3_recon_gm_iter.m.

Iterative Gauss-Newton fNIRS reconstruction through the OFFICIAL redbirdpy path
(``rb.run`` -> ``runrecon``) using the recon["isratio"] flag + Rytov
(reform='logphase').

Unlike s2 (a single fixed linear step) this updates the optical properties each
iteration and re-runs the forward + Jacobian, so it reports a genuine
multi-iteration residual curve. Homogeneous baseline (node-wise) + gray-matter
reconstruction submesh.

Usage:  python3 s3_recon_gm_iter.py [--trial 2] [--maxiter 4]
        (trial: 1=Stim C3, 2=Stim C4)

NOTE: the official runrecon path materializes the FULL dense nodal Jacobian on
the 144k-node forward mesh (both wavelengths) before remapping to the GM submesh,
so it needs substantial RAM (~16-24 GB free). s2/s4 are the memory-light
equivalents -- they stream the Jacobian one wavelength at a time and aggregate
immediately. Added recon["isratio"] support to redbirdpy/recon.py (runrecon),
mirroring the redbird-m rbrunrecon change.
"""

import argparse

import iso2mesh as i2m
import numpy as np
import redbirdpy as rb

import nirs_common as nc


def main(trial=2, maxiter=4):
    t = trial - 1  # 0-based trial index
    meas = nc.load_meas()
    tidx = meas["tidx"]
    wlkey, pairs, dOD = meas["wlkey"], meas["pairs"], meas["dOD"]
    print(
        "iterative recon: %s, t=%.2fs, maxiter=%d"
        % (meas["trials"][t], meas["taxis"][tidx], maxiter)
    )

    node, elem = nc.load_mesh()
    Nn = node.shape[0]
    face, _ = i2m.volface(elem[:, :4].astype(int))
    nv = i2m.nodesurfnorm(node, face)
    srcpos, srcdir = nc.snap2surf(node, face, nv, meas["srcpos"])
    detpos, detdir = nc.snap2surf(node, face, nv, meas["detpos"])

    # homogeneous baseline (node-wise param so multi-iter sync keeps every node)
    bhbo, bhbr = 28.0, 14.0
    cfg = {
        "node": node,
        "elem": np.hstack([elem[:, :4], np.ones((elem.shape[0], 1))]).astype(np.int64),
        "seg": np.ones(elem.shape[0], dtype=int),
        "srcpos": srcpos,
        "srcdir": srcdir,
        "detpos": detpos,
        "detdir": detdir,
        "param": {"hbo": bhbo * np.ones(Nn), "hbr": bhbr * np.ones(Nn)},
        "prop": {
            k: np.array([[0, 0, 1, 1], [0.01, 1.0, 0, 1.37]], float) for k in wlkey
        },
        "omega": 0,
    }
    cfg, sd = rb.meshprep(cfg)

    # source-detector map: activate only the measured pairs
    ns, nd = srcpos.shape[0], detpos.shape[0]
    want = {(s - 1, ns + (d - 1)) for s, d in pairs}
    for key in sd:
        tab = sd[key]
        tab[:, 2] = [1 if (int(r[0]), int(r[1])) in want else 0 for r in tab]
        sd[key] = tab

    # ratiometric data I/I0 = exp(-dOD): per-wavelength [Ndet x Nsrc], 1 on unused pairs
    ratio = {}
    for w, key in enumerate(wlkey):
        R = np.ones((nd, ns))
        R[pairs[:, 1] - 1, pairs[:, 0] - 1] = np.exp(-dOD[:, w, tidx, t])
        ratio[key] = R

    # gray-matter (label 4) reconstruction submesh
    gmel = elem[elem[:, 4] == 4, :4].astype(int)
    gmnodes = np.unique(gmel)
    newidx = np.zeros(Nn + 1, dtype=int)
    newidx[gmnodes] = np.arange(1, gmnodes.size + 1)  # 1-based recon node ids
    recon = {
        "node": node[gmnodes - 1, :],
        "elem": newidx[gmel],  # 1-based into recon.node
        "bulk": {"hbo": bhbo, "hbr": bhbr},
        "param": {"hbo": bhbo, "hbr": bhbr},
        "prop": {k: np.empty((0, 4)) for k in wlkey},
        "lambda": 0.1,
        "isratio": 1,  # ratiometric (fNIRS) data
    }
    recon["mapid"], recon["mapweight"] = i2m.tsearchn(
        recon["node"], recon["elem"], cfg["node"]
    )

    newrecon, resid, newcfg = rb.run(
        cfg,
        recon,
        ratio,
        sd,
        mode="image",
        maxiter=maxiter,
        reform="logphase",
        lambda_=recon["lambda"],
    )
    resid = np.asarray(resid).ravel()
    print("residual per iter      :", " ".join("%.4e" % r for r in resid))
    print("relative residual r/r1 :", " ".join("%.4f" % (r / resid[0]) for r in resid))

    dhbo = np.asarray(newrecon["param"]["hbo"]).ravel() - bhbo
    print(
        "final dHbO range [%.3f %.3f] uM on %d GM nodes"
        % (dhbo.min(), dhbo.max(), dhbo.size)
    )

    # map onto cortex and render
    cortex, faces, props, _ = nc.load_truth()
    tag = meas["trials"][t].replace("Stim ", "").replace(" ", "")
    cnode = nc.nearestnodes(recon["node"], cortex)
    cval = dhbo[cnode]
    truth = props["HbO_%s" % tag]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = nc.diverging_cmap()
    view = (90, 0) if tag == "C4" else (-90, 0)
    fig = plt.figure(figsize=(13, 5.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    nc.plot_surface(
        ax,
        cortex,
        faces,
        truth,
        view,
        clim=(-5, 5),
        cmap=cmap,
        title="truth %s (pk %.2f uM)" % (tag, truth.max()),
    )
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    nc.plot_surface(
        ax,
        cortex,
        faces,
        cval,
        view,
        cmap=cmap,
        title="redbirdpy %s, %d iters (pk %.2f uM)" % (tag, maxiter, cval.max()),
    )
    fig.tight_layout()
    fig.savefig(nc.FIG / ("recon_gm_iter_%s.png" % tag), dpi=110)
    print("wrote recon_gm_iter_%s.png" % tag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, default=2, help="1=Stim C3, 2=Stim C4")
    ap.add_argument("--maxiter", type=int, default=4)
    args = ap.parse_args()
    main(args.trial, args.maxiter)
