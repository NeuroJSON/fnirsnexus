#!/usr/bin/env python3
"""s2_recon_gm.py -- redbirdpy port of redbird_recon/s2_recon_gm.m.

Linear (single Gauss-Newton step) DOT image reconstruction, porting Cedalion's
ImageRecon step. A single GN iteration == a linear reconstruction; to match
Cedalion's OD-domain linear operator on a SPARSE fNIRS montage the operator is
built by hand from the low-level redbirdpy primitives (rather than the
high-level recon path, whose detector extraction assumes a dense src x det grid):

  1. load FEM head mesh + HRF dOD
  2. project optodes onto the scalp; 5-tissue baseline
  3. baseline forward solve -> detphi_base, fluence phi
  4. rb.jac -> per-wavelength mua Jacobian for the measured pairs; keep only the
     gray-matter (label 4) node columns; rb.jacchrome -> HbO/HbR Jacobian
  5. Rytov normalize each (wavelength,pair) row by detphi_base => the OD
     sensitivity (== Cedalion's Adot). Solve Anorm*[dHbO;dHbR] = -dOD with
     Tikhonov + depth weighting (rb.reginv). Anorm is built once, applied to all
     conditions.
  6. map node-wise dHbO/dHbR onto the 25k ICBM cortex surface, save binary JGIfTI

NOTE: the GM-only restriction is a deliberate redbird SIMPLIFICATION to suppress
the superficial bias; it does NOT match Cedalion (which uses cortex+scalp, see
s4_recon_surface.py).

Usage:  python3 s2_recon_gm.py [--peaktime 10.0] [--alpha 0.01] [--beta 0.01]
"""

import argparse

import iso2mesh as i2m
import jdata as jd
import numpy as np
import redbirdpy as rb

import nirs_common as nc


def build_operator(meas, cachef):
    """Build (or load) the cached linear OD->dHbO/dHbR operator on the GM nodes."""
    if cachef.exists():
        print("loading cached recon operator:", cachef.name)
        S = jd.loadjd(str(cachef))
        return (
            np.asarray(S["Anorm"]),
            np.asarray(S["rnode"]),
            int(S["nrec"]),
            int(S["npair"]),
            int(S["nwl"]),
        )

    pairs, wlkey = meas["pairs"], meas["wlkey"]
    npair, nwl = pairs.shape[0], len(wlkey)

    node, elem = nc.load_mesh()
    nseg = int(elem[:, 4].max())
    print("mesh: %d nodes, %d tets, %d regions" % (node.shape[0], elem.shape[0], nseg))
    face, _ = i2m.volface(elem[:, :4].astype(int))
    nv = i2m.nodesurfnorm(node, face)
    srcpos, srcdir = nc.snap2surf(node, face, nv, meas["srcpos"])
    detpos, detdir = nc.snap2surf(node, face, nv, meas["detpos"])

    cfg, _ = nc.make_cfg(node, elem, srcpos, srcdir, detpos, detdir, wlkey)
    ns = srcpos.shape[0]
    sd = rb.sdmap(cfg)
    for key in sd:
        sd[key][:, 2] = 1
    detphi, phi = nc.baseline_forward(cfg, sd)

    # gray-matter (label 4) nodes are the reconstruction unknowns
    gmnodes0 = np.unique(elem[elem[:, 4] == 4, :4].astype(int)) - 1  # 0-based
    rnode = node[gmnodes0, :]
    nrec = gmnodes0.size
    sdmeas = nc.measured_sd(pairs, ns)
    Jr = {}
    for key in wlkey:
        Jw, _ = rb.jac(sdmeas, phi[key], cfg["deldotdel"], cfg["elem"], cfg["evol"])
        Jr[key] = Jw[:, gmnodes0]  # keep GM columns only
        del Jw
    print(
        "Jacobian (GM only): Jr size %s of %d fwd nodes"
        % (Jr[wlkey[0]].shape, node.shape[0])
    )

    Amat = rb.matflat(rb.jacchrome(Jr, ["hbo", "hbr"]))  # [npair*nwl x 2*nrec]
    ymodel = nc.stack_ymodel(detphi, pairs, wlkey, npair)
    Anorm = Amat / ymodel[:, None]  # Rytov: d log(detphi)/d conc

    jd.savejd(
        {"Anorm": Anorm, "rnode": rnode, "nrec": nrec, "npair": npair, "nwl": nwl},
        str(cachef),
        compression="zlib",
    )
    print("cached recon operator:", cachef.name)
    return Anorm, rnode, nrec, npair, nwl


def main(peaktime=10.0, alpha=0.01, beta=0.01):
    meas = nc.load_meas(peaktime)
    tidx = meas["tidx"]
    print("reconstructing at reltime=%.2f s (index %d)" % (meas["taxis"][tidx], tidx))

    Anorm, rnode, nrec, npair, nwl = build_operator(meas, nc.HERE / "recon_gm_op.jdb")

    # depth (column) weighting: rescale columns by their sensitivity norm so deep,
    # low-sensitivity nodes are not suppressed (Cedalion alpha_spatial analog).
    cn = np.sqrt(np.sum(Anorm**2, axis=0))
    W = 1.0 / (cn + beta * cn.max())
    Aw = Anorm * W[None, :]
    lam = alpha * np.max(np.sum(Aw**2, axis=1))
    print("alpha=%.3g beta=%.3g -> lambda=%.3e" % (alpha, beta, lam))

    dOD, trials = meas["dOD"], meas["trials"]
    ntrial = len(trials)
    recon_hbo = np.zeros((nrec, ntrial))
    recon_hbr = np.zeros((nrec, ntrial))
    for t in range(ntrial):
        rhs = np.zeros(npair * nwl)
        for w in range(nwl):
            rhs[w * npair : (w + 1) * npair] = -dOD[:, w, tidx, t]
        dconc = rb.reginv(Aw, rhs, lam) * W
        recon_hbo[:, t], recon_hbr[:, t] = dconc[:nrec], dconc[nrec:]
        print(
            "%-8s ||dOD||=%.3e  dHbO range [%.3f %.3f] uM"
            % (
                trials[t],
                np.linalg.norm(rhs),
                recon_hbo[:, t].min(),
                recon_hbo[:, t].max(),
            )
        )

    # map recon-mesh result onto the 25k cortex, save as JGIfTI
    cortex, faces, _, _ = nc.load_truth()
    cnode = nc.nearestnodes(rnode, cortex)
    props = {}
    for t in range(ntrial):
        tag = trials[t].replace("Stim ", "").replace(" ", "")  # C3 / C4
        props["HbO_%s" % tag] = recon_hbo[cnode, t].astype(np.float32)
        props["HbR_%s" % tag] = recon_hbr[cnode, t].astype(np.float32)
    out = {
        "GIFTIHeader": {
            "Version": "1.0",
            "MetaData": {
                "Description": "redbirdpy linear DOT reconstruction (dHbO/dHbR uM)",
                "LengthUnit": "mm",
                "ReconTime_s": float(meas["taxis"][tidx]),
            },
        },
        "GIFTIData": {
            "MeshVertex3": {"Data": cortex.astype(np.float32), "Properties": props},
            "MeshTri3": {"Data": faces.astype(np.int32)},
        },
    }
    jd.savejd(out, str(nc.DATA / "recon_gm.bgii"), compression="zlib")
    print("saved recon_gm.bgii")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaktime", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.01)
    args = ap.parse_args()
    main(args.peaktime, args.alpha, args.beta)
