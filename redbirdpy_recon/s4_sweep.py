#!/usr/bin/env python3
"""s4_sweep.py -- redbirdpy port of redbird_recon/s4_sweep.m.

Using the cached two-surface operator (s4_op.jdb, built by s4_recon_surface.py),
sweep depth weighting (beta) and Tikhonov (alpha) to trade off scalp-leakage
suppression against cortical localization. No forward solve is run.

Usage:  python3 s4_sweep.py
"""

import jdata as jd
import numpy as np
import redbirdpy as rb

import nirs_common as nc


def main():
    cachef = nc.HERE / "s4_op.jdb"
    if not cachef.exists():
        raise SystemExit("missing %s; run s4_recon_surface.py first" % cachef.name)
    S = jd.loadjd(str(cachef))
    Anorm, cmeas = np.asarray(S["Anorm"]), np.asarray(S["cmeas"])
    ncortex, nscalp = int(S["ncortex"]), int(S["nscalp"])
    Nsurf = ncortex + nscalp

    meas = nc.load_meas()
    cortex, _, props, _ = nc.load_truth()
    pairs, wlkey, dOD, tidx = meas["pairs"], meas["wlkey"], meas["dOD"], meas["tidx"]
    npair, nwl = pairs.shape[0], len(wlkey)
    trC3, trC4 = props["HbO_C3"], props["HbO_C4"]
    itC3, itC4 = np.argmax(trC3), np.argmax(trC4)

    wrow = 1.0 / np.sqrt(cmeas + 1e-6 * cmeas.max())
    rhsC3 = np.zeros(npair * nwl)
    rhsC4 = np.zeros(npair * nwl)
    for w in range(nwl):
        rhsC3[w * npair : (w + 1) * npair] = -dOD[:, w, tidx, 0]
        rhsC4[w * npair : (w + 1) * npair] = -dOD[:, w, tidx, 1]

    print(
        "%6s %6s | %-7s %-7s %-6s | %-7s %-7s %-6s"
        % ("alpha", "beta", "errC3", "corrC3", "leakC3", "errC4", "corrC4", "leakC4")
    )
    for alpha in (0.003, 0.01, 0.05):
        for beta in (1, 0.1, 0.03, 0.01, 0.003):
            Aw = wrow[:, None] * Anorm
            cn = np.sqrt(np.sum(Aw**2, axis=0))
            Wc = 1.0 / (cn + beta * cn.max())
            Aw = Aw * Wc[None, :]
            lam = alpha * np.max(np.sum(Aw**2, axis=1))
            out = []
            for rhs, it, tr in ((rhsC3, itC3, trC3), (rhsC4, itC4, trC4)):
                d = rb.reginv(Aw, wrow * rhs, lam) * Wc
                h = d[:Nsurf]
                cv, sv = h[:ncortex], h[ncortex:]
                ir = np.argmax(cv)
                cc = np.corrcoef(cv, tr)[0, 1]
                out += [
                    np.linalg.norm(cortex[ir] - cortex[it]),
                    cc,
                    np.max(np.abs(sv)) / np.max(np.abs(cv)),
                ]
            print(
                "%6.3g %6.3g | %7.1f %7.3f %6.2f | %7.1f %7.3f %6.2f"
                % (alpha, beta, out[0], out[1], out[2], out[3], out[4], out[5])
            )


if __name__ == "__main__":
    main()
