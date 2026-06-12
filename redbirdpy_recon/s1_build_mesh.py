#!/usr/bin/env python3
"""s1_build_mesh.py -- redbirdpy port of redbird_recon/s1_build_mesh.m.

Build a tetrahedral ICBM-152 head mesh for the redbirdpy FEM recon. Cedalion's
head model is surface + voxel based (Monte-Carlo forward) and ships NO
tetrahedral mesh; redbirdpy is FEM and needs one. We build it from the 5
ICBM-152 tissue masks (../hm_icbm152) with iso2mesh v2m (CGAL), yielding a full
5-tissue model (1=scalp 2=bone 3=CSF 4=gray 5=white).

Node coordinates are mapped voxel(IJK)->RAS(mm) via the NIfTI affine so the mesh
shares the SNIRF optode frame. Isolated nodes are removed (they create zero rows
-> a singular FEM matrix).

The result is written to ../redbird_recon/head_mesh.bmsh (data files live there).
Requires iso2mesh (v2m + the CGAL meshing binaries) and jdata.

Usage:  python3 s1_build_mesh.py [outfile.bmsh]
"""

import sys

import iso2mesh as i2m
import jdata as jd
import numpy as np

import nirs_common as nc


def main(outfile=None):
    if outfile is None:
        outfile = str(nc.DATA / "head_mesh.bmsh")

    # 1. combine the 5 tissue masks into one labelled volume (inner overwrites outer)
    masknames = ["mask_skin", "mask_bone", "mask_csf", "mask_gray", "mask_white"]
    seg = None
    nii = None
    for i, name in enumerate(masknames, start=1):
        nii = jd.loadjnifti(str(nc.HM / (name + ".nii")))
        vol = np.asarray(nii["NIFTIData"])
        if seg is None:
            seg = np.zeros(vol.shape, dtype=np.uint8)
        seg[vol > 0] = i  # labels 1..5
    print("segmentation %s, labels %s" % (seg.shape, np.unique(seg).tolist()))

    # 2. tetrahedral mesh (CGAL); raise maxvol for a coarser/faster mesh
    opt = {"radbound": 2, "distbound": 1}
    maxvol = 30
    node, elem, face = i2m.v2m(
        seg, np.arange(1, seg.shape[0] + 1), opt, maxvol, method="cgalmesh"
    )

    # 3. drop isolated nodes (reindex only the index columns, keep label/bid)
    elabel = elem[:, 4:5]
    fbid = face[:, 3:4]
    node, e4, f3 = i2m.removeisolatednode(node, elem[:, :4], face[:, :3])
    elem = np.hstack([e4, elabel])
    face = np.hstack([f3, fbid])
    print(
        "mesh: %d nodes, %d tets, regions %s"
        % (node.shape[0], elem.shape[0], np.unique(elem[:, 4]).tolist())
    )

    # 4. map node coordinates voxel(IJK)->RAS(mm)
    A = nc.get_affine(nii)
    node[:, :3] = (A[:3, :3] @ (node[:, :3] - 1).T + A[:3, 3:4]).T
    print(
        "node RAS bbox: x[%.1f %.1f] y[%.1f %.1f] z[%.1f %.1f]"
        % (
            node[:, 0].min(),
            node[:, 0].max(),
            node[:, 1].min(),
            node[:, 1].max(),
            node[:, 2].min(),
            node[:, 2].max(),
        )
    )

    # 5. save as JMesh (tissue label in elem 5th column)
    i2m.savejmesh(node, face, elem, outfile)
    print("saved", outfile)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
