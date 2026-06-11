#!/usr/bin/env python3
"""Export Cedalion-side inputs into JData/JGIfTI files for the redbird-m pipeline.

This is the data-exchange stage of porting the `hmstandardization_challenge_prep`
image-reconstruction workflow from Cedalion (Python/xarray) to redbird-m (MATLAB/FEM).

It produces, in this folder:

  1. groundtruth.bgii          - JGIfTI (binary JData/BJData): ICBM-152 cortex
                                  surface (mask_brain.obj) carrying the *true*
                                  synthetic activation as per-vertex
                                  Properties.HbO_* / HbR_* (uM), one pair per
                                  stimulus condition (Stim C3 / Stim C4). Used only
                                  for scoring the reconstruction.

  2. hrf_measurement.jdb       - binary JData: the block-averaged HRF dOD that is
                                  the *input* to the reconstruction, reshaped to a
                                  tidy [pair x wavelength x time x trialtype] array
                                  plus the source/detector geometry, wavelengths,
                                  and the source-detector pair table. MATLAB reads
                                  it with loadbj()/loadjd() (no SNIRF parsing).

The volumetric FEM head mesh is *not* built here; it is generated in MATLAB from
the ICBM-152 tissue masks by s1_build_mesh.m (iso2mesh), since GIfTI/JGIfTI is a
surface-only format.

Author: ported for Qianqian Fang's redbird toolchain.
"""

from pathlib import Path

import h5py
import jdata as jd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent                      # .../imagespace
HM = DATA / "hm_icbm152"


def read_obj(fname):
    """Read a Wavefront .obj triangular surface -> (verts Nx3 float, faces Mx3 int, 1-based)."""
    verts, faces = [], []
    with open(fname) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                # faces may be "f a b c" or "f a/x b/y c/z"; take the vertex index
                faces.append([int(tok.split("/")[0]) for tok in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def export_groundtruth():
    """Write groundtruth.jgii: cortex surface + true per-vertex HbO/HbR per condition."""
    verts, faces = read_obj(HM / "mask_brain.obj")          # 25000 x3, 49992 x3 (1-based)
    nvert = verts.shape[0]
    # the .obj surfaces are in voxel/IJK coords; map to RAS (mm) with the NIfTI affine
    # so they share the frame of the FEM mesh and the SNIRF optodes.
    import nibabel as nib
    A = np.asarray(nib.load(str(HM / "mask_skin.nii")).affine)
    verts = (A[:3, :3] @ verts.T + A[:3, 3:4]).T.astype(np.float32)

    # spatial_act.csv: columns trial_type, vertex, chromo, spatial_act (uM)
    df = pd.read_csv(DATA / "spatial_act.csv")
    props = {}
    for trial in sorted(df["trial_type"].unique()):           # 'Stim C3', 'Stim C4'
        tag = trial.replace("Stim ", "").strip()              # 'C3', 'C4'
        sub = df[df["trial_type"] == trial]
        for chromo in ("HbO", "HbR"):
            arr = np.zeros(nvert, dtype=np.float32)
            rows = sub[sub["chromo"] == chromo]
            arr[rows["vertex"].to_numpy(dtype=int)] = rows["spatial_act"].to_numpy()
            props[f"{chromo}_{tag}"] = arr                    # e.g. HbO_C3, HbR_C4

    jgii = {
        "GIFTIHeader": {
            "Version": "1.0",
            "MetaData": {
                "Description": "ICBM-152 cortex with synthetic ground-truth activation",
                "SubjectID": "nn22_icbm152",
                "LengthUnit": "mm",
                "Source": "Cedalion hmstandardization_challenge_prep / spatial_act.csv",
            },
        },
        "GIFTIData": {
            "MeshVertex3": {
                "_DataInfo_": {
                    "MetaData": {
                        "AnatomicalStructurePrimary": "CortexRightAndLeft",
                        "GeometricType": "Anatomical",
                        "Name": "true_activation",
                        "Unit": "uM",
                    }
                },
                "Data": verts,
                "Properties": props,
            },
            "MeshTri3": {
                "_DataInfo_": {"MetaData": {"TopologicalType": "Closed"}},
                "Data": faces,                                # 1-based, JMesh convention
            },
        },
    }

    out = HERE / "groundtruth.bgii"
    jd.save(jgii, str(out), {"compression": "zlib"})         # binary JGIfTI (BJData)
    print(f"wrote {out.name}: {nvert} verts, {faces.shape[0]} faces, "
          f"props={list(props)}")


def export_measurement():
    """Write hrf_measurement.jdt: the block-averaged HRF dOD reshaped for redbird."""
    with h5py.File(DATA / "synthetic_hrf.snirf", "r") as f:
        n = f["nirs"]
        srcpos = n["probe/sourcePos3D"][()].astype(np.float64)        # Nsrc x3 (RAS mm)
        detpos = n["probe/detectorPos3D"][()].astype(np.float64)      # Ndet x3
        wl = n["probe/wavelengths"][()].ravel().astype(np.float64)    # [760, 850]
        d = n["data1"]
        dts = d["dataTimeSeries"][()].astype(np.float64)              # Ntime x Ncol
        taxis = d["time"][()].ravel().astype(np.float64)              # Ntime (reltime)
        # build the per-column descriptor table from measurementList*
        mls = sorted(
            (k for k in d if k.startswith("measurementList")),
            key=lambda s: int(s.replace("measurementList", "")),
        )
        cols = np.array(
            [
                [
                    int(d[k]["sourceIndex"][()]),
                    int(d[k]["detectorIndex"][()]),
                    int(d[k]["wavelengthIndex"][()]),     # 1..Nwl
                    int(d[k]["dataTypeIndex"][()]),       # 1=Stim C3, 2=Stim C4
                ]
                for k in mls
            ],
            dtype=np.int32,
        )

    ntime = dts.shape[0]
    trials = np.unique(cols[:, 3])                                    # [1,2]
    # Resolve the (src,det) pairs present (same set for every wl/trial).
    base = cols[(cols[:, 2] == 1) & (cols[:, 3] == trials[0])][:, :2]  # Npair x2
    npair = base.shape[0]
    pair_index = {(s, dd): i for i, (s, dd) in enumerate(map(tuple, base))}

    # dOD[pair, wavelength, time, trial]
    dod = np.full((npair, len(wl), ntime, len(trials)), np.nan)
    for ci, (s, dd, wi, ti) in enumerate(cols):
        p = pair_index.get((s, dd))
        if p is None:
            continue
        dod[p, wi - 1, :, np.where(trials == ti)[0][0]] = dts[:, ci]

    meas = {
        "format": "redbird-hrf-dod",
        "wavelengths": wl,                       # nm
        "srcpos": srcpos,                        # Nsrc x3, RAS mm
        "detpos": detpos,                        # Ndet x3, RAS mm
        "pairs": base,                           # Npair x2 -> [sourceIndex, detectorIndex] (1-based)
        "trialtype": ["Stim C3", "Stim C4"],
        "time": taxis,                           # Ntime reltime (s)
        "dOD": dod.astype(np.float32),           # Npair x Nwl x Ntime x Ntrial
        "Unit": "dimensionless",
        "Space": "scanner_anat_RAS_mm",
    }
    out = HERE / "hrf_measurement.jdb"
    jd.save(meas, str(out), {"compression": "zlib"})
    print(
        f"wrote {out.name}: {npair} pairs x {len(wl)} wl x {ntime} time x "
        f"{len(trials)} trials; src={srcpos.shape[0]} det={detpos.shape[0]}"
    )


if __name__ == "__main__":
    export_groundtruth()
    export_measurement()
