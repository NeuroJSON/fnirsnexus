"""Shared helpers for the redbirdpy fNIRS DOT reconstruction port.

Python mirror of the redbird-m (MATLAB) scripts in ``../redbird_recon``, using
**redbirdpy** (FEM forward/Jacobian/inverse) + **iso2mesh** (pyiso2mesh, mesh
ops) + **jdata** (pyjdata, JData/JMesh/JGIfTI I/O).

The recon DATA files stay in ``../redbird_recon`` (head_mesh.bmsh,
hrf_measurement.jdb, groundtruth.bgii); the ICBM-152 tissue masks/surfaces live
in ``../hm_icbm152``. This module only reads them.

redbirdpy, iso2mesh and jdata are assumed importable (already on PYTHONPATH);
no local toolbox paths are hard-coded here.
"""

from pathlib import Path

import iso2mesh as i2m
import jdata as jd
import numpy as np
import redbirdpy as rb
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "redbird_recon"  # recon data files stay here
HM = HERE.parent / "hm_icbm152"  # ICBM-152 masks / surfaces
FIG = (
    HERE / "figures"
)  # rendered figures (matches the MATLAB redbird_recon/figures layout)
FIG.mkdir(exist_ok=True)

# per-tissue baseline used by the forward model (label order: scalp, bone, CSF,
# gray, white). HbO/HbR in uM; scatamp/scatpow are redbird-m's 500-nm reference
# scattering (scatamp500/scatpow500).
TISSUE = {
    "hbo": np.array([25.0, 15.0, 2.0, 60.0, 40.0]),
    "hbr": np.array([12.0, 8.0, 1.0, 25.0, 18.0]),
    "scatamp": np.array([1.2, 1.35, 0.4, 1.65, 1.5]),
    "scatpow": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
}


# ---------------------------------------------------------------- I/O helpers
def meshpart(x):
    """Split a JMesh element node: (Data, Properties.Value) or (array, None)."""
    if isinstance(x, dict):
        arr = np.asarray(x["Data"], dtype=float)
        val = None
        if "Properties" in x and "Value" in x["Properties"]:
            val = np.asarray(x["Properties"]["Value"], dtype=float)
        return arr, val
    return np.asarray(x, dtype=float), None


def load_mesh(fname="head_mesh.bmsh"):
    """Load the FEM head mesh -> (node Nn x3, elem Ne x5 [4 idx 1-based + label])."""
    m = jd.loadjd(str(DATA / fname))
    node, _ = meshpart(m["MeshVertex3"])
    etet, elabel = meshpart(m["MeshTet4"])
    if elabel is None:
        elabel = np.ones((etet.shape[0], 1))
    elem = np.hstack([etet, elabel.reshape(-1, 1)])
    return node, elem


def load_meas(peaktime=10.0):
    """Load the HRF dOD measurement bundle (hrf_measurement.jdb)."""
    m = jd.loadjd(str(DATA / "hrf_measurement.jdb"))
    wl = np.asarray(m["wavelengths"], dtype=float).ravel()
    out = {
        "wl": wl,
        "wlkey": ["%g" % w for w in wl],  # '760','850' (== redbirdpy keys)
        "srcpos": np.asarray(m["srcpos"], dtype=float),
        "detpos": np.asarray(m["detpos"], dtype=float),
        "pairs": np.asarray(m["pairs"], dtype=int),  # Npair x2 [src det] (1-based)
        "dOD": np.asarray(m["dOD"], dtype=float),  # [pair, wl, time, trial]
        "taxis": np.asarray(m["time"], dtype=float).ravel(),
        "trials": [str(t) for t in m["trialtype"]],
    }
    out["tidx"] = int(np.argmin(np.abs(out["taxis"] - peaktime)))
    return out


def load_truth():
    """Load groundtruth.bgii -> (cortex Nx3, faces Mx3 1-based, props dict, raw)."""
    g = jd.loadjd(str(DATA / "groundtruth.bgii"))
    mv = g["GIFTIData"]["MeshVertex3"]
    cortex = np.asarray(mv["Data"], dtype=float)
    faces = np.asarray(g["GIFTIData"]["MeshTri3"]["Data"], dtype=int)
    props = {k: np.asarray(v, dtype=float).ravel() for k, v in mv["Properties"].items()}
    return cortex, faces, props, g


def get_affine(nii):
    """Return the 4x4 voxel->RAS affine from a jdata JNIfTI struct."""
    h = nii["NIFTIHeader"]
    A = np.asarray(h["Affine"], dtype=float)
    if A.shape[0] == 3:
        A = np.vstack([A, [0.0, 0.0, 0.0, 1.0]])
    return A


def load_scalp(withfaces=False):
    """Load the scalp surface (mask_scalp.obj, voxel coords) mapped to RAS (mm)."""
    sv, sf = i2m.readobjmesh(str(HM / "mask_scalp.obj"))
    A = get_affine(jd.loadjnifti(str(HM / "mask_skin.nii")))
    scalp = (A[:3, :3] @ np.asarray(sv, dtype=float).T + A[:3, 3:4]).T
    if withfaces:
        return scalp, np.asarray(sf, dtype=int)
    return scalp


# ----------------------------------------------------------- geometry helpers
def nearestnodes(node, q):
    """0-based row index of the nearest ``node`` for each ``q`` (== MATLAB knnsearch)."""
    _, idx = cKDTree(np.asarray(node, dtype=float)).query(
        np.asarray(q, dtype=float), k=1
    )
    return np.asarray(idx, dtype=int)


def snap2surf(node, face, nv, p, push=1.0):
    """Project optodes ``p`` onto the mesh boundary; return (pos, inward dir).

    ``face`` is 1-based (iso2mesh); ``nv`` are per-node surface normals.
    """
    surfidx = np.unique(face.astype(int).ravel()) - 1  # 0-based boundary nodes
    k = nearestnodes(node[surfidx, :], p)
    ci = surfidx[k]
    d = -nv[ci, :]
    return node[ci, :] + push * d, d


# --------------------------------------------------------------- forward model
def make_cfg(node, elem, srcpos, srcdir, detpos, detdir, wlkey, tissue=TISSUE):
    """Build a redbirdpy cfg with a per-tissue baseline, prep mesh + properties.

    Returns (cfg, sd). ``cfg['prop']`` is the wavelength-keyed [mua,musp,g,n]
    table computed from the chromophore baseline via ``rb.updateprop``.
    """
    nseg = int(elem[:, 4].max())
    sel = np.minimum(np.arange(nseg), len(tissue["hbo"]) - 1)
    scatpow = np.asarray(tissue["scatpow"])[sel]
    cfg = {
        "node": np.asarray(node, dtype=float),
        "elem": np.asarray(elem, dtype=np.int64),  # 1-based indices (+ label col)
        "seg": elem[:, 4].astype(int),
        "srcpos": np.asarray(srcpos, dtype=float),
        "srcdir": np.asarray(srcdir, dtype=float),
        "detpos": np.asarray(detpos, dtype=float),
        "detdir": np.asarray(detdir, dtype=float),
        # redbird-m musp(lam)=scatamp500*(lam/500)^-scatpow; redbirdpy uses
        # musp=scatamp*lam^-scatpow, so fold the 500-nm reference into scatamp.
        "param": {
            "hbo": np.asarray(tissue["hbo"])[sel],
            "hbr": np.asarray(tissue["hbr"])[sel],
            "scatamp": np.asarray(tissue["scatamp"])[sel] * 500.0**scatpow,
            "scatpow": scatpow,
        },
        "prop": {
            k: np.vstack([[0, 0, 1, 1]] + [[0.015, 1.0, 0, 1.37]] * nseg) for k in wlkey
        },
        "omega": 0,
    }
    cfg, sd = rb.meshprep(cfg)
    cfg["prop"] = rb.updateprop(cfg)
    return cfg, sd


def baseline_forward(cfg, sd):
    """Run the baseline FEM forward; return (detphi dict, phi dict) by wavelength."""
    detphi, phi = rb.runforward(cfg, sd=sd)
    if not isinstance(phi, dict):  # single-wavelength fallback
        phi = {list(cfg["prop"].keys())[0]: phi}
        detphi = {list(cfg["prop"].keys())[0]: detphi}
    return detphi, phi


def measured_sd(pairs, ns):
    """redbirdpy sd table for the measured pairs: [src_col, ns+det_col, 1, 1] (0-based)."""
    n = pairs.shape[0]
    return np.column_stack(
        [
            pairs[:, 0] - 1,  # source column (0-based)
            ns + (pairs[:, 1] - 1),  # detector column (0-based, offset by #sources)
            np.ones(n),
            np.ones(n),
        ]
    )


def stack_ymodel(detphi, pairs, wlkey, npair):
    """Baseline detector readings stacked by wavelength block: detphi[wv][det,src]."""
    nwl = len(wlkey)
    y = np.zeros(npair * nwl)
    di, si = pairs[:, 1] - 1, pairs[:, 0] - 1
    for w, key in enumerate(wlkey):
        bw = detphi[key] if key in detphi else detphi[float(key)]
        y[w * npair : (w + 1) * npair] = np.asarray(bw)[di, si]
    return y


# ------------------------------------------------------------------ rendering
def diverging_cmap(n=256):
    """RdBu_r-like blue->white->red diverging colormap (matches the MATLAB cmap)."""
    from matplotlib.colors import ListedColormap

    t = np.linspace(0, 1, n)[:, None]
    rgb = np.hstack(
        [np.minimum(1, 2 * t), 1 - np.abs(2 * t - 1), np.minimum(1, 2 * (1 - t))]
    )
    return ListedColormap(rgb)


def plot_surface(
    ax, verts, faces1, val, view, clim=None, cmap=None, title="", colorbar=True
):
    """Render a per-vertex scalar field on a triangular surface (faces 1-based).

    Faces are explicitly shaded with a diffuse light offset from the camera
    (matching MATLAB ``camlight``+``lighting gouraud``) so the 3-D cortical form
    stays visible even where the diverging colormap value is ~0 (white) -- without
    shading the surface washes out to white against the white background. A
    colorbar (the unshaded cmap/clim mapping) is attached to the axes like the
    MATLAB ``colorbar``.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if cmap is None:
        cmap = diverging_cmap()
    verts = np.asarray(verts, dtype=float)
    faces0 = faces1.astype(int) - 1
    val = np.asarray(val, dtype=float).ravel()
    if clim is None:
        mm = np.max(np.abs(val)) or 1.0
        clim = (-mm, mm)

    norm = Normalize(*clim)
    tris = verts[faces0]  # (Nf, 3, 3)
    fval = val[faces0].mean(axis=1)  # per-face scalar
    rgb = cmap(norm(fval))[:, :3]

    # ``view`` is given as MATLAB plotmesh view([az el]); convert to matplotlib's
    # (azim, elev) convention by matching the camera direction: the MATLAB
    # camera dir [sin(az)cos(el), -cos(az)cos(el), sin(el)] equals matplotlib's
    # [cos(el)cos(azim), cos(el)sin(azim), sin(el)] when azim = az - 90.
    azim, elev = float(view[0]) - 90.0, float(view[1])

    # diffuse shade from a light tilted ~35 deg off the camera so front-facing
    # faces span a gray->bright gradient (revealing the surface form) instead of
    # all rendering at full brightness (white).
    la, le = np.radians(azim + 25.0), np.radians(elev + 35.0)
    light = np.array([np.cos(le) * np.cos(la), np.cos(le) * np.sin(la), np.sin(le)])
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(nlen == 0, 1.0, nlen)
    shade = np.abs(nrm @ light)  # orientation-independent (closed surface)
    amb = 0.45
    rgb = np.clip(rgb * (amb + (1.0 - amb) * shade)[:, None], 0, 1)

    poly = Poly3DCollection(tris, facecolors=rgb, edgecolors="none", linewidths=0)
    ax.add_collection3d(poly)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    try:
        ax.set_box_aspect([np.ptp(verts[:, i]) for i in range(3)])
    except Exception:
        pass
    ax.set_title(title)
    if colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.0, fraction=0.04)
    return poly
