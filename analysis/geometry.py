"""Pure geometry for the board model. No I/O, no OpenCV state -- all testable.

Board frame convention: marker 201 sits at the origin with theta = 0 and the board
plane is z = 0. Every marker is coplanar, so a marker's placement is 3 DOF
(tx, ty, theta), not 6. That constraint is what makes the layout identifiable.
"""
import os

import numpy as np
import yaml

_CFG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "config")
_MEASURED = os.path.join(_CFG_DIR, "marker_sizes_measured.yaml")
_NOMINAL = os.path.join(_CFG_DIR, "marker_sizes.yaml")

# The MEASURED sizes are required, with no fallback. The board printed at ~95.9%
# scale, so silently loading the nominal table would inflate every range by 4.1% --
# a wrong answer is worse than a crash. marker_sizes.yaml is rewritten by
# collage/aruco_collage_a4.py on every run, so the correction cannot live there.
if not os.path.exists(_MEASURED):
    raise FileNotFoundError(
        f"{_MEASURED} is missing. The nominal sizes in {_NOMINAL} describe the PDF, "
        "not the printed board (~95.9% scale). See plan Task 2 Step 1."
    )
_CFG = _MEASURED

with open(_CFG) as _f:
    _RAW = yaml.safe_load(_f)

DICTIONARY = _RAW["dictionary"]
SIZES = {int(k): float(v) for k, v in _RAW["marker_size_m"].items()}
PRINT_SCALE = float(_RAW.get("print_scale_vs_nominal", 1.0))
IDS = sorted(SIZES)
IDX = {m: i for i, m in enumerate(IDS)}

assert IDS[0] == 201, "marker 201 must sort first -- it is the board-frame gauge anchor"


def local_corners(mid):
    """(4,3) corners of one marker in its own frame, in cv2.aruco order (TL,TR,BR,BL)."""
    h = SIZES[mid] / 2
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


LOC = np.stack([local_corners(m) for m in IDS])


def rodrigues(rv):
    """Vectorised Rodrigues. (N,3) rotation vectors -> (N,3,3) rotation matrices."""
    rv = np.asarray(rv, dtype=float)
    th = np.linalg.norm(rv, axis=1, keepdims=True)
    k = rv / np.where(th == 0, 1.0, th)
    K = np.zeros((len(rv), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = np.eye(3)[None].repeat(len(rv), 0)
    s, c = np.sin(th)[:, :, None], np.cos(th)[:, :, None]
    return I + s * K + (1 - c) * (K @ K)


def board_pts(layout, mrk):
    """Board-frame corners for each observation.

    layout: (9,3) of (tx, ty, theta); row IDX[201] is the gauge and must stay zero.
    mrk:    (M,) indices into IDS.
    returns (M,4,3) with z identically 0 (coplanar by construction).
    """
    mrk = np.asarray(mrk)
    l = LOC[mrk]
    th = layout[mrk, 2]
    c, s = np.cos(th), np.sin(th)
    x = c[:, None] * l[:, :, 0] - s[:, None] * l[:, :, 1] + layout[mrk, 0][:, None]
    y = s[:, None] * l[:, :, 0] + c[:, None] * l[:, :, 1] + layout[mrk, 1][:, None]
    return np.stack([x, y, np.zeros_like(x)], -1)


def board_centroid(layout):
    """Centroid of the marker centres, in board coords (metres).

    The layout's gauge is marker 201 at the origin, but that is a SOLVER constraint --
    it fixes the bundle's 3-DOF ambiguity -- not a meaningful reference point. Marker
    201 sits at the board's left edge, ~208 mm from the middle, so a pose translation
    taken raw measures distance to that corner marker rather than to the board. Offset
    by this centroid when reporting a range.
    """
    return np.asarray(layout, float)[:, :2].mean(axis=0)


def project(Xb, rv, tv, fr, fx, fy, cx, cy):
    """Pinhole projection. Xb (M,4,3) board pts, rv/tv (N,3) poses, fr (M,) frame index."""
    R = rodrigues(rv)[fr]
    Xc = np.einsum("mij,mkj->mki", R, Xb) + tv[fr][:, None, :]
    Z = np.where(np.abs(Xc[:, :, 2]) < 1e-9, 1e-9, Xc[:, :, 2])
    u = fx * Xc[:, :, 0] / Z + cx
    v = fy * Xc[:, :, 1] / Z + cy
    return np.stack([u, v], -1)
