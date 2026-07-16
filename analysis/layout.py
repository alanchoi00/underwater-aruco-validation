"""Stage 3: recover the board layout and validate the focal length.

The collage generator does not preserve arrangement ("Arrangement is not preserved;
SIZES are true"), so the layout must be estimated from the captures.

Read the three traps in the plan before touching this file. In short: never calibrate
from a single marker, never trust a low reprojection error on a planar target, and
never trust the optimizer on f -- profile it.
"""
import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from analysis import geometry as g


def bootstrap_layout(obs_by_frame, K, flip_tol_deg=25.0):
    """Per-marker PnP relative to 201, flip-filtered, robust median.

    Valid under the in-air K: tx/ty are focal-length independent because
    Z ~ S*f/s_px and X = (u-cx)*Z/f, so f cancels. Only depth would be biased.
    """
    acc = {m: [] for m in g.IDS}
    for dets in obs_by_frame.values():
        if 201 not in dets or len(dets) < 2:
            continue
        poses = {}
        for mid, q in dets.items():
            if mid not in g.SIZES:
                continue
            ok, rv, tv = cv2.solvePnP(
                g.local_corners(mid).astype(np.float32), np.asarray(q, np.float32),
                K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                poses[mid] = (cv2.Rodrigues(rv)[0], tv.ravel())
        if 201 not in poses:
            continue
        R1, t1 = poses[201]
        n1 = R1[:, 2]
        for mid, (Rj, tj) in poses.items():
            if mid == 201:
                continue
            # All markers are coplanar, so their normals must agree with 201's.
            # A large deviation is the planar-PnP two-solution flip -- reject it.
            ang = np.degrees(np.arccos(np.clip(abs(n1 @ Rj[:, 2]), -1, 1)))
            if ang > flip_tol_deg:
                continue
            Rr = R1.T @ Rj
            tr = R1.T @ (tj - t1)
            acc[mid].append([tr[0], tr[1], np.arctan2(Rr[1, 0], Rr[0, 0])])
    layout = np.zeros((9, 3))
    for mid in g.IDS:
        if mid == 201 or len(acc[mid]) < 5:
            continue
        layout[g.IDX[mid]] = np.median(np.array(acc[mid]), axis=0)
    return layout


class FixedK:
    """Bundle with fx, fy held FIXED; only layout + poses are optimised.

    Fixing K is what makes profiling possible: sweep (fx, fy) on a grid, re-optimise
    everything else, and read identifiability off the cost curve instead of trusting
    an optimizer that may silently freeze a badly-scaled variable.
    """

    def __init__(self, uv, fr, mrk, nf, cx, cy):
        self.uv, self.fr, self.mrk, self.nf = uv, fr, mrk, nf
        self.cx, self.cy = cx, cy

    def unpack(self, p):
        layout = np.zeros((9, 3))
        layout[1:] = p[:24].reshape(8, 3)
        rv = p[24:24 + 3 * self.nf].reshape(self.nf, 3)
        tv = p[24 + 3 * self.nf:24 + 6 * self.nf].reshape(self.nf, 3)
        return layout, rv, tv

    def pack(self, layout, rv, tv):
        return np.concatenate([layout[1:].ravel(), rv.ravel(), tv.ravel()])

    def resid(self, p, fx, fy):
        layout, rv, tv = self.unpack(p)
        Xb = g.board_pts(layout, self.mrk)
        px = g.project(Xb, rv, tv, self.fr, fx, fy, self.cx, self.cy)
        return (px - self.uv).ravel()

    def sparsity(self):
        M = len(self.fr)
        S = lil_matrix((2 * M * 4, 24 + 6 * self.nf), dtype=int)
        for i in range(M):
            rows = np.arange(i * 8, i * 8 + 8)
            if self.mrk[i] > 0:
                S[np.ix_(rows, (self.mrk[i] - 1) * 3 + np.arange(3))] = 1
            S[np.ix_(rows, 24 + self.fr[i] * 3 + np.arange(3))] = 1
            S[np.ix_(rows, 24 + 3 * self.nf + self.fr[i] * 3 + np.arange(3))] = 1
        return S

    def solve(self, p0, fx, fy, max_nfev=200):
        return least_squares(self.resid, p0, args=(fx, fy), jac_sparsity=self.sparsity(),
                             method="trf", loss="huber", f_scale=2.0, x_scale="jac",
                             xtol=1e-12, ftol=1e-12, max_nfev=max_nfev)


def _pnp_init(problem, layout0, K):
    """Per-frame board-PnP pose init under a candidate K, read off `problem`'s flat arrays.

    Seeding each frame from the *hypothesis* K -- not a flat zero -- matters: PnP under a
    wrong f gives a correspondingly wrong depth, which is exactly the discriminating signal
    the profile is supposed to pick up. A common zero/unit-depth seed for every frame starves
    the optimiser of that signal and, on a real multi-hundred-frame board, leaves it short of
    max_nfev before it can separate the scales -- an unidentifiability that is an artefact of
    the seed, not of the geometry.
    """
    rv0 = np.zeros((problem.nf, 3))
    tv0 = np.zeros((problem.nf, 3))
    tv0[:, 2] = 1.0
    for i in range(problem.nf):
        idx = np.where(problem.fr == i)[0]
        if len(idx) == 0:
            continue
        op = g.board_pts(layout0, problem.mrk[idx]).reshape(-1, 3)
        ip = problem.uv[idx].reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(op.astype(np.float32), ip.astype(np.float32),
                                   K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            rv0[i] = rv.ravel()
            tv0[i] = tv.ravel()
    return rv0, tv0


def profile_focal(problem, layout0, K, scales):
    """Sweep an isotropic focal scale; return {scale: cost}.

    A flat curve means f is unidentifiable and no metric claim is defensible. A clear
    minimum means f is identified and can be read straight off.
    """
    f0 = K[0, 0]
    costs = {}
    for s in scales:
        Ks = np.array([[f0 * s, 0, K[0, 2]], [0, f0 * s, K[1, 2]], [0, 0, 1]])
        rv0, tv0 = _pnp_init(problem, layout0, Ks)
        p0 = problem.pack(layout0, rv0, tv0)
        out = problem.solve(p0, f0 * s, f0 * s)
        costs[s] = float(out.cost)
    return costs


def board_pnp(layout, dets, K):
    """Multi-marker reference pose from every detected board marker in one frame."""
    op, ip = [], []
    for mid, q in dets.items():
        if mid not in g.SIZES:
            continue
        op.append(g.board_pts(layout, np.array([g.IDX[mid]]))[0])
        ip.append(np.asarray(q, dtype=float))
    if not op:
        raise ValueError("no board markers in frame")
    ok, rv, tv = cv2.solvePnP(np.concatenate(op).astype(np.float32),
                              np.concatenate(ip).astype(np.float32),
                              K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("board PnP failed")
    return rv.ravel(), tv.ravel()


def save_layout(layout, path):
    with open(path, "w") as f:
        yaml.safe_dump({
            "note": "board frame: marker 201 at origin, theta=0, plane z=0",
            "units": "tx/ty metres, theta radians",
            "layout": {int(m): [float(x) for x in layout[g.IDX[m]]] for m in g.IDS},
        }, f, sort_keys=True)


def load_layout(path):
    with open(path) as f:
        raw = yaml.safe_load(f)["layout"]
    layout = np.zeros((9, 3))
    for m, v in raw.items():
        layout[g.IDX[int(m)]] = v
    return layout
