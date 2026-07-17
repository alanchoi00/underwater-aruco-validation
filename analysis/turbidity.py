"""Turbidity: measure the water's attenuation, then synthesise more of it.

The board is already the experiment. A black patch and a white patch at the same range
share an identical veiling term in the image formation model

    I = J * exp(-beta * d) + B * (1 - exp(-beta * d))

so their difference is B-free:

    contrast(d) = (J_white - J_black) * exp(-beta * d)

Taking logs makes beta a straight-line fit that needs no assumption about backscatter.
That matters, because B itself fits badly (r^2 ~ 0.5) while beta fits well (r^2 ~ 0.96).
"""
import cv2
import numpy as np
import pandas as pd

from analysis import geometry as g

CHANNELS = ("b", "g", "r", "grey")

CANON_PX = 96          # marker side in the warped canonical view
SURROUND = 1.6         # warped view is this multiple of the marker side, to catch the sheet

# Below this apparent size the two rings are a handful of pixels each and the sample is
# dominated by the blur across the ring boundary rather than the ring itself.
MIN_PX_FOR_PHOTOMETRY = 40.0


def warp_marker(img, corners, canon_px=CANON_PX, surround=SURROUND):
    """Warp a detected marker to a canonical square, keeping the sheet around it.

    The marker maps to a centred canon_px square inside a larger view, so the white sheet
    outside the marker is sampled at the same range as the marker's own black border.
    """
    side = int(round(canon_px * surround))
    off = (side - canon_px) / 2.0
    dst = np.array([[off, off], [off + canon_px, off],
                    [off + canon_px, off + canon_px], [off, off + canon_px]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), dst)
    return cv2.warpPerspective(img, H, (side, side), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def _with_grey(bgr_means):
    """Append the Rec.601 grey of a (3,) BGR mean, matching cv2.COLOR_BGR2GRAY."""
    b, gr, r = bgr_means
    return np.array([b, gr, r, 0.114 * b + 0.587 * gr + 0.299 * r])


def patch_means(warped, canon_px=CANON_PX, surround=SURROUND):
    """Mean of the marker's black border ring and of the white sheet outside it.

    Returns (black, white), each (4,) in CHANNELS order. Both rings are eroded away from
    their boundaries: a ring sampled right up to an edge picks up the blur across it,
    which is the very thing beta is being measured through.
    """
    side = int(round(canon_px * surround))
    off = (side - canon_px) / 2.0
    cell = canon_px / 7.0

    black_ring = np.zeros((side, side), dtype=bool)
    outer = _box(off + 0.25 * cell, canon_px - 0.5 * cell, side)
    inner = _box(off + 0.75 * cell, canon_px - 1.5 * cell, side)
    black_ring[outer] = True
    black_ring[inner] = False

    white_ring = np.zeros((side, side), dtype=bool)
    w_outer = _box(off - 0.75 * cell, canon_px + 1.5 * cell, side)
    w_inner = _box(off - 0.25 * cell, canon_px + 0.5 * cell, side)
    white_ring[w_outer] = True
    white_ring[w_inner] = False

    img = warped.astype(np.float64)
    black = _with_grey([img[..., c][black_ring].mean() for c in range(3)])
    white = _with_grey([img[..., c][white_ring].mean() for c in range(3)])
    return black, white


def _box(start, length, side):
    """Boolean mask of an axis-aligned square, clipped to the view."""
    a = max(0, int(round(start)))
    b = min(side, int(round(start + length)))
    m = np.zeros((side, side), dtype=bool)
    if b > a:
        m[a:b, a:b] = True
    return m


def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_beta(d, contrast):
    """Fit contrast(d) = c0 * exp(-beta * d) by least squares on log(contrast).

    Returns (beta, c0, r2), with r2 reported in log space, which is where the fit lives.
    Non-positive contrast (noise at long range) carries no log and is dropped.
    """
    d = np.asarray(d, dtype=float)
    c = np.asarray(contrast, dtype=float)
    ok = np.isfinite(d) & np.isfinite(c) & (c > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    y = np.log(c[ok])
    slope, intercept = np.polyfit(d[ok], y, 1)
    return float(-slope), float(np.exp(intercept)), _r2(y, slope * d[ok] + intercept)


def fit_veiling(d, black, beta):
    """Fit I_black(d) = B * (1 - exp(-beta * d)) for B, with beta already known.

    Linear in B through the origin, so B is a ratio of sums. Reported with its r2 because
    that r2 is the point: it is around 0.5 on real data, and the design depends on nobody
    trusting B further than that.
    """
    d = np.asarray(d, dtype=float)
    y = np.asarray(black, dtype=float)
    x = 1 - np.exp(-beta * d)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or not np.any(x[ok] > 0):
        return float("nan"), float("nan")
    B = float(np.sum(x[ok] * y[ok]) / np.sum(x[ok] * x[ok]))
    return B, _r2(y[ok], B * x[ok])


def measure_beta(samples):
    """Per-channel beta from the black/white contrast decay across the sample set."""
    rows = []
    for ch in CHANNELS:
        sub = samples.dropna(subset=[f"white_{ch}", f"black_{ch}", "range_m"])
        beta, c0, r2 = fit_beta(sub["range_m"].to_numpy(),
                                (sub[f"white_{ch}"] - sub[f"black_{ch}"]).to_numpy())
        rows.append({"channel": ch, "beta": beta, "c0": c0, "r2": r2, "n": int(len(sub))})
    return pd.DataFrame(rows)


def measure_veiling(samples, betas):
    """Per-channel B from the black patches, given beta. betas maps channel -> beta."""
    rows = []
    for ch in CHANNELS:
        sub = samples.dropna(subset=[f"black_{ch}", "range_m"])
        B, r2 = fit_veiling(sub["range_m"].to_numpy(), sub[f"black_{ch}"].to_numpy(),
                            betas[ch])
        rows.append({"channel": ch, "B": B, "r2": r2, "n": int(len(sub))})
    return pd.DataFrame(rows)


def plane_depth_map(layout, rv, tv, K, shape):
    """Euclidean range from the camera to the board plane, per pixel.

    The board is z = 0 in board coords, so the plane in camera coords has normal
    R @ [0,0,1] and passes through tv. Each pixel's ray is s * Kinv @ [u,v,1]; solving
    n . (s*ray - tv) = 0 gives s, and the range is the length of that ray.

    This is geometric truth, not a monocular depth estimate, which is the main thing this
    synthesis has that the published ones do not.
    """
    h, w = shape[:2]
    R = g.rodrigues(np.asarray(rv, dtype=float)[None])[0]
    n = R @ np.array([0.0, 0.0, 1.0])
    t = np.asarray(tv, dtype=float)

    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(K).T

    denom = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = float(n @ t) / denom
    pts = rays * s[..., None]
    d = np.linalg.norm(pts, axis=-1)
    d[~np.isfinite(d) | (s <= 0) | (np.abs(denom) < 1e-12)] = np.nan
    return d


def board_mask(layout, rv, tv, K, shape, margin_m=0.04):
    """Pixels covered by the physical board, as the marker bounding box plus a margin.

    The margin exists because ArUco thresholds each pixel against its neighbours, so a
    marker's white sheet is part of what makes it detectable and must be degraded with it.
    The background beyond the board is left alone and is therefore physically incoherent;
    it does not reach the detector's decision, but the driver reports the caveat.
    """
    h, w = shape[:2]
    xy = np.concatenate([g.board_pts(layout, np.array([g.IDX[m]]))[0] for m in g.IDS])
    lo = xy[:, :2].min(axis=0) - margin_m
    hi = xy[:, :2].max(axis=0) + margin_m
    quad = np.array([[lo[0], lo[1], 0.0], [hi[0], lo[1], 0.0],
                     [hi[0], hi[1], 0.0], [lo[0], hi[1], 0.0]])
    R = g.rodrigues(np.asarray(rv, dtype=float)[None])[0]
    cam = quad @ R.T + np.asarray(tv, dtype=float)
    mask = np.zeros((h, w), dtype=np.uint8)
    if np.any(cam[:, 2] <= 0):
        return mask.astype(bool)
    px = (cam[:, :2] / cam[:, 2:3]) @ np.array([[K[0, 0], 0], [0, K[1, 1]]]) \
        + np.array([K[0, 2], K[1, 2]])
    cv2.fillConvexPoly(mask, np.round(px).astype(np.int32), 1)
    return mask.astype(bool)


def marker_ranges(layout, rv, tv):
    """Euclidean range from the camera to each marker's centre, given the board pose."""
    R = g.rodrigues(np.asarray(rv, dtype=float)[None])[0]
    t = np.asarray(tv, dtype=float)
    out = {}
    for mid in g.IDS:
        c = g.board_pts(layout, np.array([g.IDX[mid]]))[0].mean(axis=0)
        out[int(mid)] = float(np.linalg.norm(R @ c + t))
    return out
