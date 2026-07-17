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
from analysis.detect import apparent_size_px

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


def synthesise(img, depth, mask, B, dbeta):
    """Add dbeta of attenuation to a real frame, per channel, per pixel.

        I_new = B + (I_obs - B) * exp(-dbeta * d(x))

    This is the simplified image formation model with J eliminated. It is algebraically
    identical to recovering J = (I - B*(1-T))/T with T = exp(-beta*d) and then
    re-attenuating with beta+dbeta, but it never forms J explicitly.

    That matters because the recovered J is not representable: at beta = 0.53 (red) and
    d = 3 m, exp(-beta*d) = 0.204, and J on real pixel values ranges roughly -600 to
    +520. Any pipeline that stores that intermediate as an image clips it and destroys
    the signal. This form's output always lies between I_obs and B, so it cannot leave
    range. It also never needs beta itself, only the dbeta being added.

    B's poor fit (r^2 ~ 0.5) does not sink the method: B cancels exactly from the
    difference of any two pixels at the same range, so contrast scales by exactly
    exp(-dbeta*d) and B only sets the DC level, which ArUco's adaptive threshold
    largely rejects. That is a property of the model itself, not an advantage of this
    formulation over recovering J and re-attenuating; the two are the same function.

    Pixels outside mask, or whose depth is nan, are returned untouched.
    """
    src = np.asarray(img, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    Bv = np.asarray(B, dtype=np.float64).reshape(1, 1, 3)
    db = np.asarray(dbeta, dtype=np.float64).reshape(1, 1, 3)

    trans = np.exp(-db * np.nan_to_num(d, nan=0.0)[..., None])
    new = Bv + (src - Bv) * trans

    apply = np.asarray(mask, dtype=bool) & np.isfinite(d)
    out = np.where(apply[..., None], new, src)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _profile(gray, p, direction, reach_out, reach_in, step=0.25):
    """Bilinear intensity samples along a ray, from -reach_in to +reach_out about p.

    `direction` must be the outward normal, so negative t moves inward (into the
    marker) and positive t moves outward (into the surrounding sheet). The two reaches
    differ on purpose: outward runs into clean white sheet and can go far, inward must
    stay inside the marker's one-cell-wide black border, or the profile picks up the
    data area behind it (see edge_width).
    """
    ts = np.arange(-reach_in, reach_out + step, step)
    pts = p[None, :] + ts[:, None] * direction[None, :]
    h, w = gray.shape[:2]
    if (pts[:, 0].min() < 0 or pts[:, 0].max() > w - 2
            or pts[:, 1].min() < 0 or pts[:, 1].max() > h - 2):
        return None, None
    vals = cv2.remap(gray.astype(np.float32),
                     pts[:, 0].astype(np.float32).reshape(-1, 1),
                     pts[:, 1].astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR).ravel()
    return ts, vals


def _rise_10_90(ts, vals, reversal_tol=0.15):
    """Distance between the 10% and 90% crossings of a monotone-ish step profile.

    Rejects (returns nan for) a profile that is not a clean single step, such as the
    white-black-white hump produced when the inward reach spills past the marker's
    black border into its data area. The discriminator is the ratio of total
    variation (sum of |diffs|) to net variation (|end - start|) of the normalised
    profile: a clean step has TV approx equal to its net rise, while a hump adds a
    second excursion that inflates TV well beyond the net change. reversal_tol=0.15
    allows 15% of extra TV for sampling noise and anti-aliasing, which is generous
    against real single-step profiles but far below the roughly 2x TV a genuine
    double edge produces.
    """
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 20.0:
        return float("nan")
    norm = (vals - lo) / (hi - lo)
    t10 = _crossing(ts, norm, 0.10)
    t90 = _crossing(ts, norm, 0.90)
    if t10 is None or t90 is None:
        return float("nan")
    net = abs(float(norm[-1]) - float(norm[0]))
    if net < 1e-9:
        return float("nan")
    tv = float(np.sum(np.abs(np.diff(norm))))
    if tv > net * (1.0 + reversal_tol):
        return float("nan")
    return abs(t90 - t10)


def _crossing(ts, norm, level):
    """First linear-interpolated crossing of `level`, or None."""
    s = np.sign(norm - level)
    idx = np.nonzero(np.diff(s) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    y0, y1 = norm[i], norm[i + 1]
    if y1 == y0:
        return float(ts[i])
    return float(ts[i] + (level - y0) / (y1 - y0) * (ts[i + 1] - ts[i]))


def edge_width(gray, corners, n_samples=16, reach_px=8.0):
    """Median 10-90% rise distance, in pixels, across the marker's outer boundary.

    Measured in the ORIGINAL frame: the canonical warp resamples, and would report its own
    interpolation kernel as if it were the water. Samples several points along each of the
    four edges, crossing outward along the edge normal, and takes the median so a single
    occluded or clipped sample cannot set the answer.

    The marker's black border is one cell wide (a 7x7 grid), so a symmetric reach_px
    on both sides of the boundary contaminates small markers: at 40 px apparent size a
    cell is 5.71 px, and an 8 px inward reach lands 1.4 cells in, past the border and
    into the data area, producing a spurious white-black-white profile instead of a
    single step. The outward direction has no such limit, since it runs into clean
    white sheet. So the reach is asymmetric: outward stays at reach_px, inward is
    capped at 0.45 of the marker's own cell size, computed from its own corners, to
    stay well inside the border cell. That reduces contamination but the border is
    anti-aliased and the physical board is printed at 95.9% of nominal scale, so a
    residual hump can still reach a sample; _rise_10_90 detects and rejects it
    directly rather than trusting the reach bound alone.
    """
    c = np.asarray(corners, dtype=float)
    centre = c.mean(axis=0)
    cell = apparent_size_px(c) / 7.0
    reach_in = min(reach_px, 0.45 * cell)
    widths = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        edge = b - a
        length = np.linalg.norm(edge)
        if length < 1e-6:
            continue
        normal = np.array([-edge[1], edge[0]]) / length
        mid = (a + b) / 2.0
        if np.dot(normal, mid - centre) < 0:
            normal = -normal  # always point outward, away from the marker
        for f in np.linspace(0.25, 0.75, n_samples // 4 + 1):
            p = a + f * edge
            ts, vals = _profile(gray, p, normal, reach_px, reach_in)
            if ts is None:
                continue
            w = _rise_10_90(ts, vals)
            if np.isfinite(w):
                widths.append(w)
    return float(np.median(widths)) if widths else float("nan")


def trials_at_tau(pred, detected_by_frame):
    """Re-score a FIXED trial set against a degraded frame's detections.

    detected_by_frame maps frame_idx -> set of detected marker ids. The trial set comes
    from the original imagery and never moves, so the denominator is the same at every
    optical depth and the rates across the sweep are directly comparable.
    """
    out = pred.copy()
    out["detected"] = [int(mid in detected_by_frame.get(int(fi), ()))
                       for fi, mid in zip(out["frame_idx"], out["marker_id"])]
    return out.drop(columns=["present"])


def marker_ranges(layout, rv, tv):
    """Euclidean range from the camera to each marker's centre, given the board pose."""
    R = g.rodrigues(np.asarray(rv, dtype=float)[None])[0]
    t = np.asarray(tv, dtype=float)
    out = {}
    for mid in g.IDS:
        c = g.board_pts(layout, np.array([g.IDX[mid]]))[0].mean(axis=0)
        out[int(mid)] = float(np.linalg.norm(R @ c + t))
    return out
