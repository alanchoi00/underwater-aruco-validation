"""Stage 5: metrics.

Apparent pixel size is the primary axis because it is measured directly and needs no
calibration. Metric range is derived from it and inherits the focal length's ~+/-10%
uncertainty, so it is reported with that caveat rather than as a measurement.
"""
import cv2
import numpy as np
import pandas as pd

from analysis import geometry as g


def pixel_budget_px(grid_n=5):
    """Rule of thumb: an n x n marker needs about 3*(n+2) pixels across to decode."""
    return 3 * (grid_n + 2)


def mis_id_rate(det_df):
    """Fraction of detections carrying an id that is not on the board."""
    if not len(det_df):
        return 0.0
    off = ~det_df["marker_id"].isin(g.SIZES.keys())
    return float(off.sum()) / float(len(det_df))


def detection_rate_by_px_bin(det_df, n_frames_by_id, bins):
    """Detection count and rate per apparent-size bin, per marker.

    The denominator is the number of frames the marker could have appeared in; it must
    be supplied, since a detection table alone cannot know about missed frames.
    """
    board = det_df[det_df["marker_id"].isin(g.SIZES.keys())]
    rows = []
    for mid, sub in board.groupby("marker_id"):
        denom = n_frames_by_id.get(int(mid), 0)
        for lo, hi in zip(bins[:-1], bins[1:]):
            n = int(((sub["apparent_px"] >= lo) & (sub["apparent_px"] < hi)).sum())
            rows.append({"marker_id": int(mid), "px_lo": lo, "px_hi": hi,
                         "n_detected": n,
                         "rate": (n / denom) if denom else float("nan")})
    return pd.DataFrame(rows)


def furthest_detection_range(det_df, ranges):
    """Furthest range at which each marker was actually detected.

    `ranges` maps frame_idx -> range (m). Deliberately NOT rate-gated: a single lucky
    detection at the tail counts here. Rate gating lives in detection_rate_by_px_bin,
    where the denominator is known -- a detection table alone cannot see missed frames,
    so a "rate above threshold" range cannot honestly be computed from it.
    """
    out = {}
    board = det_df[det_df["marker_id"].isin(g.SIZES.keys())]
    for mid, sub in board.groupby("marker_id"):
        rs = [ranges[int(f)] for f in sub["frame_idx"] if int(f) in ranges]
        if rs:
            out[int(mid)] = float(max(rs))
    return out


def jitter_std(values):
    """Sample standard deviation (ddof=1) -- precision over a static dwell."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return 0.0
    return float(np.std(v, ddof=1))


def range_from_apparent_px(apparent_px, marker_id, fx):
    """Pinhole inversion: Z = f * S / s_px.

    Inherits the focal length's ~+/-10% uncertainty. Fine for a size-vs-range
    hypothesis; not a calibrated measurement.
    """
    return float(fx * g.SIZES[int(marker_id)] / float(apparent_px))


def incidence_angle_deg(rvec):
    """Angle (deg) between the board normal and the camera's line of sight; 0 = head-on.

    The board plane is z=0, so its normal is the rotation's third column. This is a PnP
    OUTPUT -- i.e. the quantity under test -- so use it for regime labelling and the
    pose-flip discussion, NOT as an independent axis for a detection-rate curve
    (see spec 3.1c / limitation 10). Value is folded to [0, 90] since a marker seen
    from the front or the back is equally oblique.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    cos = abs(float(R[2, 2]))                  # normal.z == boresight . normal
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def binned_stats(df, value_col, bin_col, bins):
    """mean/std of value_col within bins of bin_col. Empty bins report NaN, not a crash."""
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = df[(df[bin_col] >= lo) & (df[bin_col] < hi)][value_col]
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(len(sel)),
                     "mean": float(sel.mean()) if len(sel) else float("nan"),
                     "std": float(sel.std(ddof=1)) if len(sel) > 1 else float("nan")})
    return pd.DataFrame(rows)


def pose_error_vs_reference(obs_by_frame, layout, K):
    """Single-marker PnP vs the multi-marker board PnP, per frame.

    NOT accuracy -- there is no ground truth. This is self-consistency: reference and
    device-under-test share intrinsics and detector, so any common-mode bias cancels
    and stays invisible (spec 6.2). Label every output "vs board reference".

    The marker under test is excluded from its own reference, and at least two other
    board markers must remain, or the frame is skipped.

    CONFOUND -- do not break this metric down by marker size. The leave-one-out
    reference geometry is not fixed: it depends on which marker is under test, because
    that marker is the one excluded from the board PnP that generates its own reference.
    When 201 or 202 (149 mm) is under test, the reference is solved from whatever
    remains -- for this layout, the tightly-clustered small centre markers (301,
    302-305, 401, 402) -- a narrow baseline that gives a weak, noisier reference pose.
    When a 44 mm marker is under test, the reference still includes BOTH 201 and 202,
    which are 427 mm apart -- a wide baseline that gives a strong, stable reference pose.
    So grouping this metric's output by marker_id/size measures reference quality, not
    marker quality, and the size ordering it produces is real but backwards: it looks
    like "bigger markers have worse pose" when what actually varies is the strength of
    the reference each series happens to be scored against. Report this metric pooled
    across all markers only.
    """
    from analysis import layout as L        # local import avoids a circular import

    rows = []
    for (_run, fi), dets in obs_by_frame.items():
        board = {m: q for m, q in dets.items() if m in g.SIZES}
        if len(board) < 3:
            continue                        # need >=2 left after excluding the DUT
        for mid, q in board.items():
            others = {m: v for m, v in board.items() if m != mid}
            if len(others) < 2:
                continue
            try:
                rv_ref, tv_ref = L.board_pnp(layout, others, K)
            except ValueError:
                continue
            ok, rv_s, tv_s = cv2.solvePnP(
                g.local_corners(mid).astype(np.float32), np.asarray(q, np.float32),
                K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            # Where the reference says this marker's centre is, in camera coords.
            centre_board = g.board_pts(layout, np.array([g.IDX[mid]]))[0].mean(axis=0)
            R_ref = g.rodrigues(rv_ref[None])[0]
            expect = R_ref @ centre_board + tv_ref
            trans_err = float(np.linalg.norm(tv_s.ravel() - expect))
            R_s = g.rodrigues(rv_s.ravel()[None])[0]
            Rr = R_ref.T @ R_s
            ang = np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1.0, 1.0)))
            rows.append({"frame_idx": int(fi), "marker_id": int(mid),
                         "range_m": float(np.linalg.norm(expect)),
                         "trans_err_m": trans_err, "rot_err_deg": float(ang)})
    return pd.DataFrame(rows, columns=[
        "frame_idx", "marker_id", "range_m", "trans_err_m", "rot_err_deg"])


def detection_trials(obs_by_frame, layout, K, width, height):
    """Per-frame, per-marker TRIALS -- hits AND misses -- keyed on predicted apparent size.

    A detection rate needs the misses, but a detection table only records hits. The misses
    are recoverable: where other markers are detected the board pose is known, so a marker's
    apparent size can be PREDICTED whether or not it was detected.

    Leave-one-out: the pose for marker m is solved from the OTHER detected markers only, so
    m's own detection never informs the prediction of m's size (which would be circular).

    A trial only counts if the marker is FULLY inside the image. Centre-only would count a
    marker whose corners spill outside -- legitimately unseeable -- as a miss.
    """
    from analysis import layout as L        # local import avoids a circular import

    rows = []
    for (_run, fi), dets in obs_by_frame.items():
        board = {m: q for m, q in dets.items() if m in g.SIZES}
        if not board:
            continue
        for mid in g.IDS:
            others = {m: q for m, q in board.items() if m != mid}
            if not others:
                continue                    # nobody else to fix the pose
            try:
                rv, tv = L.board_pnp(layout, others, K)
            except ValueError:
                continue
            Xb = g.board_pts(layout, np.array([g.IDX[mid]]))
            px = g.project(Xb, rv[None], tv[None], np.zeros(1, dtype=int),
                           K[0, 0], K[1, 1], K[0, 2], K[1, 2])[0]
            if not (px[:, 0].min() >= 0 and px[:, 0].max() < width
                    and px[:, 1].min() >= 0 and px[:, 1].max() < height):
                continue                    # not fully visible -> not a fair trial
            side = float(np.mean([np.linalg.norm(px[i] - px[(i + 1) % 4])
                                  for i in range(4)]))
            rows.append({"frame_idx": int(fi), "marker_id": int(mid),
                         "size_mm": round(g.SIZES[mid] * 1000, 1),
                         "pred_px": side, "detected": int(mid in board)})
    return pd.DataFrame(rows, columns=["frame_idx", "marker_id", "size_mm",
                                       "pred_px", "detected"])


def _wilson(k, n, z=1.96):
    """Wilson score interval -- behaves at rate 0 and 1, where normal approx does not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate_by_bin(trials, bins, group_col="size_mm"):
    """TRUE detection rate per bin of predicted apparent size, pooled by group.

    Pooling same-size markers multiplies the samples per bin (302-305 are all 44.4 mm ->
    4x). Returns a Wilson confidence interval, which stays sane at rate 0 and 1.
    """
    rows = []
    for grp, sub in trials.groupby(group_col):
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = sub[(sub.pred_px >= lo) & (sub.pred_px < hi)]
            n, k = int(len(b)), int(b.detected.sum())
            lo_ci, hi_ci = _wilson(k, n)
            rows.append({"group": grp, "bin_lo": lo, "bin_hi": hi, "n": n,
                         "n_detected": k,
                         "rate": (k / n) if n else float("nan"),
                         "ci_lo": lo_ci, "ci_hi": hi_ci})
    return pd.DataFrame(rows)
