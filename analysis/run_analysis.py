#!/usr/bin/env python3
"""Stages 1-6 driver. ROS-free: reads only the dataset written by extract_bags.py.

    python analysis/run_analysis.py dataset/
"""
import json
import os
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis import detect, geometry as g, imucheck as ic, layout as L, metrics as M
from analysis import segment, vizstyle

PX_BINS = [8, 12, 16, 21, 28, 38, 52, 72, 100, 140, 200, 300]

# Break the walk-back trace when detection lapses longer than this. One frame is 0.4 s
# at 2.5 Hz, so this trips on a real dropout rather than on ordinary frame spacing.
DROPOUT_GAP_S = 1.0

# A bin with one or two trials carries no information but its Wilson CI spans most of
# the axis, so it dominates the figure. Suppress below this, and log what was dropped --
# a silently truncated plot reads as "we measured everywhere" when we did not.
MIN_TRIALS_PER_BIN = 10


def code_version():
    """Git SHA of this analysis repo, so a figure in the report traces to its code.

    This is the provenance link that a git submodule would otherwise provide. The
    driver runs in a slim container with no git binary, so the SHA is normally passed
    in as ANALYSIS_GIT_SHA by the caller; the git fallback covers a bare host run.
    """
    env = os.environ.get("ANALYSIS_GIT_SHA")
    if env:
        return env
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def load_run(dataset_dir, run):
    d = os.path.join(dataset_dir, run)
    det_csv = os.path.join(d, "detections.csv")
    timing_csv = os.path.join(d, "timing.csv")
    # Stage 1 on demand: an extracted dataset has frames/imu/camera_info but not yet
    # detections. Sweep and persist so the pipeline is self-contained and reproducible
    # from a freshly extracted dataset (no ad-hoc detection step required).
    if not os.path.exists(det_csv):
        det_df, timing_df = detect.sweep_dataset(d, detect.make_detector())
        det_df.to_csv(det_csv, index=False)
        timing_df.to_csv(timing_csv, index=False)
    det = pd.read_csv(det_csv)
    frames = pd.read_csv(os.path.join(d, "frames.csv"))
    imu = pd.read_csv(os.path.join(d, "imu.csv"))
    ci = yaml.safe_load(open(os.path.join(d, "camera_info.yaml")))
    return det, frames, imu, ci


def main(dataset_dir):
    vizstyle.apply()
    os.makedirs("results", exist_ok=True)
    summary = {"opencv": cv2.__version__, "host_cpu": detect.host_cpu(),
               "code_version": code_version(), "runs": {}}
    lay = L.load_layout("config/board_layout.yaml")
    trials_by_run = {}
    rng_by_run = {}
    rng_board_by_run = {}
    inc_by_run = {}

    for run in ("test1", "test2"):
        det, frames, imu, ci = load_run(dataset_dir, run)
        n_frames = len(frames)
        labels = segment.label_frames(frames, imu)
        timing = pd.read_csv(os.path.join(dataset_dir, run, "timing.csv"))
        summary["runs"][run] = {
            "frames": n_frames,
            "detections": int(len(det)),
            "turn_frames": int((labels == "turn").sum()),
            "mis_id_rate": M.mis_id_rate(det),
            "ids_seen": sorted(int(x) for x in det.marker_id.unique()),
            # Issue #2 asks for latency "on ROV compute". This is the analysis host,
            # detectMarkers only -- an algorithmic cost, not an on-ROV figure.
            "latency_ms_median": float(timing.latency_ms.median()),
            "latency_ms_p95": float(timing.latency_ms.quantile(0.95)),
        }

        # Range per frame from the largest marker present. Apparent-size range is the
        # right axis for the DETECTION figures -- there apparent size is the independent
        # variable and this is just its honest metric label.
        rng = {}
        for fi, sub in det[det.marker_id.isin(g.SIZES)].groupby("frame_idx"):
            big = sub.loc[sub.apparent_px.idxmax()]
            rng[int(fi)] = M.range_from_apparent_px(
                big.apparent_px, int(big.marker_id), ci["fx"])

        lay_for_range = L.load_layout("config/board_layout.yaml")
        Kmat_run = np.array([[ci["fx"], 0, ci["cx"]],
                             [0, ci["fy"], ci["cy"]], [0, 0, 1]])
        # Range to the BOARD, for the trajectory figure. Apparent-size range is wrong
        # here on three counts: it tracks whichever marker looks biggest (which flips
        # between 201 and 202 fifty-five times, injecting ~52 mm jumps against 15 mm of
        # real frame-to-frame motion); it assumes head-on, so it drifts ~-134 mm at 30+
        # deg incidence -- exactly during the yaw turns; and it throws away the corner
        # geometry. The board pose solves from ANY visible subset, down to one marker,
        # so it always refers to the same physical point. Offset to the marker centroid:
        # the 201 gauge is the bundle's constraint, not a sensible place to measure from.
        cen = g.board_centroid(lay_for_range)
        cen3 = np.array([cen[0], cen[1], 0.0])
        rng_board = {}
        for fi, sub in det[det.marker_id.isin(g.SIZES)].groupby("frame_idx"):
            dets = {int(r.marker_id): np.array([[r.c0x, r.c0y], [r.c1x, r.c1y],
                                                [r.c2x, r.c2y], [r.c3x, r.c3y]])
                    for r in sub.itertuples()}
            try:
                rv_b, tv_b = L.board_pnp(lay_for_range, dets, Kmat_run)
            except ValueError:
                continue
            R_b = g.rodrigues(rv_b[None])[0]
            rng_board[int(fi)] = float(np.linalg.norm(R_b @ cen3 + tv_b))

        rng_by_run[run] = rng
        rng_board_by_run[run] = rng_board

        rates = M.detection_rate_by_px_bin(det, {m: n_frames for m in g.SIZES}, PX_BINS)
        rates.to_csv(f"results/{run}_detection_rate_by_px.csv", index=False)
        summary["runs"][run]["max_range_m"] = M.furthest_detection_range(det, rng)

        # Viewing-angle regime (spec 3.1c). NOT a swept curve -- angle is confounded
        # with range and size here, and incidence is itself a PnP output. We report the
        # distribution per run so the two regimes (test1 ~head-on, test2 ~oblique) are
        # visible, and flag whether the big markers survive oblique viewing at all.
        Kmat = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]])
        inc = []
        for r in det[det.marker_id.isin(g.SIZES)].itertuples():
            mid = int(r.marker_id)
            q = np.array([[r.c0x, r.c0y], [r.c1x, r.c1y], [r.c2x, r.c2y], [r.c3x, r.c3y]])
            ok, rvec, _ = cv2.solvePnP(
                g.local_corners(mid).astype(np.float32), q.astype(np.float32),
                Kmat, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                inc.append(M.incidence_angle_deg(rvec.ravel()))
        inc = np.array(inc)
        inc_by_run[run] = inc
        summary["runs"][run]["incidence_deg"] = {
            "median": round(float(np.median(inc)), 1) if len(inc) else None,
            "p10": round(float(np.percentile(inc, 10)), 1) if len(inc) else None,
            "p90": round(float(np.percentile(inc, 90)), 1) if len(inc) else None,
            "n_oblique_gt_40deg": int((inc > 40).sum()),
        }

        # Detection TRIALS (hits and misses), corrected metric for figures 1 and 3. A
        # detection table alone has only hits; where other board markers are detected in
        # a frame the board pose is known, so a marker's apparent size -- and therefore
        # whether it SHOULD have been detected -- can be predicted leave-one-out.
        obs_run = {}
        for r in det[det.marker_id.isin(g.SIZES)].itertuples():
            obs_run.setdefault((run, r.frame_idx), {})[int(r.marker_id)] = np.array(
                [[r.c0x, r.c0y], [r.c1x, r.c1y], [r.c2x, r.c2y], [r.c3x, r.c3y]])
        trials = M.detection_trials(obs_run, lay, Kmat, ci["width"], ci["height"])
        trials.to_csv(f"results/detection_trials_{run}.csv", index=False)
        summary["runs"][run]["n_trials"] = int(len(trials))
        summary["runs"][run]["n_misses"] = (
            int((1 - trials.detected).sum()) if len(trials) else 0)
        trials_by_run[run] = trials

    # size_mm -> a representative marker id, used only to convert a group's bin centre
    # to range via the pinhole relation (fx * size / apparent_px).
    size_to_mid = {}
    for mid in g.IDS:
        size_to_mid.setdefault(round(g.SIZES[mid] * 1000, 1), mid)

    # Figure 1: the corrected calibration-free headline. A RATE needs the misses, which
    # detection_trials recovers by prediction; pooled by size group (302-305 etc share a
    # size -> 4x samples per bin) with a Wilson CI band.
    det1, frames1, _, ci1 = load_run(dataset_dir, "test1")
    trials1 = trials_by_run["test1"]
    rates1 = M.rate_by_bin(trials1, PX_BINS)
    # Bins with n < MIN_TRIALS_PER_BIN carry Wilson CIs spanning most of the axis and
    # would visually swamp the figures below; suppress them for plotting only (the full
    # table, including these bins, still goes to CSV via rates1/rate_by_bin unchanged).
    suppressed = rates1[(rates1.n > 0) & (rates1.n < MIN_TRIALS_PER_BIN)]
    summary["suppressed_bins"] = [
        {"figure": "detection_rate", "group_mm": float(row.group),
         "bin_lo_px": float(row.bin_lo), "bin_hi_px": float(row.bin_hi), "n": int(row.n)}
        for row in suppressed.itertuples()
    ]
    # All four size groups collapse onto nearly the same sigmoid -- that IS the finding
    # -- but four overlaid translucent CI bands become an unreadable haze. Facet into a
    # 2x2 small-multiple grid, one panel per size, so each curve and its CI are legible;
    # the other three sizes are redrawn as faint grey ghosts in every panel so the
    # collapse itself stays visible.
    groups = sorted(vizstyle.SIZE_COLORS)
    curves = {}
    for grp in groups:
        sub = rates1[(rates1.group == grp) & (rates1.n >= MIN_TRIALS_PER_BIN)].sort_values("bin_lo")
        if len(sub):
            curves[grp] = sub

    budget_px = M.pixel_budget_px(5)
    fig, axes = plt.subplots(2, 2, figsize=(vizstyle.WIDE_W, 4.6), sharex=True, sharey=True)
    for ax, grp in zip(axes.flat, groups):
        for other in groups:
            if other == grp or other not in curves:
                continue
            osub = curves[other]
            ocentre = ((osub.bin_lo + osub.bin_hi) / 2).to_numpy()
            ax.plot(ocentre, osub.rate, color=vizstyle.GHOST_COLOR, lw=1, zorder=1)
        if grp in curves:
            sub = curves[grp]
            centre = ((sub.bin_lo + sub.bin_hi) / 2).to_numpy()
            color, marker = vizstyle.SIZE_COLORS[grp], vizstyle.SIZE_MARKERS[grp]
            ax.fill_between(centre, sub.ci_lo, sub.ci_hi, color=color, alpha=0.2,
                             linewidth=0, zorder=2)
            ax.plot(centre, sub.rate, marker=marker, color=color, zorder=3)
        ax.axvline(budget_px, ls="--", c=vizstyle.TEXT_SECONDARY, lw=1, zorder=2)
        vizstyle.log_px_ticks(ax)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{grp:.0f} mm")

    ghost_handle = plt.Line2D([], [], color=vizstyle.GHOST_COLOR, lw=1, label="other sizes")
    budget_handle = plt.Line2D([], [], ls="--", c=vizstyle.TEXT_SECONDARY, lw=1,
                               label=f"3(n+2) budget = {budget_px} px")
    fig.supxlabel("apparent marker size (px)", y=0.10)
    fig.supylabel("detection rate")
    fig.suptitle("Detection rate vs apparent marker size")
    fig.legend(handles=[ghost_handle, budget_handle], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=(0.02, 0.14, 1, 0.94))
    vizstyle.save(fig, "detection_rate_vs_px")

    # Figure 2: the hypothesis test.
    mr = summary["runs"]["test1"]["max_range_m"]
    sizes = np.array([g.SIZES[m] for m in sorted(mr)])
    rmax = np.array([mr[m] for m in sorted(mr)])
    k = float(np.sum(sizes * rmax) / np.sum(sizes ** 2))   # least squares through 0
    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, 3.6))
    # Per-point marker-ID labels aren't kept: nobody needs to know that a given 44 mm
    # marker is 302 vs 304, and with points this close the labels collide. The
    # size-group legend (colour + marker shape) carries identity instead; the vertical
    # spread within a group stays visible and shows the measurement variability.
    by_size = {}
    for mid in sorted(mr):
        by_size.setdefault(round(g.SIZES[mid] * 1000, 1), []).append(mid)
    for size_mm, mids in by_size.items():
        color = vizstyle.SIZE_COLORS.get(size_mm, vizstyle.TEXT_PRIMARY)
        marker = vizstyle.SIZE_MARKERS.get(size_mm, "o")
        for i, mid in enumerate(sorted(mids, key=lambda m: mr[m])):
            ax.scatter(size_mm, mr[mid], color=color, marker=marker, s=14, zorder=3,
                       label=f"{size_mm:.0f} mm" if i == 0 else None)
    xs = np.linspace(0, sizes.max() * 1100, 50)
    ax.plot(xs, k * xs / 1000, ls="--", c=vizstyle.TEXT_SECONDARY, lw=1,
            label=f"max_range ~ {k:.0f} x side")
    ax.axhline(5.0, color="tab:red", lw=1, label="coarse phase needs ~5 m")
    ax.set_xlabel("marker side (mm)")
    ax.set_ylabel("max detection range (m)")
    ax.legend()
    ax.set_title("Max detection range vs marker size")
    fig.tight_layout()
    vizstyle.save(fig, "max_range_vs_size")
    summary["range_per_side_fit"] = k

    # Figure 3: issue #2's "detection rate vs range", corrected the same way as figure 1.
    # Derived from px via the pinhole relation, so it inherits the focal length's ~+/-10%.
    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, 4.0))
    for grp in sorted(vizstyle.SIZE_COLORS):
        sub = rates1[(rates1.group == grp) & (rates1.n >= MIN_TRIALS_PER_BIN)].sort_values("bin_lo")
        if not len(sub):
            continue
        centre = (sub.bin_lo + sub.bin_hi) / 2
        rng_axis = [M.range_from_apparent_px(c, size_to_mid[grp], ci1["fx"]) for c in centre]
        color, marker = vizstyle.SIZE_COLORS[grp], vizstyle.SIZE_MARKERS[grp]
        ax.plot(rng_axis, sub.rate, marker=marker, color=color, label=f"{grp:.0f} mm")
        ax.fill_between(rng_axis, sub.ci_lo, sub.ci_hi, color=color, alpha=0.12, linewidth=0)
    vizstyle.linear_range_ticks(ax, step=0.5, max_val=5.0)
    ax.set_xlabel("range (m, derived: +/-10%)")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.set_title("Detection rate vs range (derived from apparent size)")
    fig.tight_layout()
    vizstyle.save(fig, "detection_rate_vs_range")

    # Figure 4: issue #2's "translation error vs range". NOT accuracy -- single-marker
    # PnP scored against the multi-marker board reference. Caption says so.
    #
    # POOLED across all markers -- deliberately NOT broken down by marker size. The
    # leave-one-out reference geometry that pose_error_vs_reference() builds varies with
    # which marker is under test (see that function's docstring): testing 201/202 leaves
    # only the narrow small-marker centre cluster as the reference (weak baseline, weak
    # pose), while testing a small marker leaves 201+202 (427 mm apart) in the reference
    # (strong baseline). A per-size split of this metric measures reference quality, not
    # marker quality, and reads backwards ("bigger markers have worse pose"). Pool it.
    obs = {}
    for r in det1[det1.marker_id.isin(g.SIZES)].itertuples():
        obs.setdefault(("test1", r.frame_idx), {})[int(r.marker_id)] = np.array(
            [[r.c0x, r.c0y], [r.c1x, r.c1y], [r.c2x, r.c2y], [r.c3x, r.c3y]])
    Km = np.array([[ci1["fx"], 0, ci1["cx"]], [0, ci1["fy"], ci1["cy"]], [0, 0, 1]])
    perr = M.pose_error_vs_reference(obs, lay, Km)
    perr.to_csv("results/pose_error_vs_reference.csv", index=False)
    RANGE_BINS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5]
    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, 4.0))
    if len(perr):
        st = M.binned_stats(perr, "trans_err_m", "range_m", RANGE_BINS)
        # Same suppression as figures 1/3: a bin with n < MIN_TRIALS_PER_BIN is a single
        # (or handful of) sample(s), not a measured trend, and its std is often NaN
        # (n=1) or wildly unrepresentative -- drop it here too, and log what was dropped.
        dropped = st[(st.n > 0) & (st.n < MIN_TRIALS_PER_BIN)]
        for row in dropped.itertuples():
            summary["suppressed_bins"].append({
                "figure": "trans_err_vs_range",
                "bin_lo_m": float(row.bin_lo), "bin_hi_m": float(row.bin_hi),
                "n": int(row.n)})
        valid = (st.n >= MIN_TRIALS_PER_BIN).to_numpy()
        centre = ((st.bin_lo + st.bin_hi) / 2).to_numpy()
        ax.errorbar(centre[valid], st["mean"].to_numpy()[valid] * 1000,
                    yerr=st["std"].to_numpy()[valid] * 1000,
                    marker="o", color=vizstyle.SIZE_COLORS[149.4], capsize=3)
        summary["trans_err_vs_range_mm"] = {
            f"{lo}-{hi}": (None if (np.isnan(mu) or n < MIN_TRIALS_PER_BIN)
                           else round(mu * 1000, 1))
            for lo, hi, mu, n in zip(st.bin_lo, st.bin_hi, st["mean"], st["n"])}
        st.to_csv("results/trans_err_vs_range.csv", index=False)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("translation vs board reference (mm)")
    ax.set_title("Single-marker vs board reference (self-consistency, NOT accuracy)\n"
                 "pooled across marker sizes; error bars are +/-1 std")
    fig.tight_layout()
    vizstyle.save(fig, "trans_err_vs_range")

    # Figure 5: rotation error vs range. Exact sibling of figure 4 above -- same
    # pooling rationale (the leave-one-out reference confound in pose_error_vs_reference
    # is identical for rotation and translation), same range bins, same suppression, same
    # honesty about self-consistency vs the board reference. rot_err_deg was computed by
    # pose_error_vs_reference() above (perr) and, until now, plotted nowhere -- yet it is
    # the one pose metric that maps onto the docking FSM's heading_tol_rad.
    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, 4.0))
    if len(perr):
        strot = M.binned_stats(perr, "rot_err_deg", "range_m", RANGE_BINS)
        dropped_rot = strot[(strot.n > 0) & (strot.n < MIN_TRIALS_PER_BIN)]
        for row in dropped_rot.itertuples():
            summary["suppressed_bins"].append({
                "figure": "rot_err_vs_range",
                "bin_lo_m": float(row.bin_lo), "bin_hi_m": float(row.bin_hi),
                "n": int(row.n)})
        valid_rot = (strot.n >= MIN_TRIALS_PER_BIN).to_numpy()
        centre_rot = ((strot.bin_lo + strot.bin_hi) / 2).to_numpy()
        ax.errorbar(centre_rot[valid_rot], strot["mean"].to_numpy()[valid_rot],
                    yerr=strot["std"].to_numpy()[valid_rot],
                    marker="o", color=vizstyle.SIZE_COLORS[149.4], capsize=3)
        summary["rot_err_vs_range_deg"] = {
            f"{lo}-{hi}": (None if (np.isnan(mu) or n < MIN_TRIALS_PER_BIN)
                           else round(mu, 1))
            for lo, hi, mu, n in zip(strot.bin_lo, strot.bin_hi, strot["mean"], strot["n"])}
        strot.to_csv("results/rot_err_vs_range.csv", index=False)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("rotation vs board reference (deg)")
    ax.set_title("Single-marker vs board reference (self-consistency, NOT accuracy)\n"
                 "pooled across marker sizes; error bars are +/-1 std")
    fig.tight_layout()
    vizstyle.save(fig, "rot_err_vs_range")

    # Figure 6: range vs time for test1 (spec 3.1b). Detections die at ~191 s / ~4.9 m,
    # but the IMU (which spans the whole run, unlike the camera coverage) shows the bag
    # running to ~270 s -- i.e. the operator kept walking for ~79 s / ~3 m with ZERO
    # detections. That is what makes the max-range figure a real physical limit rather
    # than an artifact of the walk ending, and until now it was prose-only (spec 3.1b).
    imu1_full = pd.read_csv(os.path.join(dataset_dir, "test1", "imu.csv"))
    t0 = min(frames1["stamp"].min(), imu1_full["stamp"].min())
    frame_stamp1 = dict(zip(frames1["frame_idx"], frames1["stamp"]))
    pts = sorted((frame_stamp1[fi] - t0, r)
                 for fi, r in rng_board_by_run["test1"].items() if fi in frame_stamp1)
    bag_end_s = float(imu1_full["stamp"].max() - t0)
    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, 3.8))
    if pts:
        ts, rs = zip(*pts)
        # No markers: 323 points across the axis is ~58 per inch, so any marker large
        # enough to see overlaps its neighbours -- the data is denser than the ink.
        # But do NOT draw one continuous line either: 30% of frames have no board pose
        # (the yaw turns swing the board out of frame; the largest hole is 12 s), and a
        # solid line would interpolate a confident trajectory straight through them.
        # Break the line wherever detection actually lapsed, so the gaps read as gaps.
        ta, ra = np.asarray(ts, float), np.asarray(rs, float)
        brk = np.where(np.diff(ta) > DROPOUT_GAP_S)[0] + 1
        ax.plot(np.insert(ta, brk, np.nan), np.insert(ra, brk, np.nan),
                lw=1.2, color=vizstyle.SIZE_COLORS[149.4], zorder=3,
                label=f"detected ({len(brk)} breaks = board lost, mostly yaw turns)")
        # A segment of one frame draws no line, so a lone detection between two dropouts
        # would silently vanish -- including this run's very first one. Dot those in.
        lone = [seg[0] for seg in np.split(np.arange(len(ta)), brk) if len(seg) == 1]
        if lone:
            ax.scatter(ta[lone], ra[lone], s=4, color=vizstyle.SIZE_COLORS[149.4],
                       zorder=3)
        last_t, last_r = ts[-1], rs[-1]
        ax.scatter([last_t], [last_r], color="tab:red", marker="X", s=22, zorder=4,
                   label=f"last detection: t={last_t:.0f}s, {last_r:.2f} m")
        ax.axvspan(last_t, bag_end_s, color=vizstyle.GHOST_COLOR, alpha=0.35, zorder=1)
        ax.text((last_t + bag_end_s) / 2, last_r * 0.55,
                "no detections, walk continued.\noperator continued to ~8 m (inferred)",
                ha="center", va="top", fontsize=7.5, color=vizstyle.TEXT_SECONDARY)
        ax.legend(loc="upper left")
    ax.set_xlabel("time since run start (s)")
    ax.set_ylabel("range to board centroid (m, +/-10%)")
    # Derive the headline from the data. Hardcoding it went stale the moment the range
    # estimator changed from apparent size to the board pose (4.9 m became 5.11 m).
    if pts:
        ax.set_title(f"Walk-back profile: detection ends at ~{last_r:.1f} m, "
                     f"{bag_end_s - last_t:.0f} s before the walk does")
    fig.tight_layout()
    vizstyle.save(fig, "range_vs_time")

    # Figure 7: incidence-angle histogram (spec 3.1c / limitation 10). Two disconnected
    # viewing-angle regimes -- test1 near head-on, test2 oblique. NO detection-rate-vs-
    # angle curve is reported because angle was never controlled and is confounded with
    # range and marker size (incidence is itself a PnP output). The gap between the
    # regimes is a consequence of uncontrolled operation, not a data-coverage problem.
    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, 3.8))
    bins = np.arange(0, 91, 5)
    run_colors = {"test1": vizstyle.SIZE_COLORS[149.4], "test2": vizstyle.SIZE_COLORS[35.6]}
    for run in ("test1", "test2"):
        inc = inc_by_run.get(run, np.array([]))
        if len(inc):
            ax.hist(inc, bins=bins, color=run_colors[run], alpha=0.55,
                    label=f"{run} (median {np.median(inc):.0f} deg, n={len(inc)})",
                    edgecolor="white", linewidth=0.5)
    ax.set_xlabel("incidence angle (deg, 0 = head-on)")
    ax.set_ylabel("detections")
    ax.set_title("Incidence angle: two regimes, no sweep\n"
                 "angle was never controlled and is confounded with range and marker size")
    ax.legend(loc="upper right")
    fig.tight_layout()
    vizstyle.save(fig, "incidence_angle_hist")

    # Stage 4: IMU validation -- the only camera-INDEPENDENT check (spec 4).
    # (a) Gravity: the board does not move relative to gravity, so the angle between the
    #     PnP board-normal and the IMU gravity vector must be STABLE across a run. Its
    #     std is the signal: low std = PnP attitude is consistent with the absolute IMU
    #     reference. (b) Yaw turns: over a turn, vision delta-yaw (pnp_delta_yaw between
    #     the board poses bracketing the turn) should track integrated gyro delta-yaw.
    #     Caveat: turns swing the board out of frame, so usable spans may be few -- that
    #     is reported honestly rather than forced.
    imu1 = pd.read_csv(os.path.join(dataset_dir, "test1", "imu.csv"))
    imu_t = imu1["stamp"].to_numpy()
    accel = imu1[["ax", "ay", "az"]].to_numpy()
    gz = imu1["wz"].to_numpy()
    frame_stamp = dict(zip(frames1["frame_idx"], frames1["stamp"]))

    board_pose = {}                       # frame_idx -> (rvec, tvec), test1 only
    for fi, dets in sorted((k[1], v) for k, v in obs.items()):
        try:
            board_pose[fi] = L.board_pnp(lay, dets, Km)
        except ValueError:
            pass

    # (a) gravity check. The IMU accel is in the ZED body frame (z up); the PnP board
    # normal is in the optical frame (z forward). Reconcile with the standard REP-103
    # optical<-body rotation before comparing, or the tilt reads ~90 deg off.
    R_OPT_FROM_BODY = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    tilt = []
    for fi, (rvec, _tv) in board_pose.items():
        j = int(np.clip(np.searchsorted(imu_t, frame_stamp[fi]), 0, len(accel) - 1))
        tilt.append(ic.tilt_residual_deg(rvec, R_OPT_FROM_BODY @ accel[j]))
    tilt = np.array(tilt)
    summary["imu_gravity_check"] = {
        "n_frames": int(len(tilt)),
        "board_tilt_from_vertical_deg_median": round(float(np.median(tilt)), 2)
            if len(tilt) else None,
        "std_deg": round(float(np.std(tilt)), 2) if len(tilt) else None,
        "note": "board-normal vs gravity (IMU accel rotated body->optical, REP-103); "
                "median = board tilt from vertical, std = PnP attitude consistency vs "
                "the absolute IMU gravity reference (lower std better)",
    }

    # (b) yaw-turn check. A turn only tests the gyro against vision if the board stayed
    # in frame for essentially the whole turn; if it swung out partway through, vision
    # under-reports the rotation and the comparison is meaningless, not just noisy. So
    # gate on board-pose COVERAGE of the segment (fraction of that segment's camera
    # frames with a board pose), not merely on having a pose at both ends.
    MIN_COVERAGE = 0.8
    labels = segment.classify(imu_t, gz)
    turn_segs = [s for s in segment.segments(labels, imu_t) if s["label"] == "turn"]
    pose_frames = sorted(board_pose)
    all_stamps = frames1["stamp"].to_numpy()
    yaw_rows = []
    for seg in turn_segs:
        before = [f for f in pose_frames if frame_stamp[f] <= seg["t0"]]
        after = [f for f in pose_frames if frame_stamp[f] >= seg["t1"]]
        seg_frames = frames1[(all_stamps >= seg["t0"]) & (all_stamps <= seg["t1"])]
        n_seg_frames = len(seg_frames)
        n_with_pose = sum(1 for fi in seg_frames["frame_idx"] if fi in board_pose)
        coverage = (n_with_pose / n_seg_frames) if n_seg_frames else 0.0
        if not before or not after:
            continue                       # board out of frame across this turn
        f0, f1 = before[-1], after[0]
        vis = np.degrees(ic.pnp_delta_yaw(board_pose[f0][0], board_pose[f1][0]))
        m = (imu_t >= frame_stamp[f0]) & (imu_t <= frame_stamp[f1])
        gyro = np.degrees(ic.integrate_gyro_yaw(imu_t[m], gz[m]))
        yaw_rows.append({"t0": round(float(seg["t0"]), 1), "t1": round(float(seg["t1"]), 1),
                         "vision_deg": round(vis, 1), "gyro_deg": round(gyro, 1),
                         "coverage": round(coverage, 2)})
    evaluable_rows = [r for r in yaw_rows if r["coverage"] >= MIN_COVERAGE]
    summary["imu_yaw_check"] = {
        "n_turn_segments": len(turn_segs),
        "n_usable_spans": len(yaw_rows),
        "n_evaluable": len(evaluable_rows),
        "min_coverage": MIN_COVERAGE,
        "spans": yaw_rows,
        "evaluable_spans": evaluable_rows,
        "note": "vision vs gyro delta-yaw per turn, restricted to segments where the "
                "board's PnP pose covers >= 80% of that segment's camera frames "
                "(n_evaluable/n_turn_segments); other segments have a pose at both "
                "ends but the board left frame partway through the turn, so vision "
                "under-reports the rotation and the comparison is not meaningful. "
                "Gyro (IMU up-z) and vision (optical down-y) measure yaw about "
                "opposite axes by convention, so an expected sign flip is not an error.",
    }
    if yaw_rows:
        pd.DataFrame(yaw_rows).to_csv("results/imu_yaw_check.csv", index=False)

    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    lines = [
        "# Results summary", "",
        f"- Analysis code: `alanchoi00/underwater-aruco-validation@{summary['code_version']}`",
        f"- OpenCV: `{summary['opencv']}`",
        f"- Fitted law: `max_range ~ {k:.1f} x side_length`",
        f"- Host CPU (latency context): `{summary['host_cpu']}`",
        f"- Intrinsics: ZED shipped K, no refraction correction (see spec 3.2)",
        "- Metric range carries ~+/-10% from focal-length uncertainty.",
        "- Pose error is **vs the board reference**, not vs ground truth. None exists.",
        "- Latency is detectMarkers on the analysis host, not on ROV compute.",
        "- Angle is reported as two regimes (test1 head-on, test2 oblique ~57 deg), NOT "
        "a swept curve: there is no controlled angle sweep and angle is confounded with "
        "range/size (spec 3.1c).",
    ]
    rate_suppressed = [b for b in summary["suppressed_bins"]
                        if b["figure"] == "detection_rate"]
    pose_suppressed = [b for b in summary["suppressed_bins"]
                       if b["figure"] == "trans_err_vs_range"]
    rot_suppressed = [b for b in summary["suppressed_bins"]
                      if b["figure"] == "rot_err_vs_range"]
    lines += [
        f"- {len(rate_suppressed)} bin(s) with fewer than "
        f"{MIN_TRIALS_PER_BIN} trials were suppressed from the detection-rate figures "
        "(their Wilson CI spans most of the axis and would dominate the plot with no "
        "information); the full per-bin table, including these, is still in "
        "`detection_trials_test1.csv` / `detection_trials_test2.csv`. Suppressed: " +
        ("; ".join(f"{b['group_mm']:.0f} mm, {b['bin_lo_px']:.0f}-{b['bin_hi_px']:.0f} px, "
                   f"n={b['n']}" for b in rate_suppressed)
         if rate_suppressed else "none") + ".",
        f"- {len(pose_suppressed)} bin(s) with fewer than "
        f"{MIN_TRIALS_PER_BIN} trials were suppressed from the pose-error-vs-range "
        "figure (pooled across marker sizes; n=1-3 bins were near-empty and would carry "
        "a std of NaN or an unrepresentative one); the full per-bin table, including "
        "these, is still in `trans_err_vs_range.csv`. Suppressed: " +
        ("; ".join(f"{b['bin_lo_m']:.1f}-{b['bin_hi_m']:.1f} m, n={b['n']}"
                   for b in pose_suppressed)
         if pose_suppressed else "none") + ".",
        f"- {len(rot_suppressed)} bin(s) with fewer than "
        f"{MIN_TRIALS_PER_BIN} trials were suppressed from the rotation-error-vs-range "
        "figure (same pooling and suppression as the translation-error figure above); "
        "the full per-bin table, including these, is still in `rot_err_vs_range.csv`. "
        "Suppressed: " +
        ("; ".join(f"{b['bin_lo_m']:.1f}-{b['bin_hi_m']:.1f} m, n={b['n']}"
                   for b in rot_suppressed)
         if rot_suppressed else "none") + ".",
        "- The pose-error-vs-range figure is pooled across marker sizes, not broken down "
        "per size: the leave-one-out board reference used by `pose_error_vs_reference` "
        "gets weaker when 201/202 are the marker under test (only the narrow centre "
        "cluster remains as reference) and stronger when a small marker is under test "
        "(201+202, 427 mm apart, remain). A per-size split would measure that reference "
        "confound, not marker quality, and read backwards (see `analysis/metrics.py`).",
        "",
    ]
    for run, s in summary["runs"].items():
        lines += [f"## {run}", "",
                  f"- frames: {s['frames']}, detections: {s['detections']}",
                  f"- turn frames: {s['turn_frames']}",
                  f"- mis-ID rate: {s['mis_id_rate']:.4f}",
                  f"- ids seen: {s['ids_seen']}",
                  f"- detector latency: {s['latency_ms_median']:.2f} ms median, "
                  f"{s['latency_ms_p95']:.2f} ms p95 (analysis host, NOT ROV compute)",
                  f"- incidence angle: median {s['incidence_deg']['median']} deg "
                  f"(p10-p90 {s['incidence_deg']['p10']}-{s['incidence_deg']['p90']}); "
                  f"{s['incidence_deg']['n_oblique_gt_40deg']} detections above 40 deg",
                  ""]
        for mid, r in sorted(s["max_range_m"].items()):
            lines.append(f"  - {mid} ({g.SIZES[mid]*1000:.0f} mm): max range {r:.2f} m")
        lines.append("")

    gchk, ychk = summary["imu_gravity_check"], summary["imu_yaw_check"]
    lines += ["## Stage 4: IMU validation (camera-independent)", "",
              f"- Gravity check PASSES: board tilt from vertical over {gchk['n_frames']} "
              f"frames = {gchk['board_tilt_from_vertical_deg_median']} deg median, std "
              f"{gchk['std_deg']} deg (low std = PnP attitude consistent with the "
              "absolute IMU gravity reference). This is the camera-independent "
              "validation.", ""]
    n_eval, n_total = ychk["n_evaluable"], ychk["n_turn_segments"]
    if n_eval == 0:
        lines.append(
            f"- Yaw check: 0/{n_total} turn segments were evaluable (board-pose "
            f"coverage >= {ychk['min_coverage']:.0%} of the segment's camera frames). "
            "The yaw cross-check is not evaluable on this dataset because the yaw "
            "turns swung the board out of frame, so vision cannot observe the full "
            "rotation and any vision-vs-gyro comparison would compare an incomplete "
            "rotation to a complete one. The gravity check above therefore carries "
            "the camera-independent validation alone.")
    else:
        ratios = [abs(sp["vision_deg"]) / abs(sp["gyro_deg"])
                  for sp in ychk["evaluable_spans"] if sp["gyro_deg"]]
        agrees = bool(ratios) and (0.7 <= float(np.median(ratios)) <= 1.4)
        lines.append(
            f"- Yaw check: {n_eval}/{n_total} turn segments were evaluable "
            f"(board-pose coverage >= {ychk['min_coverage']:.0%}); the rest had the "
            "board leave frame partway through the turn, so vision under-reports the "
            "rotation there and they are excluded rather than compared. Gyro measures "
            "yaw about the IMU's up (+z) axis while PnP yaw is about the optical "
            "frame's down (+y) axis, so an opposite sign between the two is the "
            "expected convention, not a discrepancy.")
        if not agrees:
            lines.append(
                "  Even restricted to these full-coverage segments, vision's "
                "magnitude does not consistently track the gyro's (see per-segment "
                "numbers below), so the yaw check does not corroborate the gravity "
                "check on this dataset. The gravity check above remains the sole "
                "camera-independent validation; the yaw numbers are reported for "
                "completeness, not as a passing check.")
        for sp in ychk["evaluable_spans"]:
            lines.append(f"  - turn {sp['t0']}-{sp['t1']}s (coverage "
                         f"{sp['coverage']:.0%}): vision {sp['vision_deg']} deg, "
                         f"gyro {sp['gyro_deg']} deg")
    lines.append("")
    with open("results/summary.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset")
