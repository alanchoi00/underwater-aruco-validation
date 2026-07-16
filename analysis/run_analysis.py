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

# A bin with one or two trials carries no information but its Wilson CI spans most of
# the axis, so it dominates the figure. Suppress below this, and log what was dropped --
# a silently truncated plot reads as "we measured everywhere" when we did not.
MIN_TRIALS_PER_BIN = 10


def code_version():
    """Git SHA of this analysis repo, so a figure in the report traces to its code.

    This is the provenance link that a git submodule would otherwise provide. The
    docking repo does not vendor this repo (nothing there imports it), so the commit
    is recorded in the results instead.
    """
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

        # Range per frame from the largest marker present -- the most reliable one.
        rng = {}
        for fi, sub in det[det.marker_id.isin(g.SIZES)].groupby("frame_idx"):
            big = sub.loc[sub.apparent_px.idxmax()]
            rng[int(fi)] = M.range_from_apparent_px(
                big.apparent_px, int(big.marker_id), ci["fx"])

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
        {"group_mm": float(row.group), "bin_lo_px": float(row.bin_lo),
         "bin_hi_px": float(row.bin_hi), "n": int(row.n)}
        for row in suppressed.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, 4.0))
    for grp in sorted(vizstyle.SIZE_COLORS):
        sub = rates1[(rates1.group == grp) & (rates1.n >= MIN_TRIALS_PER_BIN)].sort_values("bin_lo")
        if not len(sub):
            continue
        centre = ((sub.bin_lo + sub.bin_hi) / 2).to_numpy()
        color, marker = vizstyle.SIZE_COLORS[grp], vizstyle.SIZE_MARKERS[grp]
        ax.plot(centre, sub.rate, marker=marker, color=color, label=f"{grp:.0f} mm")
        ax.fill_between(centre, sub.ci_lo, sub.ci_hi, color=color, alpha=0.15, linewidth=0)
        ax.annotate(f"{grp:.0f} mm", (centre[-1], sub.rate.to_numpy()[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=color, va="center")
    ax.axvline(M.pixel_budget_px(5), ls="--", c=vizstyle.TEXT_SECONDARY, lw=1,
               label=f"3(n+2) budget = {M.pixel_budget_px(5)} px")
    ax.set_xscale("log")
    ax.set_xlabel("apparent marker size (px)")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.set_title("Detection rate vs apparent marker size")
    fig.tight_layout()
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
            ax.scatter(size_mm, mr[mid], color=color, marker=marker, zorder=3,
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
        ax.fill_between(rng_axis, sub.ci_lo, sub.ci_hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("range (m, derived: +/-10%)")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.set_title("Detection rate vs range (derived from apparent size)")
    fig.tight_layout()
    vizstyle.save(fig, "detection_rate_vs_range")

    # Figure 4: issue #2's "translation error vs range". NOT accuracy -- single-marker
    # PnP scored against the multi-marker board reference. Caption says so.
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
        perr = perr.assign(size_mm=perr.marker_id.map(
            lambda mid: round(g.SIZES[int(mid)] * 1000, 1)))
        trans_err_summary = {}
        for grp, sub in perr.groupby("size_mm"):
            st = M.binned_stats(sub, "trans_err_m", "range_m", RANGE_BINS)
            valid = (st.n > 0).to_numpy()
            centre = ((st.bin_lo + st.bin_hi) / 2).to_numpy()
            color = vizstyle.SIZE_COLORS.get(grp, vizstyle.TEXT_PRIMARY)
            marker = vizstyle.SIZE_MARKERS.get(grp, "o")
            ax.errorbar(centre[valid], st["mean"].to_numpy()[valid] * 1000,
                        yerr=st["std"].to_numpy()[valid] * 1000,
                        marker=marker, color=color, capsize=3, label=f"{grp:.0f} mm")
            trans_err_summary[f"{grp:.0f}mm"] = {
                f"{lo}-{hi}": (None if np.isnan(mu) else round(mu * 1000, 1))
                for lo, hi, mu in zip(st.bin_lo, st.bin_hi, st["mean"])}
        summary["trans_err_vs_range_mm"] = trans_err_summary
        M.binned_stats(perr, "trans_err_m", "range_m", RANGE_BINS).to_csv(
            "results/trans_err_vs_range.csv", index=False)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("translation vs board reference (mm)")
    ax.legend()
    ax.set_title("Single-marker vs board reference (self-consistency, NOT accuracy)")
    fig.tight_layout()
    vizstyle.save(fig, "trans_err_vs_range")

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

    # (b) yaw-turn check
    labels = segment.classify(imu_t, gz)
    turn_segs = [s for s in segment.segments(labels, imu_t) if s["label"] == "turn"]
    pose_frames = sorted(board_pose)
    yaw_rows = []
    for seg in turn_segs:
        before = [f for f in pose_frames if frame_stamp[f] <= seg["t0"]]
        after = [f for f in pose_frames if frame_stamp[f] >= seg["t1"]]
        if not before or not after:
            continue                       # board out of frame across this turn
        f0, f1 = before[-1], after[0]
        vis = np.degrees(ic.pnp_delta_yaw(board_pose[f0][0], board_pose[f1][0]))
        m = (imu_t >= frame_stamp[f0]) & (imu_t <= frame_stamp[f1])
        gyro = np.degrees(ic.integrate_gyro_yaw(imu_t[m], gz[m]))
        yaw_rows.append({"t0": round(float(seg["t0"]), 1), "t1": round(float(seg["t1"]), 1),
                         "vision_deg": round(vis, 1), "gyro_deg": round(gyro, 1)})
    summary["imu_yaw_check"] = {
        "n_turn_segments": len(turn_segs),
        "n_usable_spans": len(yaw_rows),
        "spans": yaw_rows,
        "note": "vision vs gyro delta-yaw per turn; few usable spans expected because "
                "turns swing the board out of frame. Gyro (IMU up-z) and vision "
                "(optical Y) may differ in SIGN -- compare magnitude/correlation, not "
                "raw sign, until reconciled against data.",
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
        f"- {len(summary['suppressed_bins'])} bin(s) with fewer than "
        f"{MIN_TRIALS_PER_BIN} trials were suppressed from the detection-rate figures "
        "(their Wilson CI spans most of the axis and would dominate the plot with no "
        "information); the full per-bin table, including these, is still in "
        "`detection_trials_test1.csv` / `detection_trials_test2.csv`. Suppressed: " +
        ("; ".join(f"{b['group_mm']:.0f} mm, {b['bin_lo_px']:.0f}-{b['bin_hi_px']:.0f} px, "
                   f"n={b['n']}" for b in summary["suppressed_bins"])
         if summary["suppressed_bins"] else "none") + ".",
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
    lines += ["## Stage 4 - IMU validation (camera-independent)", "",
              f"- Gravity: board tilt from vertical over {gchk['n_frames']} frames = "
              f"{gchk['board_tilt_from_vertical_deg_median']} deg median, std "
              f"{gchk['std_deg']} deg (low std = PnP attitude consistent with the IMU).",
              f"- Yaw turns: {ychk['n_usable_spans']}/{ychk['n_turn_segments']} turn "
              f"segments had board poses at both ends" +
              (":" if ychk["spans"] else " -- board out of frame during turns, so the "
               "yaw check is not supported by this data (expected; see spec 3.1c)."), ""]
    for sp in ychk["spans"]:
        lines.append(f"  - turn {sp['t0']}-{sp['t1']}s: vision {sp['vision_deg']} deg, "
                     f"gyro {sp['gyro_deg']} deg (compare magnitude, not sign)")
    lines.append("")
    with open("results/summary.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset")
