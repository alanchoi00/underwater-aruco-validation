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

from analysis import detect, geometry as g, imucheck as ic, layout as L, metrics as M, segment

PX_BINS = [10, 15, 21, 30, 45, 70, 110, 180, 300]


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
    det = pd.read_csv(os.path.join(d, "detections.csv"))
    frames = pd.read_csv(os.path.join(d, "frames.csv"))
    imu = pd.read_csv(os.path.join(d, "imu.csv"))
    ci = yaml.safe_load(open(os.path.join(d, "camera_info.yaml")))
    return det, frames, imu, ci


def main(dataset_dir):
    os.makedirs("results", exist_ok=True)
    summary = {"opencv": cv2.__version__, "host_cpu": detect.host_cpu(),
               "code_version": code_version(), "runs": {}}

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

    # Figure 1: the calibration-free headline.
    det1, frames1, _, ci1 = load_run(dataset_dir, "test1")
    rates1 = M.detection_rate_by_px_bin(det1, {m: len(frames1) for m in g.SIZES}, PX_BINS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mid, sub in rates1.groupby("marker_id"):
        centre = (sub.px_lo + sub.px_hi) / 2
        ax.plot(centre, sub.rate, marker="o", label=f"{mid} ({g.SIZES[mid]*1000:.0f} mm)")
    ax.axvline(M.pixel_budget_px(5), ls="--", c="k", lw=1,
               label=f"3(n+2) budget = {M.pixel_budget_px(5)} px")
    ax.set_xscale("log"); ax.set_xlabel("apparent marker size (px)")
    ax.set_ylabel("detection rate"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Detection rate vs apparent size (calibration-free)")
    fig.tight_layout(); fig.savefig("results/detection_rate_vs_px.png", dpi=150)

    # Figure 2: the hypothesis test.
    mr = summary["runs"]["test1"]["max_range_m"]
    sizes = np.array([g.SIZES[m] for m in sorted(mr)])
    rmax = np.array([mr[m] for m in sorted(mr)])
    k = float(np.sum(sizes * rmax) / np.sum(sizes ** 2))   # least squares through 0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(sizes * 1000, rmax)
    xs = np.linspace(0, sizes.max() * 1100, 50)
    ax.plot(xs, k * xs / 1000, ls="--", c="k", lw=1, label=f"max_range ~ {k:.0f} x side")
    ax.axhline(5.0, color="tab:red", lw=1, label="coarse phase needs ~5 m")
    ax.set_xlabel("marker side (mm)"); ax.set_ylabel("max detection range (m)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Max detection range vs marker size")
    fig.tight_layout(); fig.savefig("results/max_range_vs_size.png", dpi=150)
    summary["range_per_side_fit"] = k

    # Figure 3: issue #2's "detection rate vs range". Derived from px via the pinhole
    # relation, so it inherits the focal length's ~+/-10%. The px figure above is the
    # calibration-free one; this exists because the issue asks for it.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mid, sub in rates1.groupby("marker_id"):
        centre = (sub.px_lo + sub.px_hi) / 2
        rng_axis = [M.range_from_apparent_px(c, mid, ci1["fx"]) for c in centre]
        ax.plot(rng_axis, sub.rate, marker="o",
                label=f"{mid} ({g.SIZES[mid]*1000:.0f} mm)")
    ax.set_xlabel("range (m, derived: +/-10%)"); ax.set_ylabel("detection rate")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Detection rate vs range (derived from apparent size)")
    fig.tight_layout(); fig.savefig("results/detection_rate_vs_range.png", dpi=150)

    # Figure 4: issue #2's "translation error vs range". NOT accuracy -- single-marker
    # PnP scored against the multi-marker board reference. Caption says so.
    obs = {}
    for r in det1[det1.marker_id.isin(g.SIZES)].itertuples():
        obs.setdefault(("test1", r.frame_idx), {})[int(r.marker_id)] = np.array(
            [[r.c0x, r.c0y], [r.c1x, r.c1y], [r.c2x, r.c2y], [r.c3x, r.c3y]])
    Km = np.array([[ci1["fx"], 0, ci1["cx"]], [0, ci1["fy"], ci1["cy"]], [0, 0, 1]])
    lay = L.load_layout("config/board_layout.yaml")
    perr = M.pose_error_vs_reference(obs, lay, Km)
    perr.to_csv("results/pose_error_vs_reference.csv", index=False)
    RANGE_BINS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(perr):
        st = M.binned_stats(perr, "trans_err_m", "range_m", RANGE_BINS)
        st.to_csv("results/trans_err_vs_range.csv", index=False)
        centre = (st.bin_lo + st.bin_hi) / 2
        ax.errorbar(centre, st["mean"] * 1000, yerr=st["std"] * 1000,
                    marker="o", capsize=3)
        summary["trans_err_vs_range_mm"] = {
            f"{lo}-{hi}": (None if np.isnan(mu) else round(mu * 1000, 1))
            for lo, hi, mu in zip(st.bin_lo, st.bin_hi, st["mean"])}
    ax.set_xlabel("range (m)")
    ax.set_ylabel("translation vs board reference (mm)")
    ax.grid(alpha=0.3)
    ax.set_title("Single-marker vs board reference (self-consistency, NOT accuracy)")
    fig.tight_layout(); fig.savefig("results/trans_err_vs_range.png", dpi=150)

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

    # (a) gravity check
    tilt = []
    for fi, (rvec, _tv) in board_pose.items():
        j = int(np.clip(np.searchsorted(imu_t, frame_stamp[fi]), 0, len(accel) - 1))
        tilt.append(ic.tilt_residual_deg(rvec, accel[j]))
    tilt = np.array(tilt)
    summary["imu_gravity_check"] = {
        "n_frames": int(len(tilt)),
        "board_tilt_from_vertical_deg_median": round(float(np.median(tilt)), 2)
            if len(tilt) else None,
        "std_deg": round(float(np.std(tilt)), 2) if len(tilt) else None,
        "note": "board-normal vs gravity; median = board tilt from vertical, std = PnP "
                "attitude consistency vs the absolute IMU gravity reference (lower better)",
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
        "range/size (spec 3.1c).", "",
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
