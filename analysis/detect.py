"""Stage 1: ArUco detection sweep.

Apparent size in pixels is the primary, calibration-free result axis. It is measured
straight off the image and is invariant to any intrinsics question, which is why it
carries the headline instead of metric range.
"""
import os
import platform
import time

import cv2
import numpy as np
import pandas as pd

from analysis import geometry as g

DETECTION_COLUMNS = ["frame_idx", "stamp", "marker_id", "apparent_px"] + [
    f"c{i}{ax}" for i in range(4) for ax in ("x", "y")
]
TIMING_COLUMNS = ["frame_idx", "latency_ms", "n_detected"]


def host_cpu():
    """CPU model of the analysis host. Latency is meaningless without it."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def make_detector():
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, g.DICTIONARY))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, p)


def apparent_size_px(corners):
    """Mean of the four edge lengths of the detected quad."""
    c = np.asarray(corners, dtype=float)
    return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


def detect_frame_timed(gray, detector):
    """detect_frame plus the detector's wall-clock cost in ms.

    Times ONLY detectMarkers. PNG decode is an artifact of this offline pipeline and
    would not exist on the ROV, so including it would inflate the figure.
    """
    t0 = time.perf_counter()
    out = detect_frame(gray, detector)
    return out, (time.perf_counter() - t0) * 1e3


def detect_frame(gray, detector):
    """Detect every marker, including ids outside the board.

    Off-board ids are returned rather than filtered so the mis-ID rate stays a
    measurable quantity (the pilot measured zero across 1358 detections).
    """
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    out = []
    for c, i in zip(corners, ids.flatten()):
        q = c[0].astype(float)
        out.append({"marker_id": int(i), "corners": q, "apparent_px": apparent_size_px(q)})
    return out


def sweep_dataset(dataset_dir, detector):
    """Returns (detections, timing). Timing is per-frame; detections are per-marker."""
    frames = pd.read_csv(os.path.join(dataset_dir, "frames.csv"))
    rows, timing = [], []
    for idx, stamp in zip(frames["frame_idx"], frames["stamp"]):
        path = os.path.join(dataset_dir, "frames", f"{int(idx):06d}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        dets, ms = detect_frame_timed(img, detector)
        timing.append([int(idx), ms, len(dets)])
        for d in dets:
            rows.append([int(idx), float(stamp), d["marker_id"], d["apparent_px"],
                         *d["corners"].ravel().tolist()])
    return (pd.DataFrame(rows, columns=DETECTION_COLUMNS),
            pd.DataFrame(timing, columns=TIMING_COLUMNS))
