"""Stage 2: split a run into dwell and yaw-turn segments using the IMU gyro.

Turns are kept, not discarded. They supply the tilt diversity that makes focal length
observable at all, and they are the only place the IMU yaw cross-check applies.
"""
import numpy as np
import pandas as pd

TURN_RATE_THRESHOLD = 0.15      # rad/s; ~8.6 deg/s, above hand-held jitter
MIN_TURN_S = 1.0                # shorter excursions are gyro noise, not a turn


def classify(stamps, gyro_z, turn_threshold=TURN_RATE_THRESHOLD, min_turn_s=MIN_TURN_S):
    """Per-sample label in {"dwell", "turn"}."""
    stamps = np.asarray(stamps, dtype=float)
    hot = np.abs(np.asarray(gyro_z, dtype=float)) > turn_threshold
    labels = np.full(len(stamps), "dwell", dtype=object)
    for i0, i1 in _runs(hot):
        if stamps[i1 - 1] - stamps[i0] >= min_turn_s:
            labels[i0:i1] = "turn"
    return labels


def _runs(mask):
    """Yield [start, stop) index pairs of each contiguous True run."""
    if not len(mask):
        return
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    for i0, i1 in zip(edges[::2], edges[1::2]):
        yield int(i0), int(i1)


def segments(labels, stamps):
    """Merge equal-label runs into intervals."""
    stamps = np.asarray(stamps, dtype=float)
    out = []
    if not len(labels):
        return out
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append({"label": labels[start], "t0": stamps[start],
                        "t1": stamps[i - 1], "i0": start, "i1": i})
            start = i
    return out


def label_frames(frames_df, imu_df):
    """Label each camera frame by its nearest IMU sample.

    The IMU runs at ~62 Hz against 2.5 Hz frames, so nearest-sample is well inside
    one frame period and interpolation would add nothing.
    """
    labels = classify(imu_df["stamp"].to_numpy(), imu_df["wz"].to_numpy())
    idx = np.searchsorted(imu_df["stamp"].to_numpy(), frames_df["stamp"].to_numpy())
    idx = np.clip(idx, 0, len(labels) - 1)
    return pd.Series([labels[i] for i in idx], index=frames_df.index)
