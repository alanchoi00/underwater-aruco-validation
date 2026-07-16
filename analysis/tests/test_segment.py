import numpy as np
import pandas as pd
import pytest

from analysis import segment


def _series(rate_hz=60.0, dur=30.0):
    return np.arange(0.0, dur, 1.0 / rate_hz)


def test_still_signal_is_all_dwell():
    t = _series()
    labels = segment.classify(t, np.zeros_like(t))
    assert set(labels) == {"dwell"}


def test_sustained_rotation_is_a_turn():
    t = _series()
    gz = np.zeros_like(t)
    gz[(t > 10) & (t < 14)] = 0.6           # 4 s turn, well over threshold
    labels = segment.classify(t, gz)
    assert set(labels) == {"dwell", "turn"}
    # Everything strictly inside the burst is a turn.
    assert set(labels[(t > 11) & (t < 13)]) == {"turn"}
    # Everything well outside it is not.
    assert set(labels[t < 9]) == {"dwell"}


def test_turn_detection_is_sign_agnostic():
    t = _series()
    gz = np.zeros_like(t)
    gz[(t > 10) & (t < 14)] = -0.6          # turning the other way
    assert "turn" in set(segment.classify(t, gz))


def test_brief_noise_spike_is_not_a_turn():
    """Guards flicker: a 0.1 s spike is gyro noise, not an operator turning."""
    t = _series()
    gz = np.zeros_like(t)
    gz[(t > 10) & (t < 10.1)] = 0.9
    assert set(segment.classify(t, gz, min_turn_s=1.0)) == {"dwell"}


def test_segments_merge_runs_into_intervals():
    t = _series()
    gz = np.zeros_like(t)
    gz[(t > 10) & (t < 14)] = 0.6
    segs = segment.segments(segment.classify(t, gz), t)
    turns = [s for s in segs if s["label"] == "turn"]
    assert len(turns) == 1
    assert turns[0]["t0"] == pytest.approx(10.0, abs=0.2)
    assert turns[0]["t1"] == pytest.approx(14.0, abs=0.2)


def test_label_frames_maps_each_frame_to_nearest_imu_sample():
    imu = pd.DataFrame({"stamp": _series(), "wz": np.zeros(1800)})
    imu.loc[(imu.stamp > 10) & (imu.stamp < 14), "wz"] = 0.6
    frames = pd.DataFrame({"frame_idx": [0, 1, 2], "stamp": [5.0, 12.0, 20.0]})
    out = segment.label_frames(frames, imu)
    assert out.tolist() == ["dwell", "turn", "dwell"]
