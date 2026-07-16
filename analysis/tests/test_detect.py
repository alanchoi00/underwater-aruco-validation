import cv2
import numpy as np
import pytest

from analysis import detect
from analysis import geometry as g


def _render_marker(mid, px, pad=40):
    """Render a marker at a known pixel size on white, fronto-parallel."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, g.DICTIONARY))
    img = cv2.aruco.generateImageMarker(d, mid, px)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)


def test_apparent_size_of_a_unit_square_is_its_side():
    corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    assert detect.apparent_size_px(corners) == pytest.approx(10.0)


def test_apparent_size_scales_with_the_quad():
    corners = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], dtype=float)
    assert detect.apparent_size_px(corners) == pytest.approx(40.0)


def test_detects_a_rendered_marker_and_measures_its_size():
    img = _render_marker(201, 120)
    out = detect.detect_frame(img, detect.make_detector())
    assert len(out) == 1
    assert out[0]["marker_id"] == 201
    # generateImageMarker draws the black square edge-to-edge at `px`.
    assert out[0]["apparent_px"] == pytest.approx(120, abs=2)


def test_reports_ids_outside_the_board_so_mis_ids_stay_measurable():
    img = _render_marker(7, 120)          # 7 is not a board marker
    out = detect.detect_frame(img, detect.make_detector())
    assert len(out) == 1
    assert out[0]["marker_id"] == 7
    assert out[0]["marker_id"] not in g.SIZES


def test_blank_image_detects_nothing():
    img = np.full((200, 200), 255, np.uint8)
    assert detect.detect_frame(img, detect.make_detector()) == []


def test_detection_columns_are_flat_and_stable():
    assert detect.DETECTION_COLUMNS[:4] == [
        "frame_idx", "stamp", "marker_id", "apparent_px"]
    assert len(detect.DETECTION_COLUMNS) == 12   # 4 + 8 corner coords


def test_timing_columns():
    assert detect.TIMING_COLUMNS == ["frame_idx", "latency_ms", "n_detected"]


def test_detect_frame_timed_reports_a_positive_latency():
    img = _render_marker(201, 120)
    out, ms = detect.detect_frame_timed(img, detect.make_detector())
    assert len(out) == 1
    assert ms > 0.0
    assert ms < 5000.0          # a single 200x200 frame is milliseconds, not seconds


def test_host_cpu_returns_a_non_empty_string():
    assert isinstance(detect.host_cpu(), str)
    assert detect.host_cpu()
