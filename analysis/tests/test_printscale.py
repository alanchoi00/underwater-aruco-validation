import numpy as np
import pytest

from analysis import printscale as ps


def _ortho_with_sheet(mm_per_px, h_mm, w_mm=210.0, pad_px=30):
    """Synthetic orthophoto: a white sheet on a dark background."""
    h_px = int(round(h_mm / mm_per_px))
    w_px = int(round(w_mm / mm_per_px))
    img = np.full((h_px + 2 * pad_px, w_px + 2 * pad_px), 40, np.uint8)
    img[pad_px:pad_px + h_px, pad_px:pad_px + w_px] = 235
    return img


def test_print_scale_is_unity_when_measured_matches_nominal():
    assert ps.print_scale(297.0, 297.0) == pytest.approx(1.0)


def test_print_scale_detects_a_shrunken_print():
    """A sheet measuring larger than A4 means the CONTENT was printed small."""
    assert ps.print_scale(312.6, 297.0) == pytest.approx(0.95, abs=0.005)


def test_measure_sheet_height_on_a_perfect_sheet():
    mm_per_px = 0.5
    ortho = _ortho_with_sheet(mm_per_px, 297.0)
    assert ps.measure_sheet_mm(ortho, mm_per_px, axis="height") == pytest.approx(297, abs=2)


def test_measure_sheet_width():
    mm_per_px = 0.5
    ortho = _ortho_with_sheet(mm_per_px, 297.0, w_mm=210.0)
    assert ps.measure_sheet_mm(ortho, mm_per_px, axis="width") == pytest.approx(210, abs=2)


def test_measure_detects_an_oversized_sheet():
    mm_per_px = 0.5
    ortho = _ortho_with_sheet(mm_per_px, 312.6)
    assert ps.measure_sheet_mm(ortho, mm_per_px, axis="height") == pytest.approx(312.6, abs=3)


def test_homography_maps_board_origin_to_the_expected_ortho_pixel(layout_true, K):
    """At zero rotation the board origin projects to the principal point, and the
    homography must send that back to the ortho pixel for board (0, 0) mm."""
    rv = np.array([0.0, 0.0, 0.0])
    tv = np.array([0.0, 0.0, 1.5])
    mm_per_px, origin_mm = 0.5, (-100.0, -200.0)
    H = ps.board_homography(layout_true, rv, tv, K, mm_per_px, origin_mm)

    out = H @ np.array([K["cx"], K["cy"], 1.0])
    out = out[:2] / out[2]
    expected = [(0.0 - origin_mm[0]) / mm_per_px, (0.0 - origin_mm[1]) / mm_per_px]
    np.testing.assert_allclose(out, expected, atol=1e-6)   # -> (200, 400)
