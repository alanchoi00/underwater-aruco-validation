import cv2
import numpy as np

from analysis import run_preprocessing_eval as P


def _low_contrast_gray():
    """A gradient image with a deliberately squashed dynamic range (80..140), the
    shape CLAHE is meant to act on: enough local structure that histogram
    equalisation has something to redistribute.
    """
    ramp = np.linspace(80, 140, 100, dtype=np.float64)
    img = np.tile(ramp, (100, 1))
    return img.astype(np.uint8)


def test_none_mode_returns_the_grey_image_unchanged():
    gray = _low_contrast_gray()
    out = P.preprocess(gray, "none")
    assert np.array_equal(out, gray)


def test_athresh_mode_does_not_touch_the_image_either():
    """athresh's tuning lives in the detector's parameters, not the pixels."""
    gray = _low_contrast_gray()
    out = P.preprocess(gray, "athresh")
    assert np.array_equal(out, gray)


def test_clahe_mode_changes_the_image():
    gray = _low_contrast_gray()
    out = P.preprocess(gray, "clahe")
    assert out.shape == gray.shape
    assert out.dtype == gray.dtype
    assert not np.array_equal(out, gray)


def test_tuned_detector_carries_the_tuned_adaptive_thresh_constant():
    default_constant = cv2.aruco.DetectorParameters().adaptiveThreshConstant
    tuned = P.make_tuned_detector(3.0)
    params = tuned.getDetectorParameters()
    assert params.adaptiveThreshConstant != default_constant
    assert params.adaptiveThreshConstant == 3.0


def test_tuned_detector_keeps_corner_refinement_like_the_default_detector():
    tuned = P.make_tuned_detector(5.0)
    params = tuned.getDetectorParameters()
    assert params.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_SUBPIX
