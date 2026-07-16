"""Environment guard: the study must run on the exact pinned OpenCV."""
import pathlib

import cv2


def test_opencv_is_post_aruco_rewrite():
    major, minor = (int(x) for x in cv2.__version__.split(".")[:2])
    assert (major, minor) >= (4, 7), (
        f"cv2 {cv2.__version__} predates the 4.7 ArUco rewrite; "
        "the ROS container's 4.6 must not be used for detection"
    )


def test_opencv_matches_the_pin():
    """The running cv2 must BE the pinned one, not merely satisfy a floor.

    A floor cannot catch upward drift, and upward drift is what actually happened: a
    second pip install pulled an unpinned opencv-contrib-python 5.0.0 that shadowed the
    pinned headless 4.10, so the container ran 5.0.0 while claiming 4.10, and the floor
    check above waved it through. The study's headline result IS the detector's
    behaviour, so the version is part of the claim, not a lower bound on it.

    Reads the pin rather than hardcoding it, so bumping requirements.txt moves this test
    with it instead of leaving a stale literal behind.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    req = (root / "requirements.txt").read_text()
    pinned = next(
        line.split("==")[1].strip()
        for line in req.splitlines()
        if line.startswith("opencv-contrib-python-headless==")
    )
    # The wheel is 4.10.0.84; cv2.__version__ reports the first three fields.
    assert pinned.startswith(cv2.__version__), (
        f"running cv2 {cv2.__version__} but requirements.txt pins {pinned}. "
        "Something installed a second package providing cv2 and shadowed the pin; "
        "check for an unpinned opencv-contrib-python alongside the headless one."
    )


def test_modern_aruco_api_present():
    assert hasattr(cv2.aruco, "ArucoDetector")
    assert hasattr(cv2.aruco, "generateImageMarker")


def test_original_dictionary_loads():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    assert d.markerSize == 5
