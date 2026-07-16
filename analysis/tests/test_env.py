"""Environment guard: the study must not run on the legacy (pre-4.7) ArUco API."""
import cv2


def test_opencv_is_post_aruco_rewrite():
    major, minor = (int(x) for x in cv2.__version__.split(".")[:2])
    assert (major, minor) >= (4, 7), (
        f"cv2 {cv2.__version__} predates the 4.7 ArUco rewrite; "
        "the ROS container's 4.6 must not be used for detection"
    )


def test_modern_aruco_api_present():
    assert hasattr(cv2.aruco, "ArucoDetector")
    assert hasattr(cv2.aruco, "generateImageMarker")


def test_original_dictionary_loads():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    assert d.markerSize == 5
