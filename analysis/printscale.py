"""Stage 3b: cross-check the print scale from A4 sheet geometry.

The printed markers were ruler-measured (see target/marker_sizes_measured.yaml,
k=0.9589) -- that measurement is the authoritative print scale. This stage is an
INDEPENDENT image-based cross-check of it, not a substitute: a printer scales CONTENT,
not paper (A4 is 210x297 mm by definition), so measuring a sheet in units where the
marker is assumed 155.64 mm recovers the print scale directly.

Use sheet HEIGHT: the three sheets are butted side-by-side (ambiguous vertical seams)
but their top and bottom edges are free.

NOTE (this dataset): the image method is unreliable here -- the pool wall's white grid
merges with the white sheet under Otsu, so the ruler measurement remains authoritative.
"""
import cv2
import numpy as np

from analysis import geometry as g

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0


def print_scale(measured_mm, nominal_mm):
    """Scale the CONTENT was printed at.

    A sheet that measures larger than nominal means the marker used as the length
    reference was actually smaller than assumed -- i.e. printed under 100%.
    """
    return nominal_mm / measured_mm


def board_homography(layout, rv, tv, K, mm_per_px, origin_mm):
    """Image -> board-plane orthophoto homography.

    The board is z=0, so the projection collapses to a plane-to-plane homography and
    is exactly invertible -- no depth needed.
    """
    Kmat = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]],
                    dtype=float)
    R = g.rodrigues(np.asarray(rv, float)[None])[0]
    # Board z=0 => [r1 r2 t] maps (x_board, y_board, 1) to the image.
    P = Kmat @ np.column_stack([R[:, 0], R[:, 1], np.asarray(tv, float)])
    # Orthophoto pixel <- board mm.
    S = np.array([[1.0 / mm_per_px, 0, -origin_mm[0] / mm_per_px],
                  [0, 1.0 / mm_per_px, -origin_mm[1] / mm_per_px],
                  [0, 0, 1.0]])
    # `layout` is in metres but the orthophoto is scaled in mm, so fold the unit
    # change in: inv(P @ M) == diag(1000,1000,1) @ inv(P).
    M = np.array([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1.0]])
    return S @ np.linalg.inv(P @ M)


def orthorectify(img, H, out_size):
    return cv2.warpPerspective(img, H, out_size, flags=cv2.INTER_LINEAR)


def measure_sheet_mm(ortho, mm_per_px, axis="height"):
    """Measure the white sheet's extent in the orthophoto, in millimetres.

    Otsu separates paper from the darker water/wall behind it; the largest connected
    component is the sheet.
    """
    blur = cv2.GaussianBlur(ortho, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n < 2:
        raise ValueError("no sheet found in orthophoto")
    # Skip label 0 (background); take the largest remaining component.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    extent_px = (stats[largest, cv2.CC_STAT_HEIGHT] if axis == "height"
                 else stats[largest, cv2.CC_STAT_WIDTH])
    return float(extent_px * mm_per_px)
