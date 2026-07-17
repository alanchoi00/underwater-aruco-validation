"""Turbidity: measure the water's attenuation, then synthesise more of it.

The board is already the experiment. A black patch and a white patch at the same range
share an identical veiling term in the image formation model

    I = J * exp(-beta * d) + B * (1 - exp(-beta * d))

so their difference is B-free:

    contrast(d) = (J_white - J_black) * exp(-beta * d)

Taking logs makes beta a straight-line fit that needs no assumption about backscatter.
That matters, because B itself fits badly (r^2 ~ 0.5) while beta fits well (r^2 ~ 0.96).
"""
import cv2
import numpy as np

CHANNELS = ("b", "g", "r", "grey")

CANON_PX = 96          # marker side in the warped canonical view
SURROUND = 1.6         # warped view is this multiple of the marker side, to catch the sheet

# Below this apparent size the two rings are a handful of pixels each and the sample is
# dominated by the blur across the ring boundary rather than the ring itself.
MIN_PX_FOR_PHOTOMETRY = 40.0


def warp_marker(img, corners, canon_px=CANON_PX, surround=SURROUND):
    """Warp a detected marker to a canonical square, keeping the sheet around it.

    The marker maps to a centred canon_px square inside a larger view, so the white sheet
    outside the marker is sampled at the same range as the marker's own black border.
    """
    side = int(round(canon_px * surround))
    off = (side - canon_px) / 2.0
    dst = np.array([[off, off], [off + canon_px, off],
                    [off + canon_px, off + canon_px], [off, off + canon_px]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), dst)
    return cv2.warpPerspective(img, H, (side, side), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def _with_grey(bgr_means):
    """Append the Rec.601 grey of a (3,) BGR mean, matching cv2.COLOR_BGR2GRAY."""
    b, gr, r = bgr_means
    return np.array([b, gr, r, 0.114 * b + 0.587 * gr + 0.299 * r])


def patch_means(warped, canon_px=CANON_PX, surround=SURROUND):
    """Mean of the marker's black border ring and of the white sheet outside it.

    Returns (black, white), each (4,) in CHANNELS order. Both rings are eroded away from
    their boundaries: a ring sampled right up to an edge picks up the blur across it,
    which is the very thing beta is being measured through.
    """
    side = int(round(canon_px * surround))
    off = (side - canon_px) / 2.0
    cell = canon_px / 7.0

    black_ring = np.zeros((side, side), dtype=bool)
    outer = _box(off + 0.25 * cell, canon_px - 0.5 * cell, side)
    inner = _box(off + 0.75 * cell, canon_px - 1.5 * cell, side)
    black_ring[outer] = True
    black_ring[inner] = False

    white_ring = np.zeros((side, side), dtype=bool)
    w_outer = _box(off - 0.75 * cell, canon_px + 1.5 * cell, side)
    w_inner = _box(off - 0.25 * cell, canon_px + 0.5 * cell, side)
    white_ring[w_outer] = True
    white_ring[w_inner] = False

    img = warped.astype(np.float64)
    black = _with_grey([img[..., c][black_ring].mean() for c in range(3)])
    white = _with_grey([img[..., c][white_ring].mean() for c in range(3)])
    return black, white


def _box(start, length, side):
    """Boolean mask of an axis-aligned square, clipped to the view."""
    a = max(0, int(round(start)))
    b = min(side, int(round(start + length)))
    m = np.zeros((side, side), dtype=bool)
    if b > a:
        m[a:b, a:b] = True
    return m
