import numpy as np

from analysis import turbidity as T


def _synthetic_marker(canon_px=96, surround=1.6, black=30.0, white=200.0):
    """A fronto-parallel 7x7 marker on a white sheet, rendered directly in canonical space.

    Returns (img, corners) where corners are the marker's outer quad in img pixels.
    """
    side = int(round(canon_px * surround))
    img = np.full((side, side, 3), white, dtype=np.uint8)
    off = (side - canon_px) // 2
    cell = canon_px / 7.0
    # Black border ring: the outer cell of the 7x7 grid.
    img[off:off + canon_px, off:off + canon_px] = black
    inner = int(round(cell))
    # Data area: white, so only the border ring is black.
    img[off + inner:off + canon_px - inner, off + inner:off + canon_px - inner] = white
    corners = np.array([[off, off], [off + canon_px, off],
                        [off + canon_px, off + canon_px], [off, off + canon_px]],
                       dtype=float)
    return img, corners


def test_patch_means_separate_the_black_ring_from_the_white_sheet():
    img, corners = _synthetic_marker(black=30.0, white=200.0)
    warped = T.warp_marker(img, corners)
    black, white = T.patch_means(warped)
    assert black.shape == (4,) and white.shape == (4,)
    assert np.allclose(black, 30.0, atol=2.0), black
    assert np.allclose(white, 200.0, atol=2.0), white


def test_warp_marker_is_size_invariant():
    """A marker at 2x the apparent size must yield the same patch means."""
    small, cs = _synthetic_marker(canon_px=48)
    big, cb = _synthetic_marker(canon_px=192)
    bs, ws = T.patch_means(T.warp_marker(small, cs))
    bb, wb = T.patch_means(T.warp_marker(big, cb))
    assert np.allclose(bs, bb, atol=2.0)
    assert np.allclose(ws, wb, atol=2.0)
