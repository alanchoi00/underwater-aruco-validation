#!/usr/bin/env python3
"""
aruco_collage.py — true-size ArUco collage matching the SIM dock markers,
laid out in the dock ARRANGEMENT with compressed spacing (A3 landscape).

Dictionary: DICT_ARUCO_ORIGINAL. IDs/sizes from dock_layout.py. black square =
BORDER_RATIO x total (0.7782 = 1470/1889), matching aruco.launch.py.

Page 1: front-wing pair 201/202 (200 mm), dock left/right.
Page 2: backplate cluster 301/302-305/401/402 in dock arrangement.

NOTE: spacing is COMPRESSED to fit A3 — this preserves which-marker-is-where
(arrangement), NOT the metric dock geometry. Do not use this sheet as a
calibrated multi-marker target for fused pose; use the true dock for that.
Marker SIZES are true; marker_sizes.yaml holds the black-square sizes (m).

Print at 100% (actual size). Verify scale bar = 100 mm.
"""
import argparse, os, tempfile
import cv2, numpy as np, yaml
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

BORDER_RATIO = 0.7782

# (id, total_size_mm, center_x_mm, center_z_mm) relative to page centre. x=right, z=up.
PAGES = [
    {"title": "Front-wing pair (dock left/right; spacing compressed)",
     "markers": [(201, 200.0, -106, 8), (202, 200.0, 106, 8)]},
    {"title": "Backplate cluster (dock arrangement; spacing compressed to fit A3)",
     "markers": [(302, 60.0, -150, 90), (401, 47.5, -35, 70), (402, 47.5, 35, 70),
                 (303, 60.0, 150, 90), (301, 100.0, 0, -10),
                 (305, 60.0, -150, -90), (304, 60.0, 150, -90)]},
]


def get_dictionary(name):
    const = getattr(cv2.aruco, name)
    try:
        return cv2.aruco.getPredefinedDictionary(const)
    except AttributeError:
        return cv2.aruco.Dictionary_get(const)


def gen_marker(d, mid, px):
    try:
        return cv2.aruco.generateImageMarker(d, mid, px)
    except AttributeError:
        return cv2.aruco.drawMarker(d, mid, px)


def verify(d, img, mid):
    p = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    try:
        _, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(p)
    except AttributeError:
        _, ids, _ = cv2.aruco.detectMarkers(p, d)
    assert ids is not None and mid in ids.flatten(), f"marker {mid} failed to decode"


def tile_png(d, mid, total_mm, dpi, tmp, ratio):
    total_px = max(120, int(round(total_mm / 25.4 * dpi)))
    black_px = max(60, int(round(total_px * ratio)))
    m = gen_marker(d, mid, black_px)
    tile = np.full((total_px, total_px), 255, np.uint8)
    o = (total_px - black_px) // 2
    tile[o:o + black_px, o:o + black_px] = m
    verify(d, tile, mid)
    path = os.path.join(tmp, f"m{mid}.png")
    cv2.imwrite(path, tile)
    return path


def build(out_pdf, dict_name, pages, dpi=300, margin_mm=10.0, ratio=BORDER_RATIO):
    dictionary = get_dictionary(dict_name)
    pw, ph = landscape(A3)            # 420 x 297 mm in points
    c = canvas.Canvas(out_pdf, pagesize=landscape(A3))
    tmp = tempfile.mkdtemp()
    sizes_map = {}
    cx0, cy0 = pw / 2, ph / 2

    for pi, page in enumerate(pages):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin_mm * mm, ph - margin_mm * mm,
                     f"ArUco · {dict_name} · {page['title']}")
        c.setFont("Helvetica", 8)
        c.drawString(margin_mm * mm, ph - (margin_mm + 5) * mm,
                     "PRINT AT 100% (actual size). size = TOTAL print mm; "
                     f"black square = {ratio:.4f} x. Verify scale bar = 100 mm.")
        for mid, total_mm, cx, cz in page["markers"]:
            png = tile_png(dictionary, mid, total_mm, dpi, tmp, ratio)
            w = total_mm * mm
            x = cx0 + cx * mm - w / 2
            y = cy0 + cz * mm - w / 2
            c.drawImage(png, x, y, width=w, height=w)
            c.setFont("Helvetica", 7)
            c.drawString(x, y - 5 * mm,
                         f"ID {mid} · {total_mm:g}mm (blk {total_mm * ratio:.1f})")
            sizes_map[int(mid)] = round(total_mm * ratio / 1000.0, 5)
        # scale bar
        yb = margin_mm * mm
        c.setLineWidth(1)
        c.line(pw - (margin_mm + 100) * mm, yb, pw - margin_mm * mm, yb)
        for t in (0, 50, 100):
            c.line(pw - (margin_mm + 100 - t) * mm, yb, pw - (margin_mm + 100 - t) * mm, yb + 3 * mm)
        c.setFont("Helvetica", 7)
        c.drawString(pw - (margin_mm + 100) * mm, yb - 4 * mm, "scale bar 0-50-100 mm")
        c.showPage()
    c.save()

    sidecar = os.path.join(os.path.dirname(out_pdf) or ".", "marker_sizes.yaml")
    with open(sidecar, "w") as f:
        yaml.safe_dump({"dictionary": dict_name, "note": "black-square side length (m)",
                        "marker_size_m": sizes_map}, f, sort_keys=True)
    return sidecar


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="aruco_collage_A3.pdf")
    ap.add_argument("--dict", default="DICT_ARUCO_ORIGINAL")
    args = ap.parse_args()
    sc = build(args.out, args.dict, PAGES)
    print("wrote", args.out, "and", sc)
