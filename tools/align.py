"""Align comic photos to a consistent framing.

Only rotation + uniform scale + shift + crop is applied (a similarity
transform) - no perspective correction or warping.

Usage: python3 tools/align.py [input_dir] [output_dir]
"""
import glob
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from detect import load, paper_mask  # noqa: E402
from dust import remove_dust  # noqa: E402

OUT_W, OUT_H = 3840, 2160          # output canvas (16:9)
PAGE_H = int(OUT_H * 0.82)         # paper height in every output image
DETECT_SIZE = 1000                 # long side used for detection


def find_paper(img):
    """Return (center, angle_deg, width, height) of the paper in full-res px.

    The clear page holder sits across the middle band of the spreads, so the
    rectangle is fitted only to the top and bottom parts of the paper, where
    the page edges are clean.
    """
    s = DETECT_SIZE / max(img.shape[:2])
    _, m, _ = paper_mask(img, s)
    ys = np.where(m.any(1))[0]
    y0, y1 = ys[0], ys[-1]
    h = y1 - y0
    band = m.copy()
    band[int(y0 + 0.17 * h):int(y1 - 0.15 * h)] = 0
    pts = cv2.findNonZero(band)
    hull = cv2.convexHull(pts)
    box = cv2.boxPoints(cv2.minAreaRect(hull)) / s
    # pick the most horizontal edge to define the rotation
    def tilt(i):
        d = box[(i + 1) % 4] - box[i]
        return math.degrees(math.atan2(abs(d[1]), abs(d[0])))
    i = min(range(4), key=tilt)
    a, b = box[i], box[(i + 1) % 4]
    if b[0] < a[0]:
        a, b = b, a
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    width = float(np.linalg.norm(b - a))
    height = float(np.linalg.norm(box[(i + 2) % 4] - box[(i + 1) % 4]))
    center = box.mean(0)
    full_hull = cv2.convexHull(cv2.findNonZero(m)).reshape(-1, 2) / s
    return (float(center[0]), float(center[1])), angle, width, height, box, full_hull


def _pingpong(d, band):
    """Fold distance d into [0, band) back and forth (mirror-tiling)."""
    period = 2 * band
    d = d % period
    return np.where(d < band, d, period - 1 - d)


def fill_edges(out, M, src_shape, band=150):
    """Where the photo doesn't reach the output frame, extend the mat by
    mirror-tiling a thin strip of mat next to the gap (never the paper)."""
    h, w = src_shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ M.T
    xl = int(math.ceil(max(corners[0, 0], corners[3, 0]))) + 3
    xr = int(math.floor(min(corners[1, 0], corners[2, 0]))) - 4
    yt = int(math.ceil(max(corners[0, 1], corners[1, 1]))) + 3
    yb = int(math.floor(min(corners[2, 1], corners[3, 1]))) - 4
    H, W = out.shape[:2]
    xl, yt = max(xl, 0), max(yt, 0)
    xr, yb = min(xr, W - 1), min(yb, H - 1)
    if xl == 0 and yt == 0 and xr == W - 1 and yb == H - 1:
        return out
    # strips must stay clear of the paper and the clear holder: measure how
    # far plain dark mat extends inward from each edge of the photo
    gray = cv2.GaussianBlur(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), (0, 0), 3)
    bright = gray > 60
    cols = bright[yt:yb + 1].any(0)
    rows = bright[:, xl:xr + 1].any(1)
    gap = 25

    def strip(d):
        return int(max(8, min(band, d - gap)))

    def first(a):
        idx = np.flatnonzero(a)
        return idx[0] if idx.size else len(a)
    bl = strip(first(cols[xl:xr + 1]))
    br = strip(first(cols[xl:xr + 1][::-1]))
    bt = strip(first(rows[yt:yb + 1]))
    bb = strip(first(rows[yt:yb + 1][::-1]))
    x = np.arange(W)
    ix = np.where(x < xl, xl + _pingpong(xl - x, bl),
                  np.where(x > xr, xr - _pingpong(x - xr, br), x))
    y = np.arange(H)
    iy = np.where(y < yt, yt + _pingpong(yt - y, bt),
                  np.where(y > yb, yb - _pingpong(y - yb, bb), y))
    return out[iy][:, ix]


def process(path, out_dir):
    img = load(path)
    center, angle, w, h, box, hull = find_paper(img)

    # protect the paper (whole outline, plus a margin) from dust removal
    protect = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(protect, [np.int32(hull)], 255)
    cv2.fillPoly(protect, [np.int32(box)], 255)
    protect = cv2.dilate(protect, np.ones((61, 61), np.uint8))
    clean, _, n_dust = remove_dust(img, protect)

    scale = PAGE_H / h
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += OUT_W / 2 - center[0]
    M[1, 2] += OUT_H / 2 - center[1]
    out = cv2.warpAffine(clean, M, (OUT_W, OUT_H), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT)
    out = fill_edges(out, M, img.shape)
    name = os.path.splitext(os.path.basename(path))[0] + '.jpg'
    cv2.imwrite(os.path.join(out_dir, name), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dict(file=name, angle=angle, scale=scale, dust=n_dust, box=box)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else '.'
    dst = sys.argv[2] if len(sys.argv) > 2 else 'aligned'
    os.makedirs(dst, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src, '*.jpg'))):
        r = process(p, dst)
        print(f"{r['file']:28s} rotate {r['angle']:+6.2f}°  scale {r['scale']:.3f}  dust specks {r['dust']}")


if __name__ == '__main__':
    main()
