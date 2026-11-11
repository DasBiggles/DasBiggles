"""Align comic photos to a consistent framing on a clean black background.

Geometry: only rotation + uniform scale + shift + crop (a similarity
transform) - no perspective correction or warping.

Every page is scaled to the same height and centred (spine on the centre
line). The crop is the same size for every photo: tight around the pages and
the clear clips, with an even margin all round.

Background: the mat is taken down to pure black, so dust specks vanish.
Nothing is painted in or copied; the comic and the clips are kept as shot.

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

PAGE_H = 1770        # paper height (px) in every output image
MARGIN = 90          # gap (px) between the furthest clip/page edge and the frame
DETECT_SIZE = 1000   # long side used for detection
EXTS = ('.jpg', '.jpeg', '.png')


def find_paper(img):
    """Return (center, angle_deg, height, box, hull) of the paper, full-res px.

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
    hull = cv2.convexHull(cv2.findNonZero(band))
    box = cv2.boxPoints(cv2.minAreaRect(hull)) / s

    def tilt(i):
        d = box[(i + 1) % 4] - box[i]
        return math.degrees(math.atan2(abs(d[1]), abs(d[0])))
    i = min(range(4), key=tilt)
    a, b = box[i], box[(i + 1) % 4]
    if b[0] < a[0]:
        a, b = b, a
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    height = float(np.linalg.norm(box[(i + 2) % 4] - box[(i + 1) % 4]))
    center = box.mean(0)
    full_hull = cv2.convexHull(cv2.findNonZero(m)).reshape(-1, 2) / s
    return (float(center[0]), float(center[1])), angle, height, box, full_hull


def object_mask(img, box, hull):
    """Mask of everything that is not mat: the paper and the clear clips."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(paper, [np.int32(hull)], 255)
    cv2.fillPoly(paper, [np.int32(box)], 255)
    near = cv2.dilate(paper, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (401, 401)))
    bright = np.uint8(cv2.GaussianBlur(gray, (0, 0), 2) > 60) * 255
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] > 1500 and near[lab == k].any():
            keep[k] = True
    obj = paper | (np.uint8(keep[lab]) * 255)
    # fill the clips' see-through areas so they count as one object
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
    return obj, paper


def mat_level(img, obj):
    """Brightness of the mat (the level that should become black)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    far = cv2.dilate(obj, np.ones((61, 61), np.uint8)) == 0
    vals = gray[far]
    return float(np.percentile(vals, 99.5)) if vals.size else 40.0


def blacken(img, obj, black_pt, knee=35.0):
    """Pure black outside the object; inside, a shadow curve takes the mat
    (seen through the clear plastic) down to black, leaving mid-tones and
    highlights untouched."""
    f = img.astype(np.float32)
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t = np.clip((lum - black_pt) / knee, 0, 1)
    target = np.where(lum < black_pt + knee,
                      (black_pt + knee) * t * t * (3 - 2 * t) * (lum / (black_pt + knee)),
                      lum)
    ratio = np.where(lum > 0, target / np.maximum(lum, 1e-3), 0)
    f *= ratio[..., None]
    keep = cv2.dilate(obj, np.ones((15, 15), np.uint8))
    keep = cv2.GaussianBlur(keep.astype(np.float32) / 255, (0, 0), 5)
    f *= keep[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def analyse(path):
    img = load(path)
    center, angle, h, box, hull = find_paper(img)
    obj, paper = object_mask(img, box, hull)
    scale = PAGE_H / h
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] -= center[0]                       # page centre -> (0, 0)
    M[1, 2] -= center[1]
    pts = cv2.findNonZero(obj).reshape(-1, 2).astype(np.float64)
    pts = pts[:: max(1, len(pts) // 200000)]
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
    ext = (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())
    return dict(path=path, img=None, M=M, ext=ext, scale=scale, angle=angle)


def render(info, W, H, out_dir, name):
    img = load(info['path'])
    center, angle, h, box, hull = find_paper(img)
    obj, _ = object_mask(img, box, hull)
    clean = blacken(img, obj, mat_level(img, obj))
    M = info['M'].copy()
    M[0, 2] += W / 2
    M[1, 2] += H / 2
    out = cv2.warpAffine(clean, M, (W, H), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cv2.imwrite(os.path.join(out_dir, name), out, [cv2.IMWRITE_JPEG_QUALITY, 93])


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else '.'
    dst = sys.argv[2] if len(sys.argv) > 2 else 'aligned'
    os.makedirs(dst, exist_ok=True)
    paths = sorted((p for p in glob.glob(os.path.join(src, '*'))
                    if p.lower().endswith(EXTS)),
                   key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    infos = [analyse(p) for p in paths]
    # one frame size for all: fits the widest clips / tallest page, plus margin
    half_w = max(max(-i['ext'][0], i['ext'][1]) for i in infos) + MARGIN
    half_h = max(max(-i['ext'][2], i['ext'][3]) for i in infos) + MARGIN
    W, H = 2 * int(math.ceil(half_w)), 2 * int(math.ceil(half_h))
    print(f'frame {W}x{H}')
    for i in infos:
        name = os.path.splitext(os.path.basename(i['path']))[0] + '.jpg'
        render(i, W, H, dst, name)
        e = i['ext']
        print(f"{name:28s} rotate {i['angle']:+6.2f}°  scale {i['scale']:.3f}  "
              f"extent x {e[0]:.0f}..{e[1]:.0f} y {e[2]:.0f}..{e[3]:.0f}")


if __name__ == '__main__':
    main()
