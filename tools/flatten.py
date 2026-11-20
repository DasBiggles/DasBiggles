"""Flatten comic photos into scan-like images of each page.

Spreads are split into a left and a right page; single photos (covers)
give one page. For each page:
  1. find its four corners (outer corners + both ends of the spine),
  2. undo the camera angle with a perspective transform,
  3. straighten the curved top and bottom edges (page curl near the spine)
     by stretching each column so the page edges become straight lines,
  4. crop exactly to the paper.

Usage: python3 tools/flatten.py <photo folder> <output folder>
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from detect import load, paper_mask  # noqa: E402

PAGE_H = 2400
ASPECT = 0.647          # page width / height, measured from the flat covers
PAGE_W = round(PAGE_H * ASPECT)
S = 0.5                 # detection scale
EXTS = ('.jpg', '.jpeg', '.png')


def edge_curves(m):
    """Top and bottom paper edge per column (NaN where no paper)."""
    has = m.any(0)
    top = np.where(has, m.argmax(0), np.nan).astype(float)
    bot = np.where(has, m.shape[0] - 1 - m[::-1].argmax(0), np.nan).astype(float)
    return top, bot


def fit_side(m, rows, side):
    """Fit a straight line x = a*y + b to the left/right paper edge, using only
    rows where the clips don't cover the edge."""
    ys, xs = [], []
    for y in rows:
        xx = np.flatnonzero(m[y])
        if xx.size:
            ys.append(y)
            xs.append(xx[0] if side == 'left' else xx[-1])
    ys, xs = np.array(ys, float), np.array(xs, float)
    keep = np.ones(len(ys), bool)
    for _ in range(3):
        a, b = np.polyfit(ys[keep], xs[keep], 1)
        r = np.abs(xs - (a * ys + b))
        keep = r < max(2.0, 2.5 * np.median(r[keep]))
    return a, b


def smooth(c, k=9):
    c = c.copy()
    ok = ~np.isnan(c)
    idx = np.arange(len(c))
    c[~ok] = np.interp(idx[~ok], idx[ok], c[ok])
    return cv2.medianBlur(c.astype(np.float32)[None, :], 5).ravel().astype(float)


def robust_edge(curve, x0, x1, inward, deg=2):
    """Fit a smooth curve to one page edge between x0 and x1. Dark artwork
    touching the edge makes notches that point into the page; those points
    are dropped (``inward`` = +1 for the top edge, -1 for the bottom)."""
    xs = np.arange(int(x0), int(x1))
    ys = curve[xs]
    ok = ~np.isnan(ys)
    xs, ys = xs[ok], ys[ok]
    keep = np.ones(len(xs), bool)
    for _ in range(6):
        c = np.polyfit(xs[keep], ys[keep], deg)
        r = (ys - np.polyval(c, xs)) * inward
        keep = r < max(1.5, 2.0 * np.median(np.abs(r[keep])))
    return c


def fitted(curve, xs_split, xl, xr, inward):
    """Edge curve rebuilt from smooth fits (one per page, split at the spine)."""
    out = np.full(len(curve), np.nan)
    pad = 6
    segs = [(xl, xr)] if xs_split is None else [(xl, xs_split), (xs_split, xr)]
    for i, (a, b) in enumerate(segs):
        c = robust_edge(curve, max(a, 0) + pad, min(b, len(curve)) - pad, inward)
        lo = 0 if i == 0 else int(a)
        hi = len(curve) if i == len(segs) - 1 else int(b)
        out[lo:hi] = np.polyval(c, np.arange(lo, hi))
    return out


def corner_on_curve(curve, a, b):
    """Where the straight side x = a*y + b meets the edge curve y = curve(x)."""
    x = b
    for _ in range(20):
        y = np.interp(x, np.arange(len(curve)), curve)
        x = a * y + b
    return np.array([x, np.interp(x, np.arange(len(curve)), curve)])


def find_spine(curve, xl, xr):
    """Spine = sharpest kink of the edge curve near the middle of the spread."""
    x0, x1 = int(xl + 0.3 * (xr - xl)), int(xl + 0.7 * (xr - xl))
    xs = np.arange(x0, x1)
    best, bx = -1, x0
    w = int(0.06 * (xr - xl))
    for x in xs:
        l = np.polyfit(np.arange(x - w, x), curve[x - w:x], 1)[0]
        r = np.polyfit(np.arange(x, x + w), curve[x:x + w], 1)[0]
        k = abs(l - r)
        if k > best:
            best, bx = k, x
    return np.array([bx, curve[bx]], float)


def flatten_page(img, top, bot, tl, tr, br, bl):
    """Map the page bounded by the edge curves and corners to a flat rectangle.
    tl..bl are corners in detection-scale px; top/bot are edge curves."""
    src = np.float32([tl, tr, br, bl]) / S
    dst = np.float32([[0, 0], [PAGE_W, 0], [PAGE_W, PAGE_H], [0, PAGE_H]])
    H = cv2.getPerspectiveTransform(src, dst)
    Hinv = np.linalg.inv(H)

    def rectify(curve, x0, x1):
        xs = np.linspace(x0, x1, 400)
        ys = np.interp(xs, np.arange(len(curve)), curve)
        p = cv2.perspectiveTransform(np.float32(np.c_[xs, ys] / S)[None], H)[0]
        o = np.argsort(p[:, 0])
        return p[o, 0], p[o, 1]

    tx, ty = rectify(top, tl[0], tr[0])
    bx, by = rectify(bot, bl[0], br[0])
    u = np.arange(PAGE_W, dtype=np.float32) + 0.5
    yt = np.interp(u, tx, ty)          # where the curved top edge sits, per column
    yb = np.interp(u, bx, by)
    v = np.arange(PAGE_H, dtype=np.float32)[:, None] + 0.5
    # stretch each column so the curved edges become straight
    ry = yt[None, :] + (yb - yt)[None, :] * v / PAGE_H
    rx = np.broadcast_to(u[None, :], ry.shape)
    pts = np.stack([rx, ry], -1).reshape(-1, 1, 2).astype(np.float32)
    srcpts = cv2.perspectiveTransform(pts, Hinv).reshape(PAGE_H, PAGE_W, 2)
    return cv2.remap(img, srcpts[..., 0], srcpts[..., 1], cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REPLICATE)


def process(path):
    img = load(path)
    _, m, _ = paper_mask(img, S)
    m = m > 0
    ys = np.flatnonzero(m.any(1))
    y0, y1 = ys[0], ys[-1]
    h = y1 - y0
    rows = list(range(int(y0 + 0.04 * h), int(y0 + 0.15 * h))) + \
        list(range(int(y1 - 0.13 * h), int(y1 - 0.04 * h)))
    al, bl_ = fit_side(m, rows, 'left')
    ar, br_ = fit_side(m, rows, 'right')
    top, bot = edge_curves(m)
    xl, xr = int(bl_ + al * y0), int(br_ + ar * y0)
    # only trust the edge curves between the sides
    for c in (top, bot):
        c[:max(xl, 0)] = np.nan
        c[min(xr, len(c) - 1):] = np.nan
    top, bot = smooth(top), smooth(bot)
    single = (xr - xl) < 1.0 * h      # a cover: one page, no spine
    if single:
        top, bot = fitted(top, None, xl, xr, 1), fitted(bot, None, xl, xr, -1)
    else:
        st, sb = find_spine(top, xl, xr), find_spine(bot, xl, xr)
        top = fitted(top, int(st[0]), xl, xr, 1)
        bot = fitted(bot, int(sb[0]), xl, xr, -1)
    TL, BL = corner_on_curve(top, al, bl_), corner_on_curve(bot, al, bl_)
    TR, BR = corner_on_curve(top, ar, br_), corner_on_curve(bot, ar, br_)
    if single:
        return [flatten_page(img, top, bot, TL, TR, BR, BL)], dict(corners=[TL, TR, BR, BL])
    ST = np.array([st[0], top[int(st[0])]])
    SB = np.array([sb[0], bot[int(sb[0])]])
    left = flatten_page(img, top, bot, TL, ST, SB, BL)
    right = flatten_page(img, top, bot, ST, TR, BR, SB)
    return [left, right], dict(corners=[TL, ST, TR, BR, SB, BL])


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    paths = sorted((p for p in glob.glob(os.path.join(src, '*')) if p.lower().endswith(EXTS)),
                   key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        pages, _ = process(p)
        names = [f'{stem}.jpg'] if len(pages) == 1 else [f'{stem}-left.jpg', f'{stem}-right.jpg']
        for n, pg in zip(names, pages):
            cv2.imwrite(os.path.join(dst, n), pg, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(stem, '->', ', '.join(names))


if __name__ == '__main__':
    main()
